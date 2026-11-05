(ns gen.llm.grammar-test
  "Grammars (gen.llm.grammar) and the token byte trie under their masks
   (gen.llm.bytes), against brute force.

   - The trie: each node's tokens, its subtree's, its children, against the
     vocabulary scanned token by token.
   - Masks: for random languages and random vocabularies, of bytes and of
     pieces of UTF-8 text that split characters, every DFA state's allowed
     tokens, liveness and reachability equal brute force's, which walks
     every token from every state and finds the states tokens can complete
     by a fixed point.
   - The potential's weights: the potential contract's laws
     (gen.laws.potential) over random grammars, and along random
     constrained walks, prefix is 0, complete is the language's membership
     of the text and the grammar's state is the DFA's.
   - A random language whose DFA passes :max-states is skipped, as a
     grammar that cannot be built is.
   - Misuse is loud: an empty language, a vocabulary that cannot spell the
     language.

   Mutations checked: see docs/research/llm-grammar.md, Tests."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.test.check :as tc]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [gen.llm.branching-contract :as bc]
            [gen.llm.bytes :as bytes]
            [gen.laws.potential :as potential-laws]
            [gen.llm.dfa :as dfa]
            [gen.llm.grammar :as grammar]
            [gen.llm.potential :as potential]
            [gen.llm.regex :as regex])
  (:import (java.nio.charset StandardCharsets)))

(defn- utf8 ^bytes [^String s] (.getBytes s StandardCharsets/UTF_8))

(defn- concat-bytes ^bytes [vocab tokens]
  (byte-array (mapcat #(seq ^bytes (vocab %)) tokens)))

;; ---------------------------------------------------------------------------
;; Random languages and vocabularies
;; ---------------------------------------------------------------------------

(def ^:private gen-byte-term
  "Terms over the bytes 0 to 3."
  (gen/scale #(min % 50)
   (gen/recursive-gen
    (fn [inner]
      (gen/one-of [(gen/fmap #(apply regex/cat %) (gen/vector inner 2))
                   (gen/fmap #(apply regex/alt %) (gen/vector inner 2))
                   (gen/fmap regex/star inner)
                   (gen/fmap regex/opt inner)
                   (gen/fmap #(apply regex/intersect %) (gen/vector inner 2))]))
    (gen/fmap regex/bytes-of (gen/set (gen/choose 0 3) {:min-elements 1 :max-elements 2})))))

(def ^:private gen-byte-vocab
  "Tokens of up to four bytes 0 to 4, an empty one now and then."
  (gen/fmap #(mapv (fn [bs] (byte-array (map unchecked-byte bs))) %)
            (gen/vector (gen/vector (gen/choose 0 4) 0 4) 1 24)))

(def ^:private text-chars [\a \b \é \€])

(def ^:private gen-char-term
  "Terms over a, b, é (two bytes) and € (three)."
  (gen/scale #(min % 50)
   (gen/recursive-gen
    (fn [inner]
      (gen/one-of [(gen/fmap #(apply regex/cat %) (gen/vector inner 2 3))
                   (gen/fmap #(apply regex/alt %) (gen/vector inner 2))
                   (gen/fmap regex/star inner)
                   (gen/fmap regex/opt inner)
                   (gen/let [r inner lo (gen/choose 0 2) n (gen/choose 0 2)] (regex/repeat r lo (+ lo n)))]))
    (gen/one-of [(gen/fmap regex/chars (gen/elements text-chars))
                 (gen/fmap #(apply regex/chars %) (gen/set (gen/elements text-chars) {:min-elements 1}))]))))

(defn- distinct-by-content [arrays]
  (vals (into (sorted-map) (map (fn [^bytes a] [(vec a) a])) arrays)))

(def ^:private gen-text-vocab
  "Pieces of UTF-8 text over a, b, é and €: byte ranges of random strings,
   so that tokens split characters and span several."
  (gen/let [texts (gen/vector (gen/fmap #(apply str %) (gen/vector (gen/elements text-chars) 1 4)) 1 6)
            cuts (gen/vector (gen/tuple (gen/choose 0 5) gen/nat gen/nat) 4 30)]
    (vec (distinct-by-content
          (for [[i a b] cuts
                :let [^bytes bs (utf8 (texts (mod i (count texts))))
                      lo (mod a (alength bs))
                      hi (+ lo 1 (mod b (- (alength bs) lo)))]]
            (java.util.Arrays/copyOfRange bs (int lo) (int hi)))))))

(defn- dfa-within-limit
  "r's DFA, or nil when it passes regex/dfa's :max-states, the documented
   loud error that random terms reach now and then (nested counted repeats
   of overlapping alternatives, genclj-z4jy). The properties are of the
   languages that have a DFA, as they are of the grammars that can be
   built."
  [r]
  (try (regex/dfa r)
       (catch clojure.lang.ExceptionInfo e
         (when-not (= :too-many-states (:reason (ex-data e)))
           (throw e)))))

;; ---------------------------------------------------------------------------
;; The trie against the vocabulary
;; ---------------------------------------------------------------------------

(defn- paths
  "Every node of trie with its path: [node bytes] pairs."
  [trie]
  (loop [todo [[(bytes/root trie) []]] out []]
    (if-let [[node path] (peek todo)]
      (recur (into (pop todo) (map (fn [[b c]] [c (conj path b)])) (bytes/children trie node))
             (conj out [node path]))
      out)))

(defspec the-trie-holds-the-vocabulary 200
  (prop/for-all [vocab (gen/one-of [gen-byte-vocab gen-text-vocab])]
    (let [trie (bytes/byte-trie vocab)
          unsigned (fn [^bytes bs] (mapv #(bit-and (long %) 0xff) bs))
          tokens (map-indexed (fn [t bs] [t (unsigned bs)]) vocab)
          nodes (paths trie)]
      (and (every? (fn [[node path]]
                     (and (= (bytes/depth trie node) (count path))
                          (= (bytes/tokens-at trie node) (sort (keep (fn [[t bs]] (when (and (seq bs) (= bs path)) t)) tokens)))
                          (= (set (bytes/tokens-under trie node))
                             (set (keep (fn [[t bs]] (when (and (seq bs) (= path (take (count path) bs))) t)) tokens)))
                          (every? (fn [[b c]] (= c (bytes/child trie node b))) (bytes/children trie node))
                          (every? #(= -1 (bytes/child trie node %))
                                  (remove (set (map first (bytes/children trie node))) (range 256)))))
                   nodes)
           (= (count nodes) (count (distinct (mapcat (fn [[_ bs]] (map #(subvec bs 0 %) (range (inc (count bs))))) tokens))))
           (= (bytes/bare trie) (keep (fn [[t bs]] (when (empty? bs) t)) tokens))
           (every? (fn [[t bs]] (= bs (unsigned (bytes/token-bytes trie t)))) tokens)))))

;; ---------------------------------------------------------------------------
;; Masks against brute force
;; ---------------------------------------------------------------------------

(defn- brute
  "Every token walked from every state, but the end tokens, which are no
   text: the states tokens reach from the start, those of them from which
   tokens reach acceptance (a fixed point), and each such state's allowed
   tokens."
  [d vocab end-tokens]
  (let [walk (fn [s t] (dfa/walk d s (vocab t)))
        tokens (remove (set end-tokens) (range (count vocab)))
        reachable (loop [seen #{(dfa/start d)} todo [(dfa/start d)]]
                    (if-let [s (peek todo)]
                      (let [new (remove seen (filter #(<= 0 %) (map #(walk s %) tokens)))]
                        (recur (into seen new) (into (pop todo) new)))
                      seen))
        live (loop [live (set (filter #(dfa/accepting? d %) reachable))]
               (let [more (into live (filter (fn [s] (some #(contains? live (walk s %)) tokens))) reachable)]
                 (if (= more live) live (recur more))))]
    {:reachable reachable
     :live live
     :masks (into {} (map (fn [s] [s (vec (filter #(contains? live (walk s %)) tokens))])) live)}))

(defn- masks-match?
  ([d vocab] (masks-match? d vocab #{}))
  ([d vocab end-tokens]
   (let [{:keys [live masks reachable]} (brute d vocab end-tokens)
         make #(grammar/grammar vocab d {:end-tokens end-tokens})]
     (cond (zero? (dfa/size d)) (= :empty-language (bc/reason make))
           (not (contains? live (dfa/start d))) (= :inexpressible (bc/reason make))
           :else (let [g (make)]
                   (and (= (count reachable) (:reachable (grammar/info g)))
                        (= (count live) (:live (grammar/info g)))
                        (every? (fn [s] (and (= (grammar/live? g s) (contains? live s))
                                             (= (grammar/allowed g s) (get masks s []))
                                             (every? #(= (grammar/allowed? g s %) (boolean (some #{%} (get masks s []))))
                                                     (range (count vocab)))
                                             (every? #(= -1 (grammar/advance g s %)) end-tokens)))
                                (range -1 (dfa/size d)))))))))

(defn- gen-end-tokens
  "Up to two of a vocabulary's tokens, as end tokens."
  [vocab]
  (gen/set (gen/choose 0 (dec (count vocab))) {:max-elements 2}))

(defspec byte-masks-match-brute-force 300
  (prop/for-all [r gen-byte-term
                 [vocab ends] (gen/bind gen-byte-vocab #(gen/tuple (gen/return %) (gen-end-tokens %)))]
    (let [d (dfa-within-limit r)] (or (nil? d) (masks-match? d vocab ends)))))

(defspec text-masks-match-brute-force 300
  (prop/for-all [r gen-char-term
                 [vocab ends] (gen/bind gen-text-vocab #(gen/tuple (gen/return %) (gen-end-tokens %)))]
    (let [d (dfa-within-limit r)] (or (nil? d) (masks-match? d vocab ends)))))

(deftest masks-over-a-vocabulary-of-many-words
  ;; masks are bitsets of 64-token words: a vocabulary of 97 one-character
  ;; tokens (the toy ASCII) and one of 300 pieces, against brute force
  (let [ascii (mapv #(utf8 (str (char %))) (range 32 127))
        pieces (vec (distinct-by-content (for [s ["ab" "ba" "aé" "€b" "abab" "éé"] i (range (count (utf8 s))) j (range (inc i) (inc (count (utf8 s))))]
                                           (java.util.Arrays/copyOfRange ^bytes (utf8 s) (int i) (int j)))))
        many (into ascii (map #(utf8 (str "<" % ">"))) (range 200))]
    (is (< 64 (count ascii)))
    (doseq [[re vocab] [["[a-c]+(-[0-9]{1,2})?" ascii] ["[^\"]*\"" ascii] ["(ab|é)+€?" (into many pieces)]
                        ["<1[0-9]{2}>(x|<5>)*" many]]]
      (is (masks-match? (regex/dfa re) vocab) re))))

(deftest a-token-into-a-state-tokens-cannot-complete-is-not-allowed
  ;; "ab" or "ac": the vocabulary spells "ab" but no token ends "c", so the
  ;; byte-live state after "a" is reached by "a" and then "b" only; the token
  ;; "ac?" never exists, and the token "a" stays allowed (b completes it)
  (let [vocab (mapv utf8 ["a" "b" "x"])
        g (grammar/grammar vocab "ab|ac|xc")]
    (is (= [0] (grammar/allowed g 0)) "x leads to a byte-live state no token completes")
    (is (= 0.0 (potential/prefix g [0])))
    (is (= Double/NEGATIVE_INFINITY (potential/prefix g [2])))
    (is (= 0.0 (potential/complete g [0 1])))))

;; ---------------------------------------------------------------------------
;; The potential's weights
;; ---------------------------------------------------------------------------

(defn- walk-allowed
  "A context of up to k tokens, each allowed after those before it (from
   seed)."
  [g ^long k ^long seed]
  (let [rnd (java.util.Random. seed)]
    (loop [ctx [] s (dfa/start (grammar/automaton g))]
      (let [ts (grammar/allowed g s)]
        (if (or (== k (count ctx)) (empty? ts))
          ctx
          (let [t (ts (.nextInt rnd (count ts)))]
            (recur (conj ctx t) (grammar/advance g s t))))))))

(defn- grammar-of
  "r's grammar over vocab, or nil when r has no DFA within :max-states or
   the grammar cannot be built (the vocabulary cannot spell the language)."
  [r vocab]
  (when-let [d (dfa-within-limit r)]
    (try (grammar/grammar vocab d) (catch clojure.lang.ExceptionInfo _ nil))))

(def ^:private gen-grammars
  "Entries [label grammar n-vocab] of the potential laws: random languages
   over random text vocabularies, those whose grammars can be built (about
   nine in ten), labelled by the language and the vocabulary's bytes."
  (gen/such-that some?
                 (gen/fmap (fn [[r vocab]]
                             (when-let [g (grammar-of r vocab)]
                               [{:language r :vocab (mapv vec vocab)} g (count vocab)]))
                           (gen/tuple gen-char-term gen-text-vocab))
                 50))

;; the potential contract: the weights are 0 and -Inf, so the laws'
;; tolerance does not matter
(potential-laws/deflaws grammar-weights gen-grammars 200)

(defn- weights-are-the-language?
  "Along walks from seed, the grammar of r over vocab: prefix 0, complete
   the language's membership of the text, the state the DFA's."
  [r vocab seed]
  (let [d (dfa-within-limit r)
        g (when d (grammar-of r vocab))
        contexts (when g (map #(walk-allowed g % (+ seed %)) (range 6)))]
    (or (nil? g)
        (every? (fn [ctx]
                  (and (= 0.0 (potential/prefix g ctx))
                       (= (potential/complete g ctx) (if (dfa/matches? d (concat-bytes vocab ctx)) 0.0 Double/NEGATIVE_INFINITY))
                       (= (grammar/state g ctx) (dfa/walk d (dfa/start d) (concat-bytes vocab ctx)))))
                contexts))))

(def ^:private weights-property
  (prop/for-all [r gen-char-term vocab gen-text-vocab seed gen/nat]
    (weights-are-the-language? r vocab seed)))

(defspec the-weights-are-the-language 200 weights-property)

(deftest a-language-past-max-states-is-an-expected-outcome
  ;; genclj-z4jy: the-weights-are-consistent (now the-weights-are-the-
  ;; language and the grammar-weights laws) drew a term whose DFA passes
  ;; :max-states (the shrunk one below: 25,292 derivatives, 3,397 states
  ;; minimal), and the uncaught :too-many-states failed the suite at
  ;; random, at 3 seeds in 400 of its 200 trials; text-masks-match-brute-
  ;; force draws from the same terms. Seed 128 draws one at its 61st trial.
  ;; Planted bug (2026-10-22): dfa-within-limit rethrowing every error, as
  ;; the bare regex/dfa did, fails the last three
  (let [a (regex/chars \a)
        a-or-é (regex/chars \a \é)
        shrunk (regex/repeat (regex/cat a a-or-é a-or-é (regex/repeat (regex/cat a-or-é a-or-é a-or-é) 1 2) (regex/star a) a-or-é)
                             2 4)]
    (is (= :too-many-states (bc/reason #(regex/dfa shrunk))) "the term still passes the limit")
    (is (nil? (dfa-within-limit shrunk)))
    (is (weights-are-the-language? shrunk [(utf8 "a")] 0))
    (is (:pass? (tc/quick-check 61 weights-property :seed 128)))))

(deftest a-dead-context-has-no-next-weights
  (let [g (grammar/grammar (mapv utf8 ["a" "b"]) "a+")]
    (is (= Double/NEGATIVE_INFINITY (potential/prefix g [0 1])))
    (is (= Double/NEGATIVE_INFINITY (potential/complete g [1])))
    (is (= -1 (grammar/state g [1 0])))
    (is (= :zero-prefix (bc/reason #(potential/logw-next g [1]))))
    (is (not (grammar/allowed? g -1 0)))
    (is (= [] (grammar/allowed g -1)))))

;; ---------------------------------------------------------------------------
;; Misuse
;; ---------------------------------------------------------------------------

(deftest misuse-is-loud
  (let [ascii (mapv #(utf8 (str (char %))) (range 32 127))]
    (is (= :empty-language (bc/reason #(grammar/grammar ascii (regex/intersect (regex/text "a") (regex/text "b"))))))
    (is (= :inexpressible (bc/reason #(grammar/grammar ascii "é+"))) "no token holds é's bytes")
    (is (= :inexpressible (bc/reason #(grammar/grammar (mapv utf8 ["é" "b"]) "a|ab"))) "no token is a")
    (is (= :invalid-vocab (bc/reason #(grammar/grammar [] "a"))))
    (is (= :invalid-vocab (bc/reason #(grammar/grammar ["a"] "a"))))
    (is (= :invalid-regex (bc/reason #(grammar/grammar ascii "a("))))
    (is (= :invalid-end-tokens (bc/reason #(grammar/grammar ascii "a" {:end-tokens #{100000000000000000000N}})))
        "an end token past long's range (once an IllegalArgumentException: the post-M5 review's I4-03)")))
