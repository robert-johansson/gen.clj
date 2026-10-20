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
   - The potential's weights: along random constrained walks, logw-next is
     prefix(ctx ++ [t]) - prefix(ctx) at every token, complete - prefix at
     the end, and complete is the language's membership of the text.
   - Misuse is loud: an empty language, a vocabulary that cannot spell the
     language.

   Mutations checked: see docs/research/llm-grammar.md, Tests."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [gen.llm.branching-contract :as bc]
            [gen.llm.bytes :as bytes]
            [gen.llm.dfa :as dfa]
            [gen.llm.grammar :as grammar]
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
  (gen/recursive-gen
   (fn [inner]
     (gen/one-of [(gen/fmap #(apply regex/cat %) (gen/vector inner 2))
                  (gen/fmap #(apply regex/alt %) (gen/vector inner 2))
                  (gen/fmap regex/star inner)
                  (gen/fmap regex/opt inner)
                  (gen/fmap #(apply regex/intersect %) (gen/vector inner 2))]))
   (gen/fmap regex/bytes-of (gen/set (gen/choose 0 3) {:min-elements 1 :max-elements 2}))))

(def ^:private gen-byte-vocab
  "Tokens of up to four bytes 0 to 4, an empty one now and then."
  (gen/fmap #(mapv (fn [bs] (byte-array (map unchecked-byte bs))) %)
            (gen/vector (gen/vector (gen/choose 0 4) 0 4) 1 24)))

(def ^:private text-chars [\a \b \é \€])

(def ^:private gen-char-term
  "Terms over a, b, é (two bytes) and € (three)."
  (gen/recursive-gen
   (fn [inner]
     (gen/one-of [(gen/fmap #(apply regex/cat %) (gen/vector inner 2 3))
                  (gen/fmap #(apply regex/alt %) (gen/vector inner 2))
                  (gen/fmap regex/star inner)
                  (gen/fmap regex/opt inner)
                  (gen/let [r inner lo (gen/choose 0 2) n (gen/choose 0 2)] (regex/repeat r lo (+ lo n)))]))
   (gen/one-of [(gen/fmap regex/chars (gen/elements text-chars))
                (gen/fmap #(apply regex/chars %) (gen/set (gen/elements text-chars) {:min-elements 1}))])))

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
  "Every token walked from every state: the states tokens reach from the
   start, those of them from which tokens reach acceptance (a fixed point),
   and each such state's allowed tokens."
  [d vocab]
  (let [walk (fn [s t] (dfa/walk d s (vocab t)))
        tokens (range (count vocab))
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

(defn- masks-match? [d vocab]
  (let [{:keys [live masks reachable]} (brute d vocab)]
    (cond (zero? (dfa/size d)) (= :empty-language (bc/reason #(grammar/grammar vocab d)))
          (not (contains? live (dfa/start d))) (= :inexpressible (bc/reason #(grammar/grammar vocab d)))
          :else (let [g (grammar/grammar vocab d)]
                  (and (= (count reachable) (:reachable (grammar/info g)))
                       (= (count live) (:live (grammar/info g)))
                       (every? (fn [s] (and (= (grammar/live? g s) (contains? live s))
                                            (= (grammar/allowed g s) (get masks s []))
                                            (every? #(= (grammar/allowed? g s %) (boolean (some #{%} (get masks s []))))
                                                    (range (count vocab)))))
                               (range -1 (dfa/size d))))))))

(defspec byte-masks-match-brute-force 300
  (prop/for-all [r gen-byte-term vocab gen-byte-vocab]
    (masks-match? (regex/dfa r) vocab)))

(defspec text-masks-match-brute-force 300
  (prop/for-all [r gen-char-term vocab gen-text-vocab]
    (masks-match? (regex/dfa r) vocab)))

(deftest a-token-into-a-state-tokens-cannot-complete-is-not-allowed
  ;; "ab" or "ac": the vocabulary spells "ab" but no token ends "c", so the
  ;; byte-live state after "a" is reached by "a" and then "b" only; the token
  ;; "ac?" never exists, and the token "a" stays allowed (b completes it)
  (let [vocab (mapv utf8 ["a" "b" "x"])
        g (grammar/grammar vocab "ab|ac|xc")]
    (is (= [0] (grammar/allowed g 0)) "x leads to a byte-live state no token completes")
    (is (= 0.0 (grammar/prefix g [0])))
    (is (= Double/NEGATIVE_INFINITY (grammar/prefix g [2])))
    (is (= 0.0 (grammar/complete g [0 1])))))

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

(defspec the-weights-are-consistent 200
  (prop/for-all [r gen-char-term vocab gen-text-vocab seed gen/nat]
    (let [d (regex/dfa r)
          g (try (grammar/grammar vocab d) (catch clojure.lang.ExceptionInfo _ nil))]
      (or (nil? g)
          (every? (fn [ctx]
                    (let [p (grammar/prefix g ctx)
                          ^floats row (grammar/logw-next g ctx)
                          n (count vocab)]
                      (and (= 0.0 p)
                           (= (grammar/complete g ctx) (if (dfa/matches? d (concat-bytes vocab ctx)) 0.0 Double/NEGATIVE_INFINITY))
                           (= (inc n) (alength row))
                           (every? #(= (double (aget row %)) (- (grammar/prefix g (conj ctx %)) p)) (range n))
                           (= (double (aget row n)) (- (grammar/complete g ctx) p))
                           (= (grammar/state g ctx) (dfa/walk d (dfa/start d) (concat-bytes vocab ctx))))))
                  (map #(walk-allowed g % (+ seed %)) (range 6)))))))

(deftest a-dead-context-has-no-next-weights
  (let [g (grammar/grammar (mapv utf8 ["a" "b"]) "a+")]
    (is (= Double/NEGATIVE_INFINITY (grammar/prefix g [0 1])))
    (is (= Double/NEGATIVE_INFINITY (grammar/complete g [1])))
    (is (= -1 (grammar/state g [1 0])))
    (is (= :zero-prefix (bc/reason #(grammar/logw-next g [1]))))
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
    (is (= :invalid-regex (bc/reason #(grammar/grammar ascii "a("))))))
