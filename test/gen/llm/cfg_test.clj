(ns gen.llm.cfg-test
  "Context-free grammars by Earley (gen.llm.cfg), against an oracle that
   shares nothing with it: for random grammars over a and b, the language
   D(A) and the prefix language P(A) of each rule up to five characters, by
   fixpoints over sets of strings (P(A) through the rules whose every
   symbol derives some string, so that a prefix is one of a string).

   - recognizes? and prefix? agree with D and P on every text of up to
     five characters, for 200 random grammars (nullable rules, left and
     right recursion, unproductive rules, string and regex terminals).
   - As a potential: the potential contract's laws over random grammars
     read through a vocabulary of pieces; on a regular language, the same
     weights as gen.llm.grammar's exact masks (a vocabulary with every
     byte); completions under balanced brackets are balanced.
   - Misuse is loud.

   Planted bugs (2026-11-05, each applied, the suite run, reverted): no
   skip of a nullable nonterminal (Aycock and Horspool's), and completion
   from the current set in place of the item's origin; each fails the
   oracle property."
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [gen.gfi :as gfi]
            [gen.laws.potential :as potential-laws]
            [gen.llm.branching-contract :as bc]
            [gen.llm.byte-tokenizer :as bt]
            [gen.llm.cfg :as cfg]
            [gen.llm.grammar :as grammar]
            [gen.llm.lm :as lm]
            [gen.llm.potential :as potential]
            [gen.llm.regex :as regex]
            [gen.llm.token :as token]
            [gen.llm.toy-engine :as toy]
            [gen.random :as rng])
  (:import (java.nio.charset StandardCharsets)))

(def ^:private max-len 5)

(def ^:private texts
  "Every text over a and b of up to max-len characters."
  (loop [len 0 level [""] out []]
    (if (> len max-len)
      out
      (recur (inc len) (for [t level c ["a" "b"]] (str t c)) (into out level)))))

;; Terminals: strings, and regexes as [:re pattern] in the oracle's form
(def ^:private regexes {"[ab]" #{"a" "b"} "a+" (set (map #(apply str (repeat % "a")) (range 1 (inc max-len))))
                        "b(ab)*" (set (filter #(re-matches #"b(ab)*" %) texts))})

(defn- concat-sets
  "{x ++ y} for x of xs, y of ys, of up to max-len characters."
  [xs ys]
  (into #{} (for [x xs y ys :when (<= (+ (count x) (count y)) max-len)] (str x y))))

(defn- prefixes [s] (map #(subs s 0 %) (range (inc (count s)))))

(defn- oracle
  "D and P of rules (keyword → alternatives of keywords, strings and
   [:re pattern]), up to max-len characters."
  [rules]
  (let [d-of (fn [d e] (cond (keyword? e) (d e) (string? e) #{e} :else (regexes (second e))))
        d (loop [d (zipmap (keys rules) (repeat #{}))]
            (let [d' (into {} (for [[k alts] rules]
                                [k (into (d k) (mapcat (fn [alt] (reduce concat-sets #{""} (map #(d-of d %) alt))) alts))]))]
              (if (= d d') d (recur d'))))
        ;; productivity, unbounded: a rule derives some string (of any length)
        productive (loop [known #{}]
                     (let [known' (into known (keep (fn [[k alts]]
                                                      (when (some (fn [alt] (every? #(or (not (keyword? %)) (known %)) alt)) alts) k)))
                                        rules)]
                       (if (= known known') known (recur known'))))
        p-of (fn [p e] (cond (keyword? e) (p e) (string? e) (set (prefixes e)) :else (set (mapcat prefixes (regexes (second e))))))
        p (loop [p (zipmap (keys rules) (repeat #{}))]
            (let [p' (into {} (for [[k alts] rules]
                                [k (into (p k)
                                         (for [alt alts
                                               :when (every? #(or (not (keyword? %)) (productive %)) alt)
                                               j (range (count alt))
                                               x (concat-sets (reduce concat-sets #{""} (map #(d-of d %) (take j alt)))
                                                              (p-of p (nth alt j)))]
                                           x))]))
                  p' (into {} (for [[k xs] p'] [k (cond-> xs (productive k) (conj ""))]))]
              (if (= p p') p (recur p'))))]
    {:d d :p p}))

(defn- as-cfg [start rules]
  (cfg/cfg {:start start
            :rules (into {} (for [[k alts] rules]
                              [k (mapv (fn [alt] (mapv #(if (vector? %) (cfg/regex (second %)) %) alt)) alts)]))}))

(def ^:private gen-rules
  "Rules :r0 … :rk over a and b, any referring to any (recursion), some
   unproductive."
  (gen/let [k (gen/choose 1 3)
            names (gen/return (mapv #(keyword (str "r" %)) (range k)))
            rules (apply gen/tuple
                         (repeat k (gen/vector (gen/vector (gen/one-of [(gen/elements names)
                                                                        (gen/elements ["a" "b" "ab"])
                                                                        (gen/fmap (fn [r] [:re r]) (gen/elements (keys regexes)))])
                                                           0 3)
                                               1 3)))]
    (zipmap names rules)))

(defspec earley-agrees-with-the-oracle 200
  (prop/for-all [rules gen-rules]
    (let [{:keys [d p]} (oracle rules)]
      (if-not (seq (p :r0))
        ;; the start derives nothing: an empty language is loud
        (= :empty-language (bc/reason #(as-cfg :r0 rules)))
        (let [g (as-cfg :r0 rules)]
          (every? (fn [t] (and (= (contains? (d :r0) t) (cfg/recognizes? g t))
                               (= (contains? (p :r0) t) (cfg/prefix? g t))))
                  texts))))))

(deftest examples
  (let [brackets (cfg/cfg {:start :s :rules {:s [["(" :s ")" :s] []]}})]
    (is (= [true true true false false] (mapv #(cfg/recognizes? brackets %) ["" "()" "(())()" "(()" ")("])))
    (is (= [true true false false] (mapv #(cfg/prefix? brackets %) ["(((" "(()" "())" ")"]))))
  (testing "left recursion and a regex terminal"
    (let [sums (cfg/cfg {:start :e :rules {:e [[:e "+" :n] [:n]] :n [[(cfg/regex "[0-9]+")]]}})]
      (is (every? #(cfg/recognizes? sums %) ["1" "12+3" "1+2+30"]))
      (is (not-any? #(cfg/recognizes? sums %) ["" "+" "1+" "1++2"]))
      (is (cfg/prefix? sums "1+")))))

;; ---------------------------------------------------------------------------
;; As a potential
;; ---------------------------------------------------------------------------

(defn- utf8 ^bytes [^String s] (.getBytes s StandardCharsets/UTF_8))

(def ^:private pieces (mapv utf8 ["a" "b" "ab" "ba" "bb" "aab"]))

(def ^:private gen-potential-entries
  "Potentials of random grammars over the pieces, those whose language is
   not empty."
  (gen/such-that some?
                 (gen/fmap (fn [rules]
                             (when (seq ((:p (oracle rules)) :r0))
                               [rules (cfg/potential pieces (as-cfg :r0 rules)) (count pieces)]))
                           gen-rules)
                 50))

(potential-laws/deflaws cfg-weights gen-potential-entries 100)

(deftest a-regular-language-weighs-as-the-grammar-does
  ;; a vocabulary with every byte of the language as a token: the bytes'
  ;; liveness is the tokens', so the weights are the exact masks'
  (let [vocab (mapv utf8 ["a" "b" "ab" "ba" "bab"])
        regular (cfg/potential vocab (cfg/cfg {:start :s :rules {:s [["a" :s] ["b" :t]] :t [["ab"] [:s]]}}))
        dfa-grammar (grammar/grammar vocab "a*b(a*b)*ab")]
    (doseq [ctx [[] [0] [1] [0 1] [1 2] [4] [3 0] [1 0 1]]]
      (is (= (potential/prefix dfa-grammar ctx) (potential/prefix regular ctx)) (str "prefix " ctx))
      (is (= (potential/complete dfa-grammar ctx) (potential/complete regular ctx)) (str "complete " ctx))
      (when (zero? (potential/prefix dfa-grammar ctx))
        (is (= (vec (potential/logw-next dfa-grammar ctx)) (vec (potential/logw-next regular ctx))) (str "logw-next " ctx))))))

(deftest completions-under-brackets-are-balanced
  (let [tok (bt/tokenizer ["(" ")" "()" "((" "))" "x"])
        m (lm/lm (toy/engine {:n-vocab (:n-vocab tok) :max-branches 64 :n-ctx 128}) tok {:max-cells 20000})
        brackets (cfg/cfg {:start :s :rules {:s [["(" :s ")" :s] []]}})
        completion (token/completion m {:potential (cfg/potential m brackets)})
        texts (for [seed (range 60)
                    :let [tokens (:retval (gfi/simulate completion [[(:bos tok)] 16] (rng/key seed)))]]
                [(lm/decode m (cond-> tokens (contains? (:eog tok) (peek tokens)) pop))
                 (contains? (:eog tok) (peek tokens))])]
    (is (every? (fn [[t _]] (cfg/prefix? brackets t)) texts) "every text a prefix")
    (is (every? (fn [[t ended?]] (or (not ended?) (cfg/recognizes? brackets t))) texts) "every ended text balanced")
    (is (< 20 (count (filter second texts))) "many end")))

;; ---------------------------------------------------------------------------
;; Misuse
;; ---------------------------------------------------------------------------

(deftest misuse-is-loud
  (doseq [[what f reason] [["no rules" #(cfg/cfg {:start :s :rules {}}) :invalid-grammar]
                           ["a start that is no rule" #(cfg/cfg {:start :t :rules {:s [["a"]]}}) :invalid-grammar]
                           ["a reference to no rule" #(cfg/cfg {:start :s :rules {:s [[:t]]}}) :invalid-grammar]
                           ["an alternative that is no vector" #(cfg/cfg {:start :s :rules {:s ["a"]}}) :invalid-grammar]
                           ["an element of another kind" #(cfg/cfg {:start :s :rules {:s [[42]]}}) :invalid-grammar]
                           ["an empty language" #(cfg/cfg {:start :s :rules {:s [["a" :s]]}}) :empty-language]
                           ["no grammar" #(cfg/recognizes? {:start :s} "a") :invalid-grammar]
                           ["a context with no weight" #(potential/logw-next (cfg/potential pieces (cfg/cfg {:start :s :rules {:s [["a"]]}})) [1]) :zero-prefix]
                           ["a token outside the vocabulary" #(potential/prefix (cfg/potential pieces (cfg/cfg {:start :s :rules {:s [["a"]]}})) [99]) :invalid-token]]]
    (is (= reason (bc/reason f)) what)))
