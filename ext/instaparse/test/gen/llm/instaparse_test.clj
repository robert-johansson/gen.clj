(ns gen.llm.instaparse-test
  "instaparse grammars as constraints (gen.llm.instaparse):

   - Agreement: for 200 random grammars without recursion (strings, case-
     insensitive strings, one-character regex terminals, concatenation,
     alternation, ordered choice, optional, star, plus, counted repeats,
     epsilon, hidden rules, references to later rules), instaparse's own
     whole-text parse and the term's regex/matches? agree on 20 random
     texts over a, A, b, B each, and on every text of up to three of them.
   - Sampling: completions under a tool-call grammar's potential, on the
     toy engine over pieces of the grammar's text, all parse with
     instaparse.
   - Recursion: unrolled to :max-depth (nested lists, a tail-recursive
     rule), refused without it.
   - Misuse is loud: lookahead, an unknown rule, a regex with flags or
     outside gen.llm.regex's syntax, a grammar that is neither a parser
     nor its text, a bad :max-depth.

   Planted bugs (2026-11-05, each applied, the suite run, reverted):
   ordered choice as its first branch only, and a case-insensitive string
   as its own case only, each fail the agreement property."
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [gen.gfi :as gfi]
            [gen.llm.branching-contract :as bc]
            [gen.llm.byte-tokenizer :as bt]
            [gen.llm.instaparse :as ip]
            [gen.llm.lm :as lm]
            [gen.llm.potential :as potential]
            [gen.llm.regex :as regex]
            [gen.llm.token :as token]
            [gen.llm.toy-engine :as toy]
            [gen.random :as rng]
            [instaparse.combinators :as c]
            [instaparse.core :as insta]))

;; ---------------------------------------------------------------------------
;; Random grammars without recursion, against instaparse's parser
;; ---------------------------------------------------------------------------

(def ^:private gen-terminal
  "A terminal that matches no empty text."
  (gen/one-of [(gen/fmap c/string (gen/elements ["a" "b" "ab" "ba"]))
               (gen/fmap c/string-ci (gen/elements ["a" "b" "aB"]))
               (gen/fmap c/regexp (gen/elements ["[ab]" "[aA]" "b"]))]))

(defn- gen-combinator
  "Combinators over a and b that may refer to the rules in refs. A counted
   repeat repeats a terminal: instaparse 1.5.0's repeat of a parser that
   matches the empty text rejects the empty text past one repetition
   ((rep 2 2 (opt (string \"a\"))) on \"\"), where the language has it."
  [refs]
  (gen/recursive-gen
   (fn [inner]
     (gen/one-of [(gen/fmap #(apply c/cat %) (gen/vector inner 2 3))
                  (gen/fmap #(apply c/alt %) (gen/vector inner 2 3))
                  (gen/fmap (fn [[a b]] (c/ord a b)) (gen/tuple inner inner))
                  (gen/fmap c/opt inner)
                  (gen/fmap c/star inner)
                  (gen/fmap c/plus inner)
                  (gen/let [p gen-terminal lo (gen/choose 0 2) n (gen/choose 0 2)] (c/rep lo (+ lo n) p))]))
   (gen/one-of (cond-> [gen-terminal (gen/return c/Epsilon)]
                 (seq refs) (conj (gen/fmap (fn [k] (c/nt k)) (gen/elements refs)))))))

(def ^:private gen-grammar
  "Rules :r0 … :rk, each referring only to later ones, some hidden."
  (gen/let [k (gen/choose 1 3)
            rules (apply gen/tuple (for [i (range k)]
                                     (gen/scale #(min % 6)
                                                (gen-combinator (mapv #(keyword (str "r" %)) (range (inc i) k))))))
            hidden (gen/vector gen/boolean k)]
    (into {} (map (fn [i p h] [(keyword (str "r" i)) (if h (c/hide-tag p) p)]) (range k) rules hidden))))

(def ^:private gen-text (gen/fmap str/join (gen/vector (gen/elements ["a" "A" "b" "B"]) 0 5)))

(defn- parses? [parser text] (not (insta/failure? (insta/parse parser text))))

(defn- texts-up-to
  "Every text over a, A, b, B of up to n characters."
  [n]
  (loop [len 0 level [""] out []]
    (if (> len n)
      out
      (recur (inc len) (for [t level ch ["a" "A" "b" "B"]] (str t ch)) (into out level)))))

(def ^:private short-texts (texts-up-to 3))

(defspec instaparse-and-the-term-agree 200
  (prop/for-all [rules gen-grammar
                 texts (gen/vector gen-text 20)]
    (let [parser (insta/parser rules :start :r0)
          term (ip/term parser)]
      (every? #(= (parses? parser %) (regex/matches? term %)) (concat texts short-texts)))))

;; ---------------------------------------------------------------------------
;; Completions under a grammar parse
;; ---------------------------------------------------------------------------

(def ^:private tool-call
  "call = name '(' args ')'
   name = 'get' | 'put' | 'del'
   args = (arg (<','> arg)*)?
   arg = #'[0-9]' | 'x'")

(def ^:private pieces
  ["g" "e" "t" "p" "u" "d" "l" "(" ")" "," "0" "1" "7" "x" "get" "put(" "del(" ")," "0," "(x" "q" "z"])

(deftest completions-under-a-grammar-parse
  (let [tok (bt/tokenizer pieces)
        m (lm/lm (toy/engine {:n-vocab (:n-vocab tok) :max-branches 64 :n-ctx 128}) tok {:max-cells 20000})
        g (ip/grammar m tool-call)
        completion (token/completion m {:potential g})
        parser (insta/parser tool-call)
        texts (for [seed (range 60)
                    :let [tokens (:retval (gfi/simulate completion [[(:bos tok)] 24] (rng/key seed)))]
                    :when (contains? (:eog tok) (peek tokens))]
                (lm/decode m (pop tokens)))]
    (is (< 40 (count texts)) "most completions end within 24 tokens")
    (is (< 10 (count (distinct texts))) "many texts")
    (is (every? #(parses? parser %) texts) (str "every text parses: " (vec (remove #(parses? parser %) texts))))
    (is (= 0.0 (potential/prefix g (lm/encode m "put(x,1" {:add-special? false}))))
    (is (= ##-Inf (potential/prefix g (lm/encode m "put(q" {:add-special? false}))))))

;; ---------------------------------------------------------------------------
;; Recursion
;; ---------------------------------------------------------------------------

(deftest recursion-unrolls-to-a-depth
  (let [nested "list = '[' (list | 'x')* ']'"]
    (testing "nested lists, three deep"
      (let [t (ip/term nested {:max-depth 3})]
        (is (every? #(regex/matches? t %) ["[]" "[x]" "[[x]x]" "[[[x]]]" "[[[]][]x]"]))
        (is (not-any? #(regex/matches? t %) ["[[[[x]]]]" "[" "x" "[x]]"]))))
    (testing "a tail-recursive rule"
      (let [t (ip/term "s = 'g' s | 'h'" {:max-depth 4})]
        (is (every? #(regex/matches? t %) ["h" "gh" "gggh"]))
        (is (not (regex/matches? t "ggggh")) "four rules deep: three g's")))
    (testing "refused without a depth"
      (is (= :recursive-grammar (bc/reason #(ip/term nested)))))
    (testing "a depth that leaves no string is an empty language"
      (is (= :empty-language (bc/reason #(ip/grammar [(.getBytes "x")] "s = 'x' s" {:max-depth 2})))))))

(deftest case-insensitive-strings-and-ordered-choice
  (let [t (ip/term (insta/parser "s = 'aB' / 'ab' 'c'" :string-ci true))]
    (is (every? #(regex/matches? t %) ["ab" "AB" "Ab" "abc" "ABC"]))
    (is (not (regex/matches? t "abd")))))

;; ---------------------------------------------------------------------------
;; Misuse
;; ---------------------------------------------------------------------------

(deftest misuse-is-loud
  (doseq [[what f reason] [["lookahead" #(ip/term "s = &'a' 'a'") :unsupported-combinator]
                           ["negative lookahead" #(ip/term "s = !'b' 'a'") :unsupported-combinator]
                           ["an unknown start rule" #(ip/term "s = 'a'" {:start :t}) :invalid-grammar]
                           ["a regex with flags" #(ip/term (insta/parser {:s (c/regexp (java.util.regex.Pattern/compile "a" java.util.regex.Pattern/CASE_INSENSITIVE))} :start :s)) :unsupported-regex]
                           ["no grammar" #(ip/term 42) :invalid-grammar]
                           ["a bad depth" #(ip/term "s = 'a'" {:max-depth 0}) :invalid-option]]]
    (is (= reason (bc/reason f)) what))
  (is (thrown? clojure.lang.ExceptionInfo (ip/term "s = #'a(?=b)'")) "lookaround in a regex: gen.llm.regex's error"))
