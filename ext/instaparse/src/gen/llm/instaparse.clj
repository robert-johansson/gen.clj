(ns gen.llm.instaparse
  "instaparse grammars as constraints on sampling: a grammar's regular
   language as a gen.llm.regex term, so that gen.llm.grammar's exact masks
   hold a completion to it (DESIGN §9; docs/research/llm-grammar.md, the
   instaparse addendum).

     (term grammar), (term grammar opts)     → its language, a regex term
     (grammar vocab grammar opts)            → a gen.llm.grammar of it over
                                               vocab (an LM or a byte vocabulary)

   grammar is an instaparse parser, or the text of one in instaparse's
   notation (EBNF; (insta/parser text :input-format :abnf) for ABNF). opts:
   :start, the rule (default the parser's start production); :max-depth,
   how many times a recursive rule may nest (none: a recursive grammar is
   a loud :recursive-grammar, since its language need not be regular); for
   grammar, :end-tokens and :max-states too (gen.llm.grammar's and
   gen.llm.regex/dfa's).

   The language is the grammar's as a context-free grammar, read through
   instaparse's data ((:grammar parser), a map of rules to combinators):
   a string is its text, a case-insensitive string each character in
   either case, a regex terminal its regex (gen.llm.regex/parse: a regex
   outside its syntax is loud), concatenation, alternation, ordered choice
   (as instaparse, a union: its order ranks parses, and changes no
   language), optional, star, plus, a counted repeat and epsilon theirs,
   a rule inlined where it is referred to, hidden tags and reductions
   ignored. Lookahead (&, !) is refused (:unsupported-combinator).
   A recursive rule nested past :max-depth is the empty language, so the
   language is the grammar's strings of bounded nesting.

   Two differences from instaparse's parser, both documented rather than
   hidden: a regex terminal here stands for every string its regex
   matches, where instaparse matches it once, as java.util.regex's
   lookingAt does (greedily, so #'a+' 'a' parses no text there, \"aa\"
   here); and . in a regex is any character but a newline, as
   gen.llm.regex's syntax has it, where java.util.regex's default also
   excludes \\r and Unicode's line separators."
  {:api :experimental}
  (:require [gen.llm.grammar :as grammar]
            [gen.llm.regex :as regex]
            [instaparse.core :as insta])
  (:import (java.util.regex Pattern)))

(set! *warn-on-reflection* true)

(defn- parser-of
  "grammar as an instaparse parser: itself, or the parser of its text."
  [grammar]
  (cond (and (map? grammar) (map? (:grammar grammar)) (keyword? (:start-production grammar))) grammar
        (string? grammar) (insta/parser grammar)
        :else (throw (ex-info (str "instaparse: a grammar is an instaparse parser or its text, not " (pr-str grammar))
                              {:reason :invalid-grammar :grammar grammar}))))

(defn- references
  "The rules combinator p refers to (its :nt keywords), at any depth."
  [p]
  (case (:tag p)
    :nt #{(:keyword p)}
    (:cat :alt) (into #{} (mapcat references) (:parsers p))
    :ord (into (references (:parser1 p)) (references (:parser2 p)))
    (:opt :star :plus :rep :look :neg) (references (:parser p))
    #{}))

(defn- recursive-rules
  "The rules of rules (a map of rules to combinators) that can reach
   themselves through their references."
  [rules]
  (let [edges (into {} (map (fn [[k p]] [k (references p)])) rules)
        reach (fn [k]
                (loop [todo (vec (edges k)) seen #{}]
                  (if-let [r (peek todo)]
                    (if (seen r)
                      (recur (pop todo) seen)
                      (recur (into (pop todo) (edges r)) (conj seen r)))
                    seen)))]
    (into #{} (filter #(contains? (reach %) %)) (keys rules))))

(defn- string-ci
  "Each character of s in either case."
  [^String s]
  (apply regex/cat
         (for [cp (iterator-seq (.iterator (.codePoints s)))
               :let [cp (long cp)
                     cases (distinct [cp (long (Character/toLowerCase (int cp))) (long (Character/toUpperCase (int cp)))])]]
           (if (next cases)
             (apply regex/chars (map (fn [c] [c c]) cases))
             (regex/text (String. (Character/toChars (int cp))))))))

(defn- regexp
  "A regex terminal's language: its pattern, through gen.llm.regex/parse."
  [^Pattern re]
  (when-not (zero? (.flags re))
    (throw (ex-info (str "instaparse: the regex " (pr-str (str re)) " has flags, which gen.llm.regex's syntax has not")
                    {:reason :unsupported-regex :regex (str re)})))
  (regex/parse (.pattern re)))

(defn- compile-rules
  "The language of rule start in rules, recursion unrolled to max-depth
   (nil: refused)."
  [rules start max-depth]
  (let [recursive (recursive-rules rules)
        memo (atom {})]
    (letfn [(rule [k depths]
              (when-not (contains? rules k)
                (throw (ex-info (str "instaparse: no rule " k) {:reason :invalid-grammar :rule k})))
              (let [n (long (get depths k 0))]
                (cond
                  (not (recursive k)) (memoized k depths)
                  (nil? max-depth)
                  (if (pos? n)
                    (throw (ex-info (str "instaparse: the rule " k " is recursive, so the language need not be regular;"
                                         " give :max-depth to unroll it to a depth")
                                    {:reason :recursive-grammar :rule k}))
                    (memoized k (assoc depths k 1)))
                  (>= n (long max-depth)) regex/nothing
                  :else (memoized k (assoc depths k (inc n))))))
            (memoized [k depths]
              (let [key [k depths]]
                (or (@memo key)
                    (let [t (node (rules k) depths)]
                      (swap! memo assoc key t)
                      t))))
            (node [p depths]
              (case (:tag p)
                :string (regex/text (:string p))
                :string-ci (string-ci (:string p))
                :regexp (regexp (:regexp p))
                :epsilon regex/epsilon
                :cat (apply regex/cat (map #(node % depths) (:parsers p)))
                :alt (apply regex/alt (map #(node % depths) (:parsers p)))
                :ord (regex/alt (node (:parser1 p) depths) (node (:parser2 p) depths))
                :opt (regex/opt (node (:parser p) depths))
                :star (regex/star (node (:parser p) depths))
                :plus (regex/plus (node (:parser p) depths))
                :rep (let [hi (:max p)]
                       (regex/repeat (node (:parser p) depths) (long (:min p))
                                     (when-not (and (number? hi) (Double/isInfinite (double hi))) hi)))
                :nt (rule (:keyword p) depths)
                (:look :neg) (throw (ex-info (str "instaparse: lookahead (" (name (:tag p)) ") is not supported")
                                             {:reason :unsupported-combinator :combinator (:tag p)}))
                (throw (ex-info (str "instaparse: an unknown combinator " (pr-str (:tag p)))
                                {:reason :unsupported-combinator :combinator (:tag p)}))))]
      (rule start {}))))

(defn term
  "The language of grammar (an instaparse parser or its text) as a
   gen.llm.regex term. opts: :start (default the parser's start
   production), :max-depth (default none: a recursive grammar is loud)."
  ([grammar] (term grammar {}))
  ([grammar {:keys [start max-depth]}]
   (when-not (or (nil? max-depth) (pos-int? max-depth))
     (throw (ex-info (str "instaparse: :max-depth is a positive integer, not " (pr-str max-depth))
                     {:reason :invalid-option :max-depth max-depth})))
   (let [parser (parser-of grammar)]
     (compile-rules (:grammar parser) (or start (:start-production parser)) max-depth))))

(defn grammar
  "The gen.llm.grammar of grammar's language over vocab (an LM, or a
   vector of byte arrays by token): term's opts, and gen.llm.grammar's
   :end-tokens and gen.llm.regex/dfa's :max-states."
  ([vocab g] (grammar vocab g {}))
  ([vocab g opts]
   (let [language (term g (select-keys opts [:start :max-depth]))
         d (regex/dfa language (select-keys opts [:max-states]))]
     (grammar/grammar vocab d (select-keys opts [:end-tokens])))))
