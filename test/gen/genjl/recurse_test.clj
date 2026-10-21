(ns gen.genjl.recurse-test
  "Gen.jl 0.4.8's test/modeling_library/recurse.jl: a probabilistic
   context-free grammar as a Recurse. Its nodes are 0-based (D2): Gen.jl's
   node i is node i - 1 here, at [i :production] and [i :aggregation], and
   categorical's values too: rule r is r - 1. Recurse takes one argument,
   the root's input, without Gen.jl's root index (D13)."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.choicemap :as cm]
            [gen.combinators :as comb]
            [gen.diff :as diff]
            [gen.dist :as d]
            [gen.dynamic :refer [gen]]
            [gen.genjl.support :as support :refer [approx?]]
            [gen.gfi :as gfi]
            [gen.random :as rng]
            [gen.test-support :as ts]))

(defn- numbering
  "[max-branch children child-nums parents]: recurse.jl's cases for one
   max-branch, each number one less. children are [parent j child],
   child-nums [child j] and parents [child parent]."
  [max-branch children child-nums parents]
  (doseq [[parent j child] children]
    (is (= child (comb/get-child parent j max-branch))))
  (doseq [[child j] child-nums]
    (is (= j (comb/get-child-num child max-branch))))
  (doseq [[child parent] parents]
    (is (= parent (comb/get-parent child max-branch)))))

;; recurse.jl:3
(deftest recurse-node-numbering
  (testing "max-branch 1"
    (numbering 1
               [[0 0 1] [1 0 2] [2 0 3]]
               [[1 0] [2 0] [3 0]]
               [[1 0] [2 1] [3 2]]))
  (testing "max-branch 2"
    (numbering 2
               [[0 0 1] [0 1 2] [1 0 3] [1 1 4] [2 0 5]]
               [[1 0] [2 1] [3 0] [4 1] [5 0]]
               [[1 0] [2 0] [3 1] [4 1] [5 2]]))
  (testing "max-branch 3"
    (numbering 3
               [[0 0 1] [0 1 2] [0 2 3] [1 0 4] [1 1 5] [1 2 6] [2 0 7] [2 1 8] [2 2 9] [3 0 10]]
               [[1 0] [2 1] [3 2] [4 0] [5 1] [6 2] [7 0] [8 1] [9 2] [10 0]]
               [[1 0] [2 0] [3 0] [4 1] [5 1] [6 1] [7 2] [8 2] [9 2] [10 3]])))

(def production
  "recurse.jl:68: rules aSa, bSb (one child each), aa and bb (none)."
  (gen [_]
    (let [rule (trace :rule (d/categorical [0.24 0.26 0.23 0.27]))]
      {:value rule :children (case rule (0 1) [nil] (2 3) [])})))

(def aggregation
  "recurse.jl:87: the rule's string, its prefix . or -."
  (gen [rule [child]]
    (let [prefix (if (trace :prefix (d/bernoulli 0.4)) "." "-")]
      (case rule
        0 (str "(" prefix "a" child "a)")
        1 (str "(" prefix "b" child "b)")
        2 (str "(" prefix "aa)")
        3 (str "(" prefix "bb)")))))

(def pcfg (comb/recurse production aggregation 1))

(defn- rule [i] [i :production :rule])
(defn- prefix [i] [i :aggregation :prefix])

(defn- log-product
  "log(p1) + log(p2) + ..."
  [& ps]
  (reduce + (map #(Math/log %) ps)))

(def ^:private constraints
  "recurse.jl:156: every choice of \"(.b(.a(.b(-bb)b)a)b)\", rules 2, 1, 2
   and 4 in Gen.jl's numbering."
  (cm/choicemap (rule 0) 1 (prefix 0) true
                (rule 1) 0 (prefix 1) true
                (rule 2) 1 (prefix 2) true
                (rule 3) 3 (prefix 3) false))

(defn- check-choices
  "choices[(i, Val(:production)) => :rule] and choices[(i,
   Val(:aggregation)) => :prefix] are rules[i] and prefixes[i]."
  [c rules prefixes]
  (doseq [[i r p] (map vector (range) rules prefixes)]
    (is (= r (ts/host (cm/get-choice c (rule i)))) (str "node " i "'s rule"))
    (is (= p (ts/host (cm/get-choice c (prefix i)))) (str "node " i "'s prefix"))))

;; recurse.jl:66
(deftest simple-pcfg
  (testing "an untraced call" ; recurse.jl:133
    (is (string? (pcfg nil))))
  (let [{tr :trace w :weight} (gfi/generate pcfg [nil] constraints (rng/key 1))]
    (testing "generate with every choice" ; recurse.jl:153
      (is (approx? (log-product 0.26 0.24 0.26 0.27 0.4 0.4 0.4 0.6) w))
      (is (approx? w (:score tr)))
      (is (= [nil] (:args tr)))
      (is (= "(.b(.a(.b(-bb)b)a)b)" (:retval tr)))
      (check-choices (:choices tr) [1 0 1 3] [true true true false]))
    (testing "an update of a prefix, not of the structure" ; recurse.jl:180
      (let [{t :trace w :weight rd :retdiff d :discard}
            (gfi/update tr [nil] [diff/unknown-change] (cm/choicemap (prefix 2) false) (rng/key 2))]
        (is (approx? (- (log-product 0.6) (log-product 0.4)) w))
        (is (approx? (log-product 0.26 0.24 0.26 0.27 0.4 0.4 0.6 0.6) (:score t)))
        (is (= [nil] (:args t)))
        (is (= "(.b(.a(-b(-bb)b)a)b)" (:retval t)))
        (check-choices (:choices t) [1 0 1 3] [true true false false])
        (is (true? (ts/host (cm/get-choice d (prefix 2)))))
        (is (= {:values 0 :submaps 1} (support/shallow d)))
        (is (= {:values 1 :submaps 0} (support/shallow (cm/get-submap d [2 :aggregation]))))
        (is (= diff/unknown-change rd))))
    (testing "an update of the structure: node 2's bSb becomes aa, keeping its prefix" ; recurse.jl:206
      (let [{t :trace w :weight rd :retdiff d :discard}
            (gfi/update tr [nil] [diff/unknown-change] (cm/choicemap (rule 2) 2) (rng/key 3))
            c (:choices t)]
        (is (approx? (- (log-product 0.23) (log-product 0.26 0.27 0.6)) w))
        (is (approx? (log-product 0.26 0.24 0.23 0.4 0.4 0.4) (:score t)))
        (is (= [nil] (:args t)))
        (is (= "(.b(.a(.aa)a)b)" (:retval t)))
        (check-choices c [1 0 2] [true true true])
        (is (cm/empty? (cm/get-submap c [3 :production])))
        (is (cm/empty? (cm/get-submap c [3 :aggregation])))
        (is (= 1 (ts/host (cm/get-choice d (rule 2)))))
        (is (not (support/has-value? d (prefix 2))))
        (is (= 3 (ts/host (cm/get-choice d (rule 3)))))
        (is (false? (ts/host (cm/get-choice d (prefix 3)))))
        (is (= diff/unknown-change rd))))))

;; ---------------------------------------------------------------------------
;; recurse.jl:110, the likeliest strings (statistical)
;; ---------------------------------------------------------------------------

(def ^:private likeliest
  "recurse.jl:110, test_strings: the grammar's 20 likeliest strings."
  ["(.aa)" "(-aa)" "(.bb)" "(-bb)"
   "(.a(.aa)a)" "(-a(.aa)a)" "(.a(-aa)a)" "(-a(-aa)a)"
   "(.b(.bb)b)" "(-b(.bb)b)" "(.b(-bb)b)" "(-b(-bb)b)"
   "(.a(.bb)a)" "(-a(.bb)a)" "(.a(-bb)a)" "(-a(-bb)a)"
   "(.b(.aa)b)" "(-b(.aa)b)" "(.b(-aa)b)" "(-b(-aa)b)"])

(defn- probability
  "The probability of a string of the grammar, its one derivation's: every
   node's rule and prefix."
  [s]
  (let [body (subs s 2 (dec (count s)))]
    (* (if (= \. (nth s 1)) 0.4 0.6)
       (case body
         "aa" 0.23
         "bb" 0.27
         (* (if (= \a (first body)) 0.24 0.26) (probability (subs body 1 (dec (count body)))))))))

(def ^:private runs
  "Gen.jl's 1,000 runs miss its least likely string, (.a(.aa)a) at
   p = 0.0088, with probability e^-8.9 = 1.4e-4. In 10,000 its count is 88
   with a standard error of 9.4, so 5 standard errors stay above 0."
  10000)

(defn- check-frequencies
  "test_strings, by frequency: each likeliest string's count in strings
   within 5 standard errors of runs·p, which is above 0, so it appears."
  [strings]
  (let [counts (frequencies strings)]
    (doseq [s likeliest]
      (let [p (probability s)
            n (get counts s 0)]
        (is (< (abs (- n (* runs p))) (* 5 (Math/sqrt (* runs p (- 1 p)))))
            (str s ": " n " in " runs ", p = " p))))))

(deftest the-likeliest-strings
  ;; recurse.jl:110
  (testing "generate" ; recurse.jl:136
    (check-frequencies (map #(:retval (:trace (gfi/generate pcfg [nil] cm/EMPTY %))) (rng/split (rng/key 4) runs))))
  (testing "simulate" ; recurse.jl:145
    (check-frequencies (map #(:retval (gfi/simulate pcfg [nil] %)) (rng/split (rng/key 5) runs)))))
