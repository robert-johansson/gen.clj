(ns gen.dynamic.guard-test
  "gen.dynamic.guard, the gen body rewriter, on forms (genclj-w867): if
   tests wrapped in truthy?, gen.dynamic.core's guarded functions in place
   of clojure.core's unless a local shadows them, case dispatching on
   case-key, catch clauses raising the fallback first, metadata kept, and
   lazy calls whose trace calls would never run refused. dynamic_test and
   batched_test check what the rewritten bodies do. Planted (2026-11-06):
   guard-symbol ignoring locals fails locals-shadow-the-guarded-functions
   (and a-gen-body-runs-as-rewritten);
   the if test left unwrapped fails if-tests-branch-on-truthy?; the catch
   clause without caught fails a-catch-clause-raises-the-fallback-first;
   check-lazy-bindings looking only at the body fails
   lazy-calls-whose-traces-would-not-run-are-refused."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [gen.array.batch :as batch]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dynamic :refer [gen]]
            [gen.dynamic.core]
            [gen.dynamic.guard :as guard]
            [gen.gfi :as gfi]
            [gen.random :as rng]
            [gen.test-support :as ts]))

(defn- guarded
  "form rewritten with locals bound around it, its symbols resolved in
   this namespace."
  [form & locals]
  (binding [*ns* (the-ns 'gen.dynamic.guard-test)]
    (guard/guard-branches form (set locals))))

(defn- occurs? [x form] (boolean (some #{x} (tree-seq coll? seq form))))

(deftest if-tests-branch-on-truthy?
  (is (= '(if (gen.array/truthy? x) 1 2) (guarded '(if x 1 2) 'x)))
  (is (= '(if (gen.array/truthy? x) 1) (guarded '(if x 1) 'x)) "without an else")
  (testing "macros that expand to if are guarded too"
    (is (= '(if (gen.array/truthy? x) (do 1)) (guarded '(when x 1) 'x)))
    (is (occurs? `array/truthy? (guarded '(cond x 1 y 2) 'x 'y))))
  (testing "nested: every if, in every position"
    (let [g (guarded '(let [a (if x 1 2)] (if (if y a 0) a 3)) 'x 'y)]
      (is (= 3 (count (filter #{`array/truthy?} (tree-seq coll? seq g))))))))

(deftest guarded-functions-take-clojure-core-s-place
  (is (= '(gen.dynamic.core/not x) (guarded '(not x) 'x)))
  (is (= '(gen.dynamic.core/= x 0) (guarded '(= x 0) 'x)))
  (is (= '(gen.dynamic.core/get m :k) (guarded '(get m :k) 'm)))
  (is (= '(gen.dynamic.core/number? x) (guarded '(number? x) 'x)))
  (is (= '(clojure.core/inc x) (guarded '(clojure.core/inc x) 'x)) "a function that needs no guard is left")
  (testing "as values too: a guarded function passed to another"
    (is (= '(clojure.core/map gen.dynamic.core/not xs) (guarded '(clojure.core/map not xs) 'xs))))
  (testing "every guarded name has its guarded version, which answers as the original on plain values"
    (let [table @#'guard/guarded-core]
      (is (< 30 (count table)))
      (is (every? #(var? (resolve %)) (vals table)))
      (is (every? #(var? (resolve %)) (keys table)))
      (doseq [[f args] [['not [nil]] ['= [1 1]] ['get [{:a 1} :a]] ['contains? [{:a 1} :b]] ['number? [1.5]]
                        ['every? [pos? [1 2]]] ['some [neg? [1 -2]]] ['filter [odd? [1 2 3]]]]]
        (is (= (apply @(resolve (symbol "clojure.core" (name f))) args)
               (apply @(resolve (symbol "gen.dynamic.core" (name f))) args))
            (str f))))))

(deftest locals-shadow-the-guarded-functions
  (is (= '(not x) (guarded '(not x) 'x 'not)) "a local named not is the local")
  (is (= '(fn* ([get] (get 1))) (guarded '(fn [get] (get 1)))) "a parameter too")
  (is (= '(let* [= clojure.core/+] (= 1 2)) (guarded '(let [= clojure.core/+] (= 1 2)))) "a let binding too")
  (testing "a member name is no var, even spelled like one"
    (is (= '(. m get :k) (guarded '(. m get :k) 'm)))
    (is (= '(. m (get :k)) (guarded '(. m (get :k)) 'm)))))

(deftest case-dispatches-on-a-checked-value
  ;; case expands to (let* [g k] (case* g …)): case* is rewritten to
  ;; (let* [g (case-key g)] (case* g …))
  (let [g (guarded '(case k 0 :zero :other) 'k)
        checks (filter #(and (seq? %) (= 'let* (first %)) (seq? (second (second %)))
                             (= `guard/case-key (first (second (second %)))))
                       (tree-seq coll? seq g))
        [_ [ge init] body] (first checks)]
    (is (= 1 (count checks)) "one let* binding the dispatch value to case-key's")
    (is (= (list `guard/case-key ge) init))
    (is (= ['case* ge] (take 2 body)) "case* dispatches on the checked local"))
  (testing "case-key"
    (is (= 3 (guard/case-key 3)))
    (is (= :a (guard/case-key :a)))
    (when-not (ts/jvm?)
      (is (= 2.0 (double (guard/case-key (array/array 2.0)))) "a 0-d backend array as its host value"))
    (is (:gen.array/batched-branch (ts/thrown-data (guard/case-key (batch/wrap (array/array [0 1]) [2]))))
        "a batched value cannot pick one clause")))

(deftest a-catch-clause-raises-the-fallback-first
  (let [g (guarded '(try (f) (catch Exception e (g e)) (finally (h))) 'f 'g 'h)
        [_ _ [catch-sym cls e first-form]] g]
    (is (= 'catch catch-sym))
    (is (= 'Exception cls))
    (is (= (list `gen.dynamic.core/caught e) first-form))
    (is (= '(finally (h)) (last g)))))

(deftest metadata-is-kept
  (let [form (with-meta '(f x) {:tag 'double :line 7})]
    (is (= {:tag 'double :line 7} (select-keys (meta (guarded form 'f 'x)) [:tag :line]))))
  (is (= 'long (:tag (meta (first (guarded [(with-meta '(f) {:tag 'long})] 'f))))) "inside a vector"))

(deftest lazy-calls-whose-traces-would-not-run-are-refused
  (testing "a statement's value is discarded"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"lazy and its value is discarded"
                          (guarded '(do (clojure.core/map (fn [i] (trace i 1)) xs) 1) 'xs 'trace)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"lazy and its value is discarded"
                          (guarded '(let [a 1] (for [i xs] (trace i a)) a) 'xs 'trace)) "a for, in a let body")
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"lazy and its value is discarded"
                          (guarded '(do (when ok (clojure.core/map (fn [i] (trace i 1)) xs)) 1) 'ok 'xs 'trace))
        "inside a when"))
  (testing "a binding never used"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"is bound to a lazy"
                          (guarded '(let [ys (clojure.core/map (fn [i] (trace i 1)) xs)] 1) 'xs 'trace))))
  (testing "what runs its trace calls is accepted"
    (is (guarded '(do 1 (clojure.core/map (fn [i] (trace i 1)) xs)) 'xs 'trace) "the value returned")
    (is (guarded '(let [ys (clojure.core/map (fn [i] (trace i 1)) xs)] (vec ys)) 'xs 'trace) "a binding used")
    (is (guarded '(let [ys (clojure.core/map (fn [i] (trace i 1)) xs) zs ys] (vec zs)) 'xs 'trace)
        "used by a later binding")
    (is (guarded '(do (clojure.core/mapv (fn [i] (trace i 1)) xs) 1) 'xs 'trace) "an eager call")
    (is (guarded '(do (clojure.core/map inc xs) 1) 'xs) "a lazy call with no trace calls")))

(deftest locals-of-a-gen-form
  (is (= '#{a b} (guard/locals {'a nil 'b nil}))))

(deftest a-gen-body-runs-as-rewritten
  ;; the rewriter's output in a gen body: a guarded if and a shadowed not
  (let [g (gen [x] (let [not (fn [v] (clojure.core/- v))]
                     (if (pos? x) (not x) (trace :y (d/normal 0 1)))))]
    (is (= -2.0 (:retval (gfi/simulate g [2.0] (rng/key 1)))))
    (is (= #{[:y]} (set (cm/addresses (:choices (gfi/simulate g [-2.0] (rng/key 1)))))))))
