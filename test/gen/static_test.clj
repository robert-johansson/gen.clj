(ns gen.static-test
  "The static modeling language (gen ^:static, gen.dynamic.static): the
   general laws, and every operation equal to the same body's without the
   flag (laws/static), on models with splices of Map, Unfold and other
   static functions, destructuring and parameters; update and regenerate
   skip the nodes that did not change; the graph as data and as DOT;
   batched runs; bodies that are not static are compile errors."
  (:refer-clojure :exclude [+ - *])
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [gen.array :as array :refer [+ - *]]
            [gen.batched :as batched]
            [gen.choicemap :as cm]
            [gen.combinators :as comb]
            [gen.custom :as custom]
            [gen.diff :as diff]
            [gen.dist :as d]
            [gen.dynamic :as dyn :refer [gen]]
            [gen.dynamic.static :as static]
            [gen.gfi :as gfi]
            [gen.laws :as laws]
            [gen.random :as rng]
            [gen.selection :as sel]
            [gen.test-support :as ts]))

;; ---------------------------------------------------------------------------
;; Models
;; ---------------------------------------------------------------------------

(def point (gen ^:static [x slope intercept] (let [y (trace :y (d/normal (+ (* slope x) intercept) 1))] y)))

(def linreg
  (gen ^:static [xs]
    (let [slope (trace :slope (d/normal 0 2))
          intercept (trace :intercept (d/normal 0 2))
          n (count xs)
          ys (splice :ys (comb/map point) xs (vec (repeat n slope)) (vec (repeat n intercept)))]
      ys)))

(def walk-step (gen [_t s drift] (trace :s (d/normal (+ s drift) 1))))

(def hierarchical
  (gen ^:static [n]
    (let [drift (trace :drift (d/normal 0 0.5))
          noise (trace :noise (d/gamma 2 1))
          path (splice :path (comb/unfold walk-step) n 0.0 drift)
          last-s (peek path)
          _ (trace :obs (d/normal last-s noise))]
      last-s)))

(def destructured
  (gen ^:static [{:keys [mu sd]}]
    (let [[a b] (trace :pair ((d/product d/normal d/normal) [mu sd] [0.0 1.0]))
          {:keys [lo hi]} {:lo (- a 1.0) :hi (+ a 1.0)}
          c (trace :c (d/uniform lo hi))
          flip (trace :flip (d/bernoulli 0.3))]
      [b c flip])))

(def nested
  "A static function spliced in a static function."
  (gen ^:static [m]
    (let [mu (trace :mu (d/normal m 1))
          y (splice :inner point 1.0 mu 0.0)
          z (trace :z (d/normal y 1))]
      z)))

(def models
  [[:linreg linreg [[1.0 2.0 3.0]] [[[1.0 2.5]] [[0.5 1.0 1.5 2.0]]]]
   [:hierarchical hierarchical [3] [[2] [4]]]
   [:destructured destructured [{:mu 0.0 :sd 1.0}] [[{:mu 1.0 :sd 1.0}]]]
   [:nested nested [0.0] [[1.5]]]])

(laws/deflaws static models 100)
(laws/deflaws static models 100 laws/static)

;; a static body over a callee with encapsulated randomness: its score fn
;; must split keys as the handler's assess does (assess = generate on the
;; same key), and skip them only under an exact-density operation
(def noisy-callee (gen [m] (trace :x (d/normal (+ m (untraced (d/normal 0 0.5))) 1))))

(def over-noise
  (gen ^:static [m]
    (let [mu (trace :mu (d/normal m 1))
          x (splice :inner noisy-callee mu)
          _ (trace :y (d/normal x 1))]
      x)))

(def omega-laws (merge laws/encapsulated laws/static))

(laws/deflaws static-omega [[:over-noise over-noise [0.0] [[1.0]]]] 60 omega-laws)

;; ---------------------------------------------------------------------------
;; Skipping
;; ---------------------------------------------------------------------------

(deftest update-and-regenerate-skip-what-did-not-change
  (let [runs (atom {})
        tick! (fn [k] (swap! runs update k (fnil inc 0)))
        callee (custom/deterministic {:apply (fn [[x]] (tick! :callee) (* 2.0 x))})
        model (gen ^:static [x]
                (let [a (trace :a (d/normal x 1))
                      b (trace :b (d/normal 0 1))
                      a2 (do (tick! :compute) (* a a))
                      c (splice :c callee a2)
                      d (trace :d (d/normal (+ b c) 1))]
                  d))
        tr (gfi/simulate model [0.5] (rng/key 1))
        counted (fn [f] (reset! runs {}) (f) @runs)]
    (testing "constraining b leaves a's descendants alone"
      (is (= {} (counted #(gfi/update tr [0.5] [diff/no-change] (cm/choicemap :b 0.3) (rng/key 2))))))
    (testing "constraining a reruns them"
      (is (= {:compute 1 :callee 1} (counted #(gfi/update tr [0.5] [diff/no-change] (cm/choicemap :a 0.3) (rng/key 3))))))
    (testing "a changed argument reaches a's descendants only through a's score: a keeps its value"
      (is (= {} (counted #(gfi/update tr [0.9] [diff/unknown-change] cm/EMPTY (rng/key 4))))))
    (testing "regenerating d alone"
      (is (= {} (counted #(gfi/regenerate tr (sel/select :d) (rng/key 5))))))
    (testing "the same body without ^:static reruns everything"
      (let [plain (assoc model :static nil)
            tp (gfi/simulate plain [0.5] (rng/key 1))]
        (is (= {:compute 1 :callee 1} (counted #(gfi/update tp [0.5] [diff/no-change] (cm/choicemap :b 0.3) (rng/key 2)))))))))

(deftest parameters-are-read-again-at-every-update
  ;; a store may change between runs (training), so a node reading one reruns
  (let [model (gen ^:static [] (let [w (param :w) y (trace :y (d/normal w 1))] [w y]))
        tr (gfi/simulate (dyn/with-params model {:w 1.0}) [] (rng/key 6))
        moved (:trace (gfi/update (assoc tr :gen-fn (dyn/with-params model {:w 3.0})) cm/EMPTY (rng/key 7)))]
    (is (== 3.0 (ts/number (first (:retval moved)))))))

;; ---------------------------------------------------------------------------
;; Change hints through the graph (gen.diff/propagate)
;; ---------------------------------------------------------------------------

(deftest change-hints-reach-a-callee
  (let [runs (atom 0)
        kernel (gen [x] (swap! runs inc) (trace :y (d/normal x 1)))
        model (gen ^:static [xs]
                (let [ys (splice :ys (comb/map kernel) xs)] ys))
        xs [0.1 0.2 0.3 0.4]
        xs' (assoc xs 2 0.9)
        tr (gfi/simulate model [xs] (rng/key 11))
        runs-of (fn [argdiff] (reset! runs 0)
                  [(gfi/update tr [xs'] [argdiff] cm/EMPTY (rng/key 12)) @runs])]
    (testing "a vector-diff argument: the Map reruns the element it names"
      ;; (its :y is kept, so the return value is unchanged: no-change)
      (let [[{rd :retdiff} n] (runs-of (diff/vector-diff 4 4 {2 diff/unknown-change}))]
        (is (= 1 n))
        (is (diff/no-change? rd))))
    (testing "a constraint on one element: the function's retdiff is the Map's vector-diff"
      (let [{rd :retdiff} (gfi/update tr [xs] [diff/no-change] (cm/choicemap [:ys 2 :y] 5.0) (rng/key 15))]
        (is (diff/vector-diff? rd))
        (is (= #{2} (set (keys (:updated rd)))))))
    (testing "unknown-change: every element"
      (is (= 4 (second (runs-of diff/unknown-change)))))))

(deftest a-one-call-binding-has-a-precise-diff
  ;; (nth xs 1) is unchanged when another element changes, so what depends
  ;; on it alone is skipped; (count xs) changes by the elements appended
  (let [runs (atom {})
        tick! (fn [k] (swap! runs update k (fnil inc 0)))
        model (gen ^:static [xs]
                (let [x1 (nth xs 1)
                      a (do (tick! :a) (* 2.0 x1))
                      n (count xs)
                      b (do (tick! :b) (* 1.0 n))
                      y (trace :y (d/normal (+ a b) 1))]
                  y))
        xs [0.1 0.2 0.3]
        tr (gfi/simulate model [xs] (rng/key 13))
        counted (fn [args argdiff] (reset! runs {}) (gfi/update tr args [argdiff] cm/EMPTY (rng/key 14)) @runs)]
    (is (= {} (counted [[0.1 0.2 0.7]] (diff/vector-diff 3 3 {2 diff/unknown-change}))))
    (is (= {:a 1} (counted [[0.1 0.5 0.3]] (diff/vector-diff 3 3 {1 diff/unknown-change}))))
    (is (= {:b 1} (counted [[0.1 0.2 0.3 0.4]] (diff/vector-diff 4 3 {}))))
    (is (= {:a 1 :b 1} (counted [[0.1 0.2 0.3 0.4]] diff/unknown-change)))))

;; ---------------------------------------------------------------------------
;; The graph
;; ---------------------------------------------------------------------------

(deftest the-graph-as-data-and-as-dot
  (let [{:keys [args nodes return]} (static/ir linreg)]
    (is (= '{:fixed [[xs]] :rest nil} args))
    (is (= [:trace :trace :compute :splice] (mapv :kind nodes)))
    (is (= [:slope :intercept nil :ys] (mapv :addr nodes)))
    (is (= '[[] [] [xs] [xs n slope intercept]] (mapv :inputs nodes)))
    (is (= '[ys] (:inputs return))))
  (let [dot (static/dot linreg)]
    (is (str/starts-with? dot "digraph {"))
    (is (str/includes? dot "\":slope\" -> \":ys\";"))
    (is (str/includes? dot "\":ys\" -> \"return\";")))
  (is (nil? (static/ir (gen [] 1)))))

(deftest a-batched-run-is-the-scalar-runs
  (let [args [[1.0 2.0]]
        ps (batched/simulate linreg args 5 (rng/key 8))
        {moved :particles ws :weights} (batched/update ps args [diff/no-change] (cm/choicemap :slope 0.5) (rng/key 9))]
    (is (nil? (batched/fallback-reason ps)))
    (is (every? #(ts/close? (:score %) (:weight (gfi/assess linreg args (:choices %) (rng/key 0)))) (batched/traces ps)))
    (testing "an update skips under the batch tracer as in a scalar run"
      (is (nil? (batched/fallback-reason moved)))
      (is (ts/close? (array/array (mapv #(ts/number (:weight (gfi/update % args [diff/no-change] (cm/choicemap :slope 0.5) (rng/key 9))))
                                        (batched/traces ps)))
                     ws)))))

;; ---------------------------------------------------------------------------
;; Bodies that are not static
;; ---------------------------------------------------------------------------

(defn- static-error [form]
  (try (eval form) nil
       (catch clojure.lang.Compiler$CompilerException e (ex-message (ex-cause e)))))

(deftest a-body-that-is-not-static-is-a-compile-error
  (binding [*ns* (the-ns 'gen.static-test)]
    (doseq [[why form] [["not one let" '(gen ^:static [] (trace :a (d/normal 0 1)))]
                        ["two forms" '(gen ^:static [] (let [a 1] a) 2)]
                        ["a computed address" '(gen ^:static [i] (let [a (trace [:a i] (d/normal 0 1))] a))]
                        ["a trace inside an expression" '(gen ^:static [] (let [a (+ 1 (trace :a (d/normal 0 1)))] a))]
                        ["a branch on a choice" '(gen ^:static [] (let [b (trace :b (d/bernoulli 0.5))
                                                                         a (if b (trace :x (d/normal 0 1)) 0.0)] a))]
                        ["a splice without an address" '(gen ^:static [g] (let [a (splice g)] a))]
                        ["an untraced draw" '(gen ^:static [] (let [a (untraced (d/normal 0 1))] a))]
                        ["a trace inside the return expression" '(gen ^:static [] (let [a 1] (+ 1 (trace :a (d/normal a 1)))))]]]
      (testing why
        (is (re-find #"gen \^:static" (str (static-error form))))))))

;; ---------------------------------------------------------------------------
;; The lint
;; ---------------------------------------------------------------------------

(def fits (gen [m] (let [a (trace :a (d/normal m 1)) b (* 2 a)] (trace :b (d/normal b 1)))))
(def branches (gen [] (let [b (trace :b (d/bernoulli 0.5)) x (if b (trace :x (d/normal 0 1)) 0.0)] x)))
(def loops (gen [n] (doseq [i (range n)] (trace [:y i] (d/normal 0 1)))))
(def noisy (gen [] (let [e (untraced (d/normal 0 1))] (trace :x (d/normal e 1)))))

(deftest the-lint-says-what-the-flag-would-take
  (is (= {:static? true} (static/could-be-static fits 'gen.static-test)))
  (doseq [[gf why] [[branches #"calls trace"] [loops #"one let"] [noisy #"untraced"]]]
    (let [{:keys [static? reason]} (static/could-be-static gf 'gen.static-test)]
      (is (false? static?))
      (is (re-find why reason))))
  (testing "over a namespace"
    (let [report (into {} (map (juxt (comp :name meta :var) identity)) (static/lint 'gen.static-test))]
      (is (:static? (report 'fits)))
      (is (false? (:static? (report 'loops))))
      (is (:already (report 'linreg)))))
  (testing "the verdict is the flag's: a body the lint accepts compiles with ^:static"
    (binding [*ns* (the-ns 'gen.static-test)]
      (is (dyn/dynamic-gf? (eval (list* `gen (vary-meta (first (:source fits)) assoc :static true) (rest (:source fits)))))))))
