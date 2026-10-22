(ns gen.static-test
  "The static modeling language (gen ^:static, gen.dynamic.static): the
   general laws, and every operation equal to the same body's without the
   flag (laws/static), on models with splices of Map, Unfold and other
   static functions, destructuring and parameters, as are the choice
   gradients through the compiled score; update and regenerate skip the
   nodes that did not change; the graph as data and as DOT; batched runs;
   bodies that are not static are compile errors, a binding's head is
   macroexpanded, quoted data is data, a call nested where it is evaluated
   once is hoisted, and an address's keys after the first may be computed."
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
            [gen.dist.core :as dc]
            [gen.dynamic :as dyn :refer [gen]]
            [gen.dynamic.static :as static]
            [gen.gfi :as gfi]
            [gen.gradients :as grad]
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

(def rules
  "One-call bindings of the propagate rules, each read by a choice or a Map,
   so that a rule's wrong diff changes a weight under the law's computed
   argdiffs: conj (also of two) and an appending assoc give the vectors Maps
   run over, the variadic assoc of a map a choice reads, get past a shrunk
   end, nth, count, inc, dec, + and -. (The first version led every chain
   through an assoc of a changed vector, whose unknown-change erased a
   wrong diff upstream: the confirmation review's eight rule mutations
   passed it. Those that only overstate a change, which costs a rerun and
   no weight, are the diff tests' to catch.)"
  (gen ^:static [xs i m]
    (let [a (trace :a (d/normal 0 1))
          b (trace :b (d/normal 0 1))
          n (count xs)
          ys (conj xs a)
          zs (conj xs a b)
          us (assoc xs n a)
          m2 (assoc m :p a :q b)
          q (get m2 :q 0.0)
          xi (get xs i 0.0)
          w (nth ys 1)
          k (inc n)
          j (dec n)
          s (+ n i)
          t (- n i)
          v (vec ys)
          c1 (trace :c1 (d/normal (+ q xi) 1))
          c2 (trace :c2 (d/normal w 1))
          c3 (trace :c3 (d/normal (* 0.1 (+ k j s t)) 1))
          out (splice :m (comb/map point) v (vec (repeat k w)) (vec (repeat k xi)))
          out2 (splice :z (comb/map point) zs (vec (repeat (+ n 2) 1.0)) (vec (repeat (+ n 2) 0.0)))
          out3 (splice :u (comb/map point) us (vec (repeat k 1.0)) (vec (repeat k 0.0)))]
      [out out2 out3 c1 c2 c3])))

(def rebound
  "A parameter name bound twice: the last binding is the value, as in let
   (review 3, F3: the slots shifted every later argument)."
  (gen ^:static [w w {:keys [z]} z2]
    (let [y (trace :y (d/normal (+ z z2) 0.5))]
      [w z y])))

(def rest-rebound
  "A rest parameter rebinding a fixed one: the rest is the value, and its
   diff (review 4, R10: the fixed one's diff won)."
  (gen ^:static [a & a] (let [y (trace :y (d/normal (count a) 0.5))] [a y])))

(def double-index
  "nth truncates a number index: a changed element reached by 1.0
   (confirmation review N1: the rule called it unchanged)."
  (gen ^:static [xs i]
    (let [a (trace :a (d/normal 0 1))
          ys (assoc xs 1 a)
          w (nth ys i)
          y (trace :y (d/normal w 1))]
      y)))

(def reads-w (gen ^:static [m] (let [y (trace :y (d/normal (+ m (param :w)) 1))] y)))

(def callee-store
  "A static callee with its own store under a parent with another: the
   score fn must assess it under its own (confirmation review N2)."
  (dyn/with-params (gen ^:static [m] (let [c (splice :c (dyn/with-params reads-w {:w 3.0}) m)] c)) {:w 0.0}))

(def dict-rules
  "The rules on a map argument."
  (gen ^:static [m]
    (let [a (trace :a (d/normal 0 1))
          m2 (assoc m :k a)
          c (count m2)
          g (get m2 :j 0.0)
          y (trace :y (d/normal (+ g (* 0.1 c)) 1))]
      y)))

(defmacro ^:private noisy-at
  "(trace addr (d/normal x 1)): a user macro expanding to a call."
  [addr x]
  `(~'trace ~addr (d/normal ~x 1)))

(defmacro ^:private called-at
  "(splice addr gf & args): a user macro expanding to a call."
  [addr gf & args]
  `(~'splice ~addr ~gf ~@args))

(defmacro ^:private twice-x
  "(* 2.0 x): a macro that reads a local it is not given, at a binding's
   head (review of 9d9h, 2026-10-21)."
  []
  '(* 2.0 x))

(defmacro ^:private hidden-at
  "1 plus (trace addr (d/normal x 1)): a call inside an expression."
  [addr x]
  `(+ 1 (noisy-at ~addr ~x)))

(def expanded-heads
  "Bindings and a return expression whose heads are user macros that
   expand to calls, which the dynamic gen expands (M5, Gen.jl's macros.jl:
   G11)."
  (gen ^:static [m]
    (let [a (noisy-at :a m)
          b (called-at :b point a 1.0 0.0)]
      (noisy-at :c b))))

(def quoting
  "A body returning quoted data that mentions trace and a binding, data and
   not a call (M5, Gen.jl's tilde_sugar.jl:66: G12)."
  (gen ^:static [m]
    (let [a (trace :a (d/normal m 1))
          form '(trace :x (d/normal a 1))]
      [a form])))

(def hoisting
  "Calls nested where they are evaluated exactly once, hoisted into nodes
   in evaluation order: a vector's elements, a distribution's argument, an
   if's test, a map's value and the return expression's argument (M5,
   Gen.jl's static_dsl.jl:410 and 466: G2)."
  (gen ^:static [m]
    (let [pt [(trace :x (d/normal m 1)) (trace :y (d/normal m 2))]
          z (trace :z (d/normal (+ (first pt) (trace :w (d/normal 0 1))) 1))
          [mu sd] (if (trace :b (d/bernoulli 0.5)) [m 1.0] [z 2.0])
          q {:v (trace :v (d/normal mu sd))}]
      (+ (:v q) (trace :u (d/normal z 1))))))

(def computed
  "Addresses whose keys after the first are computed (M5, Gen.jl's
   static_dsl.jl:266 and 299: G2): a choice, a call, keys that are an
   expression, the return expression's call."
  (gen ^:static [i j]
    (let [a (trace [:a i] (d/normal 0 1))
          b (splice [:b i :c] point a 1.0 0.0)
          c (trace [:c j i] (d/normal b 1))]
      (trace [:d (+ i j)] (d/normal c 1)))))

(def branch-a (gen ^:static [m] (let [a (trace :a (d/normal m 1))] a)))
(def branch-b (gen ^:static [m] (let [a (trace :a (d/normal (+ m 5.0) 1)) b (trace :b (d/normal a 1))] b)))
(def branch-c (gen ^:static [m] (let [c (trace :c (d/normal m 1))] c)))

(def switch-of-static
  "A Switch between static branches: the update from one to the other must
   not reuse the old branch's node values (review, 2026-10-20)."
  (gen [i m] (splice :s (comb/switch branch-a branch-b) i m)))

(def callee-changes
  "A splice whose callee is another static function after an update."
  (gen [pick m] (splice :x (if (= pick 0) branch-a (if (= pick 1) branch-b branch-c)) m)))

(def aliased-inner
  "An inner gen spelled through an alias, spliced twice (the post-M5
   review's I2-02): two draws, not one hoisted into the outer function."
  (dyn/gen ^:static [m] (let [g (dyn/gen [] (trace :a (d/normal m 1.0))) y (splice :y g) z (splice :z g)] [y z])))

(def models
  [[:linreg linreg [[1.0 2.0 3.0]] [[[1.0 2.5]] [[0.5 1.0 1.5 2.0]]]]
   [:hierarchical hierarchical [3] [[2] [4]]]
   [:aliased-inner aliased-inner [0.0] [[1.0]]]
   [:destructured destructured [{:mu 0.0 :sd 1.0}] [[{:mu 1.0 :sd 1.0}]]]
   [:nested nested [0.0] [[1.5]]]
   [:rules rules [[0.1 0.2 0.3] 2 {:p 0.0 :r 1.0}]
    [[[0.1 0.9 0.3] 2 {:p 0.0 :r 1.0}] [[0.1 0.2] 2 {:p 0.0 :r 1.0}] [[0.1 0.2 0.3 0.4] 1 {:p 0.0 :r 1.0}]
     [[0.1 0.2 0.3] 5 {:p 0.0}]]]
   [:double-index double-index [[0.0 0.0] 1.0] [[[0.0 0.5] 1.0]]]
   [:rebound rebound [1.0 2.0 {:z 30.0} 0.5] [[1.0 3.0 {:z 31.0} 0.5]]]
   [:rest-rebound rest-rebound [1.0 2.0] [[1.0 5.0]]]
   [:callee-store callee-store [0.0] [[1.0]]]
   [:dict-rules dict-rules [{:j 0.5 :x 1.0}] [[{:j 0.7 :x 1.0}] [{:x 1.0}] [{:j 0.5 :x 1.0 :z 2.0}]]]
   [:expanded-heads expanded-heads [0.5] [[1.5]]]
   [:quoting quoting [0.0] [[1.0]]]
   [:hoisting hoisting [0.0] [[1.0]]]
   [:computed computed [0 1] [[1 1] [0 2] [2 0]]]])

;; dynamic parents of static functions: the general laws (a static callee's
;; update from another graph's trace was wrong before the review)
(laws/deflaws static-callees [[:switch-of-static switch-of-static [0 0.5] [[1 0.5]]]
                              [:callee-changes callee-changes [0 0.5] [[1 0.5] [2 0.5]]]]
  100)

(laws/deflaws static models 100)
(laws/deflaws static models 100 laws/static)
;; the rules model alone, often enough that a case with unchanged arguments,
;; one choice constrained and the element it moves unconstrained comes up
(laws/deflaws static-rules (filterv (comp #{:rules} first) models) 300 laws/static)
;; every model but :destructured (a uniform between chosen bounds) has fixed supports
(laws/deflaws static (filterv (comp #{:linreg :hierarchical :nested :rules :dict-rules :double-index :callee-store
                                      :expanded-heads :quoting :hoisting :computed}
                                     first)
                               models)
  100 laws/fixed-support)

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
;; Gradients through the compiled score (the final law audit, F2)
;; ---------------------------------------------------------------------------

(def chain-static
  (gen ^:static [] (let [a (trace :a (d/normal 0 1)) b (trace :b (d/normal a 1)) y (trace :y (d/normal b 0.5))] y)))

(defn- continuous-gradients
  "The choice gradients of gf's trace from key k on args, over its float
   choices: a map of path to gradient."
  [gf args k]
  (let [tr (gfi/simulate gf args k)
        paths (filter #(instance? Double (ts/host (cm/get-choice (:choices tr) %))) (cm/addresses (:choices tr)))
        g (:gradients (grad/choice-gradients tr (sel/from-paths paths)))]
    (into {} (map (fn [p] [p (ts/host (cm/get-choice g p))]) paths))))

(deftest choice-gradients-through-the-compiled-score-are-the-handler-s
  ;; a static gf's assess is its compiled score fn, and so are the
  ;; gradients HMC, MALA and MAP take (DESIGN.md §9). Planted bug (the
  ;; audit's S2b): a stop-gradient on each trace node's binding in
  ;; score-form keeps every value and passed all 123 gradient, mcmc, static
  ;; and vi tests; here da is -0.3 and db 0.5
  (testing "Eq. 2.12 against the closed form: d/da = -a + (b - a) = -0.8, d/db = -(b - a) + 4 (y - b) = 5.7"
    (let [tr (:trace (gfi/generate chain-static [] (cm/choicemap :a 0.3 :b -0.2 :y 1.1) (rng/key 0)))
          g (:gradients (grad/choice-gradients tr (sel/select :a :b)))]
      (is (ts/close? -0.8 (ts/host (cm/get-choice g :a))))
      (is (ts/close? 5.7 (ts/host (cm/get-choice g :b))))))
  (testing "the same as the body without the flag, over the static models with float choices"
    (doseq [[label gf args] [[:chain chain-static []] [:linreg linreg [[1.0 2.0 3.0]]] [:hierarchical hierarchical [3]]
                              [:nested nested [0.0]] [:double-index double-index [[0.0 0.0] 1.0]]]
            seed (range 3)]
      (testing [label seed]
        (let [s (continuous-gradients gf args (rng/key seed))
              d (continuous-gradients (assoc gf :static nil) args (rng/key seed))]
          (is (seq s))
          (is (= (set (keys s)) (set (keys d))))
          (doseq [p (keys s)] (is (ts/close? (s p) (d p)) (str p))))))))

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

(deftest a-store-keeps-its-identity-and-its-skipping
  ;; with no outer store a gf's own is not copied, so the next update finds
  ;; it unchanged at once; and a gf with a store skips as one without
  ;; (review 3: nothing tested it)
  (let [store {:w 1.0}
        runs (atom 0)
        model (dyn/with-params (gen ^:static [] (let [a (trace :a (d/normal (param :w) 1))
                                                      a2 (do (swap! runs inc) (* a a))
                                                      b (trace :b (d/normal 0 1))
                                                      y (trace :y (d/normal a2 1))]
                                                  [b y]))
                               store)
        tr (gfi/simulate model [] (rng/key 67))]
    (is (identical? store (:gen.dynamic/params tr)))
    (reset! runs 0)
    (gfi/update tr (cm/choicemap :b 0.3) (rng/key 68))
    (is (= 0 @runs))))

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
;; The review's cases (2026-10-20), each wrong or silent before its fix
;; ---------------------------------------------------------------------------

(def ar-step
  "An AR(1) step whose coefficient is a parameter: it returns its choice, so
   a kept choice is an unchanged state."
  (gen [_t x] (trace :x (d/normal (* (param :w) x) 1))))

(def chain-production
  (gen [x] (let [v (trace :v (d/normal (* (param :w) x) 1))]
             {:value v :children (if (clojure.core/< x 2) [(clojure.core/+ x 1)] [])})))

(def chain-aggregation (gen [v ws] (+ v (reduce + 0.0 ws))))

(defn- consistent-trace? [gf tr]
  (ts/close? (:score tr) (:weight (gfi/assess gf (:args tr) (:choices tr) (rng/key 0)))))

(deftest a-store-from-a-choice-reaches-the-callee
  ;; an ordinary update or regenerate: a choice is the callee's store (review 3)
  (let [chain (gen [] (splice :u (comb/unfold ar-step) 3 1.0))
        model (gen [] (let [a (trace :a (d/uniform 0 1))] (splice :c (dyn/with-params chain {:w a}))))
        tr (gfi/simulate model [] (rng/key 60))
        {u :trace w :weight} (gfi/update tr (cm/choicemap :a 0.9) (rng/key 61))]
    (is (consistent-trace? model u))
    (is (ts/close? (- (:score u) (:score tr)) w))
    (is (consistent-trace? model (:trace (gfi/regenerate tr (sel/select :a) (rng/key 62)))))))

(deftest a-splice-without-an-address-brings-its-own-store
  ;; inline, the callee's own store is in effect for its calls (review 3,
  ;; F2; the kernel defined once, so the store rule alone must see it: an
  ;; inline one is another object on every run, review 4)
  (let [k (gen [x] (trace :y (d/normal (+ x (param :w)) 1)))
        inner (gen [xs] (splice :m (comb/map k) xs))
        outer (dyn/with-params (gen [g xs] (splice g xs)) {:w 0.5})
        xs [1.0 1.0]
        tr (gfi/simulate outer [(dyn/with-params inner {:w 5.0}) xs] (rng/key 63))
        {u :trace w :weight} (gfi/update tr [inner xs] [diff/unknown-change diff/no-change] cm/EMPTY (rng/key 64))]
    (is (not (ts/close? 0.0 w)))
    (is (consistent-trace? outer u))))

(deftest a-combinator-given-another-kernel-reruns-it
  ;; a trace given a gen-fn whose kernel is another object (here with
  ;; another store) keeps no step (review 3)
  (let [k1 (dyn/with-params ar-step {:w 0.5})
        k2 (dyn/with-params ar-step {:w 2.0})
        tr (gfi/simulate (comb/unfold k1) [3 1.0] (rng/key 65))
        {u :trace w :weight} (gfi/update (assoc tr :gen-fn (comb/unfold k2)) cm/EMPTY (rng/key 66))]
    (is (ts/close? (- (:weight (gfi/assess (comb/unfold k2) [3 1.0] (:choices tr) (rng/key 0))) (:score tr)) w))
    (is (consistent-trace? (comb/unfold k2) u))))

(deftest a-changed-store-reruns-what-reads-it
  ;; a callee's param read, a param in a splice argument, an aliased param,
  ;; a combinator's kernel reading one (under a static or a dynamic parent,
  ;; nested too; the confirmation review's N8): no argument shows the
  ;; change, so nothing that may read the store is skipped. The weight of an
  ;; update that keeps every choice is the change in their density.
  (let [reader (gen [] (trace :y (d/normal (param :w) 1)))
        kernel (gen [x] (trace :y (d/normal x 1)))
        reading-kernel (gen [x] (trace :y (d/normal (+ x (param :w)) 1)))
        cases {:in-a-callee (gen ^:static [] (let [c (splice :c reader)] c))
               :in-an-argument (gen ^:static [] (let [ys (splice :ys (comb/map kernel) (param :xs))] ys))
               :aliased (gen ^:static [] (let [p param y (trace :y (d/normal (p :w) 1))] y))
               :in-a-map-s-kernel (gen ^:static [] (let [ys (splice :ys (comb/map reading-kernel) [0.0 0.0])] ys))
               :in-a-map-s-kernel-dynamic (gen [] (splice :ys (comb/map reading-kernel) [0.0 0.0]))
               :in-a-nested-map (gen [] (splice :ys (comb/map (comb/map reading-kernel)) [[0.0 0.0] [0.0]]))
               ;; (review 3, F1: these kept steps and nodes, under either parent)
               :in-an-unfold-without-parameters (gen [] (splice :u (comb/unfold ar-step) 3 1.0))
               :in-a-static-unfold (gen ^:static [] (let [u (splice :u (comb/unfold ar-step) 3 1.0)] u))
               :in-a-switch-over-an-unfold (gen [] (splice :s (comb/switch (comb/unfold ar-step)) 0 3 1.0))
               :in-a-call-at-of-an-unfold (gen [] (splice :c (comb/call-at (comb/unfold ar-step)) 3 1.0 :k))
               :in-a-map-of-unfolds (gen [] (splice :m (comb/map (comb/unfold ar-step)) [3 3] [1.0 1.0]))
               :in-a-recurse (gen [] (splice :r (comb/recurse chain-production chain-aggregation 1) 1))
               :in-a-static-recurse (gen ^:static [] (let [r (splice :r (comb/recurse chain-production chain-aggregation 1) 1)] r))}
        before {:w 0.0 :xs [0.0 0.0]}
        after {:w 3.0 :xs [3.0 3.0]}]
    (doseq [[label model] cases]
      (testing label
        (let [tr (fn [gf] (gfi/simulate (dyn/with-params gf before) [] (rng/key 40)))
              weight (fn [gf] (:weight (gfi/update (assoc (tr gf) :gen-fn (dyn/with-params gf after)) cm/EMPTY (rng/key 41))))
              truth (- (:weight (gfi/assess (dyn/with-params model after) [] (:choices (tr model)) (rng/key 42)))
                       (:score (tr model)))]
          (is (not (ts/close? 0.0 truth)))
          (is (ts/close? truth (weight model)))
          (is (ts/close? truth (weight (assoc model :static nil)))))))))

(deftest stores-are-compared-without-reading-arrays
  ;; an array that is not the same object counts as changed (confirmation
  ;; review N9: = read both on the host, inside the handler); host values
  ;; and collections of them compare by value (review 5, R5-3)
  (let [same? gfi/same-value? x (array/array [1.0 2.0])]
    (is (same? {:w x :n 2.0} {:w x :n 2.0}))
    (is (not (same? {:w x} {:w (array/array [1.0 2.0])})))
    (is (not (same? {:w x} {:w x :v 1.0})))
    (is (same? {:w [1.0 2.0] :ks #{:a :b}} {:w (vector 1.0 (+ 1.0 1.0)) :ks (hash-set :b :a)}))
    (is (not (same? {:w [1.0 2.0]} {:w [1.0 3.0]})))))

(deftest a-local-is-not-the-var-it-shadows
  ;; a local bound inside a gen body is not in the inner gen's &env: count
  ;; here is the local, so its binding has no rule, and an update with a
  ;; vector-diff reruns what reads it. (The first version bound the local
  ;; outside both gens, where &env had it before the fix; the confirmation
  ;; review found it could not fail.)
  (let [outer (gen [] (let [count (fn [v] (reduce clojure.core/+ v))]
                        (gen ^:static [ys] (let [n (count ys) y (trace :y (d/normal n 1))] y))))
        g (:retval (gfi/simulate outer [] (rng/key 42)))
        update-weight (fn [gf] (let [tr (gfi/simulate gf [[1.0 2.0]] (rng/key 2))]
                                 (:weight (gfi/update tr [[1.0 5.0]] [(diff/vector-diff 2 2 {1 diff/unknown-change})] cm/EMPTY (rng/key 3)))))]
    (is (nil? (:rule (first (:nodes (:static g))))))
    (is (not (ts/close? 0.0 (update-weight g))))
    (is (ts/close? (update-weight (assoc g :static nil)) (update-weight g)))))

(defn- static-and-dynamic-updates
  "The update of gf's trace on args to args' (argdiffs), and the same
   body's without ^:static, each with its previous trace as :prev."
  [gf args args' argdiffs]
  (for [g [gf (assoc gf :static nil)]]
    (let [tr (gfi/simulate g args (rng/key 56))]
      (assoc (gfi/update tr args' argdiffs cm/EMPTY (rng/key 57)) :prev tr))))

(deftest a-name-in-the-body-is-not-the-var-it-shadows
  ;; genclj-q2g1, found by the G2 rewrites: a binding, an argument or a
  ;; name the return expression calls, named like a var with a diff rule
  ;; (nth), was read as that var, so an update kept a stale value: weight
  ;; 0 where the dynamic body's is -2.35, or claimed no change of the
  ;; return value
  (let [sum (fn [v _] (reduce + v))
        moved (diff/vector-diff 2 2 {1 diff/unknown-change})]
    (doseq [[what gf args args' argdiffs]
            [["a binding" (gen ^:static [xs] (let [nth (fn [v _] (reduce + v)) s (nth xs 0) y (trace :y (d/normal s 1))] y))
              [[1.0 2.0]] [[1.0 5.0]] [moved]]
             ["an argument" (gen ^:static [nth xs] (let [s (nth xs 0) y (trace :y (d/normal s 1))] y))
              [sum [1.0 2.0]] [sum [1.0 5.0]] [diff/no-change moved]]
             ["the return expression" (gen ^:static [xs] (let [nth (fn [v _] (reduce + v)) y (trace :y (d/normal 0 1))] (nth xs 0)))
              [[1.0 2.0]] [[1.0 5.0]] [moved]]]]
      (testing what
        (let [[static dynamic] (static-and-dynamic-updates gf args args' argdiffs)]
          (is (ts/close? (:weight dynamic) (:weight static)) "the weight")
          (is (ts/close? (:score (:trace dynamic)) (:score (:trace static))) "the score")
          (is (diff/consistent? (:retval (:prev static)) (:retval (:trace static)) (:retdiff static)) "the retdiff"))))))

(deftest destructuring-binds-what-clojure-binds
  (let [gf (gen ^:static [m] (let [{:keys [a/b c]} m y (trace :y (d/normal (+ b c) 1))] y))]
    (is (some? (:retval (gfi/simulate gf [{:a/b 1.0 :c 2.0}] (rng/key 43)))))))

(deftest a-batched-static-run-guards-as-the-body-does
  ;; one-call bindings run the guarded fns: the fallback is the dynamic body's
  (let [model (gen ^:static [] (let [b (trace :b (d/bernoulli 0.5)) nb (not b) y (trace :y (d/normal (if nb 1.0 -1.0) 1))] y))
        reason #(:message (batched/fallback-reason (batched/simulate % [] 6 (rng/key 44))))]
    (is (= (reason (assoc model :static nil)) (reason model)))))

(deftest resampled-node-values-follow-their-particles
  ;; a node value in a list is gathered with its particle (batched/take)
  (binding [batched/*spot-checks* 0]
    (let [model (gen ^:static [] (let [a (trace :a (d/normal 0 1)) pair (list a 1.0) y (trace :y (d/normal (first pair) 1))] y))
          ps (batched/take (batched/simulate model [] 4 (rng/key 45)) [3 2 1 0])
          {ws :weights} (batched/update ps [] [] (cm/choicemap :y 0.5) (rng/key 46))]
      (is (ts/close? (array/array (mapv #(ts/number (:weight (gfi/update % (cm/choicemap :y 0.5) (rng/key 46)))) (batched/traces ps))) ws)))))

(deftest a-resampled-batched-trace-still-skips
  ;; batched/take keeps the trace's generative function, whose identity a
  ;; static update checks (confirmation review N3: it rebuilt it, and every
  ;; update after a resampling reran every node)
  (let [runs (atom 0)
        model (gen ^:static [] (let [a (trace :a (d/normal 0 1))
                                     a2 (do (swap! runs inc) (* a a))
                                     y (trace :y (d/normal a2 1))
                                     z (trace :z (d/normal 0 1))]
                                 [y z]))
        ;; (no spot checks, which rerun particles alone)
        reruns (fn [ps] (binding [batched/*spot-checks* 0]
                          (reset! runs 0) (batched/update ps [] [] (cm/choicemap :z 0.5) (rng/key 49)) @runs))]
    ;; with a store of collections too, which take once rebuilt (review 3)
    (doseq [[label gf] [["no store" model] ["a vector" (dyn/with-params model {:xs [1.0 2.0]})]
                        ["a nested map" (dyn/with-params model {:layer {:b 1.0}})]]]
      (let [ps (batched/simulate gf [] 4 (rng/key 48))]
        (is (= 0 (reruns ps)) label)
        (is (= 0 (reruns (batched/take ps [3 2 1 0]))) label)))))

(deftest a-resampled-seq-is-gathered-lazily
  ;; a node value may be an infinite seq (confirmation review N4: take
  ;; forced it, and hung)
  (let [model (gen ^:static [] (let [a (trace :a (d/normal 0 1)) ones (map inc (range 5)) y (trace :y (d/normal (+ a (first ones)) 1))] y))
        taken (batched/take (batched/simulate model [] 4 (rng/key 50)) [1 0 3 2])
        ones (get-in taken [:trace :gen.dynamic/node-values 1])]
    (is (seq? ones))
    (is (not (realized? ones)))
    (is (= [1 2 3 4 5] ones))))

(deftest the-score-fn-checks-what-the-handler-checks
  ;; assess by the score fn and by the handler (the same body without the
  ;; flag) agree on every misuse (confirmation review: check-static-choices!
  ;; had no test; N2 a static callee's checks, N6 the address false, N7 a
  ;; splice of a value)
  (let [inner (gen ^:static [m] (let [y (trace :y (d/normal m 1))] y))
        outer (gen ^:static [] (let [c (splice :c inner 0.0)] c))
        paths (gen ^:static [] (let [a (trace [:p :a] (d/normal 0 1))] a))
        both-throw (fn [re gf args choices]
                     (doseq [g [gf (assoc gf :static nil)]]
                       (is (thrown-with-msg? clojure.lang.ExceptionInfo re (gfi/assess g args choices (rng/key 51)))
                           (if (:static g) "score fn" "handler"))))]
    (testing "a choice no node has"
      (both-throw #"not visited" outer [] (cm/choicemap [:c :y] 0.5 :bogus 1.0))
      (both-throw #"not visited" paths [] (cm/choicemap [:p :a] 0.5 [:p :b] 1.0))
      (both-throw #"not visited" outer [] (cm/choicemap [:c :y] 0.5 [:c :bogus] 1.0)))
    (testing "a static callee given the wrong arguments"
      (both-throw #"takes 1 argument" (gen ^:static [] (let [c (splice :c inner 0.0 1.0)] c)) [] (cm/choicemap [:c :y] 0.5)))
    (testing "a splice of a value"
      (both-throw #"needs a generative function" (gen ^:static [g] (let [c (splice :c g 1.0)] c)) [5] (cm/choicemap [:c :y] 0.5)))
    (testing "the address false"
      (let [gf (gen ^:static [] (let [a (trace false (d/normal 0 1)) b (trace :b (d/normal a 1))] b))
            tr (gfi/simulate gf [] (rng/key 52))]
        (is (ts/close? (:score tr) (:weight (gfi/assess gf [] (:choices tr) (rng/key 53)))))))))

(deftest an-inline-gen-is-a-pure-binding
  (let [gf (gen ^:static [] (let [k (gen [m] (trace :z (d/normal m 1))) z (splice :k k 0.0)] z))]
    (is (number? (ts/host (:retval (gfi/simulate gf [] (rng/key 47)))))))
  (testing "however gen is spelled (the post-M5 review's I2-02)"
    ;; The analysis knew a nested gen by its spelling, gen or gen.dynamic/gen:
    ;; through an alias its trace was hoisted into the outer function, one
    ;; draw where the body makes two. Planted, the spelling check fails the
    ;; aliased case here and the :aliased-inner laws.
    (doseq [[spelling gf] [["referred" (gen ^:static [] (let [g (gen [] (trace :a (d/normal 0.0 1.0))) y (splice :y g) z (splice :z g)] [y z]))]
                           ["aliased" (dyn/gen ^:static [] (let [g (dyn/gen [] (trace :a (d/normal 0.0 1.0))) y (splice :y g) z (splice :z g)] [y z]))]
                           ["qualified" (gen.dynamic/gen ^:static [] (let [g (gen.dynamic/gen [] (trace :a (d/normal 0.0 1.0)))
                                                                           y (splice :y g) z (splice :z g)]
                                                                       [y z]))]]]
      (testing spelling
        (let [static (gfi/simulate gf [] (rng/key 48))
              dynamic (gfi/simulate (assoc gf :static nil) [] (rng/key 48))]
          (is (= [:compute :splice :splice] (mapv :kind (:nodes (static/ir gf)))) "the inner gen is a value")
          (is (= (cm/addresses (:choices dynamic)) (cm/addresses (:choices static))))
          ;; computed through different graphs: on the host, within the tolerance
          (doseq [p (cm/addresses (:choices static))]
            (is (ts/close? (ts/host (cm/get-choice (:choices dynamic) p)) (ts/host (cm/get-choice (:choices static) p)))
                (str p)))
          (is (ts/close? (:score dynamic) (:score static))))))))

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
  (testing "the address false is a node's, as any constant"
    (is (str/includes? (static/dot (gen ^:static [] (let [a (trace false (d/normal 0 1))] a))) "\"false\" [shape=ellipse")))
  (testing "a string address is escaped: \"\\\"y\\\"\", not \"\"y\"\""
    (let [dot (static/dot (gen ^:static [x] (let [y (trace "y" (d/normal x 1))] y)))]
      (is (str/includes? dot "\"\\\"y\\\"\" [shape=ellipse"))))
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

(deftest a-call-nested-once-is-hoisted
  (let [{:keys [nodes return]} (static/ir hoisting)]
    (is (= [:x :y nil :w :z :b nil :v nil :u] (mapv :addr nodes)) "the calls in evaluation order, each before what reads it")
    (is (= [:trace :trace :compute :trace :trace :trace :compute :trace :compute :trace] (mapv :kind nodes)))
    (is (= '[pt] (:names (nth nodes 2))))
    (is (some (set (:names (nth nodes 3))) (:inputs (nth nodes 4))) ":z reads :w's node")
    (is (= (set (concat (:names (nth nodes 8)) (:names (nth nodes 9)))) (set (:inputs return))) "the return reads q and :u"))
  (testing "a body as Gen.jl's tests write it"
    (let [{tr :trace w :weight} (gfi/generate hoisting [0.0] (cm/choicemap :x 1.0 :y 2.0 :w 0.5 :z 1.0 :b true :v 0.0 :u 1.0)
                                              (rng/key 51))
          lp (fn [v mu sd] (dc/log-prob (d/normal mu sd) v))]
      (is (ts/close? (+ (lp 1.0 0.0 1) (lp 2.0 0.0 2) (lp 0.5 0 1) (lp 1.0 1.5 1) (dc/log-prob (d/bernoulli 0.5) true)
                        (lp 0.0 0.0 1.0) (lp 1.0 1.0 1))
                     w))
      (is (ts/close? 1.0 (:retval tr))))))

(deftest a-computed-address-is-a-node-at-its-constant-prefix
  (let [{:keys [nodes]} (static/ir computed)]
    (is (= [[:trace :a '[i]] [:splice :b '[i :c]] [:trace :c '[j i]] [:trace :d '[(+ i j)]]]
           (mapv (juxt :kind :addr :keys) nodes))))
  (let [tr (gfi/simulate computed [0 1] (rng/key 52))]
    (is (= #{[:a 0] [:b 0 :c :y] [:c 1 0] [:d 1]} (set (cm/addresses (:choices tr)))) "the keys as the dynamic body computes them")
    (testing "update: a node whose keys did not change is kept; one whose did is drawn at the new path"
      (let [{t :trace d :discard} (gfi/update tr [1 1] [diff/unknown-change diff/no-change] cm/EMPTY (rng/key 53))]
        (is (= #{[:a 1] [:b 1 :c :y] [:c 1 1] [:d 2]} (set (cm/addresses (:choices t)))))
        (is (= (:choices tr) d) "the old paths are discarded"))
      (let [{t :trace} (gfi/update tr [0 1] [diff/no-change diff/no-change] (cm/choicemap [:d 1] 0.5) (rng/key 54))]
        (is (identical? ((:gen.dynamic/subtraces tr) [:b 0 :c]) ((:gen.dynamic/subtraces t) [:b 0 :c])) "a kept call")
        (is (= 0.5 (t [:d 1])) "a constraint at a computed path reaches its node")))
    (testing "assess: a choice under a computed prefix but off its path is unvisited"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"not visited"
                            (gfi/assess computed [0 1] (cm/set-choice (:choices tr) [:a 5] 0.0) (rng/key 55))))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"not visited"
                            (gfi/assess computed [0 1] (cm/set-choice (:choices tr) [:b 0 :d] 0.0) (rng/key 55)))))))

(deftest a-call-not-evaluated-once-is-not-static
  (binding [*ns* (the-ns 'gen.static-test)]
    (doseq [[why form] [[#"a branch of an if calls trace.*\(trace :x" '(gen ^:static [b] (let [a (if b (trace :x (d/normal 0 1)) 0.0)] a))]
                        [#"a branch of an if calls trace" '(gen ^:static [b] (let [a (cond b 1 (trace :c (d/bernoulli 0.5)) 2 :else 3)] a))]
                        [#"a branch of an if calls splice" '(gen ^:static [b g] (let [a (when b (splice :s g))] a))]
                        [#"a function inside an expression calls trace" '(gen ^:static [xs] (let [a (mapv #(trace [:y %] (d/normal 0 1)) xs)] a))]
                        [#"a let inside an expression calls trace" '(gen ^:static [] (let [a (+ 1 (let [b 2] (trace :a (d/normal b 1))))] a))]
                        [#"an and calls trace, which runs only when the arguments before it are true" '(gen ^:static [b] (let [a (and b (trace :a (d/bernoulli 0.5)))] a))]
                        [#"an or calls trace" '(gen ^:static [b] (let [a (or b (trace :a (d/bernoulli 0.5)))] a))]
                        [#"trace's address must start with a constant key" '(gen ^:static [i] (let [a (trace [i :a] (d/normal 0 1))] a))]
                        [#"splice's address must start with a constant key" '(gen ^:static [i g] (let [a (splice [i] g)] a))]]]
      (testing (str why)
        (is (re-find why (str (static-error form))))))))

(deftest a-body-that-is-not-static-is-a-compile-error
  (binding [*ns* (the-ns 'gen.static-test)]
    (doseq [[why form] [["not one let" '(gen ^:static [] (trace :a (d/normal 0 1)))]
                        ["two forms" '(gen ^:static [] (let [a 1] a) 2)]
                        ["an address whose first key is computed" '(gen ^:static [i] (let [a (trace [i :a] (d/normal 0 1))] a))]
                        ["a computed address" '(gen ^:static [i] (let [a (trace i (d/normal 0 1))] a))]
                        ["a trace in a function" '(gen ^:static [] (let [a (mapv (fn [m] (trace :a (d/normal m 1))) [0])] a))]
                        ["a branch on a choice" '(gen ^:static [] (let [b (trace :b (d/bernoulli 0.5))
                                                                         a (if b (trace :x (d/normal 0 1)) 0.0)] a))]
                        ["a splice without an address" '(gen ^:static [g] (let [a (splice g)] a))]
                        ["an untraced draw" '(gen ^:static [] (let [a (untraced (d/normal 0 1))] a))]
                        ["a trace in the return expression's branch" '(gen ^:static [] (let [a 1] (if (pos? a) (trace :a (d/normal a 1)) 0)))]
                        ["a repeated address" '(gen ^:static [] (let [a (trace :a (d/normal 0 1)) b (trace :a (d/normal a 1))] b))]
                        ["an address under another" '(gen ^:static [g] (let [a (trace :a (d/normal 0 1)) b (splice [:a :b] g)] b))]]]
      (testing why
        (is (re-find #"gen \^:static" (str (static-error form))))))))

(deftest a-binding-s-head-is-macroexpanded
  (let [{:keys [nodes return]} (static/ir expanded-heads)]
    (is (= [:trace :splice :trace] (mapv :kind nodes)) "the user macros' calls, the return expression's last")
    (is (= [:a :b :c] (mapv :addr nodes)))
    (is (= '[(noisy-at :a m) (called-at :b point a 1.0 0.0)] (mapv :form (take 2 nodes))) "the forms as written")
    (is (= (:names (peek nodes)) (:inputs return))))
  (let [{tr :trace w :weight} (gfi/generate expanded-heads [0.5] (cm/choicemap :a 1.0 [:b :y] 2.0 :c 3.0) (rng/key 50))]
    (is (ts/close? (+ (dc/log-prob (d/normal 0.5 1) 1.0) (dc/log-prob (d/normal 1.0 1) 2.0) (dc/log-prob (d/normal 2.0 1) 3.0)) w))
    (is (ts/close? 3.0 (:retval tr))))
  (testing "a local is not the macro it shadows"
    (let [noisy-at (fn [_ x] (* 2 x))
          gf (gen ^:static [m] (let [a (noisy-at :a m)] a))]
      (is (= [:compute] (mapv :kind (:nodes (static/ir gf)))))
      (is (ts/close? 3.0 (gf 1.5)))))
  (testing "a head macro expanding to a computation: its inputs are the expansion's"
    (binding [*ns* (the-ns 'gen.static-test)]
      (let [form '(gen ^:static [x] (let [y (twice-x) z (trace :z (d/normal y 1.0))] z))]
        (is (nil? (static-error form)) "it compiles, as the dynamic body does")
        (when-let [gf (try (eval form) (catch Exception _ nil))]
          (is (= '[x] (:inputs (first (:nodes (static/ir gf))))))
          (is (= '(twice-x) (:form (first (:nodes (static/ir gf))))) "the form as written")
          (let [[static dynamic] (static-and-dynamic-updates gf [1.0] [2.0] [diff/unknown-change])]
            (is (ts/close? (:weight dynamic) (:weight static)) "an update reruns it")
            (is (ts/close? (:score (:trace dynamic)) (:score (:trace static))))))
        (is (nil? (static-error '(gen ^:static [x] (let [] (twice-x))))) "in the return expression too"))))
  (testing "a macro hiding a call inside an expression is still a compile error"
    ;; only a binding's head is expanded, so the compiler meets the trace
    ;; outside the body, where trace is not bound
    (binding [*ns* (the-ns 'gen.static-test)]
      (is (re-find #"Unable to resolve symbol: trace"
                   (str (static-error '(gen ^:static [] (let [a (hidden-at :a 0)] a)))))))))

(deftest quoted-data-is-data
  (is (= '(trace :x (d/normal a 1)) (second (quoting 0.0))))
  (let [{:keys [nodes]} (static/ir quoting)]
    (is (= [:trace :compute] (mapv :kind nodes)))
    (is (= [] (:inputs (second nodes))) "a quoted symbol is no input"))
  (is (= '(trace :x (d/normal 0 1)) ((gen ^:static [] (let [] '(trace :x (d/normal 0 1)))))) "a quoted return value"))

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
