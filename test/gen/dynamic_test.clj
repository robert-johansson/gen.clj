(ns gen.dynamic-test
  "The dynamic modeling language against the GFI's laws (Cusumano-Towner's
   thesis; docs/research/genjl-parity.md §3), on fixed-structure,
   structure-changing, spliced and discrete models."
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.test.check]
            [gen.array :as array :refer [+ - * /]]
            [gen.array.batch :as batch]
            [gen.batched :as batched]
            [gen.choicemap :as cm]
            [gen.diff :as diff]
            [gen.combinators :as comb]
            [gen.custom :as custom]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :as dyn :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.enumerative :as enum]
            [gen.inference.proposal :as proposal]
            [gen.laws :as laws]
            [gen.random :as rng]
            [gen.selection :as sel]
            [gen.test-support :as ts]
            [gen.trace :as trace]))

;; ---------------------------------------------------------------------------
;; Models
;; ---------------------------------------------------------------------------

(def linreg
  (gen [xs]
    (let [slope (trace :slope (d/normal 0 10))
          intercept (trace :intercept (d/normal 0 10))]
      (doseq [[i x] (map-indexed vector xs)]
        (trace [:y i] (d/normal (+ (* slope x) intercept) 1)))
      slope)))

(def branching
  (gen []
    (if (trace :switch (d/bernoulli 0.4))
      (trace :a (d/normal 0 1))
      (+ (trace :b (d/gamma 2 1)) (trace :c (d/poisson 3.0))))))

(def observation (gen [mu] (trace :x (d/normal mu 1))))

(def hierarchical
  (gen [n]
    (let [mu (trace :mu (d/normal 0 1))]
      (mapv #(splice [:obs %] observation mu) (range n)))))

(def discrete
  (gen []
    (let [k (trace :k (d/categorical [0.2 0.5 0.3]))
          n (trace :n (d/poisson (+ 1.0 k)))]
      (trace :flip (d/bernoulli (/ 1.0 (+ 2.0 n)))))))

(def sub-x (gen [] (trace :y (d/normal 0 1))))

(def kind-changing
  "An address that holds a choice in one branch and a splice in the other."
  (gen []
    (if (trace :s (d/bernoulli 0.5))
      (trace :x (d/normal 0 1))
      (splice :x sub-x))))

(def spliced-distribution
  "A distribution called through the GFI, its parameter an argument."
  (gen [mu]
    (splice :x (d/normal mu 1))
    (trace :y (d/normal 0 1))))

(def p-mh
  "The thesis's p_mh (p. 92): b exists only when a is true."
  (gen []
    (let [z (trace :a (d/bernoulli 0.5))
          z (and z (trace :b (d/bernoulli 0.5)))]
      (trace :c (d/bernoulli (if z 0.9 0.1))))))

(def shared-body
  "A dynamic generative function spliced without an address."
  (gen [mu]
    (let [z (trace :z (d/normal mu 1))]
      (when (trace :more (d/bernoulli 0.5))
        (trace :w (d/normal z 1)))
      z)))

(def shared-namespace
  (gen []
    (let [mu (trace :mu (d/normal 0 1))
          z (splice shared-body mu)]
      (trace :obs (d/normal z 0.5)))))

(def mixtures
  "Mixture and product distributions at trace sites."
  (gen []
    (let [x (trace :x ((d/homogeneous-mixture d/normal) [0.3 0.7] [-2.0 1.0] [0.5 1.0]))
          [u b] (trace :pair ((d/product d/normal d/bernoulli) [x 1.0] [0.4]))]
      (trace :h ((d/heterogeneous-mixture [d/normal d/uniform]) [0.5 0.5] [u 1.0] [-1.0 1.0]))
      b)))

(def array-valued
  "A return value the body builds as an array of shape [2], so an identity
   update builds it again: on the JVM what every 0-d value is on MLX (an
   array the same only as itself, DESIGN.md §9, D1). Planted bug
   (2026-10-22): the identity-update law requiring no-change again fails
   on it, as it failed on Metal."
  (gen [] (let [x (trace :x (d/normal 0 1))] (array/stack [x (+ x 1.0)]))))

(def models
  [[:linreg linreg [[1.0 2.0 3.0]] [[[1.0 2.5]] [[0.5 1.0 1.5 2.0]]]]
   [:shared-namespace shared-namespace []]
   [:mixtures mixtures []]
   [:branching branching []]
   [:hierarchical hierarchical [3] [[1] [5]]]
   [:discrete discrete []]
   [:kind-changing kind-changing []]
   [:spliced-distribution spliced-distribution [0.0] [[2.0] [-1.5]]]
   [:p-mh p-mh []]
   [:array-valued array-valued []]])

(defn- close? [x y] (ts/close? x y))

;; ---------------------------------------------------------------------------
;; Laws (gen.laws), and their checks on structure changes
;; ---------------------------------------------------------------------------

(laws/deflaws dynamic models 100 (merge laws/all laws/conventions))

;; every address has a fixed support (the mixtures' union is the reals)
(laws/deflaws dynamic (filterv (comp #{:linreg :shared-namespace :mixtures :spliced-distribution :p-mh :array-valued} first) models)
  100 laws/fixed-support)

(deftest selecting-absent-addresses-changes-nothing
  ;; p. 149: a selection may name addresses the trace does not have (the
  ;; final law audit, 4.b). A pin of the thesis's statement, which fails a
  ;; regenerate that throws on an absent address or misreads one. Planted
  ;; bug (2026-10-20): regenerate checking its selection as update checks
  ;; its constraints, loud on a selected address it does not visit
  (let [branch (gen [] (let [a (trace :a (d/bernoulli 0.3))] (when a (trace :b (d/normal 0 1))) (trace :c (d/normal 0 1))))
        tr (:trace (gfi/generate branch [] (cm/choicemap :a false :c 0.4) (rng/key 1)))
        {t :trace w :weight} (gfi/regenerate tr (sel/select :b) (rng/key 2))]
    (is (= (:choices tr) (:choices t)))
    (is (== 0.0 (ts/number w)))))

(deftest update-without-fresh-choices-is-exactly-reversible
  ;; constraints covering the new branch: w = Δscore, the reverse weight is
  ;; -w and the reverse discard is σ (Prop 2.3.1, Def 2.3.1)
  (let [tr (first (filter #(true? (ts/host (% :switch))) (map #(gfi/simulate branching [] (rng/key %)) (range 50))))
        sigma (cm/choicemap :switch false :b 2.0 :c 3)
        {t2 :trace w :weight disc :discard} (gfi/update tr sigma (rng/key 1))
        {t3 :trace w3 :weight disc3 :discard} (gfi/update t2 disc (rng/key 2))]
    (is (close? w (- (:score t2) (:score tr))))
    (is (= (ts/host (cm/choicemap :switch true :a (tr :a))) (ts/host disc)))
    (is (= (:choices tr) (:choices t3)))
    (is (close? w3 (array/negate w)))
    (is (= sigma disc3))))

(deftest a-choice-that-becomes-a-call-is-discarded
  ;; regression for the audit's E1
  (let [tr (first (filter #(true? (ts/host (% :s))) (map #(gfi/simulate kind-changing [] (rng/key %)) (range 50))))
        {t2 :trace w :weight disc :discard} (gfi/update tr (cm/choicemap :s false) (rng/key 1))]
    (is (= (ts/host (cm/choicemap :s true :x (tr :x))) (ts/host disc)))
    (is (close? w (- (- (:score t2) (:score tr)) (gfi/project t2 (sel/select [:x :y])))))
    (is (= (:choices tr) (:choices (:trace (gfi/update t2 disc (rng/key 2))))))))

(def inlined
  "hierarchical with its spliced calls written inline."
  (gen [n]
    (let [mu (trace :mu (d/normal 0 1))]
      (mapv #(trace [:obs % :x] (d/normal mu 1)) (range n)))))

(deftest splicing-is-compositional
  ;; Prop 2.3.2: a splice updates as its body written inline would
  (let [tr (gfi/simulate hierarchical [3] (rng/key 3))
        tr-inline (:trace (gfi/generate inlined [3] (:choices tr) (rng/key 4)))]
    (is (close? (:score tr) (:score tr-inline)))
    (doseq [sigma [(cm/choicemap :mu 0.7) (cm/choicemap [:obs 1 :x] -2.0) (cm/choicemap :mu -0.3 [:obs 0 :x] 1.0)]]
      (let [r (gfi/update tr sigma (rng/key 5))
            ri (gfi/update tr-inline sigma (rng/key 5))]
        (is (close? (:weight r) (:weight ri)))
        (is (= (:discard r) (:discard ri)))
        (is (= (:choices (:trace r)) (:choices (:trace ri))))))
    (testing "and with changed arguments that remove calls"
      (let [r (gfi/update tr [1] [diff/unknown-change] cm/EMPTY (rng/key 6))
            ri (gfi/update tr-inline [1] [diff/unknown-change] cm/EMPTY (rng/key 6))]
        (is (close? (:weight r) (:weight ri)))
        (is (= (:discard r) (:discard ri)))))))

;; ---------------------------------------------------------------------------
;; Structure, splices, errors
;; ---------------------------------------------------------------------------

(deftest structure-changing-update
  (let [k (rng/key 4)
        tr (first (filter #(true? (ts/host (% :switch))) (map #(gfi/simulate branching [] (rng/fold-in k %)) (range 50))))
        {t2 :trace w :weight disc :discard} (gfi/update tr (cm/choicemap :switch false) (rng/key 5))]
    (is (= #{[:switch] [:b] [:c]} (set (cm/addresses (:choices t2)))) "the other branch is sampled fresh")
    (is (= (ts/host (cm/choicemap :switch true :a (tr :a))) (ts/host disc)) "the old value and the removed choice are discarded")
    (is (close? w (- (gfi/project t2 (sel/select :switch)) (:score tr))))
    (is (= (:choices tr) (:choices (:trace (gfi/update t2 disc (rng/key 6))))) "and the update reverses")))

(deftest splices-nest-choices
  (let [tr (gfi/simulate hierarchical [2] (rng/key 1))]
    (is (= [[:mu] [:obs 0 :x] [:obs 1 :x]] (cm/addresses (:choices tr))))
    (is (= (tr [:obs 1 :x]) (second (:retval tr))))
    (let [{t2 :trace} (gfi/update tr (cm/choicemap [:obs 0 :x] 9.0) (rng/key 2))]
      (is (= 9.0 (t2 [:obs 0 :x])))
      (is (= (tr [:obs 1 :x]) (t2 [:obs 1 :x]))))
    (testing "selections reach into splices"
      (let [{t2 :trace} (gfi/regenerate tr (sel/select [:obs 1 :x]) (rng/key 3))]
        (is (= (tr :mu) (t2 :mu)))
        (is (= (tr [:obs 0 :x]) (t2 [:obs 0 :x])))
        (is (not= (tr [:obs 1 :x]) (t2 [:obs 1 :x])))))
    (testing "an update with new arguments adds calls"
      (let [{t2 :trace w :weight} (gfi/update tr [3] [diff/unknown-change] cm/EMPTY (rng/key 4))]
        (is (= 3 (count (:retval t2))))
        (is (close? w 0.0) "the new call is a fresh proposal, which cancels")))))

(deftest errors-are-loud
  (let [k (rng/key 0)]
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"not visited"
                          (gfi/generate linreg [[1.0]] (cm/choicemap :typo 1.0) k)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"used twice"
                          (gfi/simulate (gen [] (trace :x (d/normal 0 1)) (trace :x (d/normal 0 1))) [] k)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"lies under"
                          (gfi/simulate (gen [] (trace :x (d/normal 0 1)) (trace [:x :y] (d/normal 0 1))) [] k)))
    ;; the post-M5 review's P1-05; plant: the check above reading the
    ;; choices made (has-submap?), which an empty Map leaves without any
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"already holds nested choices or calls"
                          (gfi/simulate (gen [] (splice [:x :y] (comb/map observation) []) (trace :x (d/normal 0 1))) [] k))
        "a choice above a call that made no choices")
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"lies under"
                          (gfi/simulate (gen [] (trace :x (d/normal 0 1)) (splice [:x :y] (comb/map observation) [])) [] k))
        "and the other order")
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"nested choice map"
                          (gfi/generate linreg [[1.0]] (cm/choicemap [:slope :oops] 1.0) k)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"no value"
                          (gfi/assess linreg [[1.0]] (cm/choicemap :slope 1.0))))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"needs a distribution"
                          (gfi/simulate (gen [] (trace :x 3.0)) [] k)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"has shape \[3\], but the distribution's values have shape \[\]"
                          (gfi/generate (gen [] (trace :y (d/normal 0 1))) [] (cm/choicemap :y (array/array [1.0 2.0 3.0])) k))
        "a constraint of another shape is not scored as several draws")
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"nested choices"
                          (gfi/generate (gen [] (splice :x (d/normal 0 1))) [] (cm/choicemap [:x :y] 3.0) k))
        "constraints under a spliced distribution are not ignored")
    (testing "a NaN observation is an error, not a NaN weight"
      (let [m (gen [] (trace :y (d/normal 0 1)))
            nan (cm/choicemap :y ##NaN)]
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"generate: a constrained value is NaN" (gfi/generate m [] nan k)))
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"update: a constrained value is NaN"
                              (gfi/update (gfi/simulate m [] k) nan k)))
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"is NaN" (gfi/assess m [] nan)))
        ;; an array value or parameter is read only on an eager backend (rule 5)
        (when (array/concrete? (array/array [0.0]))
          (is (thrown-with-msg? clojure.lang.ExceptionInfo #"is NaN"
                                (gfi/generate (gen [] (trace :v (d/mvnormal [0 0] (array/eye 2)))) []
                                              (cm/choicemap :v (array/array [0.0 ##NaN])) k))))
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"is NaN" (gfi/generate (d/normal 0 1) [] (cm/->Value ##NaN) k)))))
    (testing "constraints are choice maps: a plain map would mean no constraints at all"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"must be a choice map.*from-map"
                            (gfi/generate linreg [[1.0]] {:slope 1.0} k)))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"must be a choice map"
                            (gfi/update (gfi/simulate linreg [[1.0]] k) {:slope 1.0} k)))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"must be a choice map" (gfi/assess linreg [[1.0]] {:slope 1.0})))
      (is (some? (gfi/generate linreg [[1.0]] nil k)) "nil is the empty choice map"))
    (testing "selections are selections, arguments are vectors"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"must be a selection"
                            (gfi/regenerate (gfi/simulate linreg [[1.0]] k) :slope k)))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"must be a vector of arguments" (gfi/simulate linreg 3 k))))
    (testing "an arity error names the generative function's parameters"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"\[xs\] takes 1 argument, given 2"
                            (gfi/simulate linreg [[1.0] 2] k))))
    (testing "an empty address path"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"must not be empty"
                            (gfi/simulate (gen [] (trace [] (d/normal 0 1))) [] k))))
    (testing "NaN parameters"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"normal: mu is NaN" (d/normal ##NaN 1)))
      (when (array/concrete? (array/array [0.0]))
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"mu is NaN" (d/normal (array/array [0.0 ##NaN]) 1))))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"mu is NaN" (d/normal [0.0 ##NaN] 1)) "host data, on every backend")
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"mu is NaN" (d/normal (float Float/NaN) 1)) "a Float too"))
    (testing "update and regenerate take one argdiff per argument"
      (let [tr (gfi/simulate linreg [[1.0]] k)]
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"1 arguments but 0 argdiffs"
                              (gfi/update tr [[2.0]] [] cm/EMPTY k)))
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"1 arguments but 2 argdiffs"
                              (gfi/regenerate tr [[2.0]] [diff/unknown-change diff/unknown-change] (sel/select :slope) k)))))))

(deftest conditionals-branch-on-values
  (let [model (gen [c] (if c (trace :a (d/normal 0 1)) (trace :b (d/normal 0 1))))]
    (testing "a 1-element array that holds false takes the else branch (plain if would not)"
      (is (= [[:b]] (cm/addresses (:choices (gfi/simulate model [(array/array [false])] (rng/key 1)))))))
    (is (= [[:a]] (cm/addresses (:choices (gfi/simulate model [true] (rng/key 1))))))
    (testing "when, cond, and and or expand to if, so they are guarded too"
      (let [m (gen [c] (when c (trace :a (d/normal 0 1))) (cond (and c true) 1 :else 2))]
        (is (cm/empty? (:choices (gfi/simulate m [(array/array [false])] (rng/key 1)))))))
    (testing "local functions shadowing macro names are not expanded"
      (let [m (gen [] (let [when (fn [x] (* 2 x))] (when 21)))]
        (is (= 42 (m)))))))

(deftest batched-handler
  (testing "the same transitions run N particles at once, under the batch tracer"
    (let [k (rng/key 7)
          {tr :trace} (gfi/run-op linreg :simulate [[1.0 2.0]] {:key k :batch-shape [5]})]
      (is (= [] (array/shape (tr :slope))) "inside the trace, values have their per-particle shapes")
      (is (= [5] (array/shape (batch/unwrap (tr :slope) [5]))) "with a leading batch axis underneath")
      (testing "each particle's score is its own assessed density"
        (doseq [[i c] (map-indexed vector (cm/unstack (cm/from-map (array/tree-map #(batch/unwrap % [5]) (cm/to-map (:choices tr)))) 5))]
          (is (close? (array/take (batch/unwrap (:score tr) [5]) i) (:weight (gfi/assess linreg [[1.0 2.0]] c))))))))
  (testing "branching on a batch of values is loud, not silently wrong"
    (is (:gen.array/batched-branch
         (ex-data (try (gfi/run-op branching :simulate [] {:key (rng/key 1) :batch-shape [4]})
                       (catch clojure.lang.ExceptionInfo e e)))))))

(deftest the-implicit-key-source
  (let [model (gen [] (trace :x (d/normal 0 1)))
        before (gfi/implicit-seed)]
    (gfi/seed! 123)
    (let [a [(model) (model)]]
      (gfi/seed! 123)
      (is (= a [(model) (model)]) "seed! makes calls without a key repeat exactly")
      (is (not= (first a) (second a)) "successive calls draw fresh keys")
      (is (= 123 (gfi/implicit-seed))))
    (testing "the stream does not repeat after 2^32 keys (the post-M5 review's P2-09)"
      (gfi/seed! 5)
      (let [k0 (gfi/next-key)]
        (.set ^java.util.concurrent.atomic.AtomicLong @#'gfi/counter (bit-shift-left 1 32))
        (is (not= k0 (gfi/next-key)) "the key at counter 2^32 is not counter 0's")))
    (gfi/seed! before)))

(deftest untraced-calls
  (is (number? (ts/host (linreg [1.0 2.0]))) "calling a generative function runs it, like Gen.jl"))

;; Host code that branches on a boolean choice, (if (tr :b) …), tested an
;; MLX array object, always true, silently (genclj-x6ox). Outside a body, a
;; trace's lookup now returns a 0-d boolean as a host boolean; inside one,
;; values stay arrays, which gen's conditionals read. Planted (2026-10-24,
;; on MLX): no conversion fails "outside"; converting inside a body too
;; fails "inside".
;; A choice map called reads as the trace called (genclj-a2t8): the same
;; host boolean outside a body, the same array inside one, which
;; gen.choicemap/*in-body* tells apart. Planted (2026-10-25): gen.dynamic
;; not binding *in-body* fails "the flag" on every backend, and on MLX
;; "inside a body"; the call without the conversion fails "outside" on MLX.
(deftest a-boolean-choice-reads-as-a-host-boolean-outside-a-body
  ;; a sampled choice (a constrained one is the host value given), false
  ;; for certain, which an array object would make true
  (let [flip (gen [] (trace :b (d/bernoulli 0.0)) (trace :x (d/normal 0 1)))
        tr (gfi/simulate flip [] (rng/key 1))
        c (:choices tr)
        seen (atom nil)]
    (testing "outside a body"
      (is (false? (tr :b)))
      (is (= :no (if (tr :b) :yes :no)))
      (is (= (ts/jvm?) (number? (tr :x))) "only booleans change: a number stays the backend's")
      (is (false? (c :b)) "the choice map called, as the trace")
      (is (= :no (if (c :b) :yes :no)))
      (is (= (ts/jvm?) (number? (c :x))))
      (is (= (ts/jvm?) (boolean? (cm/get-choice c :b))) "get-choice gives the value as stored"))
    (testing "inside a body, the backend's value"
      (gfi/simulate (gen [t] (reset! seen [(t :b) ((:choices t) :b)]) nil) [tr] (rng/key 2))
      (is (every? #(if (ts/jvm?) (false? %) (array/backend-array? %)) @seen)))
    (testing "the flag: set inside a body, not outside"
      (gfi/simulate (gen [] (reset! seen cm/*in-body*) nil) [] (rng/key 3))
      (is (true? @seen))
      (is (false? cm/*in-body*)))))

(defmacro ^:private calls
  "[n (gf 0 1 … n-1)] for n from 0 to 22: every invoke arity, the variadic
   one past twenty included, as a direct call compiles them."
  [gf]
  (vec (for [n (range 23)] [n `(fn [] (~gf ~@(range n)))])))

(defn- outcome
  "What f gives: its value, or a loud error's message; anything else (an
   AbstractMethodError, a ClassCastException) is :broken."
  [f]
  (try (f)
       (catch clojure.lang.ExceptionInfo e [:error (ex-message e)])
       (catch Throwable e [:broken (.getName (class e))])))

(deftest every-generative-function-is-called-with-any-number-of-arguments
  ;; The post-M5 review's I2-06: Map, Unfold, Switch, Recurse, defgf and
  ;; deterministic functions threw AbstractMethodError past six arguments,
  ;; and an estimated function was not callable, nor were the internal
  ;; and the exact proposal's wrappers (genclj-tp50). Planted, defgf's six
  ;; arities fail every kind from seven arguments on (a dynamic function
  ;; is a defgf too now); a wrapper as a plain record fails all its calls.
  (let [given (gen [& xs] (vec xs))
        flipped (gen [& xs] (trace :b (d/bernoulli 0.5)) (vec xs))
        kinds {:dynamic given
               :map (comb/map given)
               :unfold (comb/unfold (gen [_t s & ps] (+ s (count ps))))
               :switch (comb/switch given)
               :recurse (comb/recurse (gen [x] {:value x :children []}) (gen [v _] v) 1)
               :deterministic (custom/deterministic {:apply vec})
               :estimated (custom/estimated {:addresses [:y] :estimate (fn [_ _ _] 0.0)
                                             :simulate (fn [_ _] {:choices (cm/choicemap :y 1.0) :log-estimate 0.0})
                                             :retval (fn [args _] (vec args))})
               :internal-proposal (proposal/with-internal-proposal given (gen [_ & _] nil))
               :exact-proposal (enum/with-exact-proposal flipped [:b])}]
    (doseq [[kind gf] kinds
            [n call] (calls gf)]
      (testing (str (name kind) ", " n " arguments")
        (let [direct (outcome call)]
          (is (not (and (vector? direct) (= :broken (first direct)))) "a value or a loud error")
          (is (= (outcome #(apply gf (range n))) direct) "as apply gives"))))))

;; ---------------------------------------------------------------------------
;; Splices without an address; middleware and dispatch
;; ---------------------------------------------------------------------------

;; Gen.jl's trace[addr] at a call (M5's G1), as dynamic_dsl.jl:487 asks it
(deftest a-call's-return-value-is-at-its-address
  (doseq [static? [false true]]
    (let [bar (if static? (gen ^:static [r] (let [_ (trace :a (d/normal 0 1))] r)) (gen [r] (trace :a (d/normal 0 1)) r))
          foo (if static?
                (gen ^:static [] (let [_ (splice :x bar 1) _ (splice [:y :z] bar 2) _ (trace :u (d/normal 0 1))] 7))
                (gen [] (splice :x bar 1) (splice [:y :z] bar 2) (trace :u (d/normal 0 1)) 7))
          tr (gfi/simulate foo [] (rng/key 0))]
      (testing (if static? "static" "dynamic")
        (is (= 1 (tr :x)))
        (is (= 2 (tr [:y :z])) "a call at a path")
        (is (= 7 (tr [])) "the trace's own, at the empty path")
        (is (= (ts/host (tr [:y :z :a])) (ts/host ((trace/subtrace tr [:y :z]) :a))))
        (is (= :node (:reason (ts/thrown-data (trace/subtrace tr :y)))) "a prefix of a call's path is not a call")
        (is (= :value (:reason (ts/thrown-data (trace/subtrace tr :u)))) "nor is a choice")
        (is (= ::none (trace/subtrace tr :u ::none)) "unless given a not-found value")))))

(deftest splices-without-an-address-share-the-namespace
  (let [tr (gfi/simulate shared-namespace [] (rng/key 1))]
    (is (every? #{[:mu] [:z] [:more] [:w] [:obs]} (cm/addresses (:choices tr))) "the callee's choices join the caller's")
    (is (= 2.5 ((:trace (gfi/generate shared-namespace [] (cm/choicemap :z 2.5) (rng/key 2))) :z)) "constraints reach them")
    (let [{t2 :trace} (gfi/update tr (cm/choicemap :z -1.0) (rng/key 3))]
      (is (= -1.0 (t2 :z)))
      (is (= (tr :mu) (t2 :mu))))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"used twice"
                          (gfi/simulate (gen [] (trace :z (d/normal 0 1)) (splice shared-body 0.0)) [] (rng/key 4)))
        "addresses still may not collide")
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"without an address needs a dynamic"
                          (gfi/simulate (gen [] (splice (d/normal 0 1))) [] (rng/key 5))))))

(def doubled-site
  "Middleware for simulate: the value at :x, a normal(0, 1) draw, is drawn
   from normal(0, 2), twice it under the same key (a different model, as
   middleware may be: through the distribution it hands the transition,
   so the trace scores what it holds)."
  (fn [state p dist k]
    ((dyn/site-transitions :simulate) state p (if (= p [:x]) (d/normal 0 2) dist) k)))

(deftest middleware-of-a-callee-spliced-without-an-address-applies-to-its-body
  (let [inner (dyn/with-transitions (gen [] (trace :x (d/normal 0 1))) {:simulate doubled-site})
        outer (gen [] (splice inner) (trace :z (d/normal 0 1)))
        k (rng/key 7)
        tr (gfi/simulate outer [] k)
        plain (gfi/simulate (gen [] (splice (gen [] (trace :x (d/normal 0 1)))) (trace :z (d/normal 0 1))) [] k)]
    (is (= (* 2.0 (plain :x)) (tr :x)) "the callee's middleware doubles its :x")
    (is (= (plain :z) (tr :z)) "and the caller's own sites are unaffected")))

(deftest middleware-replaces-a-transition-for-one-operation
  (let [m (gen [] (trace :x (d/normal 0 1)))
        k (rng/key 6)
        mw (dyn/with-transitions m {:simulate doubled-site})]
    (is (= (* 2.0 ((gfi/simulate m [] k) :x)) ((gfi/simulate mw [] k) :x)))
    (is (= ((:trace (gfi/generate m [] cm/EMPTY k)) :x) ((:trace (gfi/generate mw [] cm/EMPTY k)) :x))
        "the other operations keep the standard transitions")))

(def closed-form-assess
  "A dispatcher computing linreg's assess in closed form; everything else
   is left to the handler."
  (fn [gf op [xs] {:keys [constraints batch-shape]}]
    (when (and (= op :assess) (empty? batch-shape))
      (let [c #(cm/get-choice constraints %)
            slope (c :slope) intercept (c :intercept)
            lp (fn [v mu sd] (- (- (array/log sd)) 0.9189385332046728 (* 0.5 (array/square (/ (- v mu) sd)))))]
        {:weight (+ (lp slope 0.0 10.0) (lp intercept 0.0 10.0)
                    (reduce + (map-indexed (fn [i x] (lp (c [:y i]) (+ (* slope x) intercept) 1.0)) xs)))
         :retval slope}))))

(laws/deflaws dispatched [[:linreg (dyn/with-dispatch linreg closed-form-assess) [[1.0 2.0 3.0]]]] 100
  {:dispatch-matches-handler laws/dispatch-matches-handler})

(deftest a-wrong-dispatcher-is-caught
  (let [wrong (fn [gf op args opts] (when (= op :assess) (update (closed-form-assess gf op args opts) :weight + 0.1)))
        cases (laws/cases [[:linreg (dyn/with-dispatch linreg wrong) [[1.0 2.0 3.0]]]])]
    (is (false? (:pass? (clojure.test.check/quick-check 20 (laws/dispatch-matches-handler cases)))))))

(deftest lazy-bodies-are-run-to-completion
  ;; trace calls inside a returned lazy sequence happen during the run
  (let [m (gen [] (map (fn [i] (trace [:x i] (d/normal 0 1))) (range 3)))
        tr (gfi/simulate m [] (rng/key 1))]
    (is (= [[:x 0] [:x 1] [:x 2]] (cm/addresses (:choices tr))))
    (is (close? (:score tr) (:weight (gfi/assess m [] (:choices tr))))))
  (testing "an infinite return value does not hang"
    (is (= [1.0 1.0] (vec (take 2 (:retval (gfi/simulate (gen [] (repeat 1.0)) [] (rng/key 3))))))))
  (testing "and laziness that escapes otherwise is loud"
    (let [escaped (atom nil)
          m (gen [] (reset! escaped (map (fn [i] (trace [:x i] (d/normal 0 1))) (range 3))) nil)]
      (gfi/simulate m [] (rng/key 2))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"after the generative function returned" (doall @escaped))))))

;; Return values that are lazy without being LazySeqs (the post-M5 review's
;; P1-04), at 40 draws: past the 32-element chunk that a cons or a seq
;; realizes in the body. Plant: realize! forcing LazySeqs alone fails 13 of
;; the 22 laws on these models, and every-lazy-kind-is-realized-during-the-run
;; for each kind (simulate kept 32 of conj-onto-lazy's 40 choices; generate's
;; constraints were "not visited").

(def cons-onto-lazy (gen [n] (cons 0.0 (map #(trace [:x %] (d/normal 0 1)) (range n)))))

(def conj-onto-lazy (gen [n] (conj (map #(trace [:x %] (d/normal 0 1)) (range n)) 0.0)))

(def chunk-onto-lazy (gen [n] (seq (map #(trace [:x %] (d/normal 0 1)) (range n)))))

(def repeated-lazy (gen [n] (repeat 2 (map #(trace [:x %] (d/normal 0 1)) (range n)))))

(def range-beside-cons (gen [n] [(range n) (cons 0.0 (map #(trace [:x %] (d/normal 0 1)) (range n)))]))

(laws/deflaws lazy-returns
  (mapv (fn [[label m]] [label m [40] [[3] [45]]])
        [[:cons-onto-lazy cons-onto-lazy] [:conj-onto-lazy conj-onto-lazy] [:chunk-onto-lazy chunk-onto-lazy]
         [:repeated-lazy repeated-lazy] [:range-beside-cons range-beside-cons]])
  100 (merge laws/all laws/conventions))

;; The infinite kinds (genclj-54ix), which realize! takes 10,000 elements of:
;; iterate's (its draws in the steps f traces), cycle's past the chunk cycle
;; realizes, and an infinite repeat's. Every law of laws/all, but not
;; laws/conventions's identity update reporting no-change: two infinite
;; values exhaust same-value?'s budget, so an identity update reports
;; unknown-change, which is sound (Def 2.3.2). The laws compare a return value as far as a run
;; realizes it (gen.laws' same-retval?). Plants: an unbounded comparison in
;; the retdiff (=) does not end (identity-update-changes-nothing timed out);
;; realize! forcing LazySeqs alone fails four of the 21 laws.

(def iterated (gen [n] (iterate (fn [[i x]] (if (< i n) [(inc i) (trace [:x i] (d/normal x 1))] [i x])) [0 0.0])))

(def cycled (gen [n] (cycle (map #(trace [:x %] (d/normal 0 1)) (range n)))))

(def repeated-forever (gen [n] (repeat (map #(trace [:x %] (d/normal 0 1)) (range n)))))

(laws/deflaws lazy-returns-infinite
  [[:iterated iterated [5] [[3] [8]]] [:cycled cycled [40] [[3] [45]]] [:repeated-forever repeated-forever [5] [[3] [8]]]]
  100 laws/all)

(deftest every-lazy-kind-is-realized-during-the-run
  (let [draws (set (map #(vector :x %) (range 40)))]
    (doseq [[kind m] {"iterate" (gen [] (iterate (fn [[i x]] (if (< i 40) [(inc i) (trace [:x i] (d/normal x 1))] [i x]))
                                                 [0 0.0]))
                      "cycle" (gen [] (cycle (map #(trace [:x %] (d/normal 0 1)) (range 40))))
                      "repeat" (gen [] (repeat (map #(trace [:x %] (d/normal 0 1)) (range 40))))}]
      (testing kind
        (let [tr (gfi/simulate m [] (rng/key 1))]
          (is (= draws (set (cm/addresses (:choices tr)))))
          (is (close? (:score tr) (:weight (gfi/assess m [] (:choices tr))))))
        (is (= 0.5 ((:trace (gfi/generate m [] (cm/choicemap [:x 39] 0.5) (rng/key 2))) [:x 39]))
            "a constraint in the lazy part is visited")))))

(def ^:private signed-zero
  "x · 0.0: 0.0 or -0.0 by x's sign."
  (gen [] (* (trace :x (d/normal 0.0 1.0)) 0.0)))

(deftest unchanged-is-same-value
  ;; The post-M5 review's P1-01 and P1-03 (DESIGN.md §9, D1): the retdiff
  ;; decided "unchanged" by =, so a callee's 0.0 that became -0.0, a vector
  ;; that became a list or a sorted map an array map was no-change, and a
  ;; static caller or an Unfold kept the node that tells them apart; and =
  ;; walked an infinite lazy return value forever. Planted bug (2026-10-22):
  ;; = back in the retdiff fails the four weights and the scores, and times
  ;; out both lazy updates
  (let [lp (fn [y mu] (dc/log-prob (d/normal mu 1.0) y))
        reading (fn [callee mean-of]
                  (gen ^:static [] (let [v (splice :in callee) y (trace :y (d/normal (mean-of v) 1.0))] y)))
        cases [["a signed zero" (reading signed-zero #(array/atan2 % -1.0)) {:in {:x 1.0} :y 1.0} {:in {:x -1.0}}
                (- (lp 1.0 (- Math/PI)) (lp 1.0 Math/PI))]
               ["a vector, then a list"
                (reading (gen [] (if (trace :b (d/bernoulli 0.5)) [1.0 2.0] (list 1.0 2.0))) #(first (conj % 5.0)))
                {:in {:b true} :y 0.0} {:in {:b false}} (- (lp 0.0 5.0) (lp 0.0 1.0))]
               ["a sorted map, then an array map"
                (reading (gen [] (if (trace :b (d/bernoulli 0.5)) (sorted-map :b 1.0 :a 2.0) (array-map :b 1.0 :a 2.0)))
                         #(val (first %)))
                {:in {:b true} :y 0.0} {:in {:b false}} (- (lp 0.0 1.0) (lp 0.0 2.0))]
               ;; step t's mean is atan2(state, -1), its next state x · 0.0;
               ;; step 0's x from 1 to -1, then step 1's state from 0.0 to -0.0
               ["an Unfold over a dynamic kernel"
                (gen [] (splice :in (comb/unfold (gen [_t s] (* (trace :x (d/normal (array/atan2 s -1.0) 1.0)) 0.0))) 2 0.0))
                {:in {0 {:x 1.0} 1 {:x 1.0}}} {:in {0 {:x -1.0}}}
                (+ (- (lp -1.0 Math/PI) (lp 1.0 Math/PI)) (- (lp 1.0 (- Math/PI)) (lp 1.0 Math/PI)))]]]
    (doseq [[label gf init constraints expected] cases]
      (testing label
        (let [tr (:trace (gfi/generate gf [] (cm/from-map init) (rng/key 1)))
              {t :trace w :weight} (gfi/update tr (cm/from-map constraints) (rng/key 2))]
          (is (close? expected w))
          (is (close? (:weight (gfi/assess gf [] (:choices t) (rng/key 3))) (:score t)) "the score is the new choices'")))))
  (testing "an infinite lazy return value is changed, at once"
    (doseq [[label body] [["iterate" (gen [x0] (trace :y (d/normal x0 1)) (iterate #(* 0.9 %) x0))]
                          ["repeat" (gen [x0] (trace :y (d/normal x0 1)) (repeat x0))]]]
      (let [tr (:trace (gfi/generate body [1.0] (cm/choicemap :y 0.0) (rng/key 4)))
            r (ts/within 10000 #(gfi/update tr (cm/choicemap :y 0.5) (rng/key 5)))]
        (is (diff/unknown-change? (:retdiff r)) label)))))

(deftest guarded-clojure-core-matches-clojure-core
  ;; third acceptance test: the substitution must not change ordinary code
  (let [k (rng/key 0)
        ret (fn [gf & args] (:retval (gfi/simulate gf (vec args) k)))]
    (is (false? (ret (gen [] ((some-fn neg? zero?) 5)))) "some-fn keeps false, not nil")
    (is (nil? (ret (gen [] (some #{:a} [:b :c])))))
    (is (= [1 3] (ret (gen [] (filterv odd? [1 2 3])))))
    (is (= 2 (ret (gen [filter] (splice :g (gen [x] (trace :x (d/normal 0 1)) (filter x)) 1)) inc))
        "a gen nested in a body sees the body's locals")
    (is (= 2 (ret (gen [] (.apply ^java.util.function.Function
                                  (reify java.util.function.Function (apply [_ some] (inc some))) 1))))
        "reify parameters are locals")
    (is (true? (ret (gen [] (let [x (trace :x (d/normal 0 1))] (and (number? x) (float? x) (not (integer? x))))))))
    (is (= [:a 3 nil] (ret (gen [] [(get {:k :a} :k) (get [1 2 3] 2) (get "s" 5)]))))))

(deftest bodies-keep-clojure-s-forms
  ;; fourth acceptance test B5
  (let [k (rng/key 0)
        ret (fn [gf & args] (:retval (gfi/simulate gf (vec args) k)))]
    (is (= {:foo 1} (ret (gen [] (meta ^{:foo 1} [1 2])))) "form metadata, type hints included")
    (is (= 3 (ret (gen [m] (.length ^String (get m :s))) {:s "abc"})))
    (is (= 3.0 ((gen [^double x ^double y] (+ x y)) 1 2)) "primitive-hinted parameters")
    (is (= 9 ((gen [a b c d e f g h] (+ a h)) 1 2 3 4 5 6 7 8)) "untraced calls of any arity")
    (is (= 1 (ret (gen [] (try (throw (ex-info "x" {})) (catch Exception _ 1))))) "try in scalar mode")))

(deftest discarded-lazy-calls-inside-wrappers-are-compile-errors
  (is (thrown-with-msg? Exception #"map here is lazy and its value is discarded"
                        (try (eval '(gen.dynamic/gen [xs] (when true (map #(trace [:y %] (gen.dist/normal 0 1)) xs)) 1))
                             (catch clojure.lang.Compiler$CompilerException e (throw (ex-cause e)))))))

(deftest unused-lazy-bindings-with-trace-calls-are-compile-errors
  (is (thrown-with-msg? Exception #"ys is bound to a lazy map and never used"
                        (try (eval '(gen.dynamic/gen [xs] (let [ys (map #(trace [:y %] (gen.dist/normal 0 1)) xs)] 1)))
                             (catch clojure.lang.Compiler$CompilerException e (throw (ex-cause e))))))
  (is (some? (eval '(gen.dynamic/gen [xs] (let [ys (map #(trace [:y %] (gen.dist/normal 0 1)) xs)] (vec ys)))))
      "a used one is fine"))

(deftest discarded-lazy-trace-calls-are-compile-errors
  (let [compile-error (fn [form] (try (eval form) nil
                                      (catch Throwable e (.getMessage ^Throwable (or (.getCause e) e)))))]
    (is (re-find #"lazy and its value is discarded"
                 (str (compile-error '(gen.dynamic/gen [xs] (map (fn [x] (trace [:y x] (gen.dist/normal x 1))) xs) nil)))))
    (is (re-find #"lazy and its value is discarded"
                 (str (compile-error '(gen.dynamic/gen [xs] (let [n 1] (for [x xs] (trace [:y x] (gen.dist/normal x n))) n))))))
    (testing "mapv, a lazy tail value and lazy calls without trace are fine"
      (is (nil? (compile-error '(gen.dynamic/gen [xs] (mapv (fn [x] (trace [:y x] (gen.dist/normal x 1))) xs) nil))))
      (is (nil? (compile-error '(gen.dynamic/gen [xs] (map (fn [x] (trace [:y x] (gen.dist/normal x 1))) xs)))))
      (is (nil? (compile-error '(gen.dynamic/gen [xs] (map inc xs) nil)))))))

(def returns-seven (gen [] (trace :x (d/normal 0 1)) 7.0))

(deftest a-different-callee-at-an-address-is-a-new-call
  ;; review 3: the old sub-trace was handed to the new generative function
  (let [m (gen [] (if (trace :c (d/bernoulli 0.5)) (splice :s returns-seven) (splice :s (d/normal 0 1))))
        tr (:trace (gfi/generate m [] (cm/choicemap :c true [:s :x] 0.5) (rng/key 1)))
        {t :trace w :weight disc :discard} (gfi/update tr (cm/choicemap :c false) (rng/key 2))]
    (is (not= 7.0 (t :s)) "the new distribution draws its own value")
    (is (close? w (- (- (:score t) (:score tr)) (gfi/project t (sel/select :s)))))
    (is (= (cm/choicemap :c true [:s :x] 0.5) disc))
    (is (= (:choices tr) (:choices (:trace (gfi/update t disc (rng/key 3))))) "and the update reverses")))

(deftest third-review-regressions
  (testing "B2: a splice whose dynamic callee changes keeps shared addresses"
    (let [ma (gen [] (trace :obs (d/normal 0 1)))
          mb (gen [] (trace :obs (d/normal 2 1)))
          m (gen [] (splice :s (if (trace :i (d/bernoulli 0.5)) ma mb)))
          tr (:trace (gfi/generate m [] (cm/choicemap :i true [:s :obs] 1.5) (rng/key 1)))
          {t :trace w :weight} (gfi/regenerate tr (sel/select :i) (rng/key 2))]
      (is (= 1.5 (t [:s :obs])) "the unselected observation keeps its value")
      (when-not (t :i)
        (is (close? w (- (dc/log-prob (d/normal 2 1) 1.5) (dc/log-prob (d/normal 0 1) 1.5)))))))
  (testing "B5: values of the wrong shape are errors everywhere"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"shape"
                          (gfi/generate (gen [] (splice :x (d/normal 0 1))) [] (cm/choicemap :x (array/array [1.0 2.0 3.0])) (rng/key 1))))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"shape" (gfi/assess (d/normal 0 1) [] (cm/->Value (array/array [1.0 2.0])))))
    (let [m (gen [d] (trace :v (d/normal (array/zeros [d]) 1)))
          tr (gfi/simulate m [1] (rng/key 2))]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"shape" (gfi/update tr [3] [diff/unknown-change] cm/EMPTY (rng/key 3)))
          "a retained value whose parameters change shape"))
    ;; the value shape skips host-scalar parameters (the post-M5 review's
    ;; I3-05); plant: skipping array parameters too passes both of these
    (let [m (gen [mu] (trace :y (d/normal mu 1)))
          tr (gfi/simulate m [0.0] (rng/key 2))
          wrong #"A value of shape \[\] for a distribution whose values have shape \[2\]"]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo wrong
                            (gfi/update tr [(array/array [0.0 1.0])] [diff/unknown-change] cm/EMPTY (rng/key 3)))
          "a retained value whose host-scalar parameter becomes an array")
      (is (thrown-with-msg? clojure.lang.ExceptionInfo wrong
                            (gfi/regenerate tr [(array/array [0.0 1.0])] [diff/unknown-change] sel/none (rng/key 3)))
          "and on regenerate"))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"shape"
                          (dc/log-prob ((d/product d/normal d/bernoulli) [0 1] [0.3]) [(array/array [1.0 2.0 3.0]) true]))))
  (testing "B6: trace calls in lazy splice arguments are recorded"
    (let [m (gen [] (splice :m (comb/map observation) (vec (map #(trace [:x %] (d/normal 0 1)) (range 3)))))
          tr (gfi/simulate m [] (rng/key 4))]
      (is (= 3 (count (filter #(= :x (first %)) (cm/addresses (:choices tr))))))
      (is (close? (:score tr) (:weight (gfi/assess m [] (:choices tr))))))
    (is (re-find #"lazy and its value is discarded"
                 (str (try (eval '(gen.dynamic/gen [xs] (->> xs (map-indexed (fn [i x] (trace [:y i] (gen.dist/normal x 1))))) nil))
                           nil (catch Throwable e (.getMessage ^Throwable (or (.getCause e) e))))))))
  (testing "B7: an infinite lazy return value costs little"
    ;; the handler forces only the first 10,000 elements; they are host
    ;; numbers, so the cost is the walk's, not 10,000 backend ops' (which
    ;; on MLX would measure the machine: 50 ms a run on the Jetson's cores)
    (let [m (gen [] (let [s (trace :s (d/normal 0 1))] (map #(if (zero? %) s (inc %)) (range))))
          t0 (System/nanoTime)]
      (dotimes [i 20] (gfi/simulate m [] (rng/key i)))
      (is (< (/ (- (System/nanoTime) t0) 1e6) 400.0) "under 20 ms a run"))))

;; ---------------------------------------------------------------------------
;; A choice outside its own support stops the run (DESIGN.md §9, the ten
;; decisions' fifth; genclj-q7cq, genclj-2870)
;; ---------------------------------------------------------------------------

;; The operation returns weight -Inf without running the rest of the body,
;; so a distribution downstream whose parameter the impossible value makes
;; invalid is never built: its check cannot throw. The trace holds the
;; choices and calls up to that one (recorded with them), its score -Inf
;; and its return value nil; an update discards the old choices it did not
;; reach. Planted bug (2026-10-25): weigh never stopping, the old behaviour,
;; fails every model below with the downstream parameter error (normal's
;; std, mvnormal's covariance), and the outside-support laws with it.

(def scale-then-normal
  (gen [] (let [s (trace :s (d/gamma 1 1))] (trace :y (d/normal 0 s)))))

(def scale-then-normal-static
  (gen ^:static [] (let [s (trace :s (d/gamma 1 1)) y (trace :y (d/normal 0 s))] y)))

(def scale-in-a-callee
  "The scale made by a call: the callee stops, and the caller stops at the
   call (it never sees the callee's missing return value)."
  (gen [] (let [s (splice :inner (gen [] (trace :s (d/gamma 1 1))))] (trace :y (d/normal 0 s)))))

(def length-scale-then-mvnormal
  "A covariance scaled by a length scale l (chapter 052's case, a GP's
   leaf): mvnormal's check refuses it where l is below 0."
  (gen [] (let [l (trace :l (d/gamma 2 1))]
            (trace :f (d/mvnormal (array/zeros [2]) (* l (array/array [[1.0 0.5] [0.5 1.0]])))))))

(defn- stops?
  "Does a scalar run stop at a choice outside its support? On the JVM its
   log densities are host numbers, so it does; on a lazy backend (MLX) they
   are arrays, never read, so it goes on, its score and weight -Inf all
   the same (gen.dynamic), and its trace whole."
  []
  (ts/jvm?))

(deftest a-choice-outside-its-support-stops-the-run
  (doseq [[label gf p others] [["a dynamic body" scale-then-normal [:s] {:y 0.1}]
                               ["a static body" scale-then-normal-static [:s] {:y 0.1}]
                               ["a callee" scale-in-a-callee [:inner :s] {:y 0.1}]
                               ["mvnormal downstream" length-scale-then-mvnormal [:l] {:f (array/array [0.1 -0.2])}]]]
    (testing label
      (let [given (reduce-kv (fn [c k v] (cm/set-choice c [k] v)) (cm/set-choice cm/EMPTY p 0.5) others)
            impossible (cm/set-choice given p -1.0)
            tr (:trace (gfi/generate gf [] given (rng/key 0)))
            {t2 :trace w :weight disc :discard rd :retdiff} (gfi/update tr (cm/set-choice cm/EMPTY p -1.0) (rng/key 1))]
        (is (ts/finite? (:score tr)))
        (is (= ##-Inf (ts/number w)) "update")
        (is (= ##-Inf (ts/number (:score t2))))
        (is (= -1.0 (ts/number (t2 p))))
        (is (= ##-Inf (ts/number (:weight (gfi/generate gf [] impossible (rng/key 2))))) "generate")
        (is (= ##-Inf (ts/number (:weight (gfi/assess gf [] impossible)))) "assess")
        (when (stops?)
          (is (= [p] (cm/addresses (:choices t2))) "the choices up to the impossible one")
          (is (nil? (:retval t2)))
          (is (diff/unknown-change? rd))
          (is (= (set (cm/addresses given)) (set (cm/addresses disc))) "the old choices replaced or not reached")
          (is (nil? (:retval (gfi/assess gf [] impossible))))
          (is (= ##-Inf (ts/number (:weight (gfi/assess gf [] (cm/set-choice cm/EMPTY p -1.0)))))
              "assess of the choices up to it: the rest is never asked for"))
        (testing ", and back into the support"
          (let [{t3 :trace w3 :weight} (gfi/update t2 [] [] disc (rng/key 3))]
            (is (= (ts/host (cm/to-map (:choices tr))) (ts/host (cm/to-map (:choices t3)))))
            (is (close? (:score tr) (:score t3)))
            ;; (a lazy run's downstream density was NaN at the impossible value)
            (when (stops?) (is (= ##Inf (ts/number w3))))))
        (testing ", and a move between two impossible traces has no weight (NaN, as MH's check expects)"
          (is (Double/isNaN (ts/number (:weight (gfi/update t2 (cm/set-choice cm/EMPTY p -2.0) (rng/key 4))))))))))
  (testing "regenerate: a kept choice whose support the regenerated one moves"
    (let [m (gen [] (let [a (trace :a (d/uniform 0 1)) b (trace :b (d/uniform 0 a))] (trace :c (d/normal 0 (- a b)))))
          tr (:trace (gfi/generate m [] (cm/choicemap :a 0.9 :b 0.6 :c 0.0) (rng/key 0)))
          rs (map #(gfi/regenerate tr (sel/select :a) %) (rng/split (rng/key 5) 40))]
      (is (some #(= ##-Inf (ts/number (:weight %))) rs) "a below b")
      (is (some #(ts/finite? (:weight %)) rs) "a above b")
      (when (stops?)
        (is (every? #(= (ts/finite? (:weight %)) (cm/has-value? (:choices (:trace %)) [:c])) rs)))))
  (testing "simulate, generate and assess within the support are unchanged, and other checks stay loud"
    (is (ts/finite? (:score (gfi/simulate scale-then-normal [] (rng/key 6)))))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"std must be positive"
                          (gfi/simulate (gen [] (trace :y (d/normal 0 -1.0))) [] (rng/key 7)))
        "a parameter no choice outside its support made invalid")
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"not visited"
                          (gfi/generate scale-then-normal [] (cm/choicemap :s 0.5 :z 1.0) (rng/key 8)))
        "an unvisited constraint of a run that did not stop")))

(def nested-bounds
  "b's support ends at a, and c's scale is a - b: an update that takes a
   below b (a constraint, or lo, the arguments) takes b outside its support,
   and c's scale below 0, which the run never reaches."
  (gen [lo] (let [a (trace :a (d/uniform lo (+ lo 1.0)))
                  b (trace :b (d/uniform lo a))]
              (trace :c (d/normal 0 (- a b))))))

(def nested-bounds-static
  (gen ^:static [lo] (let [a (trace :a (d/uniform lo (+ lo 1.0)))
                           b (trace :b (d/uniform lo a))
                           c (trace :c (d/normal 0 (- a b)))]
                       c)))

(def nested-bounds-in-a-callee
  (gen [lo] (let [[a b] (splice :ab (gen [lo] (let [a (trace :a (d/uniform lo (+ lo 1.0)))]
                                                [a (trace :b (d/uniform lo a))]))
                                lo)]
              (trace :c (d/normal 0 (- a b))))))

;; Models whose updates and regenerates go outside the support: the laws
;; hold of the traces they return (gen.laws' reached: an update that stops
;; reaches the constraints up to the choice that stops it).
(laws/deflaws outside-support
  [[:nested-bounds nested-bounds [0.0] [[0.5] [-0.5]]]
   [:nested-bounds-static nested-bounds-static [0.0] [[0.5] [-0.5]]]
   [:nested-bounds-in-a-callee nested-bounds-in-a-callee [0.0] [[0.5] [-0.5]]]]
  100 (merge laws/all laws/conventions))

;; ---------------------------------------------------------------------------
;; A splice's argument equal to the last is unchanged (DESIGN.md §9, the ten
;; decisions' tenth; genclj-nvo0)
;; ---------------------------------------------------------------------------

;; no-change when identical, or a host scalar, keyword or string equal to
;; the previous call's; a collection or array by identity. Planted bug
;; (2026-10-25): identity alone fails the fresh number's and string's
;; cases (a keyword is interned, and a boolean canonical, so they are
;; identical either way).
(deftest a-splice-argument-equal-to-the-last-is-unchanged
  (let [seen (atom nil)
        callee (dyn/with-dispatch (gen [_x] (trace :z (d/normal 0 1)))
                 (fn [_ op _ opts] (when (= op :update) (reset! seen (:argdiffs opts))) nil))
        argdiff (fn [make]
                  (let [g (gen [n] (splice :c callee (make n)))
                        tr (gfi/simulate g [1] (rng/key 0))]
                    (reset! seen nil)
                    (gfi/update tr [1] [diff/unknown-change] cm/EMPTY (rng/key 1))
                    (first @seen)))]
    (doseq [[label make] [["a fresh equal double" #(+ % 0.5)]
                          ["a keyword" #(keyword (str "k" %))]
                          ["a string" #(str "k" %)]
                          ["the same map" (let [m {:a 1}] (fn [_] m))]]]
      (is (diff/no-change? (argdiff make)) label))
    (doseq [[label make] [["an equal map built afresh" #(hash-map :a %)]
                          ["an equal vector built afresh" #(vector % 2)]
                          ["an equal array built afresh" #(array/array [% 2.0])]
                          ["-0.0 after 0.0" (let [calls (atom 0)] (fn [_] (if (= 1 (swap! calls inc)) 0.0 -0.0)))]
                          ["a double after a long" (let [calls (atom 0)] (fn [_] (if (= 1 (swap! calls inc)) 1 1.0)))]]]
      (is (diff/unknown-change? (argdiff make)) label))))

;; ---------------------------------------------------------------------------
;; The handler's state, and middleware's own keys (DESIGN.md §9, "The
;; handler's state and middleware"; genclj-q6sc)
;; ---------------------------------------------------------------------------

(def ^:private documented-keys
  "The state's keys, as initial-state's docstring lists them."
  #{:mode :constraints :prev :selection :argdiffs :choices :scores :subtraces :score :weight :discard
    :key :visited :omega :omega-key :omega-count})

(defn- watching
  "model with middleware that keeps, under ::steps, one step for each site:
   the state its transition received (:in), the path, distribution and key
   it was handed, and the state the standard transition returned (:out)."
  [model]
  (dyn/with-transitions model
    (into {} (for [[op t] dyn/site-transitions]
               [op (fn [state p dist k]
                     (let [[v out] (t state p dist k)]
                       [v (update out ::steps (fnil conj []) {:in state :path p :dist dist :key k :out out})]))]))))

(def ^:private one-call (gen [mu] (trace :z (d/normal mu 1))))

(def ^:private every-kind
  "A site, an untraced draw, a call, and a site after them."
  (gen []
    (let [a (trace :a (d/normal 0 1))
          u (untraced (d/normal 0 1))
          z (splice :c one-call a)]
      (trace :b (d/normal (+ z (* 0.0 u)) 1)))))

;; Planted bugs (2026-10-25): an undocumented key in the initial state
;; fails the key set; ω folded from another key than the operation's fails
;; :omega-key's case.
(deftest initial-state-holds-the-documented-keys
  (let [w (watching every-kind)
        k (rng/key 1)
        tr (gfi/simulate w [] (rng/key 0))
        cs (cm/choicemap :a 0.5)]
    (doseq [[op opts] [[:simulate {:key k}] [:generate {:key k :constraints cs}]
                       [:assess {:key k :constraints (:choices tr)}]
                       [:update {:key k :prev tr :argdiffs [] :constraints cs}]
                       [:regenerate {:key k :prev tr :argdiffs [] :selection (sel/select :a)}]]]
      (testing op
        (let [state (dyn/initial-state op opts)
              r (gfi/run-op w op [] opts)
              first-in (:in (first (::steps (or (:trace r) r))))]
          (is (= documented-keys (set (filter simple-keyword? (keys state)))) "the documented keys, and no other")
          (is (every? #(= "gen.dynamic" (namespace %)) (filter qualified-keyword? (keys state)))
              "the others are gen.dynamic's")
          (is (= op (:mode state)))
          (is (every? true? (map #(identical? (get opts % ({:constraints cm/EMPTY :selection sel/none} %)) (get state %))
                                 [:key :constraints :prev :argdiffs :selection]))
              "the operation's inputs")
          (is (identical? k (:omega-key state)))
          (is (= [nil 0] [(:omega state) (:omega-count state)]))
          (is (= [cm/EMPTY {} {} 0.0 0.0 cm/EMPTY #{}]
                 (mapv state [:choices :scores :subtraces :score :weight :discard :visited]))
              "nothing made yet")
          (is (= (dissoc state :key) (dissoc first-in :key)) "the run starts from it")
          (is (= (rng/key-data (first (rng/split k))) (rng/key-data (:key first-in)))
              "its first site has split the key"))))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"is not an operation" (dyn/initial-state :propose)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"not an operation's inputs"
                          (dyn/initial-state :generate {:constraint cs})))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"must be a choice map"
                          (dyn/initial-state :generate {:constraints {:a 0.5}})))))

;; What each key holds at the two sites of every-kind, around its untraced
;; draw and call, in every operation. Planted bug (2026-10-25): a site's
;; path added to :visited before its transition fails :visited's case.
(deftest the-state's-keys-mean-what-initial-state-says
  (let [w (watching every-kind)
        k (rng/key 2)
        tr (gfi/simulate w [] (rng/key 1))
        n01 (d/normal 0 1)
        old-a (tr :a)
        old-z (tr [:c :z])
        runs {:simulate {:trace (gfi/simulate w [] k)}
              :generate (gfi/generate w [] (cm/choicemap :a 0.5 [:c :z] 0.2) k)
              :assess (gfi/assess w [] (:choices tr) k)
              :update (gfi/update tr (cm/choicemap :a 0.5) k)
              :regenerate (gfi/regenerate tr (sel/select :a) k)}]
    (doseq [[op r] runs
            :let [made (:trace r)
                  [a b] (::steps (or made r))
                  va (cm/get-choice (:choices (:out a)) :a)
                  vz (cm/get-choice (:choices (:in b)) [:c :z])
                  lp-a (dc/log-prob n01 va)
                  lp-z (dc/log-prob (d/normal va 1) vz)
                  weights (mapv :weight [(:in a) (:out a) (:in b) (:out b)])]]
      (testing op
        (is (= [[:a] [:b]] (mapv :path [a b])))
        (is (= [op op] (mapv (comp :mode :in) [a b])) ":mode")
        (let [[after-call _] (rng/split (:key (:in a)))
              [b-next k-b] (rng/split after-call)]
          (is (= (mapv rng/key-data [b-next k-b]) (mapv rng/key-data [(:key (:in b)) (:key b)]))
              ":key: the key that the call, then the site after it, split"))
        (is (= {[:a] lp-a} (:scores (:out a))) ":scores")
        (is (= #{[:a] [:b]} (set (keys (:scores (:out b))))) ":scores: a call's are in its trace")
        (is (= #{[:a] [:c :z]} (set (cm/addresses (:choices (:in b))))) ":choices: the call's under its address")
        (is (close? (:score (:in b)) (+ lp-a lp-z)) ":score: the sites' and the call's")
        (is (close? (:score (:out b)) (+ (:score (:in b)) (get (:scores (:out b)) [:b]))))
        (if (= op :assess)
          (is (= {} (:subtraces (:in b))) ":subtraces: assess keeps none")
          (is (close? lp-z (:score (get (:subtraces (:in b)) [:c]))) ":subtraces: the call's trace"))
        (is (= [#{} #{} #{[:a] [:c]}] (mapv :visited [(:in a) (:out a) (:in b)]))
            ":visited: the paths used before the site")
        (is (= [0 1] (mapv (comp :omega-count :in) [a b])) ":omega-count: the untraced draw's")
        (is (= (rng/key-data k) (rng/key-data (:omega-key (:in b)))) ":omega-key: the operation's key")
        (is (nil? (:omega (:in a))) ":omega: nil before the first untraced draw (none kept)")
        (when made
          (is (= (rng/key-data (:omega (:in b))) (rng/key-data (::trace/omega made))) ":omega: the trace's ω"))
        (is (identical? (:constraints (:in a)) (:constraints (:in b))))
        (is (= [(boolean (#{:update :regenerate} op)) (= op :regenerate) (boolean (#{:update :regenerate} op))]
               [(some? (:prev (:in a))) (not= sel/none (:selection (:in a))) (some? (:argdiffs (:in a)))])
            ":prev, :selection and :argdiffs: the operations' that take them")
        (when (:weight r) (is (close? (:weight r) (peek weights)) ":weight: the operation's at the end"))
        (case op
          :simulate (is (every? #(close? 0.0 %) weights) ":weight: 0.0 throughout simulate")
          :generate (is (every? true? (map close? weights [0.0 lp-a (+ lp-a lp-z) (+ lp-a lp-z)]))
                        ":weight: the constrained sites' and the call's")
          :assess (is (every? true? (map close? weights (map :score [(:in a) (:out a) (:in b) (:out b)])))
                      ":weight: every site's")
          :update (let [d-a (- lp-a (dc/log-prob n01 old-a))
                        d-z (- lp-z (dc/log-prob (d/normal old-a 1) old-z))]
                    (is (every? true? (map close? weights [0.0 d-a (+ d-a d-z) (+ d-a d-z)]))
                        ":weight: the changes in log density")
                    (is (= {:a (ts/host old-a)} (ts/host (:discard (:out a)))) ":discard: the value replaced"))
          :regenerate (let [d-z (- lp-z (dc/log-prob (d/normal old-a 1) old-z))]
                        (is (every? true? (map close? weights [0.0 0.0 d-z d-z]))
                            ":weight: the unselected choices' changes")))
        (when-not (= op :update)
          (is (cm/empty? (:discard (:out b))) ":discard: update's only"))))))

;; Planted bugs (2026-10-25): a trace made without middleware's keys fails
;; simulate's, generate's, update's and regenerate's cases; an update or
;; regenerate that starts from the previous run's keys fails theirs.
(deftest middleware's-own-keys-reach-every-operation's-result
  (let [w (watching every-kind)
        k (rng/key 3)
        tr (gfi/simulate w [] (rng/key 1))
        modes (fn [x] (mapv (comp :mode :in) (::steps x)))
        u (:trace (gfi/update tr (cm/choicemap :a 0.5) k))
        r (:trace (gfi/regenerate tr (sel/select :a) k))]
    (is (= [:simulate :simulate] (modes tr)) "simulate: on its trace")
    (is (= [:generate :generate] (modes (:trace (gfi/generate w [] (cm/choicemap :a 0.5) k)))) "generate: on its trace")
    (is (= [:update :update] (modes u)) "update: on its trace, its own run's alone")
    (is (= [:regenerate :regenerate] (modes r)) "regenerate: on its trace, its own run's alone")
    (is (identical? (::steps tr) (::steps (:prev (:in (first (::steps u)))))) "the previous run's, on :prev")
    (is (= [:assess :assess] (modes (gfi/assess w [] (:choices tr) k))) "assess: in its result")
    (is (= [:simulate :simulate] (modes (gfi/propose w [] k))) "propose: in its result")
    (is (nil? (::steps (gfi/simulate every-kind [] k))) "none without middleware")))

;; Middleware that only keeps data runs the operation the standard
;; transitions run. Planted bug (2026-10-25): a middleware's transition
;; handed the state's next key in place of its site's fails every case
;; that draws (assess draws nothing at its sites).
(deftest middleware's-own-keys-change-no-operation's-meaning
  (let [w (watching every-kind)
        k (rng/key 4)
        [tw t] (map #(gfi/simulate % [] (rng/key 1)) [w every-kind])
        cs (cm/choicemap :a 0.5)
        same (fn [label & xs] (is (apply = (map ts/host xs)) label))
        parts (fn [r] (let [tr (:trace r)]
                        (cond-> (dissoc r :trace ::steps)
                          tr (assoc :choices (:choices tr) :score (:score tr) :retval (:retval tr)))))]
    (same "simulate" (parts {:trace tw}) (parts {:trace t}))
    (same "generate" (parts (gfi/generate w [] cs k)) (parts (gfi/generate every-kind [] cs k)))
    (same "assess" (parts (gfi/assess w [] (:choices t) k)) (parts (gfi/assess every-kind [] (:choices t) k)))
    (same "update" (parts (gfi/update tw cs k)) (parts (gfi/update t cs k)))
    (same "regenerate" (parts (gfi/regenerate tw (sel/select :a) k)) (parts (gfi/regenerate t (sel/select :a) k)))))

(defn- scoring
  "model with middleware that keeps, under ::scores, [operation score] after
   each site."
  [model]
  (dyn/with-transitions model
    (into {} (for [[op t] dyn/site-transitions]
               [op (fn [state p dist k]
                     (let [[v out] (t state p dist k)]
                       [v (update out ::scores (fnil conj []) [(:mode state) (:score out)])]))]))))

;; A batched run's middleware sees batched values, and its keys are one
;; value for every particle, on the batched trace. Planted bug
;; (2026-10-25): a batched run that returns no keys of middleware's fails
;; every batched case.
(deftest middleware's-own-keys-reach-batched-results
  (let [n 5
        m (scoring (gen [] (let [x (trace :x (d/normal 0 1))] (trace :y (d/normal x 1)))))
        kept (fn [tr n'] (mapv (fn [[op s]] [op (batch/unwrap s [n'])]) (::scores tr)))
        agrees? (fn [ps op]
                  (let [[[op1 s1] [op2 s2]] (kept (:trace ps) (batched/size ps))]
                    (and (batched/batched? ps) (= [op op] [op1 op2])
                         (close? s1 (dc/log-prob (d/normal 0 1) (batched/choice ps :x)))
                         (close? s2 (batched/scores ps)))))
        ps (batched/simulate m [] n (rng/key 1))]
    (is (batch/batched? (second (first (::scores (:trace ps))))) "batched values, one per particle")
    (is (agrees? ps :simulate) "simulate")
    (is (agrees? (:particles (batched/generate m [] (cm/choicemap :y 0.5) n (rng/key 2))) :generate) "generate")
    (is (agrees? (:particles (batched/update ps [] [] (cm/choicemap :y 0.3) (rng/key 3))) :update)
        "update: its own run's alone")
    (is (agrees? (:particles (batched/regenerate ps [] [] (sel/select :x) (rng/key 4))) :regenerate)
        "regenerate: its own run's alone")
    (is (agrees? (batched/take ps [3 0]) :simulate) "taken particles' values gathered with the trace")
    (let [{tr :trace} (gfi/run-op m :simulate [] {:key (rng/key 5) :batch-shape [n]})]
      (is (close? (second (peek (kept tr n))) (batch/unwrap (:score tr) [n])) "gfi/run-op, batched"))
    (is (every? #(= [:generate :generate] (mapv first (::scores %))) (batched/traces ps))
        "a rebuilt particle's trace holds the keys of the generate that rebuilt it")
    (let [branching (scoring (gen [] (if (trace :s (d/bernoulli 0.5)) (trace :y (d/normal 1 1)) (trace :z (d/normal -1 1)))))
          fell (batched/simulate branching [] 8 (rng/key 6))]
      (is (not (batched/batched? fell)))
      (is (every? #(and (= [:simulate :simulate] (mapv first (::scores %))) (close? (:score %) (second (peek (::scores %)))))
                  (:traces fell))
          "fallen back: each particle's run its own"))))

;; Planted bugs (2026-10-25): with :selection out of the checked keys its
;; case fails; without the check of a choice recorded at the path, the
;; transition that returns the state it received passes.
(deftest middleware-that-writes-a-handler-key-is-an-error
  (let [t (dyn/site-transitions :simulate)
        m (gen [] (trace :x (d/normal 0 1)))
        run (fn [step] (gfi/simulate (dyn/with-transitions m {:simulate step}) [] (rng/key 1)))
        writing (fn [k v] (fn [s p d key] (let [[x s'] (t s p d key)] [x (assoc s' k v)])))]
    (doseq [k [:mode :prev :selection :argdiffs :key :visited :subtraces :omega :omega-key :omega-count
               :gen.dynamic/above]]
      (is (= {:path [:x] :key k} (ts/thrown-data (run (writing k ::changed)))) (str k " written")))
    (is (= :key (:key (ts/thrown-data (run (fn [s p d key] (t (assoc s :key key) p d key))))))
        "or handed to the transition it wraps")
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"recorded no choice there" (run (fn [s _ _ _] [0.0 s]))))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"not \[value state'\]" (run (fn [s p d key] (first (t s p d key))))))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"left the key :count" (run (writing :count 1)))
        "an unqualified key of its own")
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"left the key :gen.trace/omega" (run (writing ::trace/omega 1)))
        "a key of gen.trace's, which would overwrite the trace's")))

;; Planted bug (2026-10-25): with :constraints among the checked keys,
;; the first generate throws.
(deftest middleware-may-condition-on-values-of-its-own
  (let [g (dyn/site-transitions :generate)
        m (gen [] (let [x (trace :x (d/normal 0 1))] (trace :y (d/normal x 1))))
        fixing (fn [path v] (dyn/with-transitions m {:generate (fn [s p d k]
                                                                 (g (update s :constraints cm/set-choice path v) p d k))}))
        {t :trace w :weight} (gfi/generate (fixing [:x] 0.5) [] cm/EMPTY (rng/key 1))]
    (is (= 0.5 (ts/host (t :x))) "the value it gives is taken")
    (is (close? w (dc/log-prob (d/normal 0 1) 0.5)) "and weighed as a constraint")
    (is (= [[:nowhere]] (:paths (ts/thrown-data (gfi/generate (fixing [:nowhere] 1.0) [] cm/EMPTY (rng/key 1)))))
        "one at a path the run never visits is the unvisited error")))

;; Planted bug (2026-10-25): a run whose middleware ran only in a callee
;; spliced without an address (its flag not on the caller's state) fails
;; the second case.
(deftest middleware's-own-keys-land-on-the-trace-of-their-run
  (let [inner (watching (gen [] (trace :x (d/normal 0 1))))
        tr (gfi/simulate (gen [] (splice :c inner) (splice inner)) [] (rng/key 1))
        st (watching (gen ^:static [] (let [x (trace :x (d/normal 0 1)) y (trace :y (d/normal x 1))] y)))
        st-tr (gfi/simulate st [] (rng/key 2))
        paths #(mapv :path (::steps %))]
    (is (= [[:x]] (paths (trace/subtrace tr :c))) "a call's, on its own trace")
    (is (= [[:x]] (paths tr)) "a splice without an address runs its callee's body in the caller's run")
    (is (= [[:x] [:y]] (paths st-tr) (paths (:trace (gfi/update st-tr (cm/choicemap :x 0.5) (rng/key 3)))))
        "a static body's, every node rerun under middleware")
    (is (= [[:x] [:y]] (paths (gfi/assess st [] (:choices st-tr) (rng/key 3)))) "a static body's assess runs the handler")))
