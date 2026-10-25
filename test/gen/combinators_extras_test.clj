(ns gen.combinators-extras-test
  "Scan, Mix and dimap (with contramap and map-retval): the GFI laws, exact
   answers (a Scan HMM's evidence against the forward algorithm, a Mix's
   marginal density), updates that skip what did not change, precise change
   hints against unknown-change, batched runs against scalar ones, stores,
   and loud errors."
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [gen.array :as array :refer [+ - * /]]
            [gen.batched :as b]
            [gen.choicemap :as cm]
            [gen.combinators :as comb]
            [gen.diff :as diff]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :as dyn :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.enumerative :as enum]
            [gen.inference.particle-filter :as pf]
            [gen.laws :as laws :refer [close?]]
            [gen.random :as rng]
            [gen.selection :as sel]
            [gen.test-support :as ts]
            [gen.trace :as trace]))

;; ---------------------------------------------------------------------------
;; Models
;; ---------------------------------------------------------------------------

(def walk
  "A Scan kernel: a noisy random walk driven by its input x and a drift,
   its carry the state and its output twice the state."
  (gen [c x drift]
    (let [s (trace :s (d/normal (+ c x drift) 1))]
      (trace :obs (d/normal s 0.5))
      [s (* 2.0 s)])))

(def scanned (comb/scan walk))

(def flipping
  "A Scan kernel whose choices vary: :extra when it flips, its output the
   flip."
  (gen [c x]
    (let [z (trace :z (d/normal (+ (* 0.5 c) x) 1))
          f (trace :flip (d/bernoulli 0.5))]
      (when f (trace :extra (d/gamma 2 1)))
      [z f])))

(def near (gen [mu] (trace :x (d/normal mu 1))))
(def far (gen [mu] (trace :x (d/normal (+ mu 3.0) 2))))
(def apart (gen [_] (+ (trace :b (d/gamma 2 1)) (trace :c (d/poisson 3.0)))))

(def mixed "Two components sharing :x." (comb/mix near far))
(def mixed-apart "Two components with different addresses." (comb/mix near apart))
(def mixed-logits (comb/mix-logits near far apart))

(def kern (gen [m] (trace :x (d/normal m 1.0))))

(def contramapped
  "A Map called on a mean and a length."
  (comb/contramap (comb/map kern) (fn [m n] [(vec (repeat n m))])))

(def summed
  "A Scan's last carry plus the sum of its outputs."
  (comb/map-retval scanned (fn [[c ys]] (reduce + c ys))))

(def dimapped (comb/dimap mixed (fn [mu] [[0.25 0.75] mu]) (fn [x] (* x x))))

(def spliced
  "A dynamic model splicing the three."
  (gen [n]
    (let [mu (trace :mu (d/normal 0 1))
          [c ys] (splice :walk scanned mu (vec (repeat n 0.1)) 0.2)
          x (splice :mix mixed [0.4 0.6] c)]
      (splice :total summed x ys 0.0))))

;; Scan's alternative arguments change its length, its initial carry, an
;; input and its parameter; Mix's its weights and its arguments
(def models
  [[:scan scanned [0.0 [0.1 0.2 0.3] 0.5]
    [[0.0 [0.1 0.2] 0.5] [0.0 [0.1 0.2 0.3 0.4] 0.5] [1.0 [0.1 0.2 0.3] 0.5] [0.0 [0.1 0.9 0.3] 0.5]
     [0.0 [0.1 0.2 0.3] -0.5]]]
   [:scan-of-a-branching-kernel (comb/scan flipping) [0.0 [0.0 1.0 2.0]] [[0.5 [0.0 1.0]] [0.0 [0.0 1.0 2.0 3.0]]]]
   [:scan-of-nothing scanned [0.0 [] 0.5] [[0.0 [0.4] 0.5]]]
   ;; steps taken out of their support and brought back (genclj-awd0)
   [:scan-of-a-bounded-kernel (comb/scan (gen [c x] (let [v (trace :v (d/uniform x (+ x 1.0)))] [(+ c v) v])))
    [0.0 [0.0 0.0]] [[0.0 [5.0 0.0]] [0.0 [0.0 -3.0]]]]
   [:mix mixed [[0.3 0.7] 0.5] [[[0.6 0.4] 0.5] [[0.3 0.7] 1.5]]]
   [:mix-of-different-addresses mixed-apart [[0.5 0.5] 0.0] [[[0.1 0.9] 0.0] [[0.5 0.5] 2.0]]]
   [:mix-logits mixed-logits [[0.0 1.0 -1.0] 0.5] [[[2.0 0.0 0.0] 0.5]]]
   [:contramap contramapped [0.5 3] [[0.5 2] [1.0 3] [0.5 4]]]
   [:map-retval summed [0.0 [0.1 0.2] 0.5] [[0.0 [0.1 0.2 0.3] 0.5]]]
   [:dimap dimapped [1.0] [[2.0]]]
   [:spliced spliced [2] [[3] [1]]]])

;; Planted bugs (2026-10-23), each failing the laws named: Scan's update
;; simulating the steps it adds, so that their constraints go unused
;; (updates-are-reversible-with-exact-discards and both update-weight laws);
;; a Scan step's discard dropped (updates-are-reversible-… and
;; update-weight-is-the-score-change-less-fresh-choices); Mix's project 0
;; (eight: both project laws, generate's two weight laws, regenerate's
;; three, and update-weight-is-the-projection-change); a dimap without an
;; argument map whose retdiff is no-change, or one whose mapped arguments
;; are called unchanged (updates-are-reversible-… and
;; regenerate-keeps-unselected-choices, by their retdiff checks); a skipped
;; Scan step whose carry is read from its output
;; (identity-update-changes-nothing, with the tests below).
(laws/deflaws extras models 100 (merge laws/all laws/conventions))
(laws/deflaws extras models 100 laws/fixed-support)

(defn- consistent? [tr]
  (close? (:score tr) (:weight (gfi/assess (:gen-fn tr) (:args tr) (:choices tr) (rng/key 0)))))

(defn- subtraces [tr] (:gen.combinators/subtraces tr))

;; ---------------------------------------------------------------------------
;; Scan
;; ---------------------------------------------------------------------------

(deftest scan-basics
  (let [args [0.0 [0.1 0.2 0.3] 0.5]
        tr (gfi/simulate scanned args (rng/key 1))
        [c ys] (:retval tr)]
    (is (= [[0 :obs] [0 :s] [1 :obs] [1 :s] [2 :obs] [2 :s]] (cm/addresses (:choices tr))) "step t's choices at t")
    (is (= 3 (count ys)))
    (is (= (ts/host (tr [2 :s])) (ts/host c)) "the last carry is the last state")
    (is (= (ts/host (* 2.0 (tr [1 :s]))) (ts/host (ys 1))) "the outputs, in order")
    (is (= (ts/host [(tr [0 :s]) (ys 0)]) (ts/host (tr 0))) "a step's return value, by its address")
    (is (consistent? tr))
    (testing "step t's carry is step t - 1's, init at step 0"
      (let [t (:trace (gfi/generate scanned args (cm/choicemap [0 :s] 1.0 [1 :s] -2.0 [2 :s] 0.5) (rng/key 2)))
            lp #(dc/log-prob (d/normal %1 1) %2)]
        (is (close? (gfi/project t (sel/select [0 :s] [1 :s] [2 :s]))
                    (+ (lp (+ 0.0 0.1 0.5) 1.0) (lp (+ 1.0 0.2 0.5) -2.0) (lp (+ -2.0 0.3 0.5) 0.5))))))
    (testing "no inputs: no steps, and the initial carry"
      (let [t (gfi/simulate scanned [7.5 [] 0.5] (rng/key 3))]
        (is (= [7.5 []] (:retval t)))
        (is (cm/empty? (:choices t)))))
    (is (= 2 (count (scanned 0.0 [1.0 2.0] 0.5))) "an untraced call returns [carry ys]")))

(defn- counting
  "A Scan kernel that counts its runs: an AR(1) state driven by x, with a
   parameter a, observed by y; the carry the state, the output y."
  [runs]
  (gen [z x a]
    (swap! runs inc)
    (let [z' (trace :z (d/normal (+ (* a z) x) 1))]
      [z' (trace :y (d/normal z' 0.5))])))

(defn- summing
  "A Scan kernel that counts its runs: its carry the running sum of its
   draws, each around its input."
  [runs]
  (gen [acc x]
    (swap! runs inc)
    (let [z (trace :z (d/normal x 1))] [(+ acc z) z])))

;; Planted bugs (2026-10-23), each failing the testings named: part-diff
;; without its same-value? check (a constrained output alone: 2 runs, its
;; step and the next); changed-steps giving every step for a vector-diff (a
;; changed input: 10 runs; the extension, 11; the shrinking); a walk that
;; never skips (twelve assertions, in all but the unknown inputs, the
;; parameter change and the carry that changes).
(deftest scan-updates-skip-the-steps-nothing-changed
  (let [runs (atom 0)
        sc (comb/scan (counting runs))
        xs (vec (repeat 10 0.0))
        nc diff/no-change
        tr (gfi/simulate sc [0.0 xs 0.9] (rng/key 1))
        runs-of (fn [f] (reset! runs 0) (let [r (f)] [@runs r]))
        kept (fn [t2 steps] (every? #(identical? ((subtraces tr) %) ((subtraces t2) %)) steps))]
    (testing "nothing changed: no step runs"
      (let [[n {rd :retdiff w :weight}] (runs-of #(gfi/update tr cm/EMPTY (rng/key 2)))]
        (is (= 0 n))
        (is (diff/no-change? rd))
        (is (zero? (ts/number w)))))
    (testing "a constrained output alone reruns its step: the carry is unchanged"
      (let [[n {t2 :trace w :weight rd :retdiff}] (runs-of #(gfi/update tr (cm/choicemap [6 :y] 1.0) (rng/key 3)))]
        (is (= 1 n))
        (is (kept t2 (remove #{6} (range 10))))
        (is (close? w (- (:score t2) (:score tr))))
        (is (= (diff/vector-diff 2 2 {1 (diff/vector-diff 10 10 {6 diff/unknown-change})}) rd))
        (is (diff/consistent? (:retval tr) (:retval t2) rd))))
    (testing "a changed state reruns its step and the next, whose state is kept: then the carry is unchanged"
      (let [[n {t2 :trace w :weight}] (runs-of #(gfi/update tr (cm/choicemap [6 :z] 1.0) (rng/key 4)))]
        (is (= 2 n))
        (is (kept t2 (concat (range 6) (range 8 10))))
        (is (close? w (- (:score t2) (:score tr))))
        (is (close? (:score t2) (:weight (gfi/assess sc [0.0 xs 0.9] (:choices t2)))))))
    (testing "a changed input reruns its step, and the next ones only while the carry changes"
      (let [xs' (assoc xs 4 0.7)
            [n {t2 :trace w :weight}] (runs-of #(gfi/update tr [0.0 xs' 0.9] [nc (diff/vector-diff 10 10 {4 diff/unknown-change}) nc]
                                                            cm/EMPTY (rng/key 5)))]
        (is (= 1 n) "step 4's state is kept, so its carry is unchanged")
        (is (close? w (- (:score t2) (:score tr))))
        (is (close? (:score t2) (:weight (gfi/assess sc [0.0 xs' 0.9] (:choices t2)))))))
    (testing "unknown-change of the inputs reruns every step"
      (is (= 10 (first (runs-of #(gfi/update tr [0.0 xs 0.9] [nc diff/unknown-change nc] cm/EMPTY (rng/key 6)))))))
    (testing "a new initial carry reruns the first step"
      (is (= 1 (first (runs-of #(gfi/update tr [1.0 xs 0.9] [diff/unknown-change nc nc] cm/EMPTY (rng/key 7)))))))
    (testing "a parameter change reruns every step"
      (let [[n {t2 :trace w :weight}] (runs-of #(gfi/update tr [0.0 xs 0.5] [nc nc diff/unknown-change] cm/EMPTY (rng/key 8)))]
        (is (= 10 n))
        (is (close? w (- (:score t2) (:score tr))))))
    (testing "extending by a step, the particle filter's move, runs the new step only"
      (let [[n {t2 :trace w :weight rd :retdiff}]
            (runs-of #(gfi/update tr [0.0 (conj xs 0.0) 0.9] [nc (diff/vector-diff 11 10 {}) nc] (cm/choicemap [10 :y] 0.3)
                                  (rng/key 9)))]
        (is (= 1 n))
        (is (kept t2 (range 10)))
        (is (close? w (gfi/project t2 (sel/select [10 :y]))) "the weight is the new observation's density")
        (is (diff/consistent? (:retval tr) (:retval t2) rd))))
    (testing "regenerating a state reruns its step and the next"
      (let [[n {t2 :trace w :weight}] (runs-of #(gfi/regenerate tr (sel/select [6 :z]) (rng/key 10)))
            later (sel/select [6 :y] [7 :z])]
        (is (= 2 n))
        (is (close? w (- (gfi/project t2 later) (gfi/project tr later))))))
    (testing "shrinking runs nothing and discards the steps dropped"
      (let [[n {t2 :trace w :weight disc :discard rd :retdiff}]
            (runs-of #(gfi/update tr [0.0 (subvec xs 0 7) 0.9] [nc (diff/vector-diff 7 10 {}) nc] cm/EMPTY (rng/key 11)))]
        (is (= 0 n))
        (is (= #{7 8 9} (set (map first (cm/addresses disc)))))
        (is (close? w (- (:score t2) (:score tr))))
        (is (diff/consistent? (:retval tr) (:retval t2) rd)))))
  (testing "a carry that changes reruns every later step"
    (let [runs (atom 0)
          sc (comb/scan (summing runs))
          tr (gfi/simulate sc [0.0 (vec (repeat 10 0.0))] (rng/key 12))
          _ (reset! runs 0)
          {t2 :trace w :weight rd :retdiff} (gfi/update tr (cm/choicemap [6 :z] 1.0) (rng/key 13))]
      (is (= 4 @runs))
      (is (close? w (- (:score t2) (:score tr))))
      (is (= #{0 1} (set (keys (:updated rd)))) "the last carry and the outputs changed")
      (is (diff/consistent? (:retval tr) (:retval t2) rd)))))

(def gen-scan-change
  "A Scan update: lengths, inputs from a few values (so that some stay),
   the initial carry and the drift, and a constraint on one step."
  (gen/let [n (gen/choose 0 6)
            n' (gen/choose 0 6)
            xs (gen/vector (gen/elements [0.0 0.5 1.0]) 6)
            xs' (gen/vector (gen/elements [0.0 0.5 1.0]) 6)
            init (gen/elements [0.0 0.5])
            drift (gen/elements [0.5 -0.2])
            t (gen/choose 0 8)
            obs? gen/boolean
            v (gen/double* {:min -3 :max 3 :NaN? false})
            seed gen/nat]
    {:xs (subvec xs 0 n) :xs' (subvec xs' 0 n') :init init :drift drift :t t :addr (if obs? :obs :s) :v v :seed seed}))

(defn- same-result? [r1 r2]
  (and (= (:choices (:trace r1)) (:choices (:trace r2)))
       (= (ts/host (:retval (:trace r1))) (ts/host (:retval (:trace r2))))
       (close? (:score (:trace r1)) (:score (:trace r2)))
       (close? (:weight r1) (:weight r2))
       (= (:discard r1) (:discard r2))))

;; Planted bug (2026-10-23): changed-steps giving no step for a vector-diff
;; (the precise hints then skip a changed input) fails it.
(defspec scan-precise-hints-match-unknown-change 200
  (prop/for-all [{:keys [xs xs' init drift t addr v seed]} gen-scan-change]
    (let [tr (gfi/simulate scanned [0.0 xs 0.5] (rng/key seed))
          k (rng/key (inc seed))
          cs (if (< t (count xs')) (cm/choicemap [t addr] v) cm/EMPTY)
          args [init xs' drift]
          precise (gfi/update tr args [(diff/compute 0.0 init) (diff/compute xs xs') (diff/compute 0.5 drift)] cs k)
          unknown (gfi/update tr args (vec (repeat 3 diff/unknown-change)) cs k)]
      (and (same-result? precise unknown)
           (close? (:score (:trace precise)) (:weight (gfi/assess scanned args (:choices (:trace precise)))))
           (diff/consistent? (:retval tr) (:retval (:trace precise)) (:retdiff precise))))))

;; An HMM: two states, a transition matrix, normal emissions
(def ^:private transitions [[0.8 0.2] [0.3 0.7]])
(def ^:private emission-means [-1.0 1.5])

(def hmm-step
  "An HMM's step: the next state from the current one's row of
   transitions, observed with noise; its carry and output the state."
  (let [rows (array/array transitions)
        means (array/array emission-means)]
    (gen [z _]
      (let [z' (trace :z (d/categorical (array/take rows z)))]
        (trace :y (d/normal (array/take means z') 1.0))
        [z' z']))))

(def hmm (comb/scan hmm-step))

(def ^:private observed [0.3 -1.2 2.0 0.7 1.1])

(defn- forward-log-evidence
  "log p(ys) by the forward algorithm on the host, the state before the
   first step z0."
  [z0 ys]
  (let [density (fn [j y] (Math/exp (clojure.core/- (clojure.core/* -0.5 (Math/pow (clojure.core/- y (emission-means j)) 2))
                                                    (clojure.core/* 0.5 (Math/log (clojure.core/* 2 Math/PI))))))
        step (fn [alpha y]
               (mapv (fn [j] (clojure.core/* (density j y)
                                             (reduce clojure.core/+ (map #(clojure.core/* (alpha %) (get-in transitions [% j]))
                                                                         [0 1]))))
                     [0 1]))]
    (Math/log (reduce clojure.core/+ (reduce step (assoc [0.0 0.0] z0 1.0) ys)))))

(defn- observations [ys] (apply cm/choicemap (mapcat (fn [t y] [[t :y] y]) (range) ys)))

;; Planted bugs (2026-10-23): every step given init as its carry, in
;; generate's steps (fails the first testing) or in assess's (the second).
(deftest scan-hmm-evidence-is-the-forward-algorithm-s
  (let [xs (vec (repeat (count observed) nil))
        exact (forward-log-evidence 1 observed)
        paths (enum/choice-vol-grid [[0 :z] [0 1]] [[1 :z] [0 1]] [[2 :z] [0 1]] [[3 :z] [0 1]] [[4 :z] [0 1]])]
    (testing "enumerating the states, through generate"
      (is (close? exact (:log-ml-estimate (enum/enumerative-inference hmm [1 xs] (observations observed) paths)))))
    (testing "assess summed over every path"
      (is (close? exact (array/logsumexp (array/stack (mapv #(:weight (gfi/assess hmm [1 xs] (cm/merge (first %) (observations observed))))
                                                            paths))))))))

;; The particle filter's estimate is unbiased: its mean over independent
;; filters, in exp space, is the evidence within 5 standard errors of the
;; filters' spread (here 0.977 ± 0.023). Planted bug (2026-10-23): a Scan
;; extension whose new step's carry is init fails it: the ratio's mean is
;; 2.40 ± 0.03, and the batch falls back (its spot checks disagree).
(deftest ^:slow scan-particle-filters-estimate-the-evidence
  (let [n 500
        filters 20
        exact (forward-log-evidence 1 observed)
        nc diff/no-change
        run (fn [k]
              (let [ks (rng/split k (count observed))
                    xs-to #(vec (repeat % nil))
                    state (pf/initialize hmm [1 (xs-to 1)] (observations (take 1 observed)) n {:batched? true :key (ks 0)})
                    state (reduce (fn [state t]
                                    (-> (pf/maybe-resample state {:key (rng/fold-in (ks t) 1)})
                                        (pf/step [1 (xs-to (inc t))] [nc (diff/vector-diff (inc t) t {})]
                                                 (cm/choicemap [t :y] (observed t)) {:key (ks t)})))
                                  state (range 1 (count observed)))]
                [(b/batched? (:particles state)) (Math/exp (clojure.core/- (ts/number (pf/log-ml-estimate state)) exact))]))
        results (mapv run (rng/split (rng/key 21) filters))
        ratios (map second results)
        mean (clojure.core// (reduce clojure.core/+ ratios) filters)
        sd (Math/sqrt (clojure.core// (reduce clojure.core/+ (map #(Math/pow (clojure.core/- % mean) 2) ratios)) (dec filters)))]
    (is (every? first results) "the filter stays batched")
    (is (< (Math/abs (clojure.core/- mean 1.0)) (clojure.core/* 5 (clojure.core// sd (Math/sqrt filters)))))))

;; ---------------------------------------------------------------------------
;; Mix
;; ---------------------------------------------------------------------------

(deftest mix-basics
  (let [tr (gfi/simulate mixed [[0.3 0.7] 0.5] (rng/key 1))]
    (is (= #{[:index] [:component :x]} (set (cm/addresses (:choices tr)))) "the index at :index, the component's choices under :component")
    (is (= (ts/host (tr [:component :x])) (ts/host (:retval tr)) (ts/host (tr :component))) "the component's return value")
    (is (consistent? tr))
    (is (= mixed (:gen-fn (trace/subtrace tr []))))
    (is (= (comb/switch near far) (:gen-fn (trace/subtrace tr :component))) "the component's call is a Switch's")
    (testing "a change of component keeps the choices both make, as a Switch does"
      (let [tr (:trace (gfi/generate mixed [[0.3 0.7] 0.5] (cm/choicemap :index 0 [:component :x] 0.3) (rng/key 2)))
            {t :trace w :weight disc :discard} (gfi/update tr (cm/choicemap :index 1) (rng/key 3))
            lp dc/log-prob]
        (is (= 0.3 (ts/host (t [:component :x]))))
        (is (= {:index 0} (ts/host disc)))
        (is (close? w (+ (- (lp (d/categorical [0.3 0.7]) 1) (lp (d/categorical [0.3 0.7]) 0))
                         (- (lp (d/normal 3.5 2) 0.3) (lp (d/normal 0.5 1) 0.3)))))))))

(def ^:private mix-inline
  "Mix's documented model, written out."
  (gen [weights & args]
    (apply splice :component (comb/switch near far) (trace :index (d/categorical weights)) args)))

;; Planted bug (2026-10-23): Mix's simulate giving its body a key folded
;; from its own fails it.
(deftest mix-is-a-categorical-and-a-switch
  (doseq [seed (range 6)]
    (let [args [[0.3 0.7] 0.5]
          k (rng/key seed)
          tr (gfi/simulate mixed args k)
          ti (gfi/simulate mix-inline args k)
          same? (fn [r ri] (and (= (:choices (:trace r)) (:choices (:trace ri))) (close? (:weight r) (:weight ri))
                                (= (:discard r) (:discard ri))))]
      (is (= (:choices tr) (:choices ti)) "the same draws, by key")
      (is (close? (:score tr) (:score ti)))
      (is (same? (gfi/update tr [[0.6 0.4] 1.0] [diff/unknown-change diff/unknown-change] (cm/choicemap :index 1) (rng/key 50))
                 (gfi/update ti [[0.6 0.4] 1.0] [diff/unknown-change diff/unknown-change] (cm/choicemap :index 1) (rng/key 50))))
      (is (same? (gfi/regenerate tr (sel/select :index) (rng/key 51)) (gfi/regenerate ti (sel/select :index) (rng/key 51)))))))

;; Planted bugs (2026-10-23): probabilities read as logits fail every
;; testing; logits read as probabilities, the third (loudly: they are not
;; probabilities).
(deftest mix-marginal-density
  (let [w [0.3 0.7]
        x 0.8
        exact (dc/log-prob ((d/heterogeneous-mixture [d/normal d/normal]) w [0.5 1] [3.5 2]) x)
        given-x (cm/choicemap [:component :x] x)]
    (testing "assess, summed over the index"
      (is (close? exact (array/logsumexp (array/stack (mapv #(:weight (gfi/assess mixed [w 0.5] (cm/merge given-x (cm/choicemap :index %))))
                                                            [0 1]))))))
    (testing "enumerating the index, through generate"
      (is (close? exact (:log-ml-estimate (enum/enumerative-inference mixed [w 0.5] given-x (enum/choice-vol-grid [:index [0 1]]))))))
    (testing "mix-logits weighs by the softmax of its logits"
      (let [logits [0.4 -1.3 2.0]
            probs (array/softmax (array/array logits))
            given (cm/choicemap :index 1 [:component :x] x)]
        (is (close? (:weight (gfi/assess (comb/mix near far apart) [probs 0.5] given))
                    (:weight (gfi/assess mixed-logits [logits 0.5] given))))))))

(deftest ^:slow ^:heavy mix-draws-components-by-their-weights
  ;; the index's frequencies and x's mean within 5 standard errors
  (let [n 4000
        trs (map #(gfi/simulate mixed [[0.3 0.7] 0.5] %) (rng/split (rng/key 30) n))
        ones (count (filter #(= 1 (long (ts/number (% :index)))) trs))
        xs (map #(ts/number (% [:component :x])) trs)
        ;; x's mixture: means 0.5 and 3.5, variances 1 and 4
        m (clojure.core/+ (clojure.core/* 0.3 0.5) (clojure.core/* 0.7 3.5))
        v (clojure.core/- (clojure.core/+ (clojure.core/* 0.3 (clojure.core/+ 1 0.25)) (clojure.core/* 0.7 (clojure.core/+ 4 12.25)))
                          (clojure.core/* m m))]
    (is (< (Math/abs (clojure.core/- (clojure.core// ones (double n)) 0.7)) (clojure.core/* 5 (Math/sqrt (clojure.core// (clojure.core/* 0.3 0.7) n)))))
    (is (< (Math/abs (clojure.core/- (clojure.core// (reduce clojure.core/+ xs) n) m)) (clojure.core/* 5 (Math/sqrt (clojure.core// v n)))))))

(deftest mix-weights-are-checked
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"one per component \(2\), not 3"
                        (gfi/simulate mixed [[0.2 0.3 0.5] 0.0] (rng/key 1))))
  (testing "GenMLX's log weights are not probabilities (mix-logits takes them)"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"probabilities" (gfi/simulate mixed [[-1.2 -0.36] 0.0] (rng/key 2))))
    (is (gfi/simulate (comb/mix-logits near far) [[-1.2 -0.36] 0.0] (rng/key 2))))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"at least one" (comb/mix)))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"generative function" (comb/mix near 3)))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"not visited"
                        (gfi/generate mixed [[0.3 0.7] 0.0] (cm/choicemap :elsewhere 1.0) (rng/key 3)))))

;; ---------------------------------------------------------------------------
;; dimap, contramap, map-retval
;; ---------------------------------------------------------------------------

(deftest dimap-traces-are-the-inner-s
  (doseq [seed (range 4)]
    (let [k (rng/key seed)
          tr (gfi/simulate dimapped [1.0] k)
          inner (gfi/simulate mixed [[0.25 0.75] 1.0] k)]
      (is (= (:choices inner) (:choices tr)) "the inner's choices at its addresses, by key")
      (is (close? (:score inner) (:score tr)))
      (is (close? (* (:retval inner) (:retval inner)) (:retval tr)) "the return value mapped")
      (is (= (ts/host (tr [:component :x])) (ts/host (tr :component))) "the inner's calls, by address")
      (is (consistent? tr)))))

;; Planted bugs (2026-10-23): inner-call giving unknown-change for every
;; mapped argument (the first testing: 3 runs); inner-call mapping the
;; arguments again though none changed (the second's count of maps).
(deftest contramap-change-hints
  (let [runs (atom 0)
        maps (atom 0)
        counted (gen [m] (swap! runs inc) (trace :x (d/normal m 1.0)))
        picked (comb/contramap (comb/map counted) (fn [ms _label] (swap! maps inc) [ms]))
        ms [0.0 1.0 2.0]
        tr (gfi/simulate picked [ms :a] (rng/key 1))
        runs-of (fn [f] (reset! runs 0) (let [r (f)] [@runs r]))]
    (testing "an argument the map drops changed: the Map's argument is identical, so unchanged"
      (let [[n {t :trace w :weight}] (runs-of #(gfi/update tr [ms :b] [diff/no-change diff/unknown-change] cm/EMPTY (rng/key 2)))]
        (is (= 0 n))
        (is (zero? (ts/number w)))
        (is (= [ms :b] (:args t)))))
    (testing "no argument changed: the inner's arguments are the previous ones, not mapped again"
      (reset! maps 0)
      (let [[n {w :weight}] (runs-of #(gfi/update tr cm/EMPTY (rng/key 3)))]
        (is (= 0 n))
        (is (= 0 @maps))
        (is (zero? (ts/number w)))))
    (testing "a new argument vector, equal or not, reruns every element"
      (let [[n {t :trace w :weight}] (runs-of #(gfi/update tr [[0.0 1.5 2.0] :a] [diff/unknown-change diff/no-change] cm/EMPTY
                                                           (rng/key 4)))]
        (is (= 3 n))
        (is (close? w (- (:score t) (:score tr))))
        (is (close? (:score t) (:weight (gfi/assess picked [[0.0 1.5 2.0] :a] (:choices t)))))))
    (testing "without an argument map, the argdiffs pass through: a vector-diff reruns the elements it names"
      (let [m (comb/map-retval (comb/map counted) count)
            tr (gfi/simulate m [ms] (rng/key 5))]
        (is (= 1 (first (runs-of #(gfi/update tr [[0.0 1.5 2.0]] [(diff/vector-diff 3 3 {1 diff/unknown-change})] cm/EMPTY
                                              (rng/key 6))))))))))

;; Planted bugs (2026-10-23): map-retval's retdiff without its same-value?
;; check fails the second testing (unknown-change); mapping the return
;; value again when the inner's is unchanged, the first's identical?.
(deftest map-retval-retdiffs
  (let [m (comb/map-retval (comb/map kern) (fn [xs] {:n (count xs)}))
        tr (gfi/simulate m [[0.0 1.0 2.0]] (rng/key 1))]
    (is (= {:n 3} (:retval tr)))
    (testing "the inner's return value unchanged: no-change, the same value"
      (let [{t :trace rd :retdiff} (gfi/update tr cm/EMPTY (rng/key 2))]
        (is (diff/no-change? rd))
        (is (identical? (:retval tr) (:retval t)))))
    (testing "the inner's changed but not the mapped value: no-change"
      (let [{rd :retdiff} (gfi/update tr (cm/choicemap [1 :x] 0.25) (rng/key 3))]
        (is (diff/no-change? rd))))
    (testing "the mapped value changed: a sound retdiff"
      (let [{t :trace rd :retdiff} (gfi/update tr [[0.0 1.0]] [diff/unknown-change] cm/EMPTY (rng/key 4))]
        (is (= {:n 2} (:retval t)))
        (is (diff/unknown-change? rd))))))

(deftest dimap-errors-are-loud
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"generative function" (comb/dimap 3 identity identity)))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"needs a function" (comb/contramap mixed 3)))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"needs a function" (comb/map-retval mixed nil)))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"a vector, not 2.0"
                        (gfi/simulate (comb/contramap kern (fn [m] (* 2.0 m))) [1.0] (rng/key 1)))))

;; ---------------------------------------------------------------------------
;; Scan's errors
;; ---------------------------------------------------------------------------

(deftest scan-errors-are-loud
  (let [k (rng/key 0)]
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"inputs, a vector" (gfi/simulate scanned [0.0 0.1 0.5] k)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"takes \(init xs & params\)" (gfi/simulate scanned [0.0] k)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"returns \[carry y\]"
                          (gfi/simulate (comb/scan (gen [c x] (trace :z (d/normal c 1)))) [0.0 [1.0]] k)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"not visited"
                          (gfi/generate scanned [0.0 [1.0] 0.5] (cm/choicemap [3 :s] 1.0) k)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"not visited"
                          (gfi/generate scanned [0.0 [1.0] 0.5] (cm/choicemap [:oops :s] 1.0) k)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"generative function" (comb/scan 3)))))

;; ---------------------------------------------------------------------------
;; Stores and identity
;; ---------------------------------------------------------------------------

;; Planted bug (2026-10-23): Mix's with-stores keeping its components'
;; stores fails the first testing.
(deftest stores-are-the-callees
  (let [held (dyn/with-params (gen [mu] (trace :x (d/normal (+ mu (param :w)) 1))) {:w 1.0})
        held-step (dyn/with-params (gen [c x] (let [s (trace :s (d/normal (+ c x (param :w)) 1))] [s s])) {:w 1.0})]
    (testing "Mix: the components', by index, as a Switch's"
      (let [m (comb/mix held far)]
        (is (= {0 {:w 1.0}} (gfi/stores m)))
        (let [m' (gfi/with-stores m {0 {:w 5.0}})
              given (cm/choicemap :index 0 [:component :x] 5.5)]
          (is (= {0 {:w 5.0}} (gfi/stores m')))
          (is (close? (+ (Math/log 0.5) (dc/log-prob (d/normal 5.5 1) 5.5)) (:weight (gfi/assess m' [[0.5 0.5] 0.5] given)))))))
    (testing "Scan: the kernel's"
      (is (= {:w 1.0} (gfi/stores (comb/scan held-step))))
      (is (= {:w 2.0} (gfi/stores (gfi/with-stores (comb/scan held-step) {:w 2.0})))))
    (testing "dimap: the inner's"
      (is (= {:w 1.0} (gfi/stores (comb/contramap held (fn [x] [x])))))
      (is (= {:w 3.0} (gfi/stores (gfi/with-stores (comb/map-retval held -) {:w 3.0})))))))

;; Planted bug (2026-10-23): Mix's same-gen-fn? ignoring logits? fails it.
(deftest rebuilt-combinators-are-the-same
  ;; so that a body building one again stays incremental
  (is (gfi/same-gen-fn? (comb/mix near far) (comb/mix near far)))
  (is (not (gfi/same-gen-fn? (comb/mix near far) (comb/mix far near))))
  (is (not (gfi/same-gen-fn? (comb/mix near far) (comb/mix-logits near far))))
  (is (gfi/same-gen-fn? (comb/scan walk) (comb/scan walk)))
  (is (not (gfi/same-gen-fn? (comb/scan walk) (comb/scan flipping))))
  (let [shifted (fn [a] (comb/contramap kern (fn [m] [(+ m a)])))]
    (is (gfi/same-gen-fn? (shifted 1.0) (shifted 1.0)) "a map built again from the same expression and value")
    (is (not (gfi/same-gen-fn? (shifted 1.0) (shifted 2.0)))))
  (testing "a Mix rebuilt in a body updates its component, which skips what did not change"
    (let [m (gen [w] (splice :m (comb/mix (comb/map kern) (comb/map kern)) w [0.0 1.0]))
          tr (gfi/simulate m [[0.3 0.7]] (rng/key 1))
          t (:trace (gfi/update tr cm/EMPTY (rng/key 2)))]
      (is (identical? (trace/subtrace tr [:m :component 1]) (trace/subtrace t [:m :component 1]))))))

;; ---------------------------------------------------------------------------
;; Batched runs
;; ---------------------------------------------------------------------------

;; Planted bugs (2026-10-23): Scan's generate simulating its steps fails the
;; Scan testing (and the dimap one, a Scan's); a skipped Scan step's carry
;; read from its output fails the Scan testing's update. After genclj-7tqw
;; (2026-10-24): a Switch scattering its groups' scores in reverse order
;; fails the per-particle Mix's scores and the Mask's batch (its spot checks
;; then disagree); a Mask whose false branch returns nil, the Mask's.
(deftest batched-runs-are-the-scalar-runs
  (testing "Scan: simulate, generate, and the particle filter's extension"
    (let [args [0.0 [0.1 0.2 0.3] 0.5]
          ps (b/simulate scanned args 4 (rng/key 1))
          obs (cm/choicemap [1 :obs] 0.4)
          {gs :particles ws :weights} (b/generate scanned args obs 4 (rng/key 2))
          args' [0.0 [0.1 0.2 0.3 0.4] 0.5]
          new-obs (cm/choicemap [3 :obs] -0.2)
          {us :particles uws :weights} (b/update ps args' [diff/no-change (diff/vector-diff 4 3 {}) diff/no-change] new-obs
                                                 (rng/key 3))]
      (is (b/batched? ps))
      (is (every? consistent? (b/traces ps)))
      (is (b/batched? gs))
      (is (every? true? (map #(close? %2 (gfi/project %1 (sel/from-choicemap obs))) (b/traces gs) (array/->clj ws))))
      (is (b/batched? us))
      (is (every? true? (map (fn [old new w] (and (consistent? new)
                                                  (= (:choices old) (sel/get-selected (:choices new) (sel/complement (sel/select 3))))
                                                  (close? w (gfi/project new (sel/from-choicemap new-obs)))))
                             (b/traces ps) (b/traces us) (array/->clj uws))))))
  (testing "Mix with an index shared by the particles"
    (let [m (gen [] (splice :m mixed [0.4 0.6] 0.0))
          obs (cm/choicemap [:m :index] 1 [:m :component :x] 2.0)
          {ps :particles ws :weights} (b/generate m [] obs 4 (rng/key 4))]
      (is (b/batched? ps))
      (is (every? true? (map #(close? %2 (:weight (gfi/generate m [] (:choices %1) (rng/key 0)))) (b/traces ps) (array/->clj ws))))))
  (testing "Mix with an index per particle: its Switch runs by partition (genclj-7tqw)"
    ;; spot checks off, so that the batched scores and weights themselves
    ;; are compared with the scalar handler's
    (binding [b/*spot-checks* 0]
      (let [args [[0.4 0.6] 0.0]
            indices (fn [ps] (mapv #(ts/host (% :index)) (b/traces ps)))
            ;; a key whose particles' indices differ under float64's draws
            ;; and float32's (key 5 drew all ones in float32)
            ps (b/simulate mixed args 8 (rng/key 8))]
        (is (b/batched? ps))
        (is (nil? (b/fallback-reason ps)))
        (is (< 1 (count (distinct (indices ps)))) "the particles' indices differ")
        (is (every? true? (map #(close? %1 (:weight (gfi/assess mixed args (:choices %2)))) (array/->clj (b/scores ps)) (b/traces ps)))
            "each particle's score is its choices' density")
        (testing "and regenerating the index moves particles between components, keeping :x"
          (let [{ps' :particles ws :weights} (b/regenerate ps args [diff/no-change diff/no-change] (sel/select :index) (rng/key 7))
                x (sel/select [:component :x])]
            (is (b/batched? ps'))
            (is (not= (indices ps) (indices ps')))
            (is (every? true? (map (fn [before after w] (and (= (ts/host (before [:component :x])) (ts/host (after [:component :x])))
                                                             (close? w (- (gfi/project after x) (gfi/project before x)))))
                                   (b/traces ps) (b/traces ps') (array/->clj ws)))))))))
  (testing "a keyed Switch on booleans, GenMLX's Mask, runs by partition when its branches return values of one kind"
    (let [masked (fn [off] (gen [] (splice :m (comb/switch {true near false (gen [& _] off)}) (trace :on (d/bernoulli 0.5)) 0.0)))
          ps (b/simulate (masked 0.0) [] 8 (rng/key 8))
          nils (b/simulate (masked nil) [] 8 (rng/key 8))]
      (is (b/batched? ps))
      (is (nil? (b/fallback-reason ps)))
      (is (< 1 (count (distinct (map #(ts/host (% :on)) (b/traces ps))))))
      (is (every? consistent? (b/traces ps)))
      (testing "nil and a number do not stack: the batch falls back, to the same particles"
        (is (not (b/batched? nils)))
        (is (every? consistent? (b/traces nils))))))
  (testing "dimap"
    (let [ps (b/simulate summed [0.0 [0.1 0.2] 0.5] 4 (rng/key 6))]
      (is (b/batched? ps))
      (is (every? consistent? (b/traces ps))))))
