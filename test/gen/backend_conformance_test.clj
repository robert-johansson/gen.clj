(ns gen.backend-conformance-test
  "The backend conformance suite: the backend contract's ops, and the
   composites gen.array builds on them, on the backend under test
   (gen.array/*backend*, chosen with -Dgen.backend) against the JVM backend,
   the reference, on the same data with the same dtypes.

   The JVM backend computes in float64 and rounds float32 results, so it is
   the correctly rounded answer. Tolerances, by op class:
   - exact: integer and boolean results, comparisons, selections, shape and
     indexing ops, rounding, extrema, and the IEEE-exact float ops (+ - * /
     sqrt are correctly rounded in both).
   - transcendental (exp, log, sin, …): a few float32 ulps.
   - accumulating (sums, products, matrix products, solves): relative to the
     magnitude of the terms, since the order of summation differs.

   Run on MLX: clojure -J-Dgen.backend=mlx -X:test :nses '[gen.backend-conformance-test]'
   (with -J-Dgen.mlx.device=cpu -J-Dgen.mlx.dtypes=64 for float64). On the
   JVM backend it compares the backend with itself."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.test-support :as ts]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [gen.array :as array]
            [gen.array.linalg :as la]
            [gen.array.protocols :as p]
            [gen.backend.jvm :as jvm]
            [gen.random :as rng]))

(def tested array/*backend*)
(def reference jvm/backend)
(def float-dt (:float (p/-defaults tested)))
(def int-dt (:int (p/-defaults tested)))
(def f64? (= :float64 float-dt))
(def flushes? (ts/flushes-subnormals?))

;; ---------------------------------------------------------------------------
;; Running one computation on both backends
;; ---------------------------------------------------------------------------

(defn- host-result
  "The host view of a result: {:shape :dtype :value} per array (the dtype of
   a 0-d result only when the backend keeps one: the JVM backend's 0-d
   arrays are host scalars), recursively for vectors of results."
  [r]
  (if (vector? r)
    (mapv host-result r)
    {:shape (array/shape r)
     :dtype (when (seq (array/shape r)) (array/dtype r))
     :value (array/->clj r)}))

(defn- run-on
  "(f & arrays) on backend b, the arrays made from the specs {:data :dtype}."
  [b f specs]
  (array/with-backend b
    (host-result (apply f (map (fn [{:keys [data dtype]}] (array/array data {:dtype dtype})) specs)))))

(defn- as-tested-defaults
  "A reference result in the tested backend's default dtypes: where the JVM
   backend's defaults (float64, int64) gave a result's dtype and the tested
   backend's are 32-bit, the value rounded (or wrapped) to them, as if
   computed in float64 and stored as float32."
  [r]
  (cond (vector? r) (mapv as-tested-defaults r)
        (= float-dt :float64) r
        :else
        (let [round (fn round [v] (cond (sequential? v) (mapv round v)
                                        (instance? Double v) (double (unchecked-float v))
                                        (integer? v) (long (unchecked-int (long v)))
                                        :else v))]
          (case (:dtype r)
            :float64 (-> r (assoc :dtype float-dt) (update :value round))
            :int64 (-> r (assoc :dtype int-dt) (update :value round))
            ;; 0-d results of the JVM backend are host scalars without a dtype
            nil (update r :value round)
            r))))

(defn- flushed
  "A reference result with its float32 subnormals flushed to zero (keeping
   their sign) where the tested backend flushes them (Metal): the square of
   5.4e-20 is one, and the generators keep only subnormal inputs out."
  [r]
  (cond (vector? r) (mapv flushed r)
        (and flushes? (#{:float32 nil} (:dtype r)))
        (update r :value (fn flush [v]
                           (cond (sequential? v) (mapv flush v)
                                 (and (instance? Double v) (< 0.0 (Math/abs (double v)) Float/MIN_NORMAL))
                                 (* 0.0 (Math/signum (double v)))
                                 :else v)))
        :else r))

(defn- close?
  "x and y equal within atol + rtol·|y| (NaN equal to NaN, infinities exact)."
  [rtol atol x y]
  (cond (and (sequential? x) (sequential? y))
        (and (= (count x) (count y)) (every? true? (map #(close? rtol atol %1 %2) x y)))
        (and (boolean? x) (boolean? y)) (= x y)
        (and (number? x) (number? y))
        (let [x (double x) y (double y)]
          (or (= x y)
              (and (Double/isNaN x) (Double/isNaN y))
              (and (Double/isFinite x) (Double/isFinite y)
                   (<= (Math/abs (- x y)) (+ atol (* rtol (Math/abs y)))))))
        :else false))

(defn- agree?
  "Do the results agree: same shapes and dtypes, values within the tolerance."
  [{:keys [rtol atol] :or {rtol 0.0 atol 0.0} :as tol} r1 r2]
  (if (vector? r1)
    (and (vector? r2) (= (count r1) (count r2)) (every? true? (map #(agree? tol %1 %2) r1 r2)))
    (and (= (:shape r1) (:shape r2))
         (or (nil? (:dtype r1)) (nil? (:dtype r2)) (= (:dtype r1) (:dtype r2)))
         (close? rtol atol (:value r1) (:value r2)))))

(defn- conforms?
  "Does (f & arrays) agree on the tested and reference backends? Errors
   must agree too: an ex-info on both (with the same :shapes when the
   reference names them), never anything else."
  [tol f & specs]
  (let [attempt (fn [b] (try {:ok (run-on b f specs)}
                             (catch clojure.lang.ExceptionInfo e {:error (ex-data e)})))
        t (attempt tested)
        r (update (attempt reference) :ok #(some-> % as-tested-defaults flushed))]
    (cond (and (:ok t) (:ok r)) (or (agree? tol (:ok t) (:ok r))
                                    (do (println "disagree:" {:tested (:ok t) :reference (:ok r) :specs specs}) false))
          (and (:error t) (:error r)) (or (not (contains? (:error r) :shapes))
                                          (= (:shapes (:error r)) (:shapes (:error t)))
                                          (do (println "errors differ:" {:tested t :reference r :specs specs}) false))
          :else (do (println "one failed:" {:tested t :reference r :specs specs}) false))))

;; tolerances
(def exact {})
(def ieee exact)
(def transcendental (if f64? {:rtol 1e-13 :atol 1e-14} {:rtol 1e-5 :atol 1e-6}))
(def loose (if f64? {:rtol 1e-9 :atol 1e-10} {:rtol 2e-4 :atol 2e-5}))

;; ---------------------------------------------------------------------------
;; Generators
;; ---------------------------------------------------------------------------

(defn- reshape-flat [flat shape]
  (if (empty? shape)
    (first flat)
    (let [step (reduce * (rest shape))]
      (mapv #(reshape-flat (subvec flat (* % step) (* (inc %) step)) (vec (rest shape)))
            (range (first shape))))))

(def gen-shape (gen/vector (gen/frequency [[9 (gen/choose 1 4)] [1 (gen/return 0)]]) 0 3))
(def gen-nonempty-shape (gen/vector (gen/choose 1 4) 1 3))

(defn- gen-operand-shape
  "A shape that broadcasts to out-shape: a suffix of it with some dims 1."
  [out-shape]
  (gen/let [k (gen/choose 0 (count out-shape))
            ones (gen/vector gen/boolean (- (count out-shape) k))]
    (mapv (fn [d one?] (if one? 1 d)) (drop k out-shape) ones)))

(defn- no-subnormal
  "x, or 0.0 for a float32 subnormal: MLX's Metal kernels flush them to zero
   (docs/array-api.md §13)."
  [x]
  (if (and (not f64?) (< 0.0 (Math/abs (double x)) Float/MIN_NORMAL)) 0.0 x))

(def gen-float
  (gen/fmap no-subnormal
            (gen/frequency [[8 (gen/double* {:infinite? false :NaN? false :min -100.0 :max 100.0})]
                            [3 (gen/fmap double (gen/choose -4 4))]
                            [1 (gen/elements [##NaN ##Inf ##-Inf 0.5 -0.5 1.5 2.5])]])))

(def gen-finite (gen/fmap no-subnormal (gen/double* {:infinite? false :NaN? false :min -10.0 :max 10.0})))
(def gen-positive (gen/double* {:infinite? false :NaN? false :min 0.01 :max 50.0}))
(def gen-int (gen/choose -20 20))

(defn- gen-spec
  "{:data :dtype} of shape with elements from gen-elem."
  [shape gen-elem dtype]
  (gen/fmap (fn [xs] {:data (reshape-flat (vec xs) shape) :dtype dtype})
            (gen/vector gen-elem (reduce * shape))))

(defn- gen-floats [gen-elem] (gen/bind gen-shape #(gen-spec % gen-elem float-dt)))

(def gen-any-spec
  "An operand of any dtype the backend defaults to."
  (gen/bind (gen/tuple gen-shape (gen/elements [:float :int :bool]))
            (fn [[shape kind]]
              (case kind
                :float (gen-spec shape gen-float float-dt)
                :int (gen-spec shape gen-int int-dt)
                :bool (gen-spec shape gen/boolean :bool)))))

(defn- gen-pair
  "Two operands that broadcast together, elements from gen-elem as dtype."
  [gen-elem dtype]
  (gen/let [out gen-shape
            sx (gen-operand-shape out)
            sy (gen-operand-shape out)
            x (gen-spec sx gen-elem dtype)
            y (gen-spec sy gen-elem dtype)]
    [x y]))

;; ---------------------------------------------------------------------------
;; Elementwise ops
;; ---------------------------------------------------------------------------

(def ^:private trials
  "Trials per property (-Dgen.conformance.trials, default 60)."
  (Long/parseLong (System/getProperty "gen.conformance.trials" "60")))

(defspec unary-exact trials
  (prop/for-all [op (gen/elements [array/negate array/abs array/sign array/floor array/ceil array/round array/isnan array/isinf array/logical-not array/square])
                 x gen-any-spec]
    (conforms? exact op x)))

(defspec unary-ieee trials
  (prop/for-all [x (gen-floats gen-float)]
    (conforms? ieee array/sqrt x)))

(defspec unary-transcendental trials
  (prop/for-all [[op x] (gen/let [[op elem] (gen/elements [[array/exp gen-finite] [array/expm1 gen-finite] [array/tanh gen-finite]
                                                           [array/sin gen-finite] [array/cos gen-finite] [array/erf gen-finite]
                                                           [array/log gen-positive] [array/log1p gen-positive]
                                                           [array/erfinv (gen/double* {:NaN? false :min -0.999 :max 0.999})]])
                                  x (gen/bind gen-shape #(gen-spec % elem float-dt))]
                          [op x])]
    (conforms? transcendental op x)))

(deftest special-functions-in-the-tails
  ;; erfinv near ±1 (where a fixed number of refinement steps must still
  ;; converge), erf beyond the series' range, lgamma and digamma near poles
  (doseq [[op xs] [[array/erfinv [0.5 -0.9 0.99 0.999999 (- 1.0 1e-12) (- 1.0 1e-14) -0.99999999 1.0 -1.0 0.0 1.5]]
                   [array/erf [-6.0 -3.0 -0.49 0.0 0.49 0.51 2.0 5.5]]
                   [array/lgamma [0.001 0.5 1.0 2.0 2.5 -0.5 -1.5 10.0 171.5]]
                   [array/digamma [0.01 0.5 1.0 1.461632 -0.5 -2.5 20.0]]]]
    (let [d (if f64? [] (mapv #(double (float %)) xs))]
      (is (conforms? (if f64? {:rtol 1e-12 :atol 1e-14} loose) op {:data (if f64? xs d) :dtype float-dt})
          (str op)))))

(defspec binary-exact trials
  (prop/for-all [op (gen/elements [array/+ array/- array/* array/maximum array/minimum array/< array/<= array/> array/>= array/== array/!=
                                   array/logical-and array/logical-or])
                 [x y] (gen/one-of [(gen-pair gen-float float-dt) (gen-pair gen-int int-dt)
                                    (gen-pair gen/boolean :bool)])]
    (conforms? exact op x y)))

(defspec binary-division trials
  (prop/for-all [op (gen/elements [array// array/mod array/floor-div])
                 [x y] (gen/one-of [(gen-pair gen-float float-dt) (gen-pair gen-int int-dt)])]
    ;; float mod and floor-div by MLX's kernels: within an ulp of the quotient
    (conforms? (if (= op array//) ieee transcendental) op x y)))

(deftest integer-division-at-the-edges
  ;; MIN / -1 overflows: x86's idiv traps on it, which killed the JVM on
  ;; MLX's x86 CPU (genclj-kw88, M2 review finding 11); Java's floorDiv
  ;; wraps to MIN and floorMod gives 0. And a zero divisor gives 0. In the
  ;; tested backend's int dtype, as the reference is compared in it
  (let [mn (if (= int-dt :int32) Integer/MIN_VALUE Long/MIN_VALUE)]
    (doseq [op [array/mod array/floor-div]]
      (is (conforms? exact op {:data [mn mn 7 -7 0 mn 5] :dtype int-dt} {:data [-1 1 -1 -1 -1 2 0] :dtype int-dt})))))

(deftest casts-to-int-at-the-edges
  ;; Java's casts (DESIGN.md §9): MLX's own is C++'s, which gave MIN for
  ;; all of these on x86's CPU (genclj-kw88). (uint32: gen.array-test, as
  ;; the reference has no uint32 arrays)
  (is (conforms? exact #(array/astype % int-dt)
                 {:data [##NaN ##Inf ##-Inf 1e20 -1e20 2.7 -2.7 -0.5 2147483648.0 -2147483648.0 4294967296.0] :dtype float-dt})))

(defspec mixed-dtypes-promote trials
  (prop/for-all [op (gen/elements [array/+ array/* array/maximum array/<])
                 x gen-any-spec
                 y gen-any-spec
                 c (gen/elements [2 2.5 true])]
    ;; operands that do not broadcast must fail alike
    (and (conforms? exact op x y)
         (conforms? exact #(op % c) x))))

(defspec power trials
  (prop/for-all [[x y] (gen/one-of [(gen-pair gen-positive float-dt)
                                    ;; negative exponents too: MLX's integer power gives 0 for all
                                    (gen-pair (gen/choose -3 6) int-dt)])]
    (conforms? transcendental array/pow x y)))

(defspec where trials
  (prop/for-all [[c x y] (gen/let [out gen-shape
                                   [sc sx sy] (gen/vector (gen-operand-shape out) 3)
                                   c (gen-spec sc gen/boolean :bool)
                                   x (gen-spec sx gen-float float-dt)
                                   y (gen-spec sy gen-float float-dt)]
                           [c x y])]
    (conforms? exact array/where c x y)))

(defspec composites trials
  (prop/for-all [[op tol x] (gen/let [[op elem tol] (gen/elements [[array/sigmoid gen-finite transcendental]
                                                                   [array/softplus gen-finite transcendental]
                                                                   [array/lgamma gen-positive loose]
                                                                   [array/digamma gen-positive loose]
                                                                   [#(array/logaddexp % 1.5) gen-finite transcendental]
                                                                   [#(array/xlogy % 2.0) gen-finite transcendental]
                                                                   [array/isfinite gen-float exact]])
                                      x (gen/bind gen-shape #(gen-spec % elem float-dt))]
                              [op tol x])]
    (conforms? tol op x)))

;; ---------------------------------------------------------------------------
;; Reductions and scans
;; ---------------------------------------------------------------------------

(defn- gen-axes
  "nil (all), one axis or a set of axes of a rank-r array, possibly negative."
  [r]
  (if (zero? r)
    (gen/return nil)
    (gen/one-of [(gen/return nil)
                 (gen/choose (- r) (dec r))
                 (gen/fmap #(vec (distinct %)) (gen/vector (gen/choose 0 (dec r)) 0 r))])))

(defspec reductions trials
  (prop/for-all [[op tol kinds] (gen/elements [[array/sum loose [:float :int :bool]] [array/prod loose [:float :int]]
                                               [array/max exact [:float :int]] [array/min exact [:float :int]]
                                               [array/all exact [:bool :float]] [array/any exact [:bool :int]]
                                               [array/mean loose [:float]] [array/logsumexp loose [:float]]])
                 shape gen-nonempty-shape
                 keep? gen/boolean
                 seed gen/nat]
    (let [kind (gen/generate (gen/elements kinds) 1 seed)
          x (gen/generate (case kind
                            :float (gen-spec shape gen-finite float-dt)
                            :int (gen-spec shape (gen/choose -3 3) int-dt)
                            :bool (gen-spec shape gen/boolean :bool))
                          10 seed)
          axes (gen/generate (gen-axes (count shape)) 10 seed)]
      (conforms? tol #(op % {:axis axes :keep-dims? keep?}) x))))

(defspec arg-reductions trials
  (prop/for-all [op (gen/elements [array/argmax array/argmin])
                 shape gen-nonempty-shape
                 seed gen/nat]
    (let [x (gen/generate (gen-spec shape (gen/one-of [gen-float (gen/fmap double (gen/choose -2 2))]) float-dt) 10 seed)
          ax (gen/generate (gen/one-of [(gen/return nil) (gen/choose 0 (dec (count shape)))]) 10 seed)]
      (conforms? exact #(op % {:axis ax}) x))))

(defspec scans trials
  (prop/for-all [[op tol] (gen/elements [[array/cumsum loose] [array/cumprod loose] [array/logcumsumexp loose]])
                 shape gen-nonempty-shape
                 reverse? gen/boolean
                 seed gen/nat]
    (let [x (gen/generate (gen-spec shape (gen/double* {:NaN? false :infinite? false :min -2.0 :max 2.0}) float-dt) 10 seed)
          ax (gen/generate (gen/choose (- (count shape)) (dec (count shape))) 10 seed)]
      (conforms? tol #(op % {:axis ax :reverse? reverse?}) x))))

;; ---------------------------------------------------------------------------
;; Shape and indexing
;; ---------------------------------------------------------------------------

(defspec shape-ops trials
  (prop/for-all [shape gen-nonempty-shape
                 seed gen/nat]
    (let [x (gen/generate (gen-spec shape gen-int int-dt) 10 seed)
          r (count shape)
          perm (vec (shuffle (range r)))]
      (and (conforms? exact #(array/reshape % [-1]) x)
           (conforms? exact #(array/transpose % perm) x)
           (conforms? exact array/transpose x)
           (conforms? exact #(array/expand-dims % 0) x)
           (conforms? exact #(array/broadcast-to % (into [2] shape)) x)
           (conforms? exact #(array/concatenate [% %] (dec r)) x)
           (conforms? exact #(array/stack [% % %] 0) x)
           (conforms? exact array/squeeze x)))))

(defspec slices trials
  (prop/for-all [shape gen-nonempty-shape
                 seed gen/nat]
    (let [x (gen/generate (gen-spec shape gen-int int-dt) 10 seed)
          r (count shape)
          [ax start stop step] (gen/generate (gen/tuple (gen/choose 0 (dec r))
                                                        (gen/one-of [(gen/return nil) (gen/choose -6 6)])
                                                        (gen/one-of [(gen/return nil) (gen/choose -6 6)])
                                                        (gen/elements [1 2 3 -1 -2]))
                                             10 seed)]
      (conforms? exact #(array/slice % ax start stop step) x))))

(defspec gathers trials
  (prop/for-all [shape gen-nonempty-shape
                 seed gen/nat]
    (let [x (gen/generate (gen-spec shape gen-finite float-dt) 10 seed)
          r (count shape)
          ax (gen/generate (gen/choose 0 (dec r)) 10 seed)
          d (nth shape ax)
          idx (gen/generate (gen-spec [3] (gen/choose (- d) (dec d)) int-dt) 10 seed)
          along (gen/generate (gen-spec (assoc shape ax 2) (gen/choose 0 (dec d)) int-dt) 10 seed)
          updates (gen/generate (gen-spec (assoc shape ax 2) gen-finite float-dt) 10 (inc seed))]
      (and (conforms? exact #(array/take %1 %2 ax) x idx)
           (conforms? exact #(array/take % (dec d) ax) x)
           (conforms? exact #(array/take-along-axis %1 %2 ax) x along)
           (conforms? loose #(array/scatter-add %1 %2 %3 ax) x along updates)
           (conforms? exact #(array/pick %1 %2) x (gen/generate (gen-spec (pop shape) (gen/choose 0 (dec (peek shape))) int-dt) 10 seed))))))

;; ---------------------------------------------------------------------------
;; Linear algebra
;; ---------------------------------------------------------------------------

(defn- spd
  "A well-conditioned symmetric positive-definite n×n matrix from m."
  [m n]
  (mapv (fn [i] (mapv (fn [j] (+ (reduce + (map #(* (nth (nth m i) %) (nth (nth m j) %)) (range n)))
                                 (if (= i j) (double n) 0.0)))
                      (range n)))
        (range n)))

(defspec matmuls trials
  (prop/for-all [[n k m] (gen/vector (gen/choose 1 4) 3)
                 batch (gen/vector (gen/choose 1 3) 0 2)
                 seed gen/nat]
    (let [x (gen/generate (gen-spec (into batch [n k]) gen-finite float-dt) 10 seed)
          y (gen/generate (gen-spec [k m] gen-finite float-dt) 10 (inc seed))
          v (gen/generate (gen-spec [k] gen-finite float-dt) 10 (+ 2 seed))
          i (gen/generate (gen-spec [n k] gen-int int-dt) 10 seed)
          j (gen/generate (gen-spec [k m] gen-int int-dt) 10 seed)]
      (and (conforms? loose array/matmul x y)
           (conforms? loose array/matmul x v)
           (conforms? exact array/matmul i j)))))

(defspec decompositions 40
  (prop/for-all [n (gen/choose 1 4)
                 seed gen/nat]
    (let [m (gen/generate (gen/vector (gen/vector gen-finite n) n) 10 seed)
          a {:data (spd m n) :dtype float-dt}
          rhs (gen/generate (gen-spec [n 2] gen-finite float-dt) 10 seed)
          tol (if f64? {:rtol 1e-9 :atol 1e-9} {:rtol 2e-3 :atol 2e-3})]
      (and (conforms? tol la/cholesky a)
           (conforms? tol #(la/solve-triangular (la/cholesky %1) %2) a rhs)
           (conforms? tol #(la/solve-triangular (la/cholesky %1) %2 {:transpose? true}) a rhs)
           (conforms? tol la/solve a rhs)
           (conforms? tol la/slogdet a)
           (conforms? tol la/inv a)))))

;; batched matrices against one right-hand side, and the reverse
(defspec batched-solves 30
  (prop/for-all [n (gen/choose 1 3)
                 seed gen/nat]
    (let [ms (gen/generate (gen/vector (gen/vector (gen/vector gen-finite n) n) 2) 10 seed)
          as {:data (mapv #(spd % n) ms) :dtype float-dt}
          ;; [1 n 2]: a matrix right-hand side broadcast over the batch ([n 2]
          ;; would be a batch of vectors, gen.array.linalg's rule)
          rhs (gen/generate (gen-spec [1 n 2] gen-finite float-dt) 10 seed)
          rhss (gen/generate (gen-spec [2 n 2] gen-finite float-dt) 10 seed)
          tol (if f64? {:rtol 1e-9 :atol 1e-9} {:rtol 2e-3 :atol 2e-3})]
      (and (conforms? tol la/solve as rhs)
           (conforms? tol #(la/solve (array/take %1 0) %2) as rhss)
           (conforms? tol #(la/solve-triangular (la/cholesky %1) %2) as rhs)
           (conforms? tol #(la/solve-triangular (la/cholesky (array/take %1 0)) %2 {:transpose? true}) as rhss)))))

(defn- non-finite? [v]
  (if (sequential? v) (every? non-finite? v) (not (Double/isFinite (double v)))))

(deftest degenerate-linear-algebra
  ;; NaN or ±Inf, never an error and never finite garbage (MLX's own kernels
  ;; return a partial Cholesky factor and finite solutions of singular
  ;; systems, and fail on others at a later evaluation)
  (let [not-pd {:data [[1.0 2.0] [2.0 1.0]] :dtype float-dt}
        singular {:data [[1.0 2.0] [2.0 4.0]] :dtype float-dt}
        zero-diag {:data [[0.0 0.0] [1.0 1.0]] :dtype float-dt}
        v {:data [1.0 1.0] :dtype float-dt}
        batch {:data [[[2.0 0.0] [0.0 2.0]] [[1.0 2.0] [2.0 1.0]]] :dtype float-dt}]
    (is (conforms? exact la/cholesky not-pd) "not positive definite: NaN")
    (is (conforms? transcendental la/cholesky batch) "only the failing matrix of a batch is NaN")
    (is (conforms? exact la/slogdet singular) "singular: sign 0, log -Inf")
    (doseq [[label f specs] [["solve" la/solve [singular v]]
                             ["inv" la/inv [singular]]
                             ["solve-triangular" la/solve-triangular [zero-diag v]]]]
      (testing label
        (doseq [b [tested reference]]
          (is (non-finite? (:value (run-on b f specs)))))))
    (testing "and later evaluations are unaffected"
      (is (conforms? exact #(array/+ % 1.0) v)))))

;; ---------------------------------------------------------------------------
;; Randomness: the same key draws the same uniforms
;; ---------------------------------------------------------------------------

(defspec uniforms-are-identical trials
  (prop/for-all [seed gen/large-integer
                 shape gen-shape]
    (let [key (rng/key seed)]
      (agree? exact
              (array/with-backend tested (host-result (rng/uniform key shape {:dtype float-dt})))
              (array/with-backend reference (host-result (rng/uniform key shape {:dtype float-dt})))))))

(deftest samplers-agree
  (let [key (rng/key 7)
        draw (fn [b f] (array/with-backend b (host-result (f))))]
    (doseq [[label f tol] [["normal" #(rng/normal key [50] {:dtype float-dt}) transcendental]
                           ["split keys" #(rng/uniform (second (rng/split key 3)) [4] {:dtype float-dt}) exact]]]
      (testing label
        (is (agree? tol (draw tested f) (draw reference f)))))))

;; ---------------------------------------------------------------------------
;; Gradients
;; ---------------------------------------------------------------------------

(defspec gradients trials
  (prop/for-all [shape gen-nonempty-shape
                 seed gen/nat]
    (let [x (gen/generate (gen-spec shape (gen/double* {:NaN? false :infinite? false :min -2.0 :max 2.0}) float-dt) 10 seed)
          f (fn [x] (array/sum (array/+ (array/* (array/sin x) (array/exp x)) (array/logsumexp x {:axis -1 :keep-dims? true}))))]
      (conforms? loose #(second ((array/value-and-grad f) %)) x))))

;; ---------------------------------------------------------------------------
;; Errors are loud and alike
;; ---------------------------------------------------------------------------

(deftest errors
  (let [fails-alike (fn [f & specs] (apply conforms? exact f specs))
        v3 {:data [1.0 2.0 3.0] :dtype float-dt}
        v2 {:data [1.0 2.0] :dtype float-dt}]
    (is (fails-alike array/+ v3 v2) "broadcasting")
    (is (fails-alike array/matmul v3 v2) "matmul inner dimensions")
    (is (fails-alike #(array/reshape % [2 2]) v3) "reshape size")
    (is (fails-alike #(array/concatenate [% (array/reshape %2 [2 1])] 0) v3 v2) "concatenate ranks")
    (is (fails-alike #(array/take % 5) v3) "a host index out of range")
    (testing "MLX errors become ex-info, and the JVM keeps running"
      (is (thrown? clojure.lang.ExceptionInfo
                   (array/with-backend tested (array/->clj (la/cholesky (array/array [[1.0 2.0 3.0]] {:dtype float-dt})))))))))
