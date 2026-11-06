(ns gen.test-support
  "Backend-neutral helpers for the test battery, so every suite runs on the
   JVM backend and on MLX (clojure -J-Dgen.backend=mlx -X:test):

   - host: a result's host value. On the JVM backend 0-d arrays are host
     scalars already; on MLX they are arrays, read on the host here.
   - close?: equal within the precision of the backend's default float
     (float64: 1e-12; float32: a few ulps, relative and absolute).
   - jvm?, float-dtype, int-dtype: for the few checks that are about one
     backend's representation.
   - correlation: the sample correlation of host numbers, for checks of
     independence.
   - thrown-data: the ex-data of a loud error, to check what it says.
   - within: a call's value, or ::timed-out, for a regression that hung.
   - mean, variance, sd, se, variance-se, within-5-se?,
     variance-within-5-se?, moments-match?, means-agree?, log-ml-se,
     self-normalized-se: sample statistics and their bands of five
     standard errors (CLAUDE.md's Test conventions), one copy for the
     battery.
   - misreported-normal: a planted bug for negative controls, a normal
     whose density is another's.
   - when-native, native-unavailable!, native-report, native-missing: the
     native suites' (MLX's binding, llama.cpp's) tests counted as run or
     skipped, which gen.test-runner prints at a run's end and
     -Dgen.require-native turns into a failure."
  (:refer-clojure :exclude [abs])
  (:require [clojure.string :as str]
            [gen.array :as array]
            [gen.array.protocols :as p]
            [gen.array.special :as special]
            [gen.choicemap :as cm]
            [gen.dist.host :as host]))

(defn backend-name [] (p/-name array/*backend*))

(defn jvm? [] (= :jvm (backend-name)))

(defn float-dtype [] (:float (p/-defaults array/*backend*)))

(defn int-dtype [] (:int (p/-defaults array/*backend*)))

(defn f32? [] (= :float32 (float-dtype)))

(defn native-grad?
  "Does the backend differentiate natively (MLX with -Dgen.mlx.grad=native),
   not with gen.array.tape?"
  []
  (and (not (jvm?)) (= "native" (System/getProperty "gen.mlx.grad"))))

(defn flushes-subnormals?
  "Does the backend flush float32 subnormals to zero? Metal does; CUDA keeps
   them, as the CPU does (found by the Thor's soak, 2026-10-20)."
  []
  (and (not (jvm?))
       (= :gpu ((requiring-resolve 'gen.backend.mlx/device) array/*backend*))
       (.startsWith (System/getProperty "os.name") "Mac")))

(defn float64?
  "Does the backend have float64 (MLX's GPUs do not)?"
  []
  (or (jvm?) (not= :gpu ((requiring-resolve 'gen.backend.mlx/device) array/*backend*))))

(defn host
  "x with every backend array in it (nested maps, vectors and seqs, and
   choice maps, as maps) read on the host: numbers, booleans and nested
   vectors."
  [x]
  (cond (instance? gen.array.protocols.IArray x) (array/->clj x)
        (cm/choicemap? x) (host (cm/to-map x))
        (map? x) (if (record? x) x (into (empty x) (map (fn [[k v]] [k (host v)])) x))
        (vector? x) (mapv host x)
        (seq? x) (doall (map host x))
        :else x))

(defn tolerance
  "{:rtol :atol} for the backend's default float: a few float32 ulps of the
   values compared, or float64 round-off."
  []
  (if (f32?) {:rtol 2e-5 :atol 2e-5} {:rtol 1e-12 :atol 1e-12}))

(defn tol
  "A tolerance band b, stated for float64: as it is on a float64 backend,
   at least float32's precision (2e-5) on a float32 one."
  ^double [b]
  (if (f32?) (max (double b) 2e-5) (double b)))

(defn close?
  "x and y elementwise within the backend's tolerance (or rtol/atol given)."
  ([x y] (array/allclose? x y (tolerance)))
  ([x y tol] (array/allclose? x y (merge (tolerance) tol))))

(defn number
  "The host double of a 0-d (or one-element) array or number."
  ^double [x]
  (double (array/item x)))

(defn abs
  "|x| of a 0-d array or number, as a host double."
  ^double [x]
  (Math/abs (number x)))

(defn finite?
  "Is the 0-d array or number x finite?"
  [x]
  (Double/isFinite (number x)))

(defmacro thrown-data
  "The ex-data of the ex-info body throws, or nil when body returns."
  [& body]
  `(try ~@body nil (catch clojure.lang.ExceptionInfo e# (ex-data e#))))

(defn within
  "(f)'s value, or :gen.test-support/timed-out after ms. f runs on a daemon
   thread, so a call that never returns is left behind without keeping the
   JVM alive; what f throws is thrown here."
  [ms f]
  (let [p (promise)]
    (doto (Thread. ^Runnable (fn [] (deliver p (try {:value (f)} (catch Throwable e {:error e})))))
      (.setDaemon true)
      (.start))
    (let [r (deref p ms ::timed-out)]
      (cond (= ::timed-out r) r
            (:error r) (throw (:error r))
            :else (:value r)))))

(defn correlation
  "The sample correlation of the coordinates of pairs [x y] of host numbers."
  [pairs]
  (let [centered (fn [xs] (let [m (/ (reduce + xs) (count xs))] (map #(- % m) xs)))
        xs (centered (map first pairs))
        ys (centered (map second pairs))
        dot (fn [as bs] (reduce + (map * as bs)))]
    (/ (dot xs ys) (Math/sqrt (* (dot xs xs) (dot ys ys))))))

;; ---------------------------------------------------------------------------
;; Sample statistics and bands of five standard errors
;; ---------------------------------------------------------------------------
;; One copy for the battery (genclj-7z2s: about twenty files had their own,
;; and fixed-percentage bands crept in beside them). They take host
;; numbers; a 0-d array among xs is read on the host.

(defn- host-doubles
  ^doubles [xs]
  (double-array (map #(if (number? %) (double %) (number %)) xs)))

(defn- mean-of ^double [^doubles a] (/ (areduce a i s 0.0 (+ s (aget a i))) (alength a)))

(defn- central-moment
  "Σ(x - m)^k / n of the doubles a about m."
  ^double [^doubles a ^double m ^long k]
  (/ (areduce a i s 0.0 (+ s (Math/pow (- (aget a i) m) k))) (alength a)))

(defn- variance-of
  "The unbiased sample variance of the doubles a."
  ^double [^doubles a]
  (let [n (alength a)] (* (central-moment a (mean-of a) 2) (/ n (- n 1.0)))))

(defn mean
  "The sample mean of xs."
  ^double [xs]
  (mean-of (host-doubles xs)))

(defn variance
  "The unbiased sample variance of xs (n - 1 in the denominator)."
  ^double [xs]
  (variance-of (host-doubles xs)))

(defn sd
  "The sample standard deviation of xs, √variance."
  ^double [xs]
  (Math/sqrt (variance xs)))

(defn se
  "The standard error of xs's mean from the sample's own spread, √(s²/n)."
  ^double [xs]
  (Math/sqrt (/ (variance xs) (count xs))))

(defn variance-se
  "The standard error of xs's sample variance s², from the sample's fourth
   central moment m4: Var(s²) = (μ4 - σ⁴(n - 3)/(n - 1))/n, with m4 and s²
   for μ4 and σ². For a normal sample it is σ²√(2/(n - 1)); for any other
   it is the distribution's own (a Gamma(3, 1)'s at n = 1000 is 0.19, not
   the normal's 0.13). It needs a finite fourth moment, and is estimable
   only where the eighth is finite too."
  ^double [xs]
  (let [a (host-doubles xs) n (alength a)
        s2 (variance-of a)
        m4 (central-moment a (mean-of a) 4)]
    (Math/sqrt (/ (max 0.0 (- m4 (* s2 s2 (/ (- n 3.0) (- n 1.0))))) n))))

(defn within-5-se?
  "Is the sample mean of xs within five standard errors of mu: √(v/n) for
   draws of variance v, or without v, se (the sample's own spread)?"
  ([xs mu]
   (<= (Math/abs (- (mean xs) (double mu))) (* 5.0 (se xs))))
  ([xs mu v]
   (<= (Math/abs (- (mean xs) (double mu))) (* 5.0 (Math/sqrt (/ (double v) (count xs)))))))

(defn variance-within-5-se?
  "Is the sample variance of xs within five standard errors (variance-se)
   of v?"
  [xs v]
  (<= (Math/abs (- (variance xs) (double v))) (* 5.0 (variance-se xs))))

(defn moments-match?
  "Do the independent draws xs have mean mu and variance v, each within
   five standard errors: the mean's √(v/n), the variance's variance-se?"
  [xs mu v]
  (and (within-5-se? xs mu v) (variance-within-5-se? xs v)))

(defn means-agree?
  "Do the means of xs and ys, two independent samples, agree within five
   standard errors of their difference, √(se_x² + se_y²) (and 1e-12, for
   samples without spread)?"
  [xs ys]
  (let [sx (se xs) sy (se ys)]
    (< (Math/abs (- (mean xs) (mean ys))) (+ 1e-12 (* 5.0 (Math/sqrt (+ (* sx sx) (* sy sy))))))))

(defn log-ml-se
  "The standard error of an importance-sampling estimate of log Z from its
   log normalized weights ŵ (host numbers), by the delta method: Ẑ's
   relative variance is (E[w²]/E[w]² - 1)/n, read from the weights as
   √(Σŵ² - 1/n)."
  ^double [log-normalized-weights]
  (let [a (host-doubles log-normalized-weights) n (alength a)]
    (Math/sqrt (max 0.0 (- (areduce a i s 0.0 (+ s (Math/exp (* 2.0 (aget a i))))) (/ 1.0 n))))))

(defn self-normalized-se
  "The standard error of a self-normalized importance-sampling estimate
   Σŵx (log normalized weights ŵ, values xs: host numbers), by the delta
   method: √(Σŵ²(x - Σŵx)²)."
  ^double [log-normalized-weights xs]
  (let [w (map #(Math/exp (double %)) (host-doubles log-normalized-weights))
        x (host-doubles xs)
        m (reduce + (map * w x))]
    (Math/sqrt (reduce + (map (fn [^double wi ^double xi] (* wi wi (- xi m) (- xi m))) w x)))))

;; ---------------------------------------------------------------------------
;; Planted bugs for negative controls
;; ---------------------------------------------------------------------------

(def misreported-normal
  "(misreported-normal mu sd sd'): draws from N(mu, sd), and reports the log
   density of N(mu, sd'), a proposal whose density is wrong unless sd' is
   sd (genclj-k3a8: a statistical check run against it must fail). Its
   parameters are host numbers, or 0-d arrays read on the host."
  (host/family {:name "misreported-normal"
                :make (fn [mu sd sd'] [(double mu) (double sd) (double sd')])
                :quantile (fn [[mu sd _] u] (+ mu (* sd (Math/sqrt 2.0) (special/erfinv (- (* 2.0 u) 1.0)))))
                :log-density (fn [[mu _ sd'] x]
                               (let [z (/ (- x mu) sd')]
                                 (- (* -0.5 z z) (Math/log sd') (* 0.5 (Math/log (* 2.0 Math/PI))))))}))

;; ---------------------------------------------------------------------------
;; Native suites: run, skipped, required (genclj-ruh8)
;; ---------------------------------------------------------------------------
;; The MLX binding's and llama.cpp's tests pass with no assertions where
;; their library (or test model) is missing. Each counts its tests here, so
;; that a run says how many skipped, and a release gate run with
;; -Dgen.require-native=mlx,llama fails instead (gen.test-runner).

(defonce ^:private native (atom {}))

(defn ^:no-doc native-ran! [suite]
  (swap! native update-in [suite :ran] (fnil inc 0)))

(defn ^:no-doc native-skipped! [suite ns-name]
  (swap! native (fn [m] (-> m (update-in [suite :skipped] (fnil inc 0)) (update-in [suite :nses] (fnil conj (sorted-set)) ns-name)))))

(defn native-unavailable!
  "Record why the native suite (:mlx or :llama) cannot run: its library,
   binding or test model missing."
  [suite why]
  (swap! native update-in [suite :why] #(or % why)))

(defmacro when-native
  "body when the native suite's availability (a delay of a boolean) holds;
   nil otherwise, the test counted as skipped. Counted either way, so a run
   tells a suite that ran from one that skipped or never ran."
  [suite available & body]
  `(if (deref ~available)
     (do (native-ran! ~suite) ~@body)
     (native-skipped! ~suite '~(ns-name *ns*))))

(defn native-counts
  "{suite {:ran :skipped :nses :why}} of this JVM's run so far."
  []
  @native)

(defn required-native
  "The suites -Dgen.require-native names (comma-separated: mlx, llama)."
  []
  (into #{} (comp (map str/trim) (remove str/blank?) (map keyword))
        (str/split (or (System/getProperty "gen.require-native") "") #",")))

(defn native-report
  "A line on the native suites counts reached ({suite {:ran :skipped :nses
   :why}}), or nil when none was."
  [counts]
  (when (seq counts)
    (str "Native suites: "
         (str/join "; " (for [[suite {:keys [ran skipped nses why]}] (sort-by key counts)]
                          (str (name suite) " ran " (or ran 0) " tests"
                               (when skipped
                                 (str ", skipped " skipped " (" (str/join " " nses) ")"
                                      (when why (str ": " why)))))))
         (when (some (comp :skipped val) counts)
           " (-J-Dgen.require-native=mlx,llama fails a run that skips them)"))))

(defn native-missing
  "The required suites (required-native) that skipped a test or ran none,
   by the counts: {suite why}."
  [counts required]
  (into (sorted-map)
        (for [suite required
              :let [{:keys [ran skipped why]} (get counts suite)]
              :when (or skipped (not ran))]
          [suite (cond skipped (str "skipped " skipped " tests" (when why (str ": " why)))
                       :else "no test of it ran")])))
