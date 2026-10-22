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
   - within: a call's value, or ::timed-out, for a regression that hung."
  (:refer-clojure :exclude [abs])
  (:require [gen.array :as array]
            [gen.array.protocols :as p]
            [gen.choicemap :as cm]))

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
