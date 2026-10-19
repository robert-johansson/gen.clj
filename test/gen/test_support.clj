(ns gen.test-support
  "Backend-neutral helpers for the test battery, so every suite runs on the
   JVM backend and on MLX (clojure -J-Dgen.backend=mlx -X:test):

   - host: a result's host value. On the JVM backend 0-d arrays are host
     scalars already; on MLX they are arrays, read on the host here.
   - close?: equal within the precision of the backend's default float
     (float64: 1e-12; float32: a few ulps, relative and absolute).
   - jvm?, float-dtype, int-dtype: for the few checks that are about one
     backend's representation."
  (:require [gen.array :as a]
            [gen.array.protocols :as p]))

(defn backend-name [] (p/-name a/*backend*))

(defn jvm? [] (= :jvm (backend-name)))

(defn float-dtype [] (:float (p/-defaults a/*backend*)))

(defn int-dtype [] (:int (p/-defaults a/*backend*)))

(defn f32? [] (= :float32 (float-dtype)))

(defn host
  "x with every backend array in it (nested maps, vectors and seqs) read
   on the host: numbers, booleans and nested vectors."
  [x]
  (cond (instance? gen.array.protocols.IArray x) (a/->clj x)
        (map? x) (if (record? x) x (into (empty x) (map (fn [[k v]] [k (host v)])) x))
        (vector? x) (mapv host x)
        (seq? x) (doall (map host x))
        :else x))

(defn tolerance
  "{:rtol :atol} for the backend's default float: a few float32 ulps of the
   values compared, or float64 round-off."
  []
  (if (f32?) {:rtol 2e-5 :atol 2e-5} {:rtol 1e-12 :atol 1e-12}))

(defn close?
  "x and y elementwise within the backend's tolerance (or rtol/atol given)."
  ([x y] (a/allclose? x y (tolerance)))
  ([x y tol] (a/allclose? x y (merge (tolerance) tol))))

(defn number
  "The host double of a 0-d (or one-element) array or number."
  ^double [x]
  (double (a/item x)))
