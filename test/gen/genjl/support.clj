(ns gen.genjl.support
  "Helpers shared by the ports of Gen.jl 0.4.8's test suite (gen.genjl.*,
   M5; docs/research/genjl-test-suite.md), beside gen.test-support's host,
   close? and tol. A helper only one port needs stays in that port.

   - approx?: Julia's isapprox, as Gen.jl's tests use it."
  (:require [gen.test-support :as ts]))

(defn- leaves
  "x's host numbers, flat: a number, or nested vectors (an array is read on
   the host first)."
  [x]
  (let [v (ts/host x)]
    (map double (if (sequential? v) (flatten v) [v]))))

(defn- norm ^double [xs] (Math/sqrt (reduce + 0.0 (map #(* (double %) (double %)) xs))))

(defn approx?
  "Julia's isapprox(x, y; atol, rtol), as Gen.jl's tests use it: x and y
   (numbers, or arrays and nested vectors of one shape) equal, or within
   max(atol, rtol · max(‖x‖, ‖y‖)) in the Euclidean norm (|x - y| for
   numbers), both finite. Julia's default rtol, √eps of the element type
   when no atol is given, is float64's (1.5e-8) on a float64 backend and
   float32's (3.5e-4) on a float32 one, where the values were computed."
  ([x y] (approx? x y {}))
  ([x y {:keys [atol rtol]}]
   (let [xs (leaves x)
         ys (leaves y)
         atol (double (or atol 0.0))
         rtol (double (or rtol (cond (pos? atol) 0.0
                                     (ts/f32?) (Math/sqrt (Math/ulp (float 1.0)))
                                     :else (Math/sqrt (Math/ulp 1.0)))))]
     (and (= (count xs) (count ys))
          (or (= xs ys)
              (and (every? #(Double/isFinite %) (concat xs ys))
                   (<= (norm (map - xs ys)) (max atol (* rtol (max (norm xs) (norm ys)))))))))))
