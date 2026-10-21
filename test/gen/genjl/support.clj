(ns gen.genjl.support
  "Helpers shared by the ports of Gen.jl 0.4.8's test suite (gen.genjl.*,
   M5; docs/research/genjl-test-suite.md), beside gen.test-support's host,
   close? and tol. A helper only one port needs stays in that port.

   - approx?: Julia's isapprox, as Gen.jl's tests use it.
   - logpdf: Gen.jl's logpdf(dist, x, params...); log-density, the same
     on the backend, to differentiate.
   - has-value?: Gen.jl's has_value(c, addr).
   - shallow: the counts of Gen.jl's get_values_shallow and
     get_submaps_shallow."
  (:require [gen.choicemap :as cm]
            [gen.gfi :as gfi]
            [gen.test-support :as ts]))

(defn- host-shape
  "The shape of a host value: [] for a number, else the nesting of its
   vectors."
  [v]
  (if (sequential? v) (into [(count v)] (when (seq v) (host-shape (first v)))) []))

(defn- leaves "A host value's numbers, flat, as doubles." [v] (map double (if (sequential? v) (flatten v) [v])))

(defn- norm ^double [xs] (Math/sqrt (reduce + 0.0 (map #(* (double %) (double %)) xs))))

(defn approx?
  "Julia's isapprox(x, y; atol, rtol), as Gen.jl's tests use it: x and y
   (numbers, or arrays and nested vectors) of one shape, and equal, or
   within max(atol, rtol · max(‖x‖, ‖y‖)) in the Euclidean norm (|x - y|
   for numbers), both finite. (Julia throws on shapes that differ; this is
   false.) Julia's default rtol, √eps of the element type when no atol is
   given, is float64's (1.5e-8) on a float64 backend and float32's
   (3.5e-4) on a float32 one, where the values were computed."
  ([x y] (approx? x y {}))
  ([x y {:keys [atol rtol]}]
   (let [x (ts/host x)
         y (ts/host y)
         xs (leaves x)
         ys (leaves y)
         atol (double (or atol 0.0))
         rtol (double (or rtol (cond (pos? atol) 0.0
                                     (ts/f32?) (Math/sqrt (Math/ulp (float 1.0)))
                                     :else (Math/sqrt (Math/ulp 1.0)))))]
     (and (= (host-shape x) (host-shape y))
          (or (= xs ys)
              (and (every? #(Double/isFinite %) (concat xs ys))
                   (<= (norm (map - xs ys)) (max atol (* rtol (max (norm xs) (norm ys)))))))))))

(defn log-density
  "logpdf's value as the backend under test holds it, not read on the
   host, so that it can be differentiated (gen.genjl.finite-diff's
   logpdf-grad)."
  [dist x & params]
  (:weight (gfi/assess (apply dist params) [] (cm/->Value x))))

(defn logpdf
  "Gen.jl's logpdf(dist, x, params...): the log density (or mass) of x
   under (apply dist params), a host double, as assess weighs it: for an
   array value, the sum of its elements', and an error for a value whose
   shape is not the distribution's (Julia's DimensionMismatch).
   (logpdf d/normal 0.5 0.0 1.0)"
  [dist x & params]
  (ts/number (apply log-density dist x params)))

(defn has-value?
  "Gen.jl's has_value(c, addr): is there a value at the address or path?"
  [c addr]
  (cm/has-value? (cm/get-submap c addr)))

(defn shallow
  "{:values n :submaps m}: how many values and submaps sit directly at the
   top of c, Gen.jl's length(collect(get_values_shallow(c))) and
   length(collect(get_submaps_shallow(c)))."
  [c]
  {:values (count (cm/values-shallow c)) :submaps (count (cm/submaps-shallow c))})
