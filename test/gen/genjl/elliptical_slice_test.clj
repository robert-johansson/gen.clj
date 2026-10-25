(ns gen.genjl.elliptical-slice-test
  "Gen.jl 0.4.8's test/inference/elliptical_slice.jl, ported: a smoke test
   of elliptical slice sampling (gen.inference.mcmc/elliptical-slice)."
  (:require [clojure.test :refer [deftest is]]
            [gen.array :as array]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.mcmc :as mcmc]
            [gen.random :as rng]
            [gen.trace :as trace]))

(def ^:private mu [0.0 0.0])

(def ^:private cov [[1.0 0.0] [0.0 1.0]])

(def ^:private foo
  (gen [] (let [x (trace :x (d/mvnormal mu cov))]
            (trace :y (d/mvnormal x [[0.01 0.0] [0.0 0.01]])))))

(deftest elliptical-slice
  ;; elliptical_slice.jl:1: runs without error
  (let [tr (:trace (gfi/generate foo [] (cm/choicemap :y (array/array [1.0 1.0])) (rng/key 1)))]
    (is (trace/trace? (:trace (mcmc/elliptical-slice tr :x mu cov (rng/key 2)))))))
