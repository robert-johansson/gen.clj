(ns gen.genjl.mala-test
  "Gen.jl 0.4.8's test/inference/mala.jl, ported: a smoke test of MALA
   (gen.inference.mcmc/mala). Gen.jl runs it again on the same function
   annotated (grad), for a return-value gradient; gen.clj differentiates
   without an annotation, so there is one function."
  (:require [clojure.test :refer [deftest is]]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.mcmc :as mcmc]
            [gen.random :as rng]
            [gen.selection :as sel]
            [gen.trace :as trace]))

(def ^:private foo (gen [] (trace :x (d/normal 0 1))))

(deftest mala
  ;; mala.jl:1: runs without error
  (let [tr (:trace (gfi/generate foo [] cm/EMPTY (rng/key 1)))]
    (is (trace/trace? (:trace (mcmc/mala tr (sel/select :x) 0.1 {:key (rng/key 2)}))))))
