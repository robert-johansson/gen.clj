(ns gen.genjl.hmc-test
  "Gen.jl 0.4.8's test/inference/hmc.jl, ported: a smoke test of HMC
   (gen.inference.mcmc/hmc). Gen.jl runs it again on the same function
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

(deftest hmc
  ;; hmc.jl:1: runs without error
  (let [tr (:trace (gfi/generate foo [] cm/EMPTY (rng/key 1)))]
    (is (trace/trace? (:trace (mcmc/hmc tr (sel/select :x) {:key (rng/key 2)}))))))
