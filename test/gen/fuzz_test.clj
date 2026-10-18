(ns gen.fuzz-test
  "The GFI laws, and batched path equivalence, over random generative
   functions (gen.random-models): random families, dependencies, branches,
   splices, shared-namespace splices, Map and Unfold."
  (:require [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [gen.batched :as b]
            [gen.gfi :as gfi]
            [gen.laws :as laws]
            [gen.random :as rng]
            [gen.random-models :as rm]))

(laws/deflaws random rm/gen-model 200)

(defspec batched-particles-of-random-models-are-their-scalar-traces 100
  (prop/for-all [[_ gf args] rm/gen-model
                 seed gen/nat]
    (let [ps (b/simulate gf args 3 (rng/key seed))]
      (every? #(laws/close? (:score %) (:weight (gfi/assess gf args (:choices %)))) (b/traces ps)))))
