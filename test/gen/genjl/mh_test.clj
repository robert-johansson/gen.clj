(ns gen.genjl.mh-test
  "Gen.jl 0.4.8's test/inference/mh.jl, ported: involutive MCMC with an
   involution written in the transform DSL (gen.inference.transform) and
   one written by hand (gen.inference.mh/involutive-mh). A hand-written
   involution takes a key (D8), and its check (Gen.jl's check=true) is the
   DSL involution's alone: gen.clj checks a transform, not a function."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.mh :as mh]
            [gen.inference.transform :as tr]
            [gen.random :as rng]
            [gen.test-support :as ts]))

(def ^:private model
  ;; mh.jl:1
  (gen [] (if (trace :x (d/bernoulli 0.5))
            (trace :y (d/normal 0 1))
            (trace :z (d/normal 1 1)))))

(def ^:private proposal
  ;; mh.jl:10
  (gen [_trace]))

(tr/deftransform ^:private dsl-involution [model-in _aux-in model-out _aux-out]
  (let [x (tr/read model-in :x :discrete)]
    (tr/write model-out :x (not x) :discrete)
    (if x
      (tr/copy model-in :y model-out :z)
      (tr/copy model-in :z model-out :y))))

(defn- hand-involution
  "x flipped, y and z swapped by update; its Jacobian is 1."
  [trace _fwd-choices _fwd-retval _proposal-args key]
  (let [x (array/item (trace :x))
        constraints (cm/choicemap :x (not x) (if x :z :y) (trace (if x :y :z)))
        {t' :trace w :weight} (gfi/update trace constraints key)]
    {:trace t' :bwd-choices cm/EMPTY :weight w}))

(deftest involutive-mcmc
  ;; mh.jl:13
  (let [trace (:trace (gfi/generate model [] (cm/choicemap :x true :y 0.5) (rng/key 1)))]
    (doseq [[label involution] [["the DSL's involution" (tr/involution dsl-involution {:check? true})]
                                ["a hand-written involution" hand-involution]]]
      (testing label
        ;; accepted: the new and old densities are the same (0.5 is halfway
        ;; between the means)
        (let [{t' :trace :keys [accepted?]} (mh/involutive-mh trace proposal [] involution (rng/key 2))]
          (is accepted?)
          (is (not (array/item (t' :x))))
          (is (= 0.5 (ts/host (t' :z)))))))))
