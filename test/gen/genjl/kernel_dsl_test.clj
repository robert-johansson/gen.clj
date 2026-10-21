(ns gen.genjl.kernel-dsl-test
  "Gen.jl 0.4.8's test/inference/kernel_dsl.jl, ported: a primitive kernel
   with metadata (Gen.jl's @pkern, here gen.inference.kernel/kernel) and a
   composite one with a loop (@kern, here kern), run with its reversal."
  (:require [clojure.test :refer [deftest is]]
            [gen.dist :as d]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.kernel :as k]
            [gen.inference.mh :as mh]
            [gen.random :as rng]
            [gen.selection :as sel]
            [gen.test-support :as ts]))

(def ^:private model
  ;; kernel_dsl.jl:1
  (gen [] (let [x (trace :x (d/normal 0.0 1.0))] (trace :y (d/normal x 1.0)))))

(defn- k1
  "kernel_dsl.jl:7, @pkern k1(tr, t): t MH moves on :x, with whether any
   was accepted as metadata."
  [t]
  (k/kernel (fn [tr key]
              (reduce (fn [{:keys [trace metadata]} key]
                        (let [{trace :trace accepted? :accepted?} (mh/mh trace (sel/select :x) key)]
                          {:trace trace :metadata (or metadata accepted?)}))
                      {:trace tr :metadata false}
                      (rng/split key t)))))

(def ^:private k1-proposal
  ;; kernel_dsl.jl:18
  (gen [_tr] (trace :x (d/normal 0.0 1.0))))

(def ^:private k1-non-primitive
  ;; kernel_dsl.jl:22, @kern k1_non_primitive(tr, t): t MH moves with k1_proposal
  (k/kern [t] (doseq [_ (range t)] (k/mh k1-proposal []))))

(deftest kernel-dsl
  ;; kernel_dsl.jl:28
  (let [tr (gfi/simulate model [] (rng/key 1))
        original (ts/host (tr :x))
        {tr :trace acc :metadata} ((k1 1) tr (rng/key 2))]
    (if acc
      (is (not= original (ts/host (tr :x))))
      (is (= original (ts/host (tr :x)))))
    (let [{tr :trace} ((k1-non-primitive 1) tr (rng/key 3))]
      ((k/reversal (k1-non-primitive 1)) tr (rng/key 4)))))
