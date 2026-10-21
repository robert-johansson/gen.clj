(ns gen.genjl.choice-at-test
  "Gen.jl 0.4.8's test/modeling_library/choice_at.jl, on the battery it
   shares with call_at.jl (gen.genjl.at-combinators): a bernoulli choice at
   a key."
  (:refer-clojure :exclude [+ - update])
  (:require [clojure.test :refer [deftest testing]]
            [gen.array :as array :refer [+ -]]
            [gen.choicemap :as cm]
            [gen.combinators :as comb]
            [gen.dist :as d]
            [gen.genjl.at-combinators :as at]
            [gen.genjl.finite-diff :refer [logpdf-grad]]
            [gen.gfi :as gfi]
            [gen.random :as rng]
            [gen.test-support :as ts]))

(def spec
  {:gf (comb/choice-at d/bernoulli)
   :addr identity
   :density (fn [prob v] (array/log (if (true? (ts/host v)) prob (- 1 prob))))
   :retval (fn [_ v] v)
   :values [true false]
   :shallow {:values 1 :submaps 0}})

;; choice_at.jl:5
(deftest assess (at/assess spec))

;; choice_at.jl:13
(deftest propose (at/propose spec))

;; choice_at.jl:21
(deftest generate (at/generate spec))

;; choice_at.jl:52
(deftest project (at/project spec))

;; choice_at.jl:59
(deftest update (at/update spec))

;; choice_at.jl:104
(deftest regenerate (at/regenerate spec))

;; choice_at.jl:142: a normal choice, whose value is the return value
(deftest choice-gradients
  (let [y 1.2
        tr (:trace (gfi/generate (comb/choice-at d/normal) [0.0 1.0 3] (cm/choicemap 3 y) (rng/key 1)))
        [dy dmu dstd] (logpdf-grad d/normal y 0.0 1.0)]
    (testing "not selected"
      (at/unselected-gradients tr 1.2 [dmu dstd nil]))
    (testing "selected, with a return-value gradient"
      (at/selected-gradients spec tr y 1.234 [dmu dstd nil] (+ dy 1.234)))))

;; choice_at.jl:175
(deftest accumulate-param-gradients
  (let [[_ dprob] (logpdf-grad d/bernoulli true 0.4)]
    (at/arg-gradients (at/initial-trace spec) nil [dprob nil])))
