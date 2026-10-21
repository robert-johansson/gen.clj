(ns gen.genjl.call-at-test
  "Gen.jl 0.4.8's test/modeling_library/call_at.jl, on the battery it shares
   with choice_at.jl (gen.genjl.at-combinators): foo called at a key, its
   choice :y under it."
  (:refer-clojure :exclude [+])
  (:require [clojure.test :refer [deftest testing]]
            [gen.array :refer [+]]
            [gen.combinators :as comb]
            [gen.dist :as d]
            [gen.dynamic :refer [gen]]
            [gen.genjl.at-combinators :as at]
            [gen.genjl.finite-diff :refer [logpdf-grad]]
            [gen.genjl.support :as support]))

(def foo
  "call_at.jl:3: x plus y ~ normal(x, 1)."
  (gen [x] (+ x (trace :y (d/normal x 1)))))

(def ^:private y
  "The value get_trace constrains (call_at.jl:51)."
  1.234)

(def spec
  {:gf (comb/call-at foo)
   :addr (fn [k] [k :y])
   :density (fn [x v] (support/logpdf d/normal v x 1))
   :retval (fn [x v] (+ x v))
   :values [y 2.345]
   :shallow {:values 0 :submaps 1}})

;; call_at.jl:9
(deftest assess (at/assess spec))

;; call_at.jl:18
(deftest propose (at/propose spec))

;; call_at.jl:26
(deftest generate (at/generate spec))

;; call_at.jl:58
(deftest project (at/project spec))

;; call_at.jl:65
(deftest force-update (at/update spec))

;; call_at.jl:115
(deftest free-update (at/regenerate spec))

;; call_at.jl:153: the objective is foo's log density plus retgrad times
;; its return value, x + y
(deftest choice-gradients
  (let [tr (at/initial-trace spec)
        [dy dx] (logpdf-grad d/normal y 0.4 1.0)
        retgrad 1.234]
    (testing "not selected"
      (at/unselected-gradients tr retgrad [(+ dx retgrad) nil]))
    (testing "selected, with a return-value gradient"
      (at/selected-gradients spec tr y retgrad [(+ dx retgrad) nil] (+ dy retgrad)))))

;; call_at.jl:182
(deftest accumulate-param-gradients
  (let [[_ dx] (logpdf-grad d/normal y 0.4 1.0)
        retgrad 1.234]
    (at/arg-gradients (at/initial-trace spec) retgrad [(+ dx retgrad) nil])))
