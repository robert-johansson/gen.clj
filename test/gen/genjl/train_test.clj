(ns gen.genjl.train-test
  "Gen.jl 0.4.8's test/inference/train.jl, ported in part: the parameter
   gradients of a student model, summed over a minibatch of a teacher's
   data, against central finite differences of assess (train.jl:82), which
   gen.genjl.finite-diff takes in float64 on the JVM backend. Gen.jl
   accumulates the gradients into each parameter
   (accumulate_param_gradients!, get_param_grad); gen.clj's are values,
   summed over the traces (D7)."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dynamic :as dyn :refer [gen]]
            [gen.gfi :as gfi]
            [gen.genjl.finite-diff :as fd]
            [gen.genjl.support :as genjl]
            [gen.gradients :as grad]
            [gen.random :as rng]
            [gen.test-support :as ts]))

(def ^:private teacher
  ;; train.jl:24: z = xor(x, y)
  (gen [] (let [x (trace :x (d/bernoulli 0.5))
                y (trace :y (d/bernoulli 0.5))]
            (trace :z (d/bernoulli (if (not= x y) 1.0 0.0))))))

(def ^:private student
  ;; train.jl:33
  (gen [z]
    (let [x (trace :x (d/bernoulli (array/sigmoid (param :theta1))))
          prob-y (cond (and z x) (array/sigmoid (param :theta2))
                       z (array/sigmoid (param :theta3))
                       x (array/sigmoid (param :theta4))
                       :else (array/sigmoid (param :theta5)))]
      (trace :y (d/bernoulli prob-y)))))

(def ^:private names [:theta1 :theta2 :theta3 :theta4 :theta5])

(def ^:private store (zipmap names (repeat 0.0)))

(defn- teacher-data
  "train.jl:52, data_generator: [inputs constraints] from one teacher run."
  [key]
  (let [{:keys [choices]} (gfi/propose teacher [] key)]
    [[(cm/get-choice choices :z)] (cm/choicemap :x (cm/get-choice choices :x) :y (cm/get-choice choices :y))]))

(defn- total-log-density
  "train.jl:86: Σ over the minibatch of assess of the student, as a
   function of its five parameters."
  [minibatch]
  (fn [& thetas]
    (let [student (dyn/with-params student (zipmap names thetas))]
      (reduce + (map (fn [[inputs constraints]]
                       (ts/number (:weight (gfi/assess student (ts/host inputs) (cm/from-map (ts/host constraints))))))
                     minibatch)))))

(deftest parameter-gradients-match-finite-differences
  ;; train.jl:73, a minibatch of 100
  (let [minibatch (mapv teacher-data (rng/split (rng/key 1) 100))
        gradients (grad/weighted-sum (repeat 1.0)
                                     (map (fn [[inputs constraints]]
                                            (grad/param-gradients
                                             (:trace (gfi/generate (dyn/with-params student store) inputs constraints
                                                                   (rng/key 2)))))
                                          minibatch))
        density (total-log-density minibatch)]
    (doseq [[i theta] (map-indexed vector names)]
      (testing (str theta)
        (is (genjl/approx? (theta gradients) (fd/finite-diff density (mapv store names) i) {:atol 1e-4}))))))

;; Pending (statistical, genclj-zgkq): train.jl:116 to :122, the parameters
;; after 2,000 epochs of SGD on the teacher's data; train.jl:126, lecture!
;; and lecture_batched!.
