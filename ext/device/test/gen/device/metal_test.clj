(ns gen.device.metal-test
  "gen.device.metal: the generator's refusals; and on MLX's GPU
   (-Dgen.backend=mlx), that thread i's chain is kernel/run's from
   (rng/split key n)[i] to float32 rounding with the same accept decisions,
   whatever n, and that n threads of one particle's generate are scalar
   importance sampling's weights."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [gen.choicemap :as cm]
            [gen.device.chain :as ch]
            [gen.device.metal :as metal]
            [gen.device.stage :as st]
            [gen.gfi :as gfi]
            [gen.gradients :as grad]
            [gen.inference.importance :as is]
            [gen.inference.kernel :as kernel]
            [gen.parity-bench :as pb]
            [gen.random :as rng]))

(def ^:private gpu?
  (and (= :mlx (array/backend-name array/*backend*)) (not= "cpu" (System/getProperty "gen.mlx.device"))))

(deftest the-generator-refuses-what-it-does-not-lower
  (let [[model args obs] (pb/models :vector)]
    (testing "float64: Metal has none"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"Metal lowering: no :float64"
                            (metal/source (binding [st/*float* :float64] (ch/weight-run model args obs))))))
    (testing "a program of anything but a key, then its data"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"a program of a key"
                            (metal/source (st/stage (fn [x] (array/exp x)) [[2] :float32])))))))

(deftest a-thread-is-kernel-runs-chain
  (if-not gpu?
    (println "gen.device.metal-test: not MLX on the GPU, the kernel tests skip (-Dgen.backend=mlx runs them)")
    (doseq [m [:vector :static]
            [kname program kern] [[:mala #(ch/mala-chain %1 %2 %3 pb/selection pb/tau %4) (kernel/mala pb/selection pb/tau)]
                                  [:hmc #(ch/hmc-chain %1 %2 %3 pb/selection {:L 8 :eps pb/eps} %4) (kernel/hmc pb/selection {:L 8 :eps pb/eps})]]]
      (testing [m kname]
        (let [[model args obs] (pb/models m)
              steps 30 d 2 n 3
              k (metal/compile (program model args obs steps))
              root (rng/key 7)
              [vs acc] (mapv array/->clj (metal/run k n (rng/key-data root)))
              proto (grad/continuous-selection (:trace (gfi/generate model args obs (rng/key 0))) pb/selection)
              addrs (mapv first (cm/leaves proto))]
          (testing "thread i does not depend on the launch's size"
            (let [[vs5 acc5] (mapv array/->clj (metal/run k 5 (rng/key-data root)))]
              (is (= vs (subvec vs5 0 n)))
              (is (= acc (subvec acc5 0 n)))))
          (doseq [i (range n)]
            (let [[k0 k1] (rng/split (nth (rng/split root n) i) 2)
                  tr (:trace (gfi/generate model args obs k0))
                  read (fn [t] (mapv #(double (array/->clj (cm/get-choice (:choices t) %))) addrs))
                  {:keys [draws]} (kernel/run kern tr {:samples steps :key k1 :keep read})]
              (doseq [t (range steps) j (range d)]
                (let [dev (get-in vs [i t j]) ref (nth (nth draws t) j)]
                  (is (< (Math/abs (- dev ref)) (* 1e-4 (+ 1.0 (Math/abs ref)))) [i t j dev ref])))
              (is (= (mapv (fn [t] (not= (nth draws t) (if (zero? t) (read tr) (nth draws (dec t))))) (range steps))
                     (nth acc i))))))))))

(deftest threads-of-a-generate-are-importance-sampling
  (if-not gpu?
    (println "gen.device.metal-test: not MLX on the GPU, the importance-sampling test skips")
    (doseq [m [:vector :static]]
      (testing m
        (let [[model args obs] (pb/models m)
              k (metal/compile (ch/weight-run model args obs))
              key (rng/key 21) n 200
              [w] (mapv array/->clj (metal/run k n (rng/key-data key)))
              ref (is/importance-sampling model args obs n {:key key})]
          (doseq [[i ref-w] (map-indexed vector (:log-weights ref))]
            (is (< (Math/abs (- (nth w i) ref-w)) (* 1e-4 (+ 1.0 (Math/abs ref-w)))) [i (nth w i) ref-w])))))))

(deftest data-of-any-size-are-inputs
  (if-not gpu?
    (println "gen.device.metal-test: not MLX on the GPU, the large-data test skips")
    (doseq [n [100 2000]]
      (testing n
        (let [xs (mapv #(/ (double %) (max 1 (dec n)) 0.1) (range n))
              ys (mapv #(+ (* 2.0 %) -1.0 (Math/sin (* 7.0 %))) xs)
              args [(array/array xs)] obs (cm/choicemap :ys (array/array ys))
              {:keys [program inputs]} (ch/weight-program pb/vector-model args obs)
              k (metal/compile program)
              key (rng/key 5)
              [w] (mapv array/->clj (metal/run k 64 (rng/key-data key) {:inputs inputs}))
              ref (is/importance-sampling pb/vector-model args obs 64 {:key key})]
          (doseq [[i ref-w] (map-indexed vector (:log-weights ref))]
            (is (< (Math/abs (- (nth w i) ref-w)) (* 1e-4 (+ 1.0 (Math/abs ref-w)))) [i (nth w i) ref-w])))))))
