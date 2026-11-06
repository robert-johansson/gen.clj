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
            [gen.array.stage :as st]
            [gen.gfi :as gfi]
            [gen.gradients :as grad]
            [gen.inference.importance :as is]
            [gen.inference.kernel :as kernel]
            [gen.parity-models :as pm]
            [gen.random :as rng]))

(def ^:private gpu?
  (and (= :mlx (array/backend-name array/*backend*)) (not= "cpu" (System/getProperty "gen.mlx.device"))))

(deftest the-generator-refuses-what-it-does-not-lower
  (let [[model args obs] (pm/models :vector)]
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
            [kname program kern] [[:mala #(ch/mala-chain %1 %2 %3 pm/selection pm/tau %4) (kernel/mala pm/selection pm/tau)]
                                  [:hmc #(ch/hmc-chain %1 %2 %3 pm/selection {:L 8 :eps pm/eps} %4) (kernel/hmc pm/selection {:L 8 :eps pm/eps})]]]
      (testing [m kname]
        (let [[model args obs] (pm/models m)
              steps 30 d 2 n 3
              k (metal/compile (program model args obs steps))
              root (rng/key 7)
              [vs acc] (mapv array/->clj (metal/run k n (rng/key-data root)))
              proto (grad/continuous-selection (:trace (gfi/generate model args obs (rng/key 0))) pm/selection)
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
        (let [[model args obs] (pm/models m)
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
              {:keys [program inputs]} (ch/weight-program pm/vector-model args obs)
              k (metal/compile program)
              key (rng/key 5)
              [w] (mapv array/->clj (metal/run k 64 (rng/key-data key) {:inputs inputs}))
              ref (is/importance-sampling pm/vector-model args obs 64 {:key key})]
          (doseq [[i ref-w] (map-indexed vector (:log-weights ref))]
            (is (< (Math/abs (- (nth w i) ref-w)) (* 1e-4 (+ 1.0 (Math/abs ref-w)))) [i (nth w i) ref-w])))))))

(def ^:private past-the-end
  "A categorical draw of 0, 1 or 2 indexing a table of two (index 2 is out
   of range)."
  (gen.dynamic/gen []
    (let [k (trace :k (gen.dist/categorical [0.2 0.3 0.5]))]
      (trace :y (gen.dist/normal (array/take (array/array [-1.0 0.0]) k) 1.0)))))

(deftest an-index-out-of-range-is-loud
  (if-not gpu?
    (println "gen.device.metal-test: not MLX on the GPU, the index-error test skips")
    (let [k (metal/compile (st/stage (fn [key] {:score (:score (gfi/simulate past-the-end [] key))}) (st/key-spec)))
          in-range? (fn [s] (try (gfi/simulate past-the-end [] (nth (rng/split (rng/key s) 1) 0)) true
                                 (catch clojure.lang.ExceptionInfo _ false)))
          good (first (filter in-range? (range 100)))]
      (testing "a launch with a run out of range throws before returning"
        (is (thrown? clojure.lang.ExceptionInfo (metal/run k 40 (rng/key-data (rng/key 3))))))
      (testing "an in-range run returns its score alone"
        (let [outs (metal/run k 1 (rng/key-data (rng/key good)))]
          (is (= 1 (count outs))))))))

(deftest a-kernel-unreachable-is-freed
  ;; genclj-2j7v: mlx_fast_metal_kernel_new's handle is freed once the
  ;; compiled program is unreachable (a Cleaner). Planted bug (2026-11-06):
  ;; the handle a bare long (1.2.0's, never freed) fails it
  (if-not gpu?
    (println "gen.device.metal-test: not MLX on the GPU, the kernel-free test skips")
    (let [before (metal/kernels-freed)
          program (st/stage (fn [key] {:x (rng/uniform key)}) (st/key-spec))]
      (dotimes [_ 4]
        (let [k (metal/compile program)]
          (array/eval! (metal/run k 2 (rng/key-data (rng/key 1))))))
      (is (loop [tries 0]
            (System/gc)
            (cond (> (metal/kernels-freed) before) true
                  (< tries 50) (do (Thread/sleep 20) (recur (inc tries)))
                  :else false))
          "a kernel freed within a second of collections"))))
