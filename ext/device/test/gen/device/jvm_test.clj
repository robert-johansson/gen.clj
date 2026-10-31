(ns gen.device.jvm-test
  "gen.device.jvm: the generator's refusals; that a run is a function of
   its key; that every method stays under HotSpot's limit for compiling;
   and, where the default backend draws float32 (-Dgen.backend=mlx), that
   the generated chain is kernel/run's from the same key, to float32
   rounding, with the same accept decisions."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [gen.choicemap :as cm]
            [gen.device.chain :as ch]
            [gen.device.jvm :as jvm]
            [gen.device.stage :as st]
            [gen.gfi :as gfi]
            [gen.gradients :as grad]
            [gen.inference.importance :as is]
            [gen.inference.kernel :as kernel]
            [gen.parity-bench :as pb]
            [gen.random :as rng]))

(deftest the-generator-refuses-what-it-does-not-lower
  (let [[model args obs] (pb/models :vector)]
    (testing "a scan over xs: only counted scans"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"scan over xs"
                            (jvm/source (ch/mala-run model args obs pb/selection pb/tau 5) "P"))))
    (testing "a program of anything but one key"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"one key"
                            (jvm/source (st/stage (fn [x] (array/exp x)) [[2] :float32]) "P"))))))

(def ^:private kernels
  "Each chain: its staged program of S steps, and kernel/run's kernel."
  {:mala [#(ch/mala-chain %1 %2 %3 pb/selection pb/tau %4) (kernel/mala pb/selection pb/tau)]
   :hmc [#(ch/hmc-chain %1 %2 %3 pb/selection {:L 8 :eps pb/eps} %4) (kernel/hmc pb/selection {:L 8 :eps pb/eps})]})

(deftest a-run-is-a-function-of-its-key
  (let [[model args obs] (pb/models :vector)
        k (jvm/compile ((first (kernels :mala)) model args obs 20))
        run #(mapv vec ((:run k) (rng/key-data (rng/key %))))]
    (is (= (run 7) (run 7)))
    (is (not= (run 7) (run 8)))
    (testing "every method small enough for the JIT"
      (is (every? #(<= % 8000) (vals (:method-sizes k))) (:method-sizes k)))))

(deftest a-cut-into-smaller-methods-runs-the-same-program
  (let [[model args obs] (pb/models :static)
        p ((first (kernels :hmc)) model args obs 10)
        k (jvm/compile p)
        small (binding [] (System/setProperty "gen.jvm.chunk" "40") (try (jvm/compile p) (finally (System/clearProperty "gen.jvm.chunk"))))
        run (fn [k] (mapv vec ((:run k) (rng/key-data (rng/key 3)))))]
    (is (< (:chunk small) (:chunk k)))
    (is (< (:methods k) (:methods small)))
    (is (= (run k) (run small)))))

(deftest the-chain-is-kernel-runs
  (if-not (= :float32 (:float (array/defaults array/*backend*)))
    (println "gen.device.jvm-test: the backend draws float64, the comparison with kernel/run skips (-Dgen.backend=mlx runs it)")
    (doseq [m [:vector :static]
            [kname [program kern]] kernels]
      (testing [m kname]
        (let [[model args obs] (pb/models m)
              steps 50 d 2
              k (jvm/compile (program model args obs steps))
              proto (grad/continuous-selection (:trace (gfi/generate model args obs (rng/key 0))) pb/selection)
              addrs (mapv first (cm/leaves proto))]
          (doseq [s (range 3)]
            (let [key (rng/key (+ 100 s))
                  [vs acc] ((:run k) (rng/key-data key))
                  [k0 k1] (rng/split key 2)
                  tr (:trace (gfi/generate model args obs k0))
                  read (fn [t] (mapv #(double (array/->clj (cm/get-choice (:choices t) %))) addrs))
                  {:keys [draws]} (kernel/run kern tr {:samples steps :key k1 :keep read})]
              (doseq [t (range steps) j (range d)]
                (let [dev (aget ^floats vs (+ (* t d) j)) ref (nth (nth draws t) j)]
                  (is (< (Math/abs (- dev ref)) (* 1e-4 (+ 1.0 (Math/abs ref)))) [s t j dev ref])))
              (is (= (mapv (fn [t] (not= (nth draws t) (if (zero? t) (read tr) (nth draws (dec t))))) (range steps))
                     (vec acc))))))))))

(deftest run-i-of-many-is-the-run-on-split-key-i
  (let [[model args obs] (pb/models :vector)
        k (jvm/compile ((first (kernels :mala)) model args obs 10))
        root (rng/key 11) n 7
        {[vs acc] :outputs} (jvm/run-many k n (rng/key-data root) {:block 2})]
    (doseq [i (range n)]
      (let [[v a] ((:run k) (rng/key-data (nth (rng/split root n) i)))]
        (is (= (vec v) (vec (java.util.Arrays/copyOfRange ^floats vs (int (* i 20)) (int (* (inc i) 20))))) i)
        (is (= (vec a) (vec (java.util.Arrays/copyOfRange ^booleans acc (int (* i 10)) (int (* (inc i) 10))))) i)))
    (testing "an instance's accept is the same runs"
      (let [^java.util.function.LongConsumer inst ((:instance k) n)]
        (.accept inst (jvm/key-long (rng/key-data root)))
        (is (= (vec vs) (vec (first (.get ^java.util.function.Supplier inst)))))))))

(deftest many-runs-do-not-depend-on-the-threads
  (let [[model args obs] (pb/models :vector)
        k (jvm/compile (ch/weight-run model args obs))
        root (rng/key-data (rng/key 5))
        one (jvm/run-many k 1000 root {:threads 1 :lse 0})
        all (jvm/run-many k 1000 root {:lse 0})]
    (is (= (vec (first (:outputs one))) (vec (first (:outputs all)))))
    (is (== (:log-sum-exp one) (:log-sum-exp all)))))

(deftest importance-sampling-is-the-handlers
  (if-not (= :float32 (:float (array/defaults array/*backend*)))
    (println "gen.device.jvm-test: the backend draws float64, the comparison with importance sampling skips")
    (doseq [m [:vector :static]]
      (testing m
        (let [[model args obs] (pb/models m)
              k (jvm/compile (ch/weight-run model args obs))
              key (rng/key 21) n 200
              {[w] :outputs lse :log-sum-exp} (jvm/run-many k n (rng/key-data key) {:lse 0})
              ref (is/importance-sampling model args obs n {:key key})]
          (doseq [[i ref-w] (map-indexed vector (:log-weights ref))]
            (is (< (Math/abs (- (aget ^floats w i) ref-w)) (* 1e-4 (+ 1.0 (Math/abs ref-w)))) [i (aget ^floats w i) ref-w]))
          (is (< (Math/abs (- (- lse (Math/log n)) (:log-ml-estimate ref))) 1e-4)))))))

(deftest a-float64-program-is-the-jvm-backends-run-bit-for-bit
  (if-not (= :jvm (array/backend-name array/*backend*))
    (println "gen.device.jvm-test: not the JVM backend, the float64 comparison skips")
    (doseq [m [:vector :static]]
      (testing m
        (let [[model args obs] (pb/models m)
              proto (grad/continuous-selection (:trace (gfi/generate model args obs (rng/key 0))) pb/selection)
              addrs (mapv first (cm/leaves proto))
              key (rng/key 31) steps 30
              [k0 k1] (rng/split key 2)
              tr (:trace (gfi/generate model args obs k0))
              read (fn [t] (mapv #(double (array/->clj (cm/get-choice (:choices t) %))) addrs))]
          (doseq [[kname [program kern]] kernels]
            (testing kname
              (let [k (jvm/compile (binding [st/*float* :float64] (program model args obs steps)))
                    [vs acc] ((:run k) (rng/key-data key))
                    {:keys [draws]} (kernel/run kern tr {:samples steps :key k1 :keep read})]
                (is (= (vec (apply concat draws)) (vec vs)))
                (is (= (mapv (fn [t] (not= (nth draws t) (if (zero? t) (read tr) (nth draws (dec t))))) (range steps))
                       (vec acc))))))
          (testing "importance sampling"
            (let [k (jvm/compile (binding [st/*float* :float64] (ch/weight-run model args obs)))
                  n 100
                  {[w] :outputs lse :log-sum-exp} (jvm/run-many k n (rng/key-data key) {:lse 0})
                  ref (is/importance-sampling model args obs n {:key key})]
              (is (= (:log-weights ref) (vec w)))
              (is (== (:log-ml-estimate ref) (- lse (Math/log n)))))))))))
