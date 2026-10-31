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
