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
            [gen.array.stage :as st]
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
    (testing "a program of anything but a key, then its data"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"a program of a key"
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

(defn- regression-data
  "The vector model's arguments and observations on n points."
  [n]
  (let [xs (mapv #(/ (double %) (max 1 (dec n)) 0.1) (range n))
        ys (mapv #(+ (* 2.0 %) -1.0 (Math/sin (* 7.0 %))) xs)]
    [[(array/array xs)] (cm/choicemap :ys (array/array ys))]))

(deftest data-of-any-size-are-inputs-and-the-code-does-not-grow
  (if-not (= :jvm (array/backend-name array/*backend*))
    (println "gen.device.jvm-test: not the JVM backend, the large-data comparison skips")
    (let [sizes (atom [])]
      (doseq [n [100 1000]]
        (testing n
          (let [[args obs] (regression-data n)
                model pb/vector-model
                {:keys [program inputs]} (binding [st/*float* :float64] (ch/weight-program model args obs))
                k (jvm/compile program)
                jin (mapv #(jvm/input-array % :float64) inputs)
                key (rng/key 5)
                {[w] :outputs} (jvm/run-many k 30 (rng/key-data key) {:inputs jin})]
            (swap! sizes conj (:statements k))
            (is (= (:log-weights (is/importance-sampling model args obs 30 {:key key})) (vec w)))
            (testing "a MALA chain, the data its inputs"
              (let [{cp :program cin :inputs} (binding [st/*float* :float64] (ch/chain-program model args obs pb/selection [:mala 1e-4] 4))
                    [vs acc] ((:run (jvm/compile cp)) (rng/key-data key) (mapv #(jvm/input-array % :float64) cin))
                    [k0 k1] (rng/split key 2)
                    tr (:trace (gfi/generate model args obs k0))
                    addrs (mapv first (cm/leaves (grad/continuous-selection tr pb/selection)))
                    read (fn [t] (mapv #(double (array/->clj (cm/get-choice (:choices t) %))) addrs))
                    {:keys [draws]} (kernel/run (kernel/mala pb/selection 1e-4) tr {:samples 4 :key k1 :keep read})]
                (is (= (vec (apply concat draws)) (vec vs)))
                (is (= (mapv (fn [t] (not= (nth draws t) (if (zero? t) (read tr) (nth draws (dec t))))) (range 4)) (vec acc))))))))
      (testing "the code's size does not depend on the data's"
        (is (apply = @sizes) @sizes)))))

(deftest large-constants-are-passed-not-written
  (if-not (= :jvm (array/backend-name array/*backend*))
    (println "gen.device.jvm-test: not the JVM backend, the large-constant comparison skips")
    (let [[args obs] (regression-data 500)
          k (jvm/compile (binding [st/*float* :float64] (ch/weight-run pb/vector-model args obs)))
          key (rng/key 6)
          {[w] :outputs} (jvm/run-many k 10 (rng/key-data key) {})]
      (is (= 2 (count (:constants k))))
      (is (= (:log-weights (is/importance-sampling pb/vector-model args obs 10 {:key key})) (vec w))))))

(def ^:private indexed-mean
  "A categorical draw indexing a table of means (a gather on a draw)."
  (gen.dynamic/gen []
    (let [k (trace :k (gen.dist/categorical [0.2 0.3 0.5]))]
      (trace :y (gen.dist/normal (array/take (array/array [-1.0 0.0 2.0]) k) 1.0)))))

(def ^:private two-component
  (gen.dynamic/gen [x]
    (trace :m ((gen.dist/homogeneous-mixture gen.dist/normal) [0.4 0.6] (array/stack [x (array/+ x 2.0)] -1) [1.0 0.5]))))

(deftest gathers-and-categoricals-are-the-handlers
  (if-not (= :jvm (array/backend-name array/*backend*))
    (println "gen.device.jvm-test: not the JVM backend, the gather comparison skips")
    (doseq [[label model args obs] [[:indexed-mean-free indexed-mean [] cm/EMPTY]
                                    [:indexed-mean-k indexed-mean [] (cm/choicemap :k 2)]
                                    [:indexed-mean-y indexed-mean [] (cm/choicemap :y 1.5)]
                                    [:mixture-free two-component [-0.8] cm/EMPTY]
                                    [:mixture-observed two-component [-0.8] (cm/choicemap :m 2.2)]]]
      (testing label
        (let [k (jvm/compile (binding [st/*float* :float64]
                               (st/stage (fn [key] (let [{:keys [trace weight]} (gfi/generate model (st/constants args) (st/constants obs) key)]
                                                     {:weight weight :score (:score trace)}))
                                         (st/key-spec))))]
          (doseq [s (range 20)]
            (let [key (rng/key s)
                  [w sc] ((:run k) (rng/key-data key))
                  {:keys [trace weight]} (gfi/generate model args obs key)]
              (is (= [weight (:score trace)] [(aget ^doubles w 0) (aget ^doubles sc 0)]) s))))))))

(deftest equality-on-a-staged-value-is-loud
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"clojure.core/= on a staged value"
                        (st/stage (fn [key] {:x (if (= (array/argmax (gen.random/uniform key [3])) 0) 1.0 2.0)}) (st/key-spec))))
  (testing "a keyword or nil is unequal, as on the JVM backend"
    (is (some? (st/stage (fn [key] (let [k (array/argmax (gen.random/uniform key [3]))]
                                     {:x (if (or (= k :missing) (= k nil)) 1.0 2.0)}))
                         (st/key-spec))))))

(def ^:private past-the-end
  "A categorical draw of 0, 1 or 2 indexing a table of two: index 2 is out of
   range, which the handler raises."
  (gen.dynamic/gen []
    (let [k (trace :k (gen.dist/categorical [0.2 0.3 0.5]))]
      (trace :y (gen.dist/normal (array/take (array/array [-1.0 0.0]) k) 1.0)))))

(deftest an-index-out-of-range-is-the-handlers-error
  (if-not (= :jvm (array/backend-name array/*backend*))
    (println "gen.device.jvm-test: not the JVM backend, the index-error test skips")
    (let [handler-error (fn [s] (try (gfi/simulate past-the-end [] (rng/key s)) nil (catch clojure.lang.ExceptionInfo e (ex-message e))))
          bad (first (filter handler-error (range 100)))
          good (first (remove handler-error (range 100)))
          k (jvm/compile (binding [st/*float* :float64]
                           (st/stage (fn [key] {:score (:score (gfi/simulate past-the-end [] key))}) (st/key-spec))))]
      (testing "a run whose draw is out of range raises the handler's own error"
        (is (some? bad))
        (is (thrown-with-msg? clojure.lang.ExceptionInfo (re-pattern (java.util.regex.Pattern/quote (handler-error bad)))
                              ((:run k) (rng/key-data (rng/key bad))))))
      (testing "an in-range run returns its score, the flag's output dropped"
        (let [outs ((:run k) (rng/key-data (rng/key good)))]
          (is (= 1 (count outs)))
          (is (= (:score (gfi/simulate past-the-end [] (rng/key good))) (aget ^doubles (first outs) 0)))))
      (testing "many runs: the first bad one's error"
        (is (thrown? clojure.lang.ExceptionInfo (jvm/run-many k 40 (rng/key-data (rng/key 3)) {})))))))
