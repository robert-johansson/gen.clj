(ns gen.device.cuda-test
  "gen.device.cuda, parked (DESIGN §9, 1.2.1's owner calls, genclj-z1gj):
   the generator's text and refusals everywhere. The tests that need a
   CUDA device skip, each by name, unless -Dgen.cuda.parked=false with
   NVRTC (-Dgen.cuda.nvrtc=<libnvrtc.so>) runs them: that a thread is the
   per-chain program on its own key, whatever the launch's size; and, where
   the default backend draws float32 (-Dgen.backend=mlx), that thread i's
   chain is kernel/run's from (rng/split key n)[i], to float32 rounding,
   with the same accept decisions."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [gen.choicemap :as cm]
            [gen.device.chain :as ch]
            [gen.device.cuda :as cuda]
            [gen.gfi :as gfi]
            [gen.gradients :as grad]
            [gen.inference.kernel :as kernel]
            [gen.parity-models :as pm]
            [gen.random :as rng]))

(def ^:private parked?
  "Is the CUDA lowering parked here? Unless -Dgen.cuda.parked=false."
  (not= "false" (System/getProperty "gen.cuda.parked")))

(def ^:private gpu? (some? (System/getProperty "gen.cuda.nvrtc")))

(defn- skipped?
  "Does the test test-name, which needs a CUDA device, skip here? Saying
   so, by name, and why."
  [test-name]
  (when-let [why (cond parked? "the CUDA lowering is parked (DESIGN §9, genclj-z1gj; -Dgen.cuda.parked=false runs it)"
                       (not gpu?) "no -Dgen.cuda.nvrtc")]
    (println (str "gen.device.cuda-test/" test-name ": skipped, " why))
    true))

(deftest the-generator-refuses-what-it-does-not-lower
  (let [[model args obs] (pm/models :vector)]
    (testing "a scan over xs: only counted scans run in a thread"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"scan over xs"
                            (cuda/source (ch/mala-run model args obs pm/selection pm/tau 5)))))
    (testing "a program of anything but a key, then its data"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"a program of a key"
                            (cuda/source (gen.array.stage/stage (fn [x] (array/exp x)) [[2] :float32])))))))

(deftest a-negated-literal-is-c
  ;; genclj-z1gj: the negation of a literal (which the generator writes in
  ;; place, unbound) was (--1.5f), a decrement of a constant, which NVRTC
  ;; refuses; as the JVM and Metal dialects write it, (- -1.5f)
  (let [{:keys [source]} (cuda/source (gen.array.stage/stage (fn [key] {:x (array/+ (rng/uniform key) (array/negate (array/array -1.5)))})
                                                             (gen.array.stage/key-spec)))]
    (is (= "(- -1.5f)" (re-find #"\(- ?-1\.5f\)" source)))
    (is (nil? (re-find #"\(--[^)]*\)" source)))))

(defn- launch
  "The outputs [values accepted] of n threads of k on root, as host vectors."
  [k n root]
  (let [bufs (cuda/buffers k n)]
    (try
      (cuda/launch! k bufs n (rng/key-data root))
      [(vec (cuda/read-floats (first bufs))) (mapv #(== 1 %) (cuda/read-bytes (second bufs)))]
      (finally (cuda/free-buffers bufs)))))

(deftest a-thread-is-its-own-chain
  (when-not (skipped? 'a-thread-is-its-own-chain)
    (let [[model args obs] (pm/models :vector)
          steps 20 d 2
          k (cuda/compile (cuda/source (ch/mala-chain model args obs pm/selection pm/tau steps)))
          root (rng/key 7)]
      (try
        (let [[v3 a3] (launch k 3 root)
              [v5 a5] (launch k 5 root)
              [v3' a3'] (launch k 3 root)]
          (testing "the same key, the same draws"
            (is (= v3 v3')) (is (= a3 a3')))
          (testing "thread i's chain does not depend on the launch's size"
            (is (= v3 (subvec v5 0 (* 3 steps d))))
            (is (= a3 (subvec a5 0 (* 3 steps))))))
        (when (= :float32 (:float (array/defaults array/*backend*)))
          (testing "thread i is kernel/run's chain from (split key n)[i]"
            (let [n 3
                  [vs acc] (launch k n root)
                  proto (grad/continuous-selection (:trace (gfi/generate model args obs (rng/key 0))) pm/selection)
                  addrs (mapv first (cm/leaves proto))]
              (doseq [i (range n)]
                (let [[k0 k1] (rng/split (nth (rng/split root n) i) 2)
                      tr (:trace (gfi/generate model args obs k0))
                      read (fn [t] (mapv #(double (array/->clj (cm/get-choice (:choices t) %))) addrs))
                      {:keys [draws]} (kernel/run (kernel/mala pm/selection pm/tau) tr {:samples steps :key k1 :keep read})]
                  (doseq [t (range steps) j (range d)]
                    (let [dev (nth vs (+ (* i steps d) (* t d) j)) ref (nth (nth draws t) j)]
                      (is (< (Math/abs (- dev ref)) (* 1e-4 (+ 1.0 (Math/abs ref)))) [i t j dev ref])))
                  (is (= (mapv (fn [t] (not= (nth draws t) (if (zero? t) (read tr) (nth draws (dec t))))) (range steps))
                         (subvec acc (* i steps) (* (inc i) steps)))))))))
        (finally (cuda/unload! k))))))

(deftest the-log-ml-reduced-on-the-device-is-the-hosts
  (when-not (skipped? 'the-log-ml-reduced-on-the-device-is-the-hosts)
    (let [[model args obs] (pm/models :vector)
          k (cuda/compile (cuda/source (ch/weight-run model args obs)))]
      (try
        (doseq [n [1 7 1000 1025 100000]]
          (let [bufs (cuda/buffers k n)]
            (try
              (cuda/launch! k bufs n (rng/key-data (rng/key n)))
              (let [^floats w (cuda/read-floats (first bufs))
                    m (reduce max (map double w))
                    host (- (+ m (Math/log (reduce + (map #(Math/exp (- (double %) m)) w)))) (Math/log n))
                    dev (cuda/log-mean-exp! k bufs n 0)]
                (is (< (Math/abs (- dev host)) (* 1e-6 (+ 1.0 (Math/abs host)))) [n dev host])
                (testing "the same weights, the same result"
                  (is (== dev (cuda/log-mean-exp! k bufs n 0)))))
              (finally (cuda/free-buffers bufs)))))
        (finally (cuda/unload! k))))))
