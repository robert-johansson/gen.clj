(ns gen.device.run-test
  "kernel/run's and importance sampling's :device option (gen.inference.device)
   through the lowerings (gen.device): on the JVM backend in float64 the
   results are the handler's bit for bit, their traces whole (an update from
   one is the handler's); a model that does not stage falls back to the
   handler with the reason; a compiled program is reused."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [gen.array.protocols]
            [gen.batched :as b]
            [gen.choicemap :as cm]
            [gen.combinators :as comb]
            [gen.dist :as d]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.device :as device]
            [gen.inference.importance :as is]
            [gen.inference.kernel :as kernel]
            [gen.inference.mcmc :as mcmc]
            [gen.parity-bench :as pb]
            [gen.random :as rng]
            [gen.selection :as sel]
            [gen.test-support :as ts]))

(def ^:private jvm? (= :jvm (array/backend-name array/*backend*)))

(defn- leaves-of [t] (mapv (fn [[p v]] [p (array/->clj v)]) (cm/leaves (:choices t))))

(def ^:private dense-metric
  "A dense inverse metric for the parity model's slope and intercept: a
   staged chain takes its factors as constants."
  (mcmc/dense-inv-metric [:slope :intercept] [[0.004 -0.01] [-0.01 0.05]]))

(deftest kernel-run-on-device-is-the-handlers
  (if-not jvm?
    (println "gen.device.run-test: the comparison runs on the JVM backend")
    (doseq [m [:vector :static]
            [kname k] [[:mala (kernel/mala pb/selection pb/tau)] [:hmc (kernel/hmc pb/selection {:L 8 :eps pb/eps})]
                       [:hmc-dense (kernel/hmc pb/selection {:L 8 :eps pb/eps :inv-metric dense-metric})]
                       [:mala-dense (kernel/mala pb/selection pb/tau {:inv-metric dense-metric})]]]
      (testing [m kname]
        (let [[model args obs] (pb/models m)
              tr (:trace (gfi/generate model args obs (rng/key 0)))
              opts {:samples 40 :burn 7 :thin 3 :key (rng/key 3)}
              h (kernel/run k tr opts)
              dv (kernel/run k tr (assoc opts :device :jvm))]
          (is (= :jvm (:device dv)) (:device-fallback dv))
          (is (= (map leaves-of (:traces h)) (map leaves-of (:traces dv))))
          (is (= (map :score (:traces h)) (map :score (:traces dv))))
          (is (= (:acceptance-rate h) (:acceptance-rate dv)))
          (is (= (:divergences h) (:divergences dv)))
          (testing ":keep"
            (is (= (:draws (kernel/run k tr (assoc opts :keep #(% :slope))))
                   (:draws (kernel/run k tr (assoc opts :keep #(% :slope) :device :jvm))))))
          (testing "a returned trace is whole: an update from it is the handler's"
            (let [c (cm/choicemap :slope 1.5)
                  uh (gfi/update (last (:traces h)) c (rng/key 9))
                  ud (gfi/update (last (:traces dv)) c (rng/key 9))]
              (is (= (:weight uh) (:weight ud)))
              (is (= (:score (:trace uh)) (:score (:trace ud)))))))))))

(deftest importance-sampling-on-device-is-the-handlers
  (if-not jvm?
    (println "gen.device.run-test: the comparison runs on the JVM backend")
    (doseq [m [:vector :static]]
      (testing m
        (let [[model args obs] (pb/models m)
              h (is/importance-sampling model args obs 100 {:key (rng/key 5)})
              dv (is/importance-sampling model args obs 100 {:key (rng/key 5) :device :jvm})]
          (is (= :jvm (:device dv)) (:device-fallback dv))
          (is (= (:log-weights h) (:log-weights dv)))
          (is (= (:log-ml-estimate h) (:log-ml-estimate dv)))
          (is (= (:ess h) (:ess dv)))
          (is (= (map leaves-of (:traces h)) (map leaves-of (:traces dv))))
          (is (= (:score (:trace (gfi/update (first (:traces h)) (cm/choicemap :slope 0.5) (rng/key 1))))
                 (:score (:trace (gfi/update (first (:traces dv)) (cm/choicemap :slope 0.5) (rng/key 1)))))))))))

(defn- close?
  "jvm-test's band for a float32 chain against the handler's: within 1e-4
   of 1 + |ref|."
  [dev ref]
  (< (Math/abs (- (double dev) (double ref))) (* 1e-4 (+ 1.0 (Math/abs (double ref))))))

(defn- leaf-specs [t] (mapv (fn [[p v]] [p (array/dtype v) (array/shape v)]) (cm/leaves (:choices t))))

(defn- leaf-values [t] (mapv (fn [[_ v]] (array/->clj v)) (cm/leaves (:choices t))))

(deftest kernel-run-on-device-is-the-handlers-to-float32-rounding
  ;; where the backend draws float32 (MLX): the device result is built on
  ;; the host from the program's outputs, as arrays of the backend
  (if-not (= :float32 (:float (array/defaults array/*backend*)))
    (println "gen.device.run-test: the backend draws float64, the float32 comparison skips (-Dgen.backend=mlx runs it)")
    (doseq [m [:vector :static]
            [kname k] [[:mala (kernel/mala pb/selection pb/tau)] [:hmc (kernel/hmc pb/selection {:L 8 :eps pb/eps})]
                       [:hmc-dense (kernel/hmc pb/selection {:L 8 :eps pb/eps :inv-metric dense-metric})]]]
      (testing [m kname]
        (let [[model args obs] (pb/models m)
              tr (:trace (gfi/generate model args obs (rng/key 0)))
              opts {:samples 40 :burn 7 :thin 3 :key (rng/key 3)}
              h (kernel/run k tr opts)
              dv (kernel/run k tr (assoc opts :device :jvm))]
          (is (= :jvm (:device dv)) (:device-fallback dv))
          (testing "the same accept decisions"
            (is (= (:acceptance-rate h) (:acceptance-rate dv)))
            (is (= (:divergences h) (:divergences dv))))
          (testing "the traces hold the handler's kinds of arrays, its values to float32 rounding"
            (is (= (map leaf-specs (:traces h)) (map leaf-specs (:traces dv))))
            (is (every? true? (map close? (flatten (map leaf-values (:traces dv))) (flatten (map leaf-values (:traces h))))))
            (is (every? true? (map #(close? (array/->clj (:score %1)) (array/->clj (:score %2))) (:traces dv) (:traces h)))))
          (testing ":keep, read on the host"
            (let [kh (:draws (kernel/run k tr (assoc opts :keep #(% :slope))))
                  kd (:draws (kernel/run k tr (assoc opts :keep #(% :slope) :device :jvm)))]
              (is (every? number? kd))
              (is (every? true? (map close? kd kh)))))
          (testing "a returned trace is whole: an update from it is the handler's"
            (let [c (cm/choicemap :slope 1.5)]
              (is (close? (array/->clj (:weight (gfi/update (last (:traces dv)) c (rng/key 9))))
                          (array/->clj (:weight (gfi/update (last (:traces h)) c (rng/key 9)))))))))))))

(deftest importance-sampling-on-device-is-the-handlers-to-float32-rounding
  ;; where the backend draws float32 (MLX): weights and traces made on the
  ;; host from the program's outputs (:metal's read once from the GPU), the
  ;; observations the given arrays, every trace as the handler's
  (if-not (= :float32 (:float (array/defaults array/*backend*)))
    (println "gen.device.run-test: the backend draws float64, the float32 importance-sampling comparison skips")
    (doseq [m [:vector :static]
            device [:jvm :metal]]
      (testing [m device]
        (let [[model args obs] (pb/models m)
              h (is/importance-sampling model args obs 50 {:key (rng/key 5)})
              dv (is/importance-sampling model args obs 50 {:key (rng/key 5) :device device})]
          (if (and (= :metal device) (:device-fallback dv))
            (println "gen.device.run-test: :metal fell back (" (:reason (:device-fallback dv)) "), its comparison skips")
            (do (is (= device (:device dv)) (:device-fallback dv))
                (is (every? true? (map close? (:log-weights dv) (:log-weights h))))
                (is (close? (:log-ml-estimate dv) (:log-ml-estimate h)))
                (is (= (map leaf-specs (:traces h)) (map leaf-specs (:traces dv))))
                (is (every? true? (map close? (flatten (map leaf-values (:traces dv))) (flatten (map leaf-values (:traces h))))))
                (is (close? (array/->clj (:weight (gfi/update (first (:traces dv)) (cm/choicemap :slope 0.5) (rng/key 1))))
                            (array/->clj (:weight (gfi/update (first (:traces h)) (cm/choicemap :slope 0.5) (rng/key 1)))))))))))))

(def ^:private chain-kernels
  [[:mala (kernel/mala pb/selection pb/tau)]
   [:hmc (kernel/hmc pb/selection {:L 8 :eps pb/eps})]
   [:hmc-dense (kernel/hmc pb/selection {:L 8 :eps pb/eps :inv-metric dense-metric})]])

(defn- chain-starts
  "n traces of model from the prior on their own keys: the chains' starts."
  [[model args obs] n]
  (mapv #(:trace (gfi/generate model args obs (rng/key (+ 40 %)))) (range n)))

(defn- by-chain
  "kernel/run on the handler per chain, chain i on (rng/split key n)[i]."
  [k starts opts]
  (mapv #(kernel/run k %1 (assoc opts :key %2)) starts (rng/split (:key opts) (count starts))))

(deftest run-chains-are-kernel-runs-per-chain
  ;; every chain from its own start (a per-run input of the program),
  ;; equal to its kernel/run on the handler bit for bit on the JVM backend
  (if-not jvm?
    (println "gen.device.run-test: the run-chains comparison runs on the JVM backend")
    (doseq [m [:vector :static]
            [kname k] chain-kernels]
      (testing [m kname]
        (let [starts (chain-starts (pb/models m) 5)
              opts {:samples 20 :burn 3 :thin 2 :key (rng/key 7)}
              h (by-chain k starts opts)
              dv (device/run-chains k starts (assoc opts :device :jvm))]
          (is (= :jvm (:device dv)) (:device-fallback dv))
          (is (= (mapv #(map leaves-of (:traces %)) h) (mapv #(map leaves-of %) (:traces dv))))
          (is (= (mapv #(map :score (:traces %)) h) (mapv #(map :score %) (:traces dv))))
          (is (= (mapv :acceptance-rate h) (:acceptance-rates dv)))
          (is (= (mapv :divergences h) (:divergences dv)))
          (testing ":keep"
            (is (= (mapv :draws (by-chain k starts (assoc opts :keep #(% :slope))))
                   (:draws (device/run-chains k starts (assoc opts :keep #(% :slope) :device :jvm))))))
          (testing "without :device, kernel/run per chain"
            (is (= (mapv :acceptance-rate h) (:acceptance-rates (device/run-chains k starts opts))))))))))

(deftest a-device-run-counts-the-handlers-divergences
  ;; genclj-gqd3: an HMC or MALA proposal whose energy error passes 1000
  ;; diverged (gen.inference.mcmc's verdict), and a device chain counts
  ;; the :diverged? its kernel's step returns. At the leapfrog's limit
  ;; here (an eps near 0.117, twice the posterior's narrowest standard
  ;; deviation) some transitions' energy errors pass 1000, finite, which
  ;; no rule counted before. Planted bugs (2026-11-04): the rule gone fails
  ;; the counts' positivity; the device's :div output dropped
  ;; (gen.inference.device) fails the device's counts
  (if-not jvm?
    (println "gen.device.run-test: the divergence comparison runs on the JVM backend")
    (doseq [m [:vector :static]
            [kname k] [[:hmc (kernel/hmc pb/selection {:L 8 :eps 0.117})]
                       [:mala (kernel/mala pb/selection (/ (* 0.1172 0.1172) 2.0))]]]
      (testing [m kname]
        (let [[model args obs] (pb/models m)
              tr (:trace (gfi/generate model args obs (rng/key 0)))
              opts {:samples 40 :burn 7 :thin 3 :key (rng/key 3)}
              h (kernel/run k tr opts)
              dv (kernel/run k tr (assoc opts :device :jvm))]
          (is (= :jvm (:device dv)) (:device-fallback dv))
          (is (< 0 (:divergences h) 127) "some transitions diverge, not all")
          (is (= (:divergences h) (:divergences dv)))
          (is (= (:acceptance-rate h) (:acceptance-rate dv)))
          (is (= (map leaves-of (:traces h)) (map leaves-of (:traces dv))))
          (testing "run-chains"
            (let [starts (chain-starts (pb/models m) 5)
                  opts {:samples 20 :burn 3 :thin 2 :key (rng/key 7)}
                  h (by-chain k starts opts)
                  dv (device/run-chains k starts (assoc opts :device :jvm))]
              (is (= :jvm (:device dv)) (:device-fallback dv))
              (is (pos? (reduce + (map :divergences h))))
              (is (= (mapv :divergences h) (:divergences dv)))
              (is (= (mapv :acceptance-rate h) (:acceptance-rates dv))))))))))

(deftest run-chains-on-a-float32-backend
  (if-not (= :float32 (:float (array/defaults array/*backend*)))
    (println "gen.device.run-test: the backend draws float64, the float32 run-chains comparison skips")
    (doseq [m [:vector]
            [kname k] chain-kernels
            device [:jvm :metal]]
      (testing [m kname device]
        (let [starts (chain-starts (pb/models m) 6)
              opts {:samples 20 :burn 3 :thin 2 :key (rng/key 7)}
              h (by-chain k starts opts)
              dv (device/run-chains k starts (assoc opts :device device))]
          (is (= device (:device dv)) (:device-fallback dv))
          (is (= (mapv :acceptance-rate h) (:acceptance-rates dv)))
          (is (= (mapv #(map leaf-specs (:traces %)) h) (mapv #(map leaf-specs %) (:traces dv))))
          (is (every? true? (map close? (flatten (mapv #(map leaf-values (:traces %)) h))
                                 (flatten (mapv #(map leaf-values %) (:traces dv)))))))))))

(deftest run-chains-keep-choices-without-traces
  ;; :keep-choices reads the kept choices from the program's outputs, as
  ;; :keep {address #(% address)} reads them from traces: a latent choice
  ;; (an output), an observation (the start's, no step changes it)
  (let [[model args obs] (pb/models :vector)
        starts (chain-starts (pb/models :vector) 4)
        opts {:samples 15 :burn 2 :thin 2 :key (rng/key 9)}
        addrs [:slope :intercept :ys]
        expected (mapv :draws (by-chain (kernel/hmc pb/selection {:L 8 :eps pb/eps}) starts
                                        (assoc opts :keep (into {} (map (fn [a] [a #(% a)])) addrs))))]
    (doseq [device (if jvm? [:jvm] [:jvm :metal])]
      (testing device
        (let [r (device/run-chains (kernel/hmc pb/selection {:L 8 :eps pb/eps}) starts
                                   (assoc opts :keep-choices addrs :device device))]
          (is (= device (:device r)) (:device-fallback r))
          (if jvm?
            (is (= expected (:draws r)))
            (doseq [a addrs]
              (is (every? true? (map close? (flatten (mapv a (:draws r))) (flatten (mapv a expected)))) (str a))))))))
  (testing "loud: an address with no choice, or both :keep and :keep-choices"
    (let [starts (chain-starts (pb/models :vector) 2)
          k (kernel/mala pb/selection pb/tau)]
      (is (thrown? clojure.lang.ExceptionInfo (device/run-chains k starts {:samples 3 :key (rng/key 1) :keep-choices [:nope] :device :jvm})))
      (is (= :keep-and-keep-choices
             (:reason (ex-data (try (device/run-chains k starts {:samples 3 :key (rng/key 1) :keep identity :keep-choices [:slope]})
                                    (catch clojure.lang.ExceptionInfo e e)))))))))

(def ^:private wide
  "A latent of 100 values (past gen.device.scalar's 64: a large per-run
   input, each chain's own)."
  (gen [] (let [z (trace :z (d/normal (array/zeros [100]) 1.0))] (trace :y (d/normal (array/sum z) 3.0)))))

(def ^:private big-data
  "200 observations (a large input): shared by every chain when they are
   one object, each chain's own otherwise."
  (gen [xs] (let [slope (trace :slope (d/normal 0 1))] (trace :ys (d/normal (array/* slope xs) 1.0)))))

(def ^:private xs200 (array/array (mapv #(* 0.01 %) (range 200))))

(defn- compare-chains
  "run-chains on device against kernel/run per chain: bit for bit on the
   JVM backend (float64), else to float32 rounding with the same
   acceptance."
  [k starts opts device]
  (let [h (by-chain k starts opts)
        dv (device/run-chains k starts (assoc opts :device device))]
    (is (= device (:device dv)) (:device-fallback dv))
    (is (= (mapv :acceptance-rate h) (:acceptance-rates dv)))
    (if jvm?
      (is (= (mapv #(map leaves-of (:traces %)) h) (mapv #(map leaves-of %) (:traces dv))))
      (is (every? true? (map close? (flatten (mapv #(map leaf-values (:traces %)) h))
                             (flatten (mapv #(map leaf-values %) (:traces dv)))))))))

(deftest run-chains-with-large-inputs
  (let [opts {:samples 6 :burn 1 :thin 1 :key (rng/key 11)}
        devices (if jvm? [:jvm] [:jvm :metal])]
    (doseq [device devices]
      (testing [device "a large latent, each chain's own start (copied into the run's scratch)"]
        (compare-chains (kernel/mala (sel/select :z) 0.02)
                        (mapv #(:trace (gfi/generate wide [] (cm/choicemap :y 2.0) (rng/key (+ 60 %)))) (range 4))
                        opts device))
      (testing [device "large data, one object every chain shares"]
        (let [obs (cm/choicemap :ys (array/* 1.5 xs200))]
          (compare-chains (kernel/hmc (sel/select :slope) {:L 4 :eps 0.02})
                          (mapv #(:trace (gfi/generate big-data [xs200] obs (rng/key (+ 70 %)))) (range 4))
                          opts device)))
      (testing [device "large data, each chain its own copy (per run)"]
        (compare-chains (kernel/hmc (sel/select :slope) {:L 4 :eps 0.02})
                        (mapv #(:trace (gfi/generate big-data [xs200] (cm/choicemap :ys (array/* 1.5 xs200)) (rng/key (+ 80 %)))) (range 4))
                        opts device)))))

(deftest run-chains-on-mlx-refuse-millions-of-traces
  ;; on a lazy backend (MLX), a trace a kept step and chain past 50,000 is
  ;; a loud error naming :keep-choices, which reads values with no trace
  (if-not (gen.array.protocols/-lazy? array/*backend*)
    (println "gen.device.run-test: an eager backend makes any number of traces, the trace limit skips")
    (let [starts (chain-starts (pb/models :vector) 200)
          k (kernel/mala pb/selection pb/tau)
          opts {:samples 300 :key (rng/key 2) :device :jvm}]
      (is (= :too-many-traces (:reason (ts/thrown-data (device/run-chains k starts opts)))))
      (is (= :too-many-traces (:reason (ts/thrown-data (device/run-chains k starts (assoc opts :keep #(% :slope)))))))
      (is (= 200 (count (:draws (device/run-chains k starts (assoc opts :keep-choices [:slope])))))))))

(deftest run-chains-of-other-structures-fall-back
  (let [[model args obs] (pb/models :vector)
        [static-model sargs sobs] (pb/models :static)
        starts [(:trace (gfi/generate model args obs (rng/key 1))) (:trace (gfi/generate static-model sargs sobs (rng/key 2)))]
        k (kernel/mala pb/selection pb/tau)
        r (device/run-chains k starts {:samples 5 :key (rng/key 3) :device :jvm})]
    (is (nil? (:device r)))
    (is (re-find #"different generative functions" (:reason (:device-fallback r))))
    (is (= :chain-by-chain (:ran (:device-fallback r))) "two models: no one batched run")
    (is (= 2 (count (:acceptance-rates r))))))

(def ^:private two-observed
  "Two observed arrays of one shape, other values: one array each, never
   one input (aliasing holds only for one array a trace holds twice)."
  (gen [xs] (let [slope (trace :slope (d/normal 0 1))]
              (trace :ya (d/normal (array/* slope xs) 1.0))
              (trace :yb (d/normal (array/* (array/negate slope) xs) 1.0)))))

(deftest leaves-of-one-shape-are-not-aliased
  (let [xs (array/array [0.0 1.0 2.0])
        obs (cm/choicemap :ya (array/array [0.1 1.9 4.2]) :yb (array/array [-0.2 -2.1 -3.8]))
        k (kernel/mala (sel/select :slope) 0.05)
        starts (mapv #(:trace (gfi/generate two-observed [xs] obs (rng/key (+ 50 %)))) (range 3))
        opts {:samples 8 :key (rng/key 6)}]
    (doseq [device (if jvm? [:jvm] [:jvm :metal])]
      (testing device
        (compare-chains k starts opts device)
        (when jvm?
          (let [h (kernel/run k (first starts) (assoc opts :key (rng/key 6)))
                dv (kernel/run k (first starts) (assoc opts :key (rng/key 6) :device device))]
            (is (= (map leaves-of (:traces h)) (map leaves-of (:traces dv))))))))))

(def ^:private unstaged
  "z ~ mvnormal: its density's Cholesky is not staged (the staging backend
   has no :cholesky), so the chains fall back; it runs batched."
  (gen [] (let [z (trace :z (d/mvnormal [0.0 0.0] [[1.0 0.5] [0.5 1.0]]))]
            (trace :y (d/normal (array/sum z) 1.0)))))

(deftest run-chains-fall-back-to-one-batched-run
  ;; a model the device cannot run: the kernel's batched form over every
  ;; chain at once (owner, 2026-11-04), its draws the batched run's: equal
  ;; to stepping the batched form by hand on (rng/split key steps)
  (let [k (kernel/mala (sel/select :z) 0.3)
        starts (mapv #(:trace (gfi/generate unstaged [] (cm/choicemap :y 1.0) (rng/key (+ 30 %)))) (range 5))
        opts {:samples 6 :burn 1 :thin 2 :key (rng/key 4)}
        r (device/run-chains k starts (assoc opts :device :jvm :keep-choices [:z]))
        bf (kernel/batched-form k)
        steps 13
        by-hand (reduce (fn [[ps kept] [i key]]
                          (let [ps' (:particles (bf ps key))]
                            [ps' (if (and (> (inc i) 1) (zero? (mod (- (inc i) 1) 2))) (conj kept (array/->clj (b/choice ps' :z))) kept)]))
                        [(:particles (b/generate unstaged [] cm/EMPTY 5 (rng/key 0) {:per-particle (mapv :choices starts)})) []]
                        (map vector (range steps) (rng/split (rng/key 4) steps)))]
    (is (nil? (:device r)))
    (is (= :batched (:ran (:device-fallback r))) (:device-fallback r))
    (is (re-find #"cholesky" (:reason (:device-fallback r))))
    (is (= (mapv (fn [c] {:z (mapv #(nth % c) (second by-hand))}) (range 5)) (:draws r)))
    (is (every? #(<= 0.0 % 1.0) (:acceptance-rates r)))))

(def ^:private point (gen ^:static [x slope] (let [y (trace :y (d/normal (array/* slope x) 1))] y)))

(def ^:private linreg
  ;; n, a host count of the arguments, is a node value the static language
  ;; reads on update: a staged chain keeps it a constant
  (gen ^:static [xs]
    (let [slope (trace :slope (d/normal 0 2))
          n (count xs)
          ys (splice :ys (comb/map point) xs (vec (repeat n slope)))]
      ys)))

(deftest a-host-count-in-the-trace-stays-a-constant
  (when jvm?
    (let [xs [1.0 2.0 3.0]
          obs (reduce (fn [m [i y]] (cm/set-choice m [:ys i :y] y)) cm/EMPTY (map-indexed vector [2.1 3.9 6.2]))
          tr (:trace (gfi/generate linreg [xs] obs (rng/key 0)))
          k (kernel/mala (sel/select :slope) 0.05)
          opts {:samples 8 :key (rng/key 3)}
          h (kernel/run k tr opts)
          dv (kernel/run k tr (assoc opts :device :jvm))
          ih (is/importance-sampling linreg [xs] obs 20 {:key (rng/key 5)})
          iv (is/importance-sampling linreg [xs] obs 20 {:key (rng/key 5) :device :jvm})]
      (is (= :jvm (:device dv)) (:device-fallback dv))
      (is (= (map leaves-of (:traces h)) (map leaves-of (:traces dv))))
      (is (= (map :score (:traces h)) (map :score (:traces dv))))
      (is (= :jvm (:device iv)) (:device-fallback iv))
      (is (= (:log-weights ih) (:log-weights iv))))))

(def ^:private branching
  (gen []
    (let [b (trace :b (d/bernoulli 0.4))]
      (if b (trace :x (d/normal 0 1)) (trace :x (d/normal 3 1))))))

(def ^:private branching-on-x
  (gen []
    (let [x (trace :x (d/normal 0 1))]
      (if (array/truthy? (array/> x 0)) (trace :y (d/normal 1 1)) (trace :y (d/normal -1 1))))))

(deftest a-model-that-does-not-stage-falls-back-to-the-handler
  (let [tr (:trace (gfi/generate branching-on-x [] (cm/choicemap :y 0.5) (rng/key 1)))
        k (kernel/mala (sel/select :x) 0.1)
        opts {:samples 6 :key (rng/key 2)}
        r (kernel/run k tr (assoc opts :device :jvm))]
    (is (= :jvm (:device (:device-fallback r))))
    (is (re-find #"staged value" (:reason (:device-fallback r))))
    (is (nil? (:device r)))
    (is (= (map leaves-of (:traces (kernel/run k tr opts))) (map leaves-of (:traces r))))))

(deftest a-callee-made-anew-each-run-falls-back-with-the-reason
  (let [model (gen []
                (let [x (trace :x (d/normal 0 1))
                      inner (gen [m] (trace :y (d/normal m 1)))]
                  (splice :sub inner x)))
        tr (:trace (gfi/generate model [] (cm/choicemap [:sub :y] 0.5) (rng/key 1)))
        r (kernel/run (kernel/mala (sel/select :x) 0.1) tr {:samples 4 :key (rng/key 2) :device :jvm})]
    (is (re-find #"anew each run" (:reason (:device-fallback r))))))

(deftest a-branch-on-a-boolean-the-chain-does-not-move-runs-on-the-device
  (when jvm?
    (let [tr (:trace (gfi/generate branching [] (cm/choicemap :b true) (rng/key 1)))
          k (kernel/mala (sel/select :x) 0.1)
          opts {:samples 6 :key (rng/key 2)}
          r (kernel/run k tr (assoc opts :device :jvm))]
      (is (= :jvm (:device r)) (:device-fallback r))
      (is (= (map leaves-of (:traces (kernel/run k tr opts))) (map leaves-of (:traces r)))))))

(deftest a-compiled-program-is-reused
  (when jvm?
    (device/clear-cache!)
    (let [[model args obs] (pb/models :vector)
          tr (:trace (gfi/generate model args obs (rng/key 0)))
          programs #(count @@#'gen.inference.device/cache)
          run #(kernel/run %1 %2 {:samples %3 :key (rng/key 1) :device :jvm})]
      (run (kernel/mala pb/selection pb/tau) tr 5)
      (is (= 1 (programs)))
      (run (kernel/mala pb/selection pb/tau) tr 5)
      (is (= 1 (programs)) "a kernel rebuilt with the same parameters: one program")
      (run (kernel/mala pb/selection pb/tau) (:trace (gfi/generate model (into [] args) obs (rng/key 4))) 5)
      (is (= 1 (programs)) "a new argument vector around the same arrays: one program")
      (run (kernel/mala pb/selection (* 2 pb/tau)) tr 5)
      (is (= 2 (programs)) "another step size, another program")
      (run (kernel/mala pb/selection pb/tau) tr 6)
      (is (= 3 (programs)) "another length, another program")
      (is (= :jvm (:device (run (kernel/hmc pb/selection {:L 8 :eps pb/eps}) tr 5))))
      (run (kernel/hmc pb/selection {:L 8 :eps pb/eps}) tr 5)
      (is (= 4 (programs)) "an HMC kernel rebuilt alike: one program"))))
