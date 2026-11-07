(ns gen.inference.options-test
  "The inference functions refuse an option key they do not read
   (gen.inference.options; genclj-vux3), where they used to ignore it and
   run on their defaults: the health check's probes ({:steps 50 :epsilon
   0.3} ran hmc at L 10 and eps 0.1, {:burnin 50} burned nothing,
   {:adapt-delta 0.95} kept 0.8) among them. Every entry point is called
   with a key it does not read, and its error names the function, that
   key and the keys it takes, which pins each function's options. Planted
   (2026-11-07): kernel/run's check removed, its row fails (run's own
   :samples error is thrown instead), and so does the shared map's."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.batched :as b]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.importance :as is]
            [gen.inference.kernel :as k]
            [gen.inference.mcmc :as mcmc]
            [gen.inference.mh :as mh]
            [gen.inference.nuts :as nuts]
            [gen.inference.options :as options]
            [gen.inference.particle-filter :as pf]
            [gen.inference.transform :as tf :refer [copy]]
            [gen.inference.vi :as vi]
            [gen.learning :as learn]
            [gen.random :as rng]
            [gen.selection :as sel]
            [gen.test-support :as ts]))

(def model (gen [] (let [x (trace :x (d/normal 0 1))] (trace :y (d/normal x 1)) x)))
(def observations (cm/choicemap :y 1.3))
(def x-sel (sel/select :x))
(def proposal (gen [tr] (trace :x (d/normal (tr :x) 0.5))))
(def family (gen [] (trace :x (d/normal (param :m) 1))))
(def store {:m 0.0})

(def start (delay (:trace (gfi/generate model [] observations (rng/key 1)))))
(def particles (delay (:particles (b/generate model [] observations 4 (rng/key 2)))))
(def state (delay (pf/initialize model [] observations 4 {:key (rng/key 3)})))

(def ^:private hmc-keys [:L :compile? :eps :inv-metric :jitter :key])
(def ^:private mala-keys [:compile? :inv-metric :key])
(def ^:private nuts-keys [:compile? :eps :inv-metric :jitter :key :max-depth])
(def ^:private map-optimize-keys [:max-step-size :min-step-size :tau])
(def ^:private adapt-keys [:L :adapt-metric? :compile? :eps :gamma :init-buffer :inv-metric :jitter :kappa :kernel :key
                           :max-depth :t0 :target-accept :term-buffer :warmup :window])
(def ^:private resample-keys [:key :method :n :priority])

(def entry-points
  "[f (fn [opts] a call of f with the options opts) the keys f takes]: every
   stable inference function that takes an options map."
  [[`mcmc/hmc #(mcmc/hmc @start x-sel % (rng/key 4)) hmc-keys]
   [`mcmc/mala #(mcmc/mala @start x-sel 0.1 %) mala-keys]
   [`mcmc/nuts #(mcmc/nuts @start x-sel % (rng/key 4)) nuts-keys]
   [`mcmc/map-optimize #(mcmc/map-optimize @start x-sel %) map-optimize-keys]
   [`mcmc/hmc-particles #(mcmc/hmc-particles @particles x-sel % (rng/key 4)) hmc-keys]
   [`mcmc/mala-particles #(mcmc/mala-particles @particles x-sel 0.1 %) mala-keys]
   [`mcmc/map-optimize-particles #(mcmc/map-optimize-particles @particles x-sel %) map-optimize-keys]
   [`k/kernel #(k/kernel (k/mh x-sel) nil %) [:particles :reverse-particles]]
   [`k/hmc #(k/hmc x-sel %) (remove #{:key} hmc-keys)]
   [`k/mala #(k/mala x-sel 0.1 %) (remove #{:key} mala-keys)]
   [`k/nuts #(k/nuts x-sel %) (remove #{:key} nuts-keys)]
   [`k/map-optimize #(k/map-optimize x-sel %) map-optimize-keys]
   [`k/run #(k/run (k/mh x-sel) @start %) [:burn :device :float :keep :key :samples :thin]]
   [`nuts/adapt #(nuts/adapt @start x-sel % (rng/key 4)) adapt-keys]
   [`nuts/sample #(nuts/sample @start x-sel % (rng/key 4)) (sort (conj adapt-keys :keep :samples))]
   [`mh/move #(mh/move @start proposal [] % (rng/key 4) (rng/key 5)) [:backward :backward-args :fwd-choices :involution]]
   [`tf/run #(tf/run (tf/transform [mi ai mo ao] (copy mi :x mo :x)) (cm/choicemap :x 1.0) nil %) [:aux-retval :retained]]
   [`is/importance-sampling #(is/importance-sampling model [] observations 4 %)
    [:batched? :device :float :key :proposal :proposal-args]]
   [`is/importance-resampling #(is/importance-resampling model [] observations 4 %) [:batched? :key :proposal :proposal-args]]
   [`is/estimate #(is/estimate (fn [t] (t :x)) model [] observations 4 %) [:batched? :key :proposal :proposal-args]]
   [`is/weighted-sample #(is/weighted-sample model [] observations % (rng/key 4)) [:proposal :proposal-args]]
   [`pf/initialize #(pf/initialize model [] observations 4 %) [:batched? :key :layout :proposal :proposal-args :strata]]
   [`pf/step #(pf/step @state [] [] (cm/choicemap) %) [:backward :backward-args :key :proposal :proposal-args]]
   [`pf/resample #(pf/resample @state %) resample-keys]
   [`pf/resize #(pf/resize @state 2 %) (remove #{:n} resample-keys)]
   [`pf/maybe-resample #(pf/maybe-resample @state %) (sort (conj resample-keys :ess-threshold))]
   [`pf/replicate #(pf/replicate @state 2 %) [:layout]]
   [`pf/dereplicate #(pf/dereplicate @state 2 %) [:key :layout :method]]
   [`pf/coalesce #(pf/coalesce @state %) [:by]]
   [`pf/introduce #(pf/introduce @state observations 2 %) [:args :key :model :proposal :proposal-args]]
   [`pf/reweighting #(pf/reweighting proposal [] %) [:backward :backward-args :involution]]
   [`vi/elbo-gradient #(vi/elbo-gradient model [] observations family [] store 2 %)
    [:baseline :batched? :estimator :key :model-store :traces?]]
   [`vi/vimco-gradient #(vi/vimco-gradient model [] observations family [] store 2 %)
    [:baseline :batched? :key :model-store :traces?]]
   [`vi/black-box-vi #(vi/black-box-vi model [] observations family [] store (learn/sgd {:lr 0.01}) %)
    [:baseline :batched? :callback :estimator :iters :key :model-optimizer :model-state :model-store :samples :state]]
   [`vi/black-box-vimco #(vi/black-box-vimco model [] observations family [] store (learn/sgd {:lr 0.01}) %)
    [:baseline :batched? :callback :estimates :iters :key :model-optimizer :model-state :model-store :samples :state]]])

(deftest every-entry-point-refuses-a-key-it-does-not-read
  (doseq [[f call accepted] entry-points]
    (is (= {:reason :unknown-options :fn f :unknown [:sample] :accepted (vec (sort accepted))}
           (ts/thrown-data (call {:sample 10})))
        (str f))))

(deftest the-health-checks-probes-are-loud
  (testing "hmc's {:steps 50 :epsilon 0.3} ran at L 10 and eps 0.1"
    (is (= [:steps :epsilon] (:unknown (ts/thrown-data (mcmc/hmc @start x-sel {:steps 50 :epsilon 0.3} (rng/key 4))))))
    (is (= [:steps] (:unknown (ts/thrown-data (k/hmc x-sel {:steps 50 :eps 0.3}))))
        "the kernel refuses it when it is made, before a chain runs"))
  (testing "run's {:burnin 50} burned nothing"
    (is (= [:burnin] (:unknown (ts/thrown-data (k/run (k/mh x-sel) @start {:samples 10 :burnin 50}))))))
  (testing "nuts' {:adapt-delta 0.95} kept 0.8"
    (is (= [:adapt-delta] (:unknown (ts/thrown-data (nuts/sample @start x-sel {:samples 10 :adapt-delta 0.95} (rng/key 4))))))))

(deftest a-shared-options-map-is-refused
  (let [shared {:samples 5 :burn 2 :warmup 10 :key (rng/key 6)}
        refused (ts/thrown-data (nuts/sample @start x-sel shared))]
    (is (= {:fn `nuts/sample :unknown [:burn]} (select-keys refused [:fn :unknown])))
    (is (= [:burn] (:unknown (ts/thrown-data (nuts/adapt @start x-sel (dissoc shared :samples))))) "adapt's too")
    (is (= [:warmup] (:unknown (ts/thrown-data (k/run (k/mh x-sel) @start shared)))))
    (testing "each function given its own keys, the :accepted the error names"
      (is (= 5 (count (:traces (nuts/sample @start x-sel (select-keys shared (:accepted refused)))))))
      (is (= 5 (count (:traces (k/run (k/mh x-sel) @start (dissoc shared :warmup)))))))))

(deftest the-check
  (testing "a map or nil, its keys among the accepted, passes as it is"
    (is (nil? (options/check `f nil #{:a})))
    (is (= {:a 1} (options/check `f {:a 1} #{:a :b})))
    (is (= {} (options/check `f {} #{}))))
  (testing "anything else is loud"
    (is (= {:reason :invalid-options :fn `f :value [:a 1]} (ts/thrown-data (options/check `f [:a 1] #{:a}))))
    (is (= :invalid-options (:reason (ts/thrown-data (options/check `f (rng/key 0) #{:a}))))))
  (testing "the unknown keys in the map's order, the accepted sorted, and a qualified key is no exception"
    (is (= {:reason :unknown-options :fn `f :unknown [:c ::d] :accepted [:a :b]}
           (ts/thrown-data (options/check `f (array-map :a 1 :c 2 ::d 3) #{:b :a}))))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"f: unknown options \[:c\]; the options are :a :b"
                          (options/check `f {:c 1} #{:a :b})))))
