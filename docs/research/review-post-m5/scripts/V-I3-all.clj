;; Verifier's independent checks of idiom-inference.md (I3). JVM backend.
;; Run from the snapshot: nice -n 10 clojure -J-Xmx1g -M:test:commons-math -i <this>
(require '[gen.array :as array] '[gen.dist :as d] '[gen.dist.core :as dc :refer [defdist]]
         '[gen.dist.dsl :refer [dist]] '[gen.dist.host :as host] '[gen.dist.commons-math :as cmath]
         '[gen.dynamic :refer [gen]] '[gen.gfi :as gfi] '[gen.random :as rng] '[gen.choicemap :as cm]
         '[gen.selection :as sel] '[gen.combinators :as comb] '[gen.diff :as diff]
         '[gen.inference.particle-filter :as pf] '[gen.inference.importance :as is]
         '[gen.inference.mh :as mh] '[gen.inference.kernel :as k] '[gen.inference.mcmc :as mcmc]
         '[gen.inference.vi :as vi] '[gen.learning :as learn])
(import '[org.apache.commons.math3.distribution PoissonDistribution])

(defmacro show [label & body]
  `(println ~label "=>" (try (pr-str (do ~@body))
                             (catch Throwable e# (str (.getSimpleName (class e#)) ": " (ex-message e#)
                                                      (when (ex-data e#) (str " " (pr-str (ex-data e#)))))))))
(defn time-ns [n f]
  (dotimes [_ n] (f))
  (let [t0 (System/nanoTime)] (dotimes [_ n] (f)) (/ (- (System/nanoTime) t0) (double n))))
(defn section [s] (println) (println "=====" s))

;; ---------------------------------------------------------------------------
(section "I3-02 (a): exceptions per log-prob of a discrete pushforward without a finite support")
(dist shifted-poisson [lam] (array/+ (d/poisson lam) 1))
(dist shifted-geometric [p] (array/* 2 (d/geometric p)))
(let [thrown (atom 0)
      orig dc/support]
  (with-redefs [dc/support (fn [d] (try (orig d) (catch clojure.lang.ExceptionInfo e (swap! thrown inc) (throw e))))]
    (dotimes [_ 10] (dc/log-prob (shifted-poisson 3.0) 4))
    (dotimes [_ 10] (dc/log-prob (shifted-geometric 0.3) 4)))
  (println "ex-infos thrown by 20 log-probs (10 shifted-poisson, 10 shifted-geometric):" @thrown))
(show "shifted-poisson log-prob at 4 (log P(X=3), lam 3; expect -1.4959)" (dc/log-prob (shifted-poisson 3.0) 4))
(show "shifted-geometric log-prob at 4 (log P(G=2), p .3 = log(.3·.49) = -1.9173)" (dc/log-prob (shifted-geometric 0.3) 4))
(let [p (d/poisson 3.0) sp (shifted-poisson 3.0)]
  (dotimes [r 3]
    (println "round" r "poisson log-prob ns" (Math/round (time-ns 200000 #(dc/log-prob p 3)))
             " shifted-poisson log-prob ns" (Math/round (time-ns 200000 #(dc/log-prob sp 4))))))

(section "I3-02 (b): a host family in a dist body, as the public API allows it")
(def host-poisson (cmath/integer "host-poisson" (fn [lam] (PoissonDistribution. nil (double lam) 1e-12 10000000))))
(show "(dist shifted-host [lam] (array/+ (host-poisson lam) 1)), unmarked var"
      (eval '(gen.dist.dsl/dist shifted-host [lam] (gen.array/+ (user/host-poisson lam) 1))))
(show "is host-poisson's var marked as a constructor?" (boolean (::dc/constructor (meta #'host-poisson))))
(show "host-poisson's own support (no finite support)" (dc/support (host-poisson 3.0)))
(def ^{::dc/constructor true} marked-host-poisson host-poisson)
(dist shifted-marked [lam] (array/+ (marked-host-poisson lam) 1))
(show "only with the private marker planted: log-prob at 4" (dc/log-prob (shifted-marked 3.0) 4))

;; ---------------------------------------------------------------------------
(section "I3-03: labeled categorical log-prob vs the categorical's, and a rebuilt draw")
(dist labeled-cat [labels probs] (nth labels (d/categorical probs)))
(dist shifted-exp [rate shift] (array/+ (d/exponential rate) shift))
(doseq [kk [10 100 1000]]
  (let [labels (mapv #(keyword (str "l" %)) (range kk))
        probs (vec (repeat kk (/ 1.0 kk)))
        c (d/categorical probs) lc (labeled-cat labels probs)
        calls (atom 0) orig dc/log-prob]
    (with-redefs [dc/log-prob (fn [dd v] (swap! calls inc) (orig dd v))]
      (dc/log-prob lc :l3))
    (println "K" kk "categorical us" (format "%.2f" (/ (time-ns 2000 #(dc/log-prob c 3)) 1000.0))
             " labeled-cat us" (format "%.1f" (/ (time-ns (max 20 (quot 20000 kk)) #(dc/log-prob lc :l3)) 1000.0))
             " dc/log-prob calls in one labeled log-prob:" @calls)))
(show "labeled-cat log-prob of :l3 at K 10 (expect log 0.1 = -2.3026)"
      (dc/log-prob (labeled-cat (mapv #(keyword (str "l" %)) (range 10)) (vec (repeat 10 0.1))) :l3))
(let [e (d/exponential 2.0) se (shifted-exp 2.0 1.0)]
  (dotimes [r 3]
    (println "round" r "exponential log-prob ns" (Math/round (time-ns 300000 #(dc/log-prob e 1.5)))
             " shifted-exp ns" (Math/round (time-ns 300000 #(dc/log-prob se 2.5)))
             " (d/exponential 2.0) construction ns" (Math/round (time-ns 300000 #(d/exponential 2.0))))))
;; the proposed one-call scoring of the support, on a draw with an array parameter
(let [b (d/bernoulli (array/array [0.1 0.2 0.3]))]
  (show "bernoulli(p of shape [3]) support" (dc/support b))
  (show "one call over the support, (dc/log-prob b (array/array support))" (array/->clj (dc/log-prob b (array/array (dc/support b)))))
  (show "per-value scoring, stacked (what dist does now)" (array/->clj (array/stack (mapv #(dc/log-prob b %) (dc/support b))))))

;; ---------------------------------------------------------------------------
(section "I3-04: array/eval! calls in the scalar particle-filter paths (50 particles)")
(let [kernel (gen [t s] (let [x (trace :x (d/normal s 1.0))] x))
      model (gen [n] (splice :steps (comb/unfold kernel) n 0.0))
      obs (fn [t] (cm/from-map {:steps {t {:x 0.5}}}))
      st (pf/initialize model [1] (obs 0) 50 {:key (rng/key 1)})
      st2 (pf/step st [2] [diff/unknown-change] (obs 1) {:key (rng/key 2)})
      n (atom 0)
      orig array/eval!
      counting (fn [f] (reset! n 0) (with-redefs [array/eval! (fn [x] (swap! n inc) (orig x))] (f)) @n)]
  (println "pf/step:" (counting #(pf/step st [2] [diff/unknown-change] (obs 1) {:key (rng/key 2)})))
  (println "pf/rejuvenate:" (counting #(pf/rejuvenate st2 (k/mh (sel/select [:steps 1 :x])) (rng/key 3))))
  (println "pf/move-reweight:" (counting #(pf/move-reweight st2 (pf/reweighting (sel/select [:steps 1 :x])) (rng/key 4))))
  (println "importance-sampling (for comparison):" (counting #(is/importance-sampling model [1] (obs 0) 50 {:key (rng/key 5)})))
  (println "pf/initialize:" (counting #(pf/initialize model [1] (obs 0) 50 {:key (rng/key 1)}))))

;; ---------------------------------------------------------------------------
(section "I3-05: value-shape check cost, alternating rounds")
(let [nd (d/normal 0.0 1.0)]
  (dotimes [r 3]
    (println "round" r "check-value-shape ns" (Math/round (time-ns 1000000 #(dc/check-value-shape nd 0.5)))
             " value-shape ns" (Math/round (time-ns 1000000 #(dc/value-shape nd)))
             " log-prob ns" (Math/round (time-ns 1000000 #(dc/log-prob nd 0.5))))))
(def model200 (gen [] (let [mu (trace :mu (d/normal 0.0 1.0))]
                        (mapv (fn [i] (trace [:y i] (d/normal mu 1.0))) (range 200)))))
(let [obs (cm/from-map {:y (zipmap (range 200) (repeat 0.3))})
      tr (:trace (gfi/generate model200 [] obs (rng/key 0)))
      s (sel/select :mu) key (rng/key 1)]
  (dotimes [r 4]
    (let [with (/ (time-ns 4000 #(gfi/regenerate tr s key)) 1000.0)
          without (with-redefs [dc/value-shape (constantly nil)] (/ (time-ns 4000 #(gfi/regenerate tr s key)) 1000.0))]
      (println "round" r "regenerate :mu (200 kept sites) us: with check" (format "%.1f" with)
               " without" (format "%.1f" without) " share" (format "%.1f%%" (* 100 (/ (- with without) with)))))))

;; ---------------------------------------------------------------------------
(section "I3-06: defdist and unknown clauses")
(defdist v-exp "misspelled log-prob" [rate] (sample [key shape] (array/exp 1.0)) (logprob [v] (array/- v)))
(show "construct v-exp (clause logprob)" (v-exp 1.0))
(show "its log-prob" (dc/log-prob (v-exp 1.0) 0.5))
(defdist v-pos "misspelled validate" [rate]
  (valdate (dc/check-positive "v-pos" "rate" rate))
  (sample [key shape] (rng/exponential key shape))
  (log-prob [v] (array/- v)))
(show "construct v-pos with rate -1 (validation silently skipped?)" (v-pos -1.0))
(defdist v-disc "misspelled discrete?" [p]
  (sample [key shape] (rng/bernoulli key p shape))
  (log-prob [v] 0.0)
  (discret? [] true))
(show "v-disc discrete? (declared true under a misspelled clause)" (dc/discrete? (v-disc 0.5)))

;; ---------------------------------------------------------------------------
(section "I3-07: every gen.dist constructor declares a value shape")
(let [ctors (for [[s v] (ns-publics 'gen.dist) :when (::dc/constructor (meta v)) :when (not= s 'broadcasted-normal)] s)
      default (get-method dc/-value-shape :default)
      missing (remove #(not= default (get-method dc/-value-shape (keyword "gen.dist" (name %)))) ctors)
      discrete (filter #(not= (get-method dc/-discrete? :default) (get-method dc/-discrete? (keyword "gen.dist" (name %)))) ctors)]
  (println "constructors:" (count ctors) " without a value-shape method:" (vec missing))
  (println "with a discrete? method:" (count discrete) (vec (sort discrete))))

;; ---------------------------------------------------------------------------
(section "I3-10: compiled-step cache")
(def cache @#'gen.inference.mcmc/compiled-steps)
(def model-n (gen [n] (let [mu (trace :mu (d/normal 0.0 1.0))]
                        (mapv (fn [i] (trace [:y i] (d/normal mu 1.0))) (range n)))))
(reset! cache {})
(doseq [n (range 1 18)]
  (let [tr (:trace (gfi/generate model-n [n] (cm/from-map {:y {0 0.1}}) (rng/key n)))]
    (mcmc/hmc tr (sel/select :mu) {:L 2 :eps 0.1 :compile? true} (rng/key 0))))
(println "entries after 17 distinct keys:" (count @cache) "(16 then wholesale eviction: expect 1)")

;; ---------------------------------------------------------------------------
(section "I3-12: piecewise-uniform with leading axes; the mixture's element shape")
(let [pu (d/piecewise-uniform (array/array [[0.0 1.0 2.0] [10.0 11.0 12.0]]) (array/array [[0.5 0.5] [0.5 0.5]]))]
  (show "value-shape" (dc/value-shape pu))
  (show "sample" (array/->clj (dc/sample pu (rng/key 3))))
  (show "sample, key 4" (array/->clj (dc/sample pu (rng/key 4))))
  (show "log-prob [0.5 10.5]" (array/->clj (dc/log-prob pu (array/array [0.5 10.5])))))
(let [nrm (d/normal (array/zeros [3]) 1.0)
      mvn (d/mvnormal [0.0 0.0] [[1.0 0.0] [0.0 1.0]])]
  (doseq [[label c] [["normal(mu [3])" nrm] ["mvnormal(d 2)" mvn]]]
    (let [x (dc/sample c (rng/key 0))]
      (println label " value-shape" (dc/value-shape c) " log-density shape (the mixture's elements)"
               (array/shape (dc/log-prob c x))))))

;; ---------------------------------------------------------------------------
(section "I3-13: nil parameters")
(show "(d/normal 0.0 nil)" (d/normal 0.0 nil))
(show "  sampled" (dc/sample (d/normal 0.0 nil) (rng/key 0)))
(show "(d/bernoulli nil)" (d/bernoulli nil))
(show "(d/categorical nil)" (d/categorical nil))

;; ---------------------------------------------------------------------------
(section "I3-14: case on user options")
(show "learn/init :rmsprop" (learn/init {:kind :rmsprop} {:w 1.0}))
(show "learn/step :rmsprop" (learn/step {:kind :rmsprop} nil {:w 1.0} {:w 1.0}))
(let [model (gen [] (trace :z (d/normal 0.0 1.0)))
      q (gen [] (trace :z (d/normal 0.0 1.0)))]
  (show "vi/elbo-gradient :baseline :loo" (vi/elbo-gradient model [] cm/EMPTY q [] {} 3 {:baseline :loo :key (rng/key 0)})))
(let [kernel (gen [t s] (trace :x (d/normal s 1.0)))
      model (gen [n] (splice :steps (comb/unfold kernel) n 0.0))
      st (pf/initialize model [1] cm/EMPTY 4 {:key (rng/key 1)})]
  (show "pf/dereplicate :method :first" (pf/dereplicate st 2 {:method :first})))

;; ---------------------------------------------------------------------------
(section "Missed? hmc's third argument given a key")
(let [m (gen [] (let [x (trace :x (d/normal 0.0 1.0))] (trace :y (d/normal x 1.0))))
      tr (:trace (gfi/generate m [] (cm/choicemap :y 1.0) (rng/key 0)))
      s (sel/select :x)
      v (fn [r] (array/item ((:trace r) :x)))]
  (println "mala, key last, twice:" (v (mcmc/mala tr s 0.1 (rng/key 7))) (v (mcmc/mala tr s 0.1 (rng/key 7))))
  (println "hmc, opts then key, twice:" (v (mcmc/hmc tr s {} (rng/key 7))) (v (mcmc/hmc tr s {} (rng/key 7))))
  (println "hmc, a key as its third argument, twice:" (v (mcmc/hmc tr s (rng/key 7))) (v (mcmc/hmc tr s (rng/key 7)))))

(println)
(println "===== done")
(shutdown-agents)
