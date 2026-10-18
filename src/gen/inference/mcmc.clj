(ns gen.inference.mcmc
  "Gradient-based MCMC and optimization over a selection of continuous
   choices, as Gen.jl's hmc, mala, map_optimize and elliptical_slice. Each
   works on any generative function, nested addresses included, through
   gen.gradients/choice-gradients.

     (hmc trace selection {:L 10 :eps 0.1})    → {:trace :accepted? :log-alpha}
     (mala trace selection tau)                → {:trace :accepted? :log-alpha}
     (map-optimize trace selection opts)       → trace
     (elliptical-slice trace addr mu cov)      → trace

   The selected choices must keep the trace's structure: HMC and MALA move
   them jointly and leave every other choice alone. Keys are an optional last
   argument (or :key in an options map); omitting one draws from the implicit
   key source."
  (:refer-clojure :exclude [+ - * /])
  (:require [gen.array :as a :refer [+ - * /]]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.gfi :as gfi]
            [gen.gradients :as grad]
            [gen.random :as rng]))

(set! *warn-on-reflection* true)

(defn- accept [trace proposed log-alpha key]
  (let [la (double (a/item log-alpha))]
    (when (Double/isNaN la)
      (throw (ex-info "The acceptance ratio is NaN" {:log-alpha la})))
    (let [accepted? (or (>= la 0.0) (< (Math/log (double (a/item (rng/uniform key)))) la))]
      {:trace (if accepted? proposed trace) :accepted? accepted? :log-alpha la})))

(defn- flat-gradient
  "[values-choicemap flat-values flat-gradient] of the selection at trace."
  [trace selection]
  (let [{:keys [values gradients]} (grad/choice-gradients trace selection)]
    [values (cm/to-array values) (cm/to-array gradients)]))

(defn- move-to
  "trace with the selected choices (shaped like proto) set to the flat array
   v: the new trace and the update's weight."
  [trace proto v key]
  (let [{t :trace w :weight} (gfi/update trace (cm/from-array proto v) key)]
    [t w]))

(defn- normal-lp
  "log N(x; mu, sd) summed over elements."
  [x mu sd]
  (a/sum (dc/log-prob (d/normal mu sd) x)))

(defn mala
  "Metropolis-adjusted Langevin: propose v' ~ N(v + tau·∇, √(2 tau)) for the
   selected choices v, with the reverse move's density from the gradient at
   v'."
  ([trace selection tau] (mala trace selection tau (gfi/next-key)))
  ([trace selection tau key]
   (let [[k-propose k-update k-accept] (rng/split key 3)
         sd (Math/sqrt (* 2.0 (double tau)))
         [proto v g] (flat-gradient trace selection)
         fwd-mu (+ v (* tau g))
         v' (+ fwd-mu (* sd (rng/normal k-propose (a/shape v))))
         [t' weight] (move-to trace proto v' k-update)
         [_ _ g'] (flat-gradient t' selection)
         bwd-mu (+ v' (* tau g'))]
     (accept trace t' (+ (- weight (normal-lp v' fwd-mu sd)) (normal-lp v bwd-mu sd)) k-accept))))

(defn hmc
  "Hamiltonian Monte Carlo on the selected choices: L leapfrog steps of size
   eps with standard normal momenta. Options: :L (10), :eps (0.1)."
  ([trace selection] (hmc trace selection nil))
  ([trace selection opts] (hmc trace selection opts (gfi/next-key)))
  ([trace selection {:keys [L eps] :or {L 10 eps 0.1}} key]
   (let [[k-momenta k-update k-accept] (rng/split key 3)
         [proto v g] (flat-gradient trace selection)
         p0 (rng/normal k-momenta (a/shape v))
         [t' p] (loop [i 0 t trace v v g g p p0]
                  (if (< i (long L))
                    (let [p (+ p (* (/ eps 2.0) g))
                          v (+ v (* eps p))
                          [t _] (move-to t proto v (rng/fold-in k-update i))
                          [_ _ g] (flat-gradient t selection)]
                      (recur (inc i) t v g (+ p (* (/ eps 2.0) g))))
                    [t p]))
         kinetic (fn [p] (normal-lp p 0.0 1.0))]
     (accept trace t' (+ (- (:score t') (:score trace)) (- (kinetic p) (kinetic p0))) k-accept))))

(defn map-optimize
  "One step of gradient ascent on the log density of the selected choices,
   with backtracking: try the step size max-step-size, shrinking it by tau
   until the score does not decrease (Gen.jl's map_optimize). Options:
   :max-step-size (0.1), :tau (0.5), :min-step-size (1e-16)."
  ([trace selection] (map-optimize trace selection nil))
  ([trace selection {:keys [max-step-size tau min-step-size] :or {max-step-size 0.1 tau 0.5 min-step-size 1e-16}}]
   (let [[proto v g] (flat-gradient trace selection)
         score (double (a/item (:score trace)))]
     (loop [step (double max-step-size)]
       (let [[t _] (move-to trace proto (+ v (* step g)) (rng/key 0))]
         (cond (>= (double (a/item (:score t))) score) t
               (< step (double min-step-size)) trace
               :else (recur (clojure.core/* step (double tau)))))))))

(defn elliptical-slice
  "Elliptical slice sampling (Murray, Adams and MacKay 2010) of the choice at
   addr, whose prior is N(mu, cov): a new value on the ellipse through the
   current one and a prior draw, accepted on the slice of the likelihood (the
   update weight less the prior's own ratio, which the ellipse accounts for)."
  ([trace addr mu cov] (elliptical-slice trace addr mu cov (gfi/next-key)))
  ([trace addr mu cov key]
   (let [[k-nu k-u k-theta] (rng/split key 3)
         prior (d/mvnormal mu cov)
         f0 (trace addr)
         x (- f0 mu)
         nu (- (dc/sample prior k-nu) mu)
         log-u (Math/log (double (a/item (rng/uniform k-u))))
         lp0 (dc/log-prob prior f0)
         at (fn [theta]
              (let [f (+ mu (+ (* x (Math/cos theta)) (* nu (Math/sin theta))))
                    {t :trace w :weight} (gfi/update trace (cm/choicemap addr f) (rng/key 0))]
                [t (double (a/item (- w (- (dc/log-prob prior f) lp0))))]))]
     (loop [i 0
            theta (* 2.0 Math/PI (double (a/item (rng/uniform (rng/fold-in k-theta 0)))))
            lo (- theta (* 2.0 Math/PI))
            hi theta]
       (let [[t log-lik-ratio] (at theta)]
         (if (> log-lik-ratio log-u)
           t
           (let [[lo hi] (if (neg? theta) [theta hi] [lo theta])
                 u (double (a/item (rng/uniform (rng/fold-in k-theta (inc i)))))]
             (recur (inc i) (+ lo (* u (- hi lo))) lo hi))))))))
