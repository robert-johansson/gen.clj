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
  (:require [gen.array :as array :refer [+ - * /]]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.gfi :as gfi]
            [gen.gradients :as grad]
            [gen.random :as rng]
            [gen.selection :as sel]))

(set! *warn-on-reflection* true)

(defn- accept [trace proposed log-alpha key]
  (let [la (double (array/item log-alpha))]
    (when (Double/isNaN la)
      (throw (ex-info "The acceptance ratio is NaN" {:log-alpha la})))
    (let [accepted? (or (>= la 0.0) (< (Math/log (double (array/item (rng/uniform key)))) la))]
      {:trace (if accepted? proposed trace) :accepted? accepted? :log-alpha la})))

(defn- nonempty! [v selection]
  (when (zero? (array/size v))
    (throw (ex-info "The selection selects no choices of the trace" {:selection selection})))
  v)

(defn- flat-gradient
  "[values-choicemap flat-values flat-gradient] of the selection at trace."
  [trace selection]
  (let [{:keys [values gradients]} (grad/choice-gradients trace selection)]
    [values (nonempty! (cm/to-array values) selection) (cm/to-array gradients)]))

(defn- flat-values
  "[values-choicemap flat-values others] of the selection at trace, others
   the other choices as a map: what a compiled step takes."
  [trace selection]
  (let [values (grad/continuous-selection trace selection)]
    [values (nonempty! (cm/to-array values) selection)
     (cm/to-map (sel/get-selected (:choices trace) (sel/complement selection)))]))

(defn- move-to
  "trace with the selected choices (shaped like proto) set to the flat array
   v: the new trace and the update's weight."
  [trace proto v key]
  (let [{t :trace w :weight} (gfi/update trace (cm/from-array proto v) key)]
    [t w]))

(defn- normal-lp
  "log N(x; mu, sd) summed over elements."
  [x mu sd]
  (array/sum (dc/log-prob (d/normal mu sd) x)))

(def ^:private compiled-steps
  "Compiled kernel steps by kernel and settings, model, arguments and choice
   structure (the most recent few), or ::plain where compiling failed."
  (atom {}))

(defn- compiled-step
  "The kernel step (build) compiled for this trace's model, arguments and
   choice structure (cached under settings) and applied to args, or nil when
   the model cannot be compiled (it reads a value on the host), which is
   remembered."
  [trace proto others settings build & args]
  (let [structure (fn [m] (array/tree-map (fn [x] [(array/shape x) (array/dtype x)]) m))
        k [settings (:gen-fn trace) (:args trace) (structure others) (structure (cm/to-map proto))]
        cached (get @compiled-steps k)]
    (when-not (= cached ::plain)
      (let [f (or cached (let [f (array/compile (build))]
                           (swap! compiled-steps (fn [m] (assoc (if (>= (count m) 16) {} m) k f)))
                           f))]
        (try (apply f args)
             (catch Exception e
               ;; the first call traces the step: whatever fails there (a host
               ;; read, a value the trace holds as an array that the model
               ;; needs on the host) makes the step plain from now on, and the
               ;; plain kernel raises it again if it is the model's own error
               (if (or (nil? cached) (= :compile (:op (ex-data e))))
                 (do (swap! compiled-steps assoc k ::plain) nil)
                 (throw e))))))))

(defn- log-density
  "The log density of gf on args as a function of arrays only (for
   array/compile): [v others] → log p, with v the flat selected values (shaped
   by proto) and others the other choices as a map."
  [gf args proto]
  (fn [v others]
    (try
      (gfi/with-exact-density "A gradient-based move (HMC, MALA)"
        (:weight (gfi/assess gf args (cm/merge (cm/from-map others) (cm/from-array proto v)) (rng/key 0))))
      (catch clojure.lang.ExceptionInfo e
        (if-let [p (when (:missing-choice (ex-data e)) (:path (ex-data e)))]
          (throw (ex-info (str "HMC and MALA move the selected choices within the trace's structure: at the values"
                               " tried the model visits " p ", which the trace does not hold (" (ex-message e) ")")
                          {:path p} e))
          (throw e))))))

(defn- mala-proposal
  "MALA's proposal as a function of arrays only: [v noise others] →
   [v' log q(v | v') - log q(v' | v)]."
  [gf args proto tau]
  (let [grad-v (array/grad (log-density gf args proto))
        sd (Math/sqrt (* 2.0 (double tau)))]
    (fn [v noise others]
      (let [fwd-mu (+ v (* tau (grad-v v others)))
            v' (+ fwd-mu (* sd noise))
            bwd-mu (+ v' (* tau (grad-v v' others)))]
        [v' (- (normal-lp v bwd-mu sd) (normal-lp v' fwd-mu sd))]))))

(defn- step-opts
  "A kernel's last argument, a key or an options map, as an options map
   with a key."
  [key-or-opts]
  (let [opts (if (map? key-or-opts) key-or-opts {:key key-or-opts})]
    (assoc opts :key (or (:key opts) (gfi/next-key)))))

(defn mala
  "Metropolis-adjusted Langevin: propose v' ~ N(v + tau·∇, √(2 tau)) for the
   selected choices v, with the reverse move's density from the gradient at
   v'. The last argument is a key or an options map: :key, and :compile?
   (false) to compute the proposal and its densities as one compiled
   function, as hmc's."
  ([trace selection tau] (mala trace selection tau nil))
  ([trace selection tau key-or-opts]
   (let [{:keys [key compile?]} (step-opts key-or-opts)
         [k-propose k-update k-accept] (rng/split key 3)
         [proto v others] (flat-values trace selection)
         noise (rng/normal k-propose (array/shape v))
         build #(mala-proposal (:gen-fn trace) (:args trace) proto tau)
         ;; the proposal as a function of the selected values, compiled or not
         [v' log-q] (or (when compile? (compiled-step trace proto others [:mala tau] build v noise others))
                        ((build) v noise others))
         [t' weight] (move-to trace proto v' k-update)]
     (accept trace t' (+ weight log-q) k-accept))))

(defn- leapfrogs
  "L leapfrog steps as a function of arrays only: [v p others] → [v' p']."
  [gf args proto L eps]
  (let [grad-v (array/grad (log-density gf args proto))
        half (/ eps 2.0)]
    (fn [v p others]
      (loop [i 0 v v p p g (grad-v v others)]
        (if (< i (long L))
          (let [p (+ p (* half g))
                v (+ v (* eps p))
                g (grad-v v others)]
            (recur (inc i) v (+ p (* half g)) g))
          [v p])))))

(defn hmc
  "Hamiltonian Monte Carlo on the selected choices: L leapfrog steps of size
   eps with standard normal momenta. Options: :L (10), :eps (0.1),
   :compile? (false): run the L steps as one compiled function (array/compile:
   one MLX graph on the MLX backend, the same steps on the JVM backend), a
   function of the selected values and the other choices, cached per model,
   arguments and structure. A model that reads a value on the host cannot
   be compiled and runs the same function uncompiled. Either way the trace
   is updated once, after the L steps."
  ([trace selection] (hmc trace selection nil))
  ([trace selection opts] (hmc trace selection opts (or (:key opts) (gfi/next-key))))
  ([trace selection {:keys [L eps compile?] :or {L 10 eps 0.1}} key]
   (let [[k-momenta k-update k-accept] (rng/split key 3)
         kinetic (fn [p] (normal-lp p 0.0 1.0))
         finish (fn [t' p0 p]
                  (accept trace t' (+ (- (:score t') (:score trace)) (- (kinetic p) (kinetic p0))) k-accept))]
     (let [[proto v others] (flat-values trace selection)
           p0 (rng/normal k-momenta (array/shape v))
           build #(leapfrogs (:gen-fn trace) (:args trace) proto L eps)
           ;; the L steps as a function of the selected values, compiled or
           ;; not: the trace is updated once, at the end
           [v' p] (or (when compile? (compiled-step trace proto others [:hmc L eps] build v p0 others))
                      ((build) v p0 others))]
       (finish (first (move-to trace proto v' k-update)) p0 p)))))

(defn map-optimize
  "One step of gradient ascent on the log density of the selected choices,
   with backtracking: try the step size max-step-size, shrinking it by tau
   until the score does not decrease (Gen.jl's map_optimize). Options:
   :max-step-size (0.1), :tau (0.5), :min-step-size (1e-16)."
  ([trace selection] (map-optimize trace selection nil))
  ([trace selection {:keys [max-step-size tau min-step-size] :or {max-step-size 0.1 tau 0.5 min-step-size 1e-16}}]
   (let [[proto v g] (flat-gradient trace selection)
         score (double (array/item (:score trace)))]
     (loop [step (double max-step-size)]
       (let [[t _] (move-to trace proto (+ v (* step g)) (rng/key 0))]
         (cond (>= (double (array/item (:score t))) score) t
               (< step (double min-step-size)) trace
               :else (recur (clojure.core/* step (double tau)))))))))

(defn elliptical-slice
  "Elliptical slice sampling (Murray, Adams and MacKay 2010) of the choice at
   addr, whose prior is N(mu, cov): a new value on the ellipse through the
   current one and a prior draw, accepted on the slice of the likelihood (the
   update weight less the prior's own ratio, which the ellipse accounts for)."
  ([trace addr mu cov] (elliptical-slice trace addr mu cov (gfi/next-key)))
  ([trace addr mu cov key]
   (let [mu (dc/->array mu)
         cov (dc/->array cov)
         [k-nu k-u k-theta] (rng/split key 3)
         prior (d/mvnormal mu cov)
         f0 (or (trace addr)
                (throw (ex-info (str "elliptical-slice: the trace has no choice at " (pr-str addr)) {:addr addr})))
         x (- f0 mu)
         nu (- (dc/sample prior k-nu) mu)
         log-u (Math/log (double (array/item (rng/uniform k-u))))
         lp0 (dc/log-prob prior f0)
         at (fn [theta]
              (let [f (+ mu (+ (* x (Math/cos theta)) (* nu (Math/sin theta))))
                    {t :trace w :weight} (gfi/with-exact-density "elliptical-slice" (gfi/update trace (cm/choicemap addr f) (rng/key 0)))]
                [t (double (array/item (- w (- (dc/log-prob prior f) lp0))))]))]
     (loop [i 0
            theta (* 2.0 Math/PI (double (array/item (rng/uniform (rng/fold-in k-theta 0)))))
            lo (- theta (* 2.0 Math/PI))
            hi theta]
       (let [[t log-lik-ratio] (at theta)]
         (if (> log-lik-ratio log-u)
           t
           (let [[lo hi] (if (neg? theta) [theta hi] [lo theta])
                 u (double (array/item (rng/uniform (rng/fold-in k-theta (inc i)))))]
             (recur (inc i) (+ lo (* u (- hi lo))) lo hi))))))))
