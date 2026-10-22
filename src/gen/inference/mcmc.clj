(ns gen.inference.mcmc
  "Gradient-based MCMC and optimization over a selection of continuous
   choices, as Gen.jl's hmc, mala, map_optimize and elliptical_slice. Each
   works on any generative function, nested addresses included, through
   gen.gradients/choice-gradients.

     (hmc trace selection {:L 10 :eps 0.1})    → {:trace :accepted? :log-alpha :diverged?}
     (mala trace selection tau)                → {:trace :accepted? :log-alpha :diverged?}
     (map-optimize trace selection opts)       → trace
     (elliptical-slice trace addr mu cov)      → trace

   The selected choices must keep the trace's structure: HMC and MALA move
   them jointly and leave every other choice alone. A proposal whose
   position, momentum, gradient or energy turns non-finite, or where a
   distribution's parameter fails its check, diverged: it is rejected,
   with acceptance 0 (:log-alpha -Inf), and reported as :diverged?
   (gen.inference.kernel/run counts them), as Stan treats a divergence. A
   starting trace whose log density or gradient is not finite (NaN, or a
   probability of 0) is an error, as Stan refuses such a start. Keys are an
   optional last argument (or :key in an options map); omitting one draws
   from the implicit key source."
  (:refer-clojure :exclude [+ - * /])
  (:require [gen.array :as array :refer [+ - * /]]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.gfi :as gfi]
            [gen.gradients :as grad]
            [gen.inference.mh :as mh]
            [gen.random :as rng]
            [gen.selection :as sel]))

(set! *warn-on-reflection* true)

(defn- diverged
  "The result of a proposal that diverged: rejected, with acceptance 0."
  [trace]
  {:trace trace :accepted? false :log-alpha ##-Inf :diverged? true})

(defn- accept
  "mh/accept's decision on proposed, of log acceptance ratio log-alpha: a
   ratio that is not finite (an energy that overflowed or is NaN) diverged."
  [trace proposed log-alpha key]
  (if (Double/isFinite (double (array/item log-alpha)))
    (assoc (mh/accept trace proposed log-alpha key) :diverged? false)
    (diverged trace)))

(defn- finite
  "Are the elements of xs all finite? A boolean array, read on the host only
   when the kernel finishes."
  [& xs]
  (reduce array/logical-and (map #(array/all (array/isfinite %)) xs)))

(defn- check-start!
  "Throws unless start-ok, the kernel's step's verdict on the starting
   trace: a log density or gradient there that is not finite would make
   every proposal diverge, and the chain never move (DESIGN §9,
   PROVISIONAL)."
  [kernel start-ok trace selection]
  (when-not (array/truthy? start-ok)
    (throw (ex-info (str kernel ": the starting trace's log density or its gradient at the selection is not finite"
                         " (score " (array/->clj (:score trace)) "), so no move from it can be weighed")
                    {:reason :non-finite-start :kernel (keyword kernel) :score (:score trace) :selection selection}))))

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
  "log N(x; mu, sd) summed over elements, sd a host number. By arithmetic,
   not gen.dist's normal, which refuses a NaN mean: a diverged gradient's
   NaN must reach the divergence check."
  [x mu ^double sd]
  (let [z (/ (- x mu) sd)]
    (array/sum (- (* -0.5 (* z z)) (+ (Math/log sd) (* 0.5 (Math/log (* 2.0 Math/PI))))))))

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
        (let [{:keys [path missing-choice unvisited]} (ex-data e)]
          (if (and path (or missing-choice unvisited))
            (throw (ex-info (str "HMC and MALA move the selected choices within the trace's structure: at the values"
                                 " tried the model "
                                 (if missing-choice
                                   (str "visits " path ", which the trace does not hold")
                                   (str "no longer visits " path ", which the trace holds"))
                                 " (" (ex-message e) ")")
                            {:path path} e))
            (throw e)))))))

(defn- tried
  "f, a log density's gradient (or value and gradient) of [v others], at
   positions a move tries: a distribution's parameter that fails its check
   there (a scale that underflowed to 0, say) leaves the density
   undefined, so f gives (nan v), a divergence, as Stan rejects a proposal
   at a domain error (DESIGN §9, PROVISIONAL). Any other error, a
   structure change or a value's shape, is raised."
  [f nan]
  (fn [v others]
    (try (f v others)
         (catch clojure.lang.ExceptionInfo e
           (if (let [{:keys [distribution param]} (ex-data e)] (and distribution param))
             (nan v)
             (throw e))))))

(defn- mala-proposal
  "MALA's proposal as a function of arrays only: [v noise others] →
   [v' log q(v | v') - log q(v' | v) diverged start-ok], diverged true when
   v', the gradients or the densities are not finite (or a parameter fails
   its check at v', tried), start-ok whether the log density and its
   gradient at v are. The reverse gradient is taken at v' only when v' is
   finite (at v otherwise: assess would refuse a NaN), so a compiled
   proposal needs no branch."
  [gf args proto tau]
  (let [density (log-density gf args proto)
        value-and-grad (array/value-and-grad density)
        grad-tried (tried (array/grad density) #(* % ##NaN))
        sd (Math/sqrt (* 2.0 (double tau)))]
    (fn [v noise others]
      (let [[lp g] (value-and-grad v others)
            fwd-mu (+ v (* tau g))
            v' (+ fwd-mu (* sd noise))
            ok (finite g v')
            v' (array/where ok v' v)
            g' (grad-tried v' others)
            bwd-mu (+ v' (* tau g'))
            log-q (- (normal-lp v bwd-mu sd) (normal-lp v' fwd-mu sd))]
        [v' log-q (array/logical-not (array/logical-and ok (finite g' log-q))) (finite lp g)]))))

(defn- options
  "opts, an options map or nil (loud for anything else, such as a key in
   its place, which the kernel would not see)."
  [opts]
  (if (or (nil? opts) (map? opts))
    opts
    (throw (ex-info (str "The kernel takes an options map here, not " (pr-str opts)) {:value opts}))))

(defn- step-opts
  "A kernel's last argument, a key or an options map (loud otherwise), as an
   options map with a key: the options' :key, or one from the implicit
   source."
  [key-or-opts]
  (let [opts (cond (rng/key? key-or-opts) {:key key-or-opts}
                   (or (nil? key-or-opts) (map? key-or-opts)) key-or-opts
                   :else (throw (ex-info (str "The kernel's last argument is a key or an options map, not "
                                              (pr-str key-or-opts))
                                         {:value key-or-opts})))]
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
         [v' log-q diverged? start-ok] (or (when compile? (compiled-step trace proto others [:mala tau] build v noise others))
                                           ((build) v noise others))]
     (check-start! "mala" start-ok trace selection)
     (if (array/truthy? diverged?)
       (diverged trace)
       (let [[t' weight] (move-to trace proto v' k-update)]
         (accept trace t' (+ weight log-q) k-accept))))))

(defn- leapfrogs
  "L leapfrog steps as a function of arrays only: [v p others] → [v' p'
   diverged start-ok]. diverged is true once a position, momentum,
   gradient or log density turns non-finite (or a parameter fails its
   check, tried), and from that step on the state stays at its last finite
   values: no gradient is taken at a non-finite position (assess would
   refuse a NaN), and a compiled step needs no early exit. start-ok is
   whether the log density and its gradient at v are finite."
  [gf args proto L eps]
  (let [value-and-grad (array/value-and-grad (log-density gf args proto))
        tried-value-and-grad (tried value-and-grad (fn [v] [##NaN (* v ##NaN)]))
        half (/ eps 2.0)]
    (fn [v p others]
      (let [[lp g] (value-and-grad v others)
            start-ok (finite lp g)]
        (loop [i 0 v v p p g g diverged (array/logical-not start-ok)]
          (if (< i (long L))
            (let [p-half (+ p (* half g))
                  v-next (+ v (* eps p-half))
                  diverged (array/logical-or diverged (array/logical-not (finite p-half v-next)))
                  v-next (array/where diverged v v-next)
                  [lp g-next] (tried-value-and-grad v-next others)
                  p-next (+ p-half (* half g-next))
                  diverged (array/logical-or diverged (array/logical-not (finite lp g-next p-next)))]
              (recur (inc i) (array/where diverged v v-next) (array/where diverged p p-next)
                     (array/where diverged g g-next) diverged))
            [v p diverged start-ok]))))))

(defn hmc
  "Hamiltonian Monte Carlo on the selected choices: L leapfrog steps of size
   eps with standard normal momenta. The third argument is an options map
   or a key, and a key may follow the options. Options: :L (10), :eps (0.1),
   :key, :compile? (false): run the L steps as one compiled function (array/compile:
   one MLX graph on the MLX backend, the same steps on the JVM backend), a
   function of the selected values and the other choices, cached per model,
   arguments and structure. A model that reads a value on the host cannot
   be compiled and runs the same function uncompiled. Either way the trace
   is updated once, after the L steps."
  ([trace selection] (hmc trace selection nil))
  ([trace selection key-or-opts] (let [opts (step-opts key-or-opts)] (hmc trace selection opts (:key opts))))
  ([trace selection opts key]
   (let [{:keys [L eps compile?] :or {L 10 eps 0.1}} (options opts)
         [k-momenta k-update k-accept] (rng/split key 3)
         kinetic (fn [p] (normal-lp p 0.0 1.0))
         [proto v others] (flat-values trace selection)
         p0 (rng/normal k-momenta (array/shape v))
         build #(leapfrogs (:gen-fn trace) (:args trace) proto L eps)
         ;; the L steps as a function of the selected values, compiled or
         ;; not: the trace is updated once, at the end
         [v' p diverged? start-ok] (or (when compile? (compiled-step trace proto others [:hmc L eps] build v p0 others))
                                       ((build) v p0 others))]
     (check-start! "hmc" start-ok trace selection)
     (if (array/truthy? diverged?)
       (diverged trace)
       (let [[t'] (move-to trace proto v' k-update)]
         (accept trace t' (+ (- (:score t') (:score trace)) (- (kinetic p) (kinetic p0))) k-accept))))))

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
         f0 (or (trace addr nil)
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
