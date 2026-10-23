(ns gen.inference.mcmc
  "Gradient-based MCMC and optimization over a selection of continuous
   choices, as Gen.jl's hmc, mala, map_optimize and elliptical_slice, and
   Stan's No-U-Turn Sampler. Each works on any generative function, nested
   addresses included, through gen.gradients/choice-gradients.

     (hmc trace selection {:L 10 :eps 0.1})    → {:trace :accepted? :log-alpha :diverged?}
     (mala trace selection tau)                → {:trace :accepted? :log-alpha :diverged?}
     (nuts trace selection {:eps 0.1})         → {:trace :accept-stat :tree-depth :n-leapfrog
                                                  :diverged? :energy :eps}
     (map-optimize trace selection opts)       → trace
     (elliptical-slice trace addr mu cov)      → trace

     (hmc-particles ps selection opts)          → {:particles :accepted? :log-alpha :diverged?}
     (mala-particles ps selection tau)          → {:particles :accepted? :log-alpha :diverged?}
     (map-optimize-particles ps selection opts) → particles

   The selected choices must keep the trace's structure: HMC, MALA and NUTS
   move them jointly and leave every other choice alone. They take a
   diagonal metric, :inv-metric (Stan's inv_metric, the identity by
   default): a variance for each selected value, in the order of
   (cm/to-array (grad/continuous-selection trace selection)), or one number
   for all. Momenta are drawn with those precisions and a position moves by
   inv-metric ⊙ momentum, so variances that match the target's make its
   scales alike (gen.inference.nuts adapts them). A proposal whose
   position, momentum, gradient or energy turns non-finite, or where a
   distribution's parameter fails its check, diverged: it is rejected,
   with acceptance 0 (:log-alpha -Inf), and reported as :diverged?
   (gen.inference.kernel/run counts them), as Stan treats a divergence;
   NUTS's trajectory also diverges at an energy error over 1000, as
   Stan's. A starting trace whose log density or gradient is not finite
   (NaN, or a probability of 0) is an error, as Stan refuses such a start.
   Keys are an optional last argument (or :key in an options map); omitting
   one draws from the implicit key source.

   The -particles kernels move every particle of a gen.batched collection,
   n independent chains each accepted on its own, at once: one batched
   gradient per step for all of them where the model runs batched. Each
   kernel's move is one function of arrays and keys (hmc-move, mala-move,
   map-move), which the scalar kernel runs on its trace and the -particles
   kernel on the batched trace, under the batch tracer
   (gen.batched/map-particles); where the model does not run batched, each
   particle's move is the scalar kernel's, from its own key (the key split n
   ways)."
  (:refer-clojure :exclude [+ - * / < >=])
  (:require [gen.array :as array :refer [+ - * / < >=]]
            [gen.array.batch :as batch]
            [gen.batched :as batched]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.gfi :as gfi]
            [gen.gradients :as grad]
            [gen.random :as rng]
            [gen.selection :as sel]))

(set! *warn-on-reflection* true)

(defn- diverged
  "The result of a proposal that diverged: rejected, with acceptance 0."
  [trace]
  {:trace trace :accepted? false :log-alpha ##-Inf :diverged? true})

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
  "log N(x; mu, sd) summed over elements, sd a host number or an array. By
   arithmetic, not gen.dist's normal, which refuses a NaN mean: a diverged
   gradient's NaN must reach the divergence check."
  [x mu sd]
  (let [z (/ (- x mu) sd)]
    (array/sum (- (* -0.5 (* z z)) (+ (array/log sd) (* 0.5 (Math/log (* 2.0 Math/PI))))))))

(defn- inverse-metric
  "The :inv-metric option for the flat values v: the host number 1.0 (the
   identity, under which hmc and mala move as they did without one) for
   nil, a positive number, or a vector or array of v's shape. Loud unless
   every variance is positive and finite."
  [inv v]
  (let [m (cond (nil? inv) 1.0
                (sequential? inv) (array/astype (array/array inv) (array/dtype v))
                (array/backend-array? inv) (array/astype inv (array/dtype v))
                :else inv)]
    (when-not (and (array/array? m)
                   (or (array/host? m) (= (array/shape m) (array/shape v)))
                   (array/truthy? (array/all (array/logical-and (array/isfinite m) (array/> m 0)))))
      (throw (ex-info (str ":inv-metric must be a positive variance, or one for each of the " (array/size v)
                           " selected values, not " (pr-str inv))
                      {:reason :invalid-inv-metric :inv-metric inv :shape (array/shape v)})))
    m))

(def ^:private compiled-steps
  "Compiled kernel steps by kernel and settings, model, arguments and choice
   structure (the most recent few), or ::plain where compiling failed."
  (atom {}))

(defn- compiled
  "The kernel step (build) compiled for this trace's model, arguments and
   choice structure (cached under settings): a function of the step's
   arguments, which returns nil when the model cannot be compiled (it reads
   a value on the host), and that is remembered."
  [trace proto others settings build]
  (let [structure (fn [m] (array/tree-map (fn [x] [(array/shape x) (array/dtype x)]) m))
        k [settings (:gen-fn trace) (:args trace) (structure others) (structure (cm/to-map proto))]]
    (fn [& args]
      (let [cached (get @compiled-steps k)]
        (when-not (= cached ::plain)
          (let [f (or cached (let [f (array/compile (build))]
                               (swap! compiled-steps (fn [m] (assoc (if (>= (count m) 16) {} m) k f)))
                               f))]
            (try (apply f args)
                 (catch Exception e
                   ;; the first call traces the step: whatever fails there (a
                   ;; host read, a value the trace holds as an array that the
                   ;; model needs on the host) makes the step plain from now
                   ;; on, and the plain kernel raises it again if it is the
                   ;; model's own error
                   (if (or (nil? cached) (= :compile (:op (ex-data e))))
                     (do (swap! compiled-steps assoc k ::plain) nil)
                     (throw e))))))))))

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
  "f, a log density's value and gradient of [v others], at positions a move
   tries: a distribution's parameter that fails its check there (a scale
   that underflowed to 0, say) leaves the density undefined, so f gives
   (nan v), a divergence, as Stan rejects a proposal at a domain error
   (DESIGN §9, PROVISIONAL). Any other error, a structure change or a
   value's shape, is raised. In a batched run the check may fail for some
   particles only: the batched-branch error, so that each particle moves
   on its own."
  [f nan]
  (fn [v others]
    (try (f v others)
         (catch clojure.lang.ExceptionInfo e
           (cond (not (let [{:keys [distribution param]} (ex-data e)] (and distribution param))) (throw e)
                 (batch/batch-backend? array/*backend*)
                 (throw (array/batched-branch "A parameter fails its check at a position tried" {} e))
                 :else (nan v))))))

(def ^:private nan-value-and-grad
  "A tried value and gradient's divergence."
  (fn [v] [##NaN (* v ##NaN)]))

(defn- verdict
  "A move's proposal v' from v (flat values shaped by proto) as a function
   of arrays only, so per particle in a batched run: {:values the selected
   choices at v' if accepted, at v if not, :accepted?, :log-alpha (-Inf
   when diverged), :diverged? (also when log-alpha is not finite: an energy
   that overflowed, or NaN), :start-ok}. Accepted with probability
   min(1, exp(log-alpha)) by the uniform of key, as mh/accept."
  [proto v v' log-alpha diverged start-ok key]
  (let [diverged (array/logical-or diverged (array/logical-not (array/isfinite log-alpha)))
        accepted (array/logical-and (array/logical-not diverged)
                                    (array/logical-or (>= log-alpha 0.0) (< (array/log (rng/uniform key)) log-alpha)))]
    {:values (cm/from-array proto (array/where accepted v' v))
     :accepted? accepted
     :log-alpha (array/where diverged ##-Inf log-alpha)
     :diverged? diverged
     :start-ok start-ok}))

(defn- moved
  "A scalar kernel's result from its move's verdict: the trace updated to
   the values (with key) when accepted, the trace itself when not."
  [trace {:keys [values accepted? log-alpha diverged?]} key]
  (cond (array/truthy? diverged?) (diverged trace)
        (array/truthy? accepted?) {:trace (:trace (gfi/update trace values key)) :accepted? true
                                   :log-alpha (double (array/item log-alpha)) :diverged? false}
        :else {:trace trace :accepted? false :log-alpha (double (array/item log-alpha)) :diverged? false}))

(defn- mala-proposal
  "MALA's proposal as a function of arrays only: [v noise inv others] →
   [v' log q(v | v') - log q(v' | v) diverged start-ok lp lp'],
   preconditioned by the inverse metric inv (v' ~ N(v + tau·inv ⊙ ∇,
   2 tau·inv)), diverged true when v', the gradients or the densities are
   not finite (or a parameter fails its check at v', tried), start-ok
   whether the log density and its gradient at v are, lp and lp' the log
   densities at v and v'. The reverse gradient is taken at v' only when v'
   is finite (at v otherwise: assess would refuse a NaN), so a compiled
   proposal needs no branch."
  [gf args proto tau]
  (let [density (log-density gf args proto)
        value-and-grad (array/value-and-grad density)
        tried-value-and-grad (tried value-and-grad nan-value-and-grad)
        root-2tau (Math/sqrt (* 2.0 (double tau)))]
    (fn [v noise inv others]
      (let [sd (* root-2tau (array/sqrt inv))
            [lp g] (value-and-grad v others)
            fwd-mu (+ v (* tau (* inv g)))
            v' (+ fwd-mu (* sd noise))
            ok (finite g v')
            v' (array/where ok v' v)
            [lp' g'] (tried-value-and-grad v' others)
            bwd-mu (+ v' (* tau (* inv g')))
            log-q (- (normal-lp v bwd-mu sd) (normal-lp v' fwd-mu sd))]
        [v' log-q (array/logical-not (array/logical-and ok (finite g' log-q))) (finite lp g) lp lp']))))

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

(defn- mala-move
  "MALA's move of trace as a function of arrays and keys only (verdict), so
   that it runs batched over particles too: options and keys mala's."
  [trace selection tau {:keys [compile? inv-metric]} key]
  (let [[k-propose _ k-accept] (rng/split key 3)
        [proto v others] (flat-values trace selection)
        inv (inverse-metric inv-metric v)
        noise (rng/normal k-propose (array/shape v))
        build #(mala-proposal (:gen-fn trace) (:args trace) proto tau)
        ;; the proposal as a function of the selected values, compiled or not
        [v' log-q diverged start-ok lp lp'] (or (when compile? ((compiled trace proto others [:mala tau] build) v noise inv others))
                                               ((build) v noise inv others))]
    (verdict proto v v' (+ (- lp' lp) log-q) diverged start-ok k-accept)))

(defn mala
  "Metropolis-adjusted Langevin: propose v' ~ N(v + tau·∇, √(2 tau)) for the
   selected choices v, with the reverse move's density from the gradient at
   v'. The last argument is a key or an options map: :key, :inv-metric (the
   identity; with variances M⁻¹, v' ~ N(v + tau·M⁻¹∇, 2 tau·M⁻¹)), and
   :compile? (false) to compute the proposal and its densities as one
   compiled function, as hmc's."
  ([trace selection tau] (mala trace selection tau nil))
  ([trace selection tau key-or-opts]
   (let [{:keys [key] :as opts} (step-opts key-or-opts)
         move (mala-move trace selection tau opts key)]
     (check-start! "mala" (:start-ok move) trace selection)
     (moved trace move (second (rng/split key 3))))))

(defn- leapfrog
  "One leapfrog step of size eps (signed: negative runs time backwards)
   under the inverse metric inv, as a function of arrays only: [v p g eps
   inv others] → [v' p' g' lp ok], g the log density's gradient at v and g'
   and lp its gradient and value at v'. ok is false when the position,
   momentum, gradient or log density turns non-finite, or a parameter fails
   its check at v' (tried). A non-finite v' is replaced by v before the
   gradient is taken, so no gradient is taken at a non-finite position
   (assess would refuse a NaN), and a compiled step needs no branch."
  [value-and-grad]
  (let [value-and-grad (tried value-and-grad (fn [v] [##NaN (* v ##NaN)]))]
    (fn [v p g eps inv others]
      (let [half (/ eps 2.0)
            p-half (+ p (* half g))
            v' (+ v (* eps (* inv p-half)))
            ok (finite p-half v')
            v' (array/where ok v' v)
            [lp g'] (value-and-grad v' others)
            p' (+ p-half (* half g'))]
        [v' p' g' lp (array/logical-and ok (finite lp g' p'))]))))

(defn- leapfrogs
  "L leapfrog steps as a function of arrays only: [v p eps inv others] →
   [v' p' diverged start-ok lp lp'], lp and lp' the log densities at v and
   v'. diverged is true once a step is not ok (a non-finite value, or a
   parameter failing its check), and from that step on the state stays at
   its last finite values, so a compiled step needs no early exit. start-ok
   is whether the log density and its gradient at v are finite."
  [gf args proto L]
  (let [value-and-grad (array/value-and-grad (log-density gf args proto))
        step (leapfrog value-and-grad)]
    (fn [v p eps inv others]
      (let [[lp g] (value-and-grad v others)
            start-ok (finite lp g)]
        (loop [i 0 v v p p g g lp' lp diverged (array/logical-not start-ok)]
          (if (< i (long L))
            (let [[v-next p-next g-next lp-next ok] (step v p g eps inv others)
                  diverged (array/logical-or diverged (array/logical-not ok))]
              (recur (inc i) (array/where diverged v v-next) (array/where diverged p p-next)
                     (array/where diverged g g-next) (array/where diverged lp' lp-next) diverged))
            [v p diverged start-ok lp lp']))))))

(defn- hmc-move
  "HMC's move of trace as a function of arrays and keys only (verdict), so
   that it runs batched over particles too: options and keys hmc's."
  [trace selection {:keys [L eps compile? inv-metric] :or {L 10 eps 0.1}} key]
  (let [[k-momenta _ k-accept] (rng/split key 3)
        [proto v others] (flat-values trace selection)
        inv (inverse-metric inv-metric v)
        ;; the momenta's scale, √M
        sd (/ 1.0 (array/sqrt inv))
        kinetic (fn [p] (normal-lp p 0.0 sd))
        p0 (* sd (rng/normal k-momenta (array/shape v)))
        build #(leapfrogs (:gen-fn trace) (:args trace) proto L)
        ;; the L steps as a function of the selected values, compiled or
        ;; not: a trace is updated once, at the end
        [v' p diverged start-ok lp lp'] (or (when compile? ((compiled trace proto others [:hmc L] build) v p0 eps inv others))
                                           ((build) v p0 eps inv others))]
    (verdict proto v v' (+ (- lp' lp) (- (kinetic p) (kinetic p0))) diverged start-ok k-accept)))

(defn- uniform
  "A uniform draw on [0, 1) on key, as a host double."
  ^double [key]
  (double (array/item (rng/uniform key))))

(defn- check-jitter!
  "jitter, loud unless 0 ≤ jitter < 1."
  [jitter]
  (when-not (and (number? jitter) (<= 0.0 (double jitter)) (< (double jitter) 1.0))
    (throw (ex-info (str ":jitter must be at least 0 and below 1, not " (pr-str jitter))
                    {:reason :invalid-jitter :jitter jitter})))
  jitter)

(defn- jitter-step
  "eps·(1 + jitter·(2u - 1)): the step size of a transition jittered by u."
  [eps jitter u]
  (* eps (+ 1.0 (* jitter (- (* 2.0 u) 1.0)))))

(defn- jittered
  "One transition's step size, Stan's stepsize_jitter (Neal 2011's remedy
   for a trajectory of fixed length that returns near its start):
   eps·(1 + jitter·(2u - 1)), u uniform on the transition's key at index 3,
   beside the three keys it splits, read on the host. With jitter 0, eps,
   and nothing is drawn. Loud unless 0 ≤ jitter < 1."
  [eps jitter key]
  (check-jitter! jitter)
  (if (zero? (double jitter))
    eps
    (jitter-step eps (double jitter) (uniform (rng/fold-in key 3)))))

(defn hmc
  "Hamiltonian Monte Carlo on the selected choices: L leapfrog steps of size
   eps, with momenta drawn from N(0, M), M the inverse of :inv-metric (the
   identity by default: standard normal momenta). The third argument is an
   options map or a key, and a key may follow the options. Options: :L
   (10), :eps (0.1), :inv-metric, :jitter (0): each transition steps by
   eps·(1 + jitter·(2u - 1)), u uniform, Stan's step-size jitter, which
   keeps L steps from returning near the start (the result then holds the
   step taken, :eps), :key, :compile? (false): run the L steps as one
   compiled function (array/compile: one MLX graph on the MLX backend, the
   same steps on the JVM backend), a function of the selected values and
   the other choices, cached per model, arguments and structure. A model
   that reads a value on the host cannot be compiled and runs the same
   function uncompiled. Either way the trace is updated once, after the L
   steps."
  ([trace selection] (hmc trace selection nil))
  ([trace selection key-or-opts] (let [opts (step-opts key-or-opts)] (hmc trace selection opts (:key opts))))
  ([trace selection opts key]
   (let [{:keys [eps jitter] :or {eps 0.1 jitter 0} :as opts} (options opts)
         step (jittered eps jitter key)
         move (hmc-move trace selection (assoc opts :eps step) key)]
     (check-start! "hmc" (:start-ok move) trace selection)
     (cond-> (moved trace move (second (rng/split key 3)))
       (pos? (double jitter)) (assoc :eps step)))))

;; ---------------------------------------------------------------------------
;; NUTS
;; ---------------------------------------------------------------------------

(def ^:private max-energy-error
  "The energy error past which a NUTS trajectory diverged (Stan's
   max_deltaH)."
  1000.0)

(defn- energy
  "The Hamiltonian at log density lp, momentum p and velocity sharp (inv ⊙
   p): -lp + ½ p·sharp, Stan's."
  [lp p sharp]
  (- (* 0.5 (array/sum (* p sharp))) lp))

(defn- nuts-step
  "NUTS's leapfrog step as a function of arrays only: [v p g eps inv others]
   → [v' p' g' sharp' h'], with sharp' = inv ⊙ p', the velocity the
   no-U-turn criterion reads, and h' the energy there, +Inf when the step
   diverged."
  [gf args proto]
  (let [step (leapfrog (array/value-and-grad (log-density gf args proto)))]
    (fn [v p g eps inv others]
      (let [[v p g lp ok] (step v p g eps inv others)
            sharp (* inv p)]
        [v p g sharp (array/where ok (energy lp p sharp) ##Inf)]))))

(defn- continuing?
  "Stan's compute_criterion: do the velocities a and b (sharp momenta) both
   point along rho, the momenta summed between them? A boolean array."
  [a b rho]
  (array/logical-and (array/> (array/sum (* a rho)) 0) (array/> (array/sum (* b rho)) 0)))

(defn- no-u-turn?
  "The generalized no-U-turn criterion (Betancourt 2017, appendix A) for
   the subtrees a and b joined, b built on from a's :end, rho their summed
   momenta, with Stan's extra checks across the join: a's first point with
   b's, its momenta summed with b's first momentum, and a's last point with
   b's, b's momenta summed with a's last momentum. One host read."
  [a b rho]
  (array/truthy?
   (array/logical-and (continuing? (:sharp (:beg a)) (:sharp (:end b)) rho)
                      (array/logical-and (continuing? (:sharp (:beg a)) (:sharp (:beg b)) (+ (:rho a) (:p (:beg b))))
                                         (continuing? (:sharp (:end a)) (:sharp (:end b)) (+ (:rho b) (:p (:end a))))))))

(defn- subtree
  "Stan's build_tree: 2^depth leapfrog steps (step, of one signed step
   size) from the point z, h0 the trajectory's initial energy. Returns
   {:beg :end :rho :log-weight :proposal :valid? :diverged? :n-leapfrog
   :accept-sum}: its first and last points, its summed momenta, the log of
   its points' summed weights e^(h0 - h), a point drawn from them
   multinomially (between two halves in proportion to their weights), and
   whether it is valid: no step diverged (an energy error over
   max-energy-error, or a non-finite step) and neither it nor a subtree in
   it turned. The counts cover every step taken, an invalid subtree's too,
   and the acceptance sum adds each step's min(1, e^(h0 - h)). An invalid
   half ends the subtree, which stops there."
  [step z depth h0 key]
  (if (zero? (long depth))
    (let [z (step z)
          log-weight (- h0 (:h z))
          diverged? (clojure.core/> (- (:h z) h0) max-energy-error)]
      {:beg z :end z :rho (:p z) :log-weight log-weight :proposal z
       :valid? (not diverged?) :diverged? diverged? :n-leapfrog 1 :accept-sum (Math/min 1.0 (Math/exp log-weight))})
    (let [[k-init k-final k-pick] (rng/split key 3)
          init (subtree step z (dec (long depth)) h0 k-init)]
      (if-not (:valid? init)
        init
        (let [final (subtree step (:end init) (dec (long depth)) h0 k-final)
              counts {:n-leapfrog (+ (:n-leapfrog init) (:n-leapfrog final))
                      :accept-sum (+ (:accept-sum init) (:accept-sum final))}]
          (if-not (:valid? final)
            (merge final counts)
            (let [log-weight (array/logaddexp (:log-weight init) (:log-weight final))
                  rho (+ (:rho init) (:rho final))]
              (merge counts
                     {:beg (:beg init) :end (:end final) :rho rho :log-weight log-weight
                      :proposal (if (clojure.core/< (uniform k-pick) (Math/exp (- (:log-weight final) log-weight)))
                                  (:proposal final)
                                  (:proposal init))
                      :valid? (no-u-turn? init final rho) :diverged? false}))))))))

(defn- trajectory
  "Stan's base_nuts transition from the point z0 with step (a function of
   the signed step size and a point): the trajectory doubles in a random
   direction until it turns, a subtree is invalid or it reaches max-depth.
   Each valid new subtree's proposal replaces the sample with probability
   min(1, its weight over the old trajectory's), biased progressive
   sampling. Returns {:sample :tree-depth :n-leapfrog :accept-sum
   :diverged?}."
  [step z0 max-depth key]
  (let [h0 (:h z0)]
    (loop [depth 0
           traj {:bck z0 :fwd z0 :rho (:p z0) :log-weight 0.0}
           sample z0
           n 0
           accept-sum 0.0]
      (let [done (fn [depth diverged?] {:sample sample :tree-depth depth :n-leapfrog n :accept-sum accept-sum
                                        :diverged? diverged?})]
        (if (= depth max-depth)
          (done depth false)
          (let [[k-dir k-tree k-pick] (rng/split (rng/fold-in key depth) 3)
                forward? (clojure.core/> (uniform k-dir) 0.5)
                ;; the trajectory, oriented so that its :end is the end it grows from
                a (if forward?
                    {:beg (:bck traj) :end (:fwd traj) :rho (:rho traj)}
                    {:beg (:fwd traj) :end (:bck traj) :rho (:rho traj)})
                b (subtree (partial step (if forward? 1.0 -1.0)) (:end a) depth h0 k-tree)
                n (clojure.core/+ n (long (:n-leapfrog b)))
                accept-sum (clojure.core/+ accept-sum (double (:accept-sum b)))]
            (if-not (:valid? b)
              (assoc (done depth (:diverged? b)) :n-leapfrog n :accept-sum accept-sum)
              (let [sample (if (clojure.core/< (uniform k-pick) (Math/exp (- (:log-weight b) (:log-weight traj))))
                             (:proposal b)
                             sample)
                    rho (+ (:rho traj) (:rho b))
                    traj {:bck (if forward? (:bck traj) (:end b)) :fwd (if forward? (:end b) (:fwd traj))
                          :rho rho :log-weight (array/logaddexp (:log-weight traj) (:log-weight b))}]
                (if (no-u-turn? a b rho)
                  (recur (inc depth) traj sample n accept-sum)
                  {:sample sample :tree-depth (inc depth) :n-leapfrog n :accept-sum accept-sum :diverged? false})))))))))

(defn nuts
  "One transition of the No-U-Turn Sampler on the selected choices, Stan's
   (base_nuts.hpp; Betancourt 2017): from momenta drawn as hmc's, the
   trajectory doubles forwards or backwards at random, by leapfrog steps of
   size eps, until it makes a U-turn by the generalized criterion (with
   Stan's extra checks across merged subtrees), a step diverges, or it
   holds 2^max-depth - 1 steps. The next state is drawn from the
   trajectory's points in proportion to e^-energy: multinomially within a
   subtree, and between each new subtree and the old trajectory by biased
   progressive sampling, which favours the new subtree. A subtree that
   diverges or turns within itself is left out. The third argument is an
   options map or a key, and a key may follow the options. Options: :eps
   (0.1), :max-depth (10), :inv-metric, :jitter (0, as hmc's), :key,
   :compile? (false): run each leapfrog step as one compiled function (as
   hmc's, the tree's logic on the host). The trace is updated once, at the
   end.

   Returns {:trace :accept-stat :tree-depth :n-leapfrog :diverged? :energy
   :eps}, Stan's statistics: the mean over every step taken of
   min(1, e^(h0 - h)) (the statistic step-size adaptation targets), the
   number of doublings kept, the leapfrog steps (each one gradient, plus
   one at the start), whether a step diverged (an energy error over 1000,
   or a non-finite step), the energy of the state drawn and the step size
   taken. No divergence does not mean the target was explored (see
   gen.inference.nuts): read gen.inference.diagnostics too."
  ([trace selection] (nuts trace selection nil))
  ([trace selection key-or-opts] (let [opts (step-opts key-or-opts)] (nuts trace selection opts (:key opts))))
  ([trace selection opts key]
   (let [{:keys [eps max-depth compile? inv-metric jitter] :or {eps 0.1 max-depth 10 jitter 0}} (options opts)
         _ (when-not (and (number? eps) (Double/isFinite (double eps)) (pos? (double eps)))
             (throw (ex-info (str "nuts: :eps must be a positive number, not " (pr-str eps)) {:reason :invalid-eps :eps eps})))
         eps (jittered eps jitter key)
         _ (when-not (pos-int? max-depth)
             (throw (ex-info (str "nuts: :max-depth must be a positive integer, not " (pr-str max-depth))
                             {:reason :invalid-max-depth :max-depth max-depth})))
         [k-momenta k-tree k-update] (rng/split key 3)
         [proto v others] (flat-values trace selection)
         inv (inverse-metric inv-metric v)
         [lp g] ((array/value-and-grad (log-density (:gen-fn trace) (:args trace) proto)) v others)
         _ (check-start! "nuts" (finite lp g) trace selection)
         p (/ (rng/normal k-momenta (array/shape v)) (array/sqrt inv))
         sharp (* inv p)
         z0 {:v v :p p :g g :sharp sharp :h (double (array/item (energy lp p sharp)))}
         build #(nuts-step (:gen-fn trace) (:args trace) proto)
         plain (build)
         run (if compile?
               (let [f (compiled trace proto others [:nuts] build)]
                 (fn [& args] (or (apply f args) (apply plain args))))
               plain)
         step (fn [direction z]
                (let [[v p g sharp h] (run (:v z) (:p z) (:g z) (* direction (double eps)) inv others)
                      h (double (array/item h))]
                  ;; a NaN energy is an infinite one (Stan's)
                  {:v v :p p :g g :sharp sharp :h (if (Double/isNaN h) ##Inf h)}))
         {:keys [sample tree-depth n-leapfrog accept-sum diverged?]} (trajectory step z0 max-depth k-tree)]
     {:trace (if (identical? sample z0) trace (first (move-to trace proto (:v sample) k-update)))
      :accept-stat (/ (double accept-sum) (long n-leapfrog))
      :tree-depth tree-depth :n-leapfrog n-leapfrog :diverged? diverged? :energy (:h sample) :eps eps})))

(defn- map-move
  "map-optimize's line search as a function of arrays only, so that it runs
   batched over particles too, each with its own step size: {:values the
   selected choices it moves to, :moved?}. The search ends when every
   particle has its step, read across the batch (array/all-true?)."
  [trace selection {:keys [max-step-size tau min-step-size] :or {max-step-size 0.1 tau 0.5 min-step-size 1e-16}}]
  (let [[proto v others] (flat-values trace selection)
        density (log-density (:gen-fn trace) (:args trace) proto)
        g ((array/grad density) v others)
        score (:score trace)]
    (loop [step max-step-size to v moved false done false]
      (let [candidate (+ v (* step g))
            better (>= (density candidate others) score)
            taken (array/logical-and better (array/logical-not done))
            to (array/where taken candidate to)
            moved (array/logical-or moved taken)
            ;; a step below the least is tried before the search gives up
            done (array/logical-or done (array/logical-or better (< step min-step-size)))]
        (if (array/all-true? done)
          {:values (cm/from-array proto to) :moved? moved}
          (recur (array/where done step (* step tau)) to moved done))))))

(defn map-optimize
  "One step of gradient ascent on the log density of the selected choices,
   with backtracking: try the step size max-step-size, shrinking it by tau
   until the score does not decrease (Gen.jl's map_optimize). Options:
   :max-step-size (0.1), :tau (0.5), :min-step-size (1e-16)."
  ([trace selection] (map-optimize trace selection nil))
  ([trace selection opts]
   (let [{:keys [values moved?]} (map-move trace selection opts)]
     (if (array/truthy? moved?) (:trace (gfi/update trace values (rng/key 0))) trace))))

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

;; ---------------------------------------------------------------------------
;; Every particle at once
;; ---------------------------------------------------------------------------

(defn- uncompiled!
  "A -particles kernel's options, which cannot ask for :compile?: a batched
   step runs under the batch tracer, which array/compile does not see
   through."
  [kernel opts]
  (when (:compile? opts)
    (throw (ex-info (str kernel " does not compile its steps: they run under the batch tracer") {:kernel kernel})))
  opts)

(defn- check-starts!
  "check-start! for every particle, start-ok an [n] bool array."
  [kernel start-ok selection]
  (when-not (array/all-true? start-ok)
    (let [bad (vec (keep-indexed #(when-not %2 %1) (array/->clj start-ok)))]
      (throw (ex-info (str kernel ": the log density or its gradient at the selection is not finite at the start of"
                           " particles " bad ", so no move from them can be weighed")
                      {:reason :non-finite-start :kernel (keyword kernel) :particles bad :selection selection})))))

(defn- particle-moves
  "Every particle's move (move, of a trace and a key, as hmc-move) at once,
   by gen.batched/map-particles, each accepted on its own: {:particles,
   :accepted? and :diverged? (n booleans each), :log-alpha (an [n] array)},
   and :eps (an [n] array) when the moves report their step sizes."
  [kernel ps selection move key]
  (let [{:keys [values accepted? log-alpha diverged? start-ok eps]} (batched/map-particles ps move key)
        accepted (mapv true? (array/->clj accepted?))]
    (check-starts! kernel start-ok selection)
    (cond-> {:particles (if (some true? accepted) (batched/set-choices ps values) ps)
             :accepted? accepted
             :log-alpha log-alpha
             :diverged? (mapv true? (array/->clj diverged?))}
      eps (assoc :eps eps))))

(defn hmc-particles
  "hmc on every particle of a gen.batched collection, n independent chains
   each accepted on its own: one batched gradient per leapfrog step for all
   of them, where the model runs batched. Arguments as hmc's, without
   :compile?; with :jitter each chain's step size is jittered by its own u,
   as its own hmc's would be (DESIGN §9, PROVISIONAL). Returns {:particles,
   :accepted? and :diverged? (n booleans each), :log-alpha (an [n] array)},
   and with :jitter :eps, the steps taken (an [n] array); a start that is
   not finite is an error, as hmc's."
  ([ps selection] (hmc-particles ps selection nil))
  ([ps selection key-or-opts] (let [opts (step-opts key-or-opts)] (hmc-particles ps selection opts (:key opts))))
  ([ps selection opts key]
   (let [{:keys [eps jitter] :or {eps 0.1 jitter 0} :as opts} (uncompiled! "hmc-particles" (options opts))
         jitter (double (check-jitter! jitter))
         ;; jittered's step, its u drawn per particle (an array, not read)
         move (fn [t k]
                (if (zero? jitter)
                  (hmc-move t selection opts k)
                  (let [step (jitter-step eps jitter (rng/uniform (rng/fold-in k 3)))]
                    (assoc (hmc-move t selection (assoc opts :eps step) k) :eps step))))]
     (particle-moves "hmc" ps selection move key))))

(defn mala-particles
  "mala on every particle of a gen.batched collection, as hmc-particles:
   one batched gradient at the start and one at the proposal for all of
   them. Arguments as mala's, without :compile?; returns as hmc-particles."
  ([ps selection tau] (mala-particles ps selection tau nil))
  ([ps selection tau key-or-opts]
   (let [{:keys [key] :as opts} (uncompiled! "mala-particles" (step-opts key-or-opts))]
     (particle-moves "mala" ps selection #(mala-move %1 selection tau opts %2) key))))

(defn map-optimize-particles
  "map-optimize on every particle of a gen.batched collection, each with
   its own line search: one batched gradient, and one batched score per
   step size tried, for all of them where the model runs batched. Options
   as map-optimize's. Returns the particles."
  ([ps selection] (map-optimize-particles ps selection nil))
  ([ps selection opts]
   (let [{:keys [values moved?]} (batched/map-particles ps (fn [t _] (map-move t selection opts)) nil)]
     (if (array/any-true? moved?) (batched/set-choices ps values) ps))))
