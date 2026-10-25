(ns gen.inference.enumerative
  "Exact inference by enumeration: Gen.jl's enumerative_inference and
   choice_vol_grid, and generative functions whose internal proposal draws
   finite choices from their exact conditional.

     (choice-vol-grid [addr vals] [addr ends :continuous] … {:anchor :midpoint})
       → [[choices log-volume] …]
     (enumerative-inference model args observations grid)
       → {:traces :log-normalized-weights :log-ml-estimate}
     (enumerative-inference model args observations grid {:batched? true})
       → {:batches :log-normalized-weights :log-ml-estimate}
     (traces result)                     → the scalar traces, batched or not
     (marginal result addr)              → {value probability}
     (with-exact-proposal gf addrs)      → a generative function

   enumerative-inference generates the model once per grid point,
   constrained by the observations and the point's choices, and weights each
   trace by generate's weight plus the point's log-volume. Normalized, the
   weights are the posterior probabilities of the points (of the regions
   they stand for, on continuous grids). log-ml-estimate is log Σ exp(w):
   exact when the grid is discrete and covers every unobserved choice.
   Choices the grid leaves out are drawn by the model's internal proposal.
   With {:batched? true} the grid's points are the particles of one batched
   run (gen.batched), each constrained by its own point (per-particle
   constraints), in chunks of :chunk-size points if given: point by point
   the same weights when the grid and the observations give every choice.

   with-exact-proposal keeps gf's distribution and traces and changes only
   its internal proposal (thesis pp. 145-146): generate draws the finite
   choices at addrs from their exact conditional given everything else, by
   enumerating them. Its weight is log p(observations) exactly when every
   other choice is observed. regenerate on those choices is a Gibbs move
   (weight 0); it keeps every unselected choice, and a selection that would
   leave one unvisited is a loud error (gen.inference.proposal). It
   satisfies the general GFI laws (gen.laws), not the forward-sampling
   ones.

   Inside an Unfold or a Map, each step's weight is exact given the step's
   arguments. So the total is the exact likelihood, with the enumerated
   choices summed out step by step (and a particle filter's increments are
   exact), only when no step's return value depends on its enumerated
   choices: an Unfold's next state must come from observed choices alone.
   Otherwise later steps condition on the choices drawn, and the total is a
   sequential importance weight, unbiased in exp space (the locally optimal
   proposal) but not exact: it changes with the key. {:check-retval? true}
   makes that case an error.

   Keys are an optional last argument (or :key in an options map)."
  (:refer-clojure :exclude [-])
  (:require [gen.array :as array :refer [-]]
            [gen.batched :as batched]
            [gen.choicemap :as cm]
            [gen.dist.core :as dc]
            [gen.dynamic :as dyn]
            [gen.gfi :as gfi]
            [gen.inference.proposal :as proposal]
            [gen.inference.weights :as w]
            [gen.random :as rng]
            [gen.selection :as sel]
            [gen.trace :as trace]))

(set! *warn-on-reflection* true)

(defn- product
  "The Cartesian product of colls as vectors, the last varying fastest."
  [colls]
  (reduce (fn [acc coll] (vec (for [xs acc x coll] (conj xs x)))) [[]] colls))

;; ---------------------------------------------------------------------------
;; Grids
;; ---------------------------------------------------------------------------

(defn- intervals
  "[anchor log-width] of each interval between consecutive endpoints."
  [ends anchor]
  (let [xs (mapv double ends)]
    (when-not (and (>= (count xs) 2) (apply < xs))
      (throw (ex-info "choice-vol-grid: a continuous choice needs two or more increasing interval endpoints"
                      {:endpoints ends})))
    (mapv (fn [lo hi]
            [(case anchor :left lo :right hi :midpoint (+ lo (/ (clojure.core/- hi lo) 2.0)))
             (Math/log (clojure.core/- hi lo))])
          xs (rest xs))))

(defn- grid-axis
  "One spec's [value log-volume] pairs."
  [[addr vals support dims :as spec] anchor]
  (let [support (or support :discrete)
        dims (or dims 1)]
    (when (or (< (count spec) 2) (> (count spec) 4) (nil? addr))
      (throw (ex-info "choice-vol-grid: a spec is [addr vals], [addr vals support] or [addr vals :continuous dims]"
                      {:spec spec})))
    (case support
      :discrete (do (when (empty? vals)
                      (throw (ex-info (str "choice-vol-grid: no values for " (pr-str addr)) {:spec spec})))
                    (mapv (fn [v] [v 0.0]) vals))
      :continuous (if (= 1 dims)
                    (intervals vals anchor)
                    (do (when-not (= dims (count vals))
                          (throw (ex-info (str "choice-vol-grid: " dims " dimensions need " dims " endpoint vectors")
                                          {:spec spec})))
                        (mapv (fn [corner] [(array/array (mapv first corner)) (reduce + (map second corner))])
                              (product (mapv #(intervals % anchor) vals)))))
      (throw (ex-info "choice-vol-grid: the support must be :discrete or :continuous" {:spec spec})))))

(defn choice-vol-grid
  "Gen.jl's choice_vol_grid: the points of a grid over choices, each a
   [choices log-volume] pair. A spec is [addr vals], [addr vals support] or
   [addr vals :continuous dims]:
   - :discrete (the default): each of vals, with log-volume 0;
   - :continuous: the intervals between consecutive endpoints vals, each at
     its anchor, with the log of its width;
   - :continuous with dims n > 1: vals holds n endpoint vectors, and the
     points are the anchors of the boxes of their product (array values),
     with the log of each box's volume.
   A point's log-volume is the sum of its specs'. A trailing options map
   takes :anchor (:left, :right or :midpoint, the default). Points come in
   row-major order: the last spec varies fastest (Gen.jl's iterator varies
   the first fastest)."
  [& specs]
  (let [[specs {:keys [anchor] :or {anchor :midpoint}}] (if (map? (last specs))
                                                          [(butlast specs) (last specs)]
                                                          [specs nil])
        addrs (map first specs)]
    (when-not (#{:left :right :midpoint} anchor)
      (throw (ex-info "choice-vol-grid: :anchor is :left, :right or :midpoint" {:anchor anchor})))
    (when (empty? specs)
      (throw (ex-info "choice-vol-grid needs at least one spec" {})))
    (when-not (apply distinct? (map #(if (vector? %) % [%]) addrs))
      (throw (ex-info "choice-vol-grid: an address appears twice" {:addresses (vec addrs)})))
    (mapv (fn [points]
            [(reduce (fn [c [addr [v]]] (cm/set-choice c addr v)) cm/EMPTY (map vector addrs points))
             (reduce + (map second points))])
          (product (mapv #(grid-axis % anchor) specs)))))

;; ---------------------------------------------------------------------------
;; Enumerative inference
;; ---------------------------------------------------------------------------

(defn- check-points!
  "Throw unless the grid has points, and none sets an observed choice."
  [points observations]
  (when (empty? points)
    (throw (ex-info "enumerative-inference: the grid has no points" {})))
  (let [observed (set (cm/addresses observations))]
    (doseq [[choices] points]
      (when-let [p (some observed (cm/addresses choices))]
        (throw (ex-info (str "enumerative-inference: the grid sets the observed choice at " p) {:path p}))))))

(defn- one-at-a-time
  "The points generated one at a time: {:traces :weights}, each weight
   generate's plus the point's log-volume."
  [model args observations points key]
  (let [results (mapv (fn [[choices log-volume] k]
                        (let [{:keys [trace weight]} (gfi/generate model args (cm/merge observations choices) k)]
                          (gfi/eval-trace! trace weight)
                          [trace (+ (double (array/item weight)) (double log-volume))]))
                      points (rng/split key (count points)))]
    {:traces (mapv first results) :weights (mapv second results)}))

(defn- in-batches
  "The points generated in batched runs of at most chunk-size particles,
   particle i constrained by point i's choices: {:batches :weights}, one
   particle collection per run."
  [model args observations points chunk-size key]
  (let [chunks (mapv vec (partition-all chunk-size points))
        runs (mapv (fn [chunk k]
                     (let [{ps :particles ws :weights}
                           (batched/generate model args observations (count chunk) k {:per-particle (mapv first chunk)})]
                       [ps (mapv #(+ (double %1) (double %2)) (array/->clj ws) (map second chunk))]))
                   chunks (rng/split key (count chunks)))]
    {:batches (mapv first runs) :weights (into [] (mapcat second) runs)}))

(defn enumerative-inference
  "Gen.jl's enumerative_inference: model generated once per point of grid
   (choice-vol-grid, or any collection of [choices log-volume] pairs),
   constrained by the observations and the point's choices, which must not
   overlap them. Returns the traces in grid order, their normalized log
   weights (generate's weight plus the log-volume) and log-ml-estimate,
   log Σ exp(weight). Options:
   - :key, for choices the grid leaves out;
   - :batched? (false): the points are particles of one batched run
     (gen.batched), each constrained by its own point, so the handler's
     cost is paid once for the grid. The result holds :batches, particle
     collections in grid order, in place of :traces; traces gives the
     scalar traces either way. The weights are the scalar loop's, point by
     point, when the observations and the grid give every choice (a model
     that cannot run batched, one that branches on a grid value say, falls
     back to one point at a time); choices the grid leaves out are drawn
     from the same distribution, but not as the scalar loop draws them;
   - :chunk-size: with :batched?, at most this many points per batched run
     (and one collection per run), so that a run's working memory is that
     many particles' (the result holds every point's trace, as the scalar
     loop's does). The weights are those of one run when the grid and the
     observations give every choice; choices the grid leaves out are drawn
     with a key for each run."
  ([model args observations grid] (enumerative-inference model args observations grid nil))
  ([model args observations grid {:keys [key batched? chunk-size]}]
   (let [points (vec grid)
         key (or key (gfi/next-key))]
     (check-points! points observations)
     (when-not (or (nil? chunk-size) (pos-int? chunk-size))
       (throw (ex-info "enumerative-inference: :chunk-size is a positive integer" {:chunk-size chunk-size})))
     (let [run (if batched?
                 (in-batches model args observations points (or chunk-size (count points)) key)
                 (one-at-a-time model args observations points key))
           {:keys [log-total log-weights]} (w/normalize (:weights run))]
       (-> run
           (dissoc :weights)
           (assoc :log-normalized-weights log-weights :log-ml-estimate log-total))))))

(defn traces
  "The scalar traces of an enumerative-inference result in grid order,
   batched or not."
  [result]
  (if-let [batches (:batches result)]
    (into [] (mapcat batched/traces) batches)
    (:traces result)))

(defn- choice-values
  "The host values of the choice at addr in a result's traces, in grid
   order: a batched collection's read at once."
  [{:keys [traces batches]} addr]
  (let [of-traces (fn [trs] (map #(array/->clj (cm/get-choice (:choices %) addr)) trs))]
    (if batches
      (into [] (mapcat #(if (batched/batched? %) (array/->clj (batched/choice % addr)) (of-traces (batched/traces %))))
            batches)
      (vec (of-traces traces)))))

(defn marginal
  "The posterior probability of each value of the choice at addr, from an
   enumerative-inference result, batched or not (or any {:traces
   :log-normalized-weights}): a map from the value (on the host) to its
   probability."
  [result addr]
  (reduce (fn [m [v lw]] (update m v (fnil + 0.0) (Math/exp lw)))
          {} (map vector (choice-values result addr) (:log-normalized-weights result))))

;; ---------------------------------------------------------------------------
;; Exact proposals
;; ---------------------------------------------------------------------------

(defn- watching
  "gf whose generate records in seen (a volatile map) the support of the
   distribution at each of paths, and checks that every run finds the same."
  [gf paths seen]
  (let [t (or (get (:transitions gf) :generate) (dyn/site-transitions :generate))]
    (dyn/with-transitions gf
      (assoc (:transitions gf) :generate
             (fn [state p dist key]
               (when (contains? paths p)
                 (let [s (try (dc/support dist)
                              (catch clojure.lang.ExceptionInfo e
                                (throw (ex-info (str "with-exact-proposal: the choice at " p " is not finite ("
                                                     (ex-message e) ")")
                                                {:path p} e))))]
                   (if-let [s0 (get @seen p)]
                     (when (not= s0 s)
                       (throw (ex-info (str "with-exact-proposal: the support of the choice at " p
                                            " depends on other choices")
                                       {:path p :supports [s0 s]})))
                     (vswap! seen assoc p s))))
               (t state p dist key))))))

(defn- check-visited! [free seen choices]
  (doseq [p free]
    (when-not (contains? @seen p)
      (throw (ex-info (str "with-exact-proposal: the choice at " p
                           (if (cm/has-value? choices p)
                             " is made by a generative function the model calls at an address; wrap that one"
                             " was not visited"))
                      {:path p})))))

(defn- assignments
  "Every assignment of values to the free paths, as maps from path to value
   (row-major)."
  [free seen limit]
  (let [supports (mapv @seen free)
        n (reduce * 1 (map count supports))]
    (when (> n limit)
      (throw (ex-info (str "with-exact-proposal: " n " assignments to enumerate (the limit is " limit
                           "; raise :max-assignments)")
                      {:assignments n :limit limit})))
    (mapv #(zipmap free %) (product supports))))

(defn- structure-error [c cause]
  (ex-info (str "with-exact-proposal: the model's structure depends on the enumerated choices (under " (pr-str c) ")")
           {:assignment c} cause))

(defn- assigned
  "The choice map m with the assignment c (a map from path to value)."
  [m c]
  (reduce (fn [m [p v]] (cm/set-choice m p v)) m c))

(defn- checked
  "tr, the run under the assignment c, unless it made other choices than
   the n given, or (a run stopped at a choice outside its support,
   gfi/impossible?) others than those given: unconstrained, the
   unconstrained ones of those (base's less sigma's, and c's), must all be
   there unless one of them stopped it (its projection on them -Inf), or
   the forward proposal's density r of them is not known (a constraint
   inside a call that stopped the run, which unweighing cannot reach)."
  [tr c n given unconstrained]
  (let [made (cm/addresses (:choices tr))]
    (if (gfi/impossible? (:score tr))
      (do (when-not (every? #(cm/has-value? given %) made)
            (throw (structure-error c nil)))
          (when-not (or (every? #(cm/has-value? (:choices tr) %) unconstrained)
                        (gfi/impossible? (gfi/project tr (sel/from-paths unconstrained))))
            (throw (ex-info (str "with-exact-proposal: under " (pr-str c) " a constraint inside a call is outside its"
                                 " support, so the forward proposal's density of the choices after it is unknown:"
                                 " observe it at the model's own address")
                            {:assignment c}))))
      (when-not (= n (count made))
        (throw (structure-error c nil)))))
  tr)

;; A choice of sigma, the constraints, with log density 0: the forward
;; proposal's own density, which weighs only the other choices. A run stops
;; at a choice outside its support (DESIGN.md §9), but the forward
;; proposal's density of the other choices is defined where sigma is
;; impossible, and the enumeration needs it there (r_c of an assignment c
;; under which sigma is impossible: its share of logsumexp(r)).
(defmethod dc/-log-prob ::unweighed [_ _] 0.0)
(defmethod dc/-sample ::unweighed [d key batch-shape] (dc/-sample (:of (:params d)) key batch-shape))
(defmethod dc/-value-shape ::unweighed [d] (dc/value-shape (:of (:params d))))
(defmethod dc/-discrete? ::unweighed [d] (dc/discrete? (:of (:params d))))
(defmethod dc/-support ::unweighed [d] (dc/enumerable-support (:of (:params d))))

(defn- unweighing
  "g, a dynamic generative function, whose generate gives each of its own
   choices that sigma constrains log density 0, and adds the choice's own
   to lps (a volatile number): the run's score is then the trace's less
   those, which lps holds, and a choice of sigma outside its support stops
   no run. (sigma's choices in a call keep their densities.)"
  [g sigma lps]
  (let [t (or (get (:transitions g) :generate) (dyn/site-transitions :generate))]
    (dyn/with-transitions g
      (assoc (:transitions g) :generate
             (fn [state p dist key]
               (if (cm/has-value? sigma p)
                 (let [lp (dc/log-prob dist (cm/get-choice sigma p))]
                   (vswap! lps array/+ (if (pos? (array/ndim lp)) (array/sum lp) lp))
                   (t state p (dc/->Distribution ::unweighed {:of dist}) key))
                 (t state p dist key)))))))

(defn- one-run-each
  "s, r and the return values of g on args, one run under each assignment
   of cs with the other choices base, sigma's unweighed (unweighing):
   checked to make the n given choices, the unconstrained ones (base's less
   sigma's, and the free paths) all of them unless the run stopped there."
  [g args base sigma free cs n]
  (let [unconstrained (into (vec (remove #(cm/has-value? sigma %) (cm/addresses base))) free)
        runs (mapv (fn [c]
                     (let [lps (volatile! 0.0)
                           given (assigned base c)
                           tr (try (:trace (gfi/with-exact-density "with-exact-proposal"
                                             (gfi/generate (unweighing g sigma lps) args given (rng/key 0))))
                                   (catch clojure.lang.ExceptionInfo e
                                     ;; a choice of the other run goes unused under c
                                     (throw (if (:unvisited (ex-data e)) (structure-error c e) e))))]
                       (checked tr c n given unconstrained)
                       {:s (array/+ (:score tr) @lps) :r (gfi/project tr (sel/from-paths unconstrained)) :retval (:retval tr)}))
                   cs)]
    {:assignments cs :s (array/stack (mapv :s runs)) :r (array/stack (mapv :r runs)) :retvals (mapv :retval runs)}))

(defn- one-batch
  "g's particles on args, one per assignment of cs (per-particle
   constraints) with the other choices base, from one batched run; nil when
   it fails, for the runs one at a time to raise the error, naming the
   assignment. Inside a batched run, where no other can nest, gen.batched
   runs them one at a time, under that run's tracer."
  [g args base cs]
  (try (:particles (gfi/with-exact-density "with-exact-proposal"
                     (batched/generate g args base (count cs) (rng/key 0) {:per-particle (mapv #(assigned cm/EMPTY %) cs)})))
       (catch clojure.lang.ExceptionInfo _ nil)))

(defn- enumerated
  "For each assignment c of the free paths, with sigma (the constraints) and
   others (the other choices): s_c, the log density of the trace, and r_c,
   the log density of gf's forward proposal for others and c given sigma
   (project on the unconstrained addresses; finite where s_c is -Inf
   because sigma is impossible under c, which a run that stops there does
   not reach: unweighing). Every choice is given, so the runs are
   deterministic. {:assignments :s :r :retvals}, with s and r stacked.
   batched? enumerates in one batched run, without :retvals, which does not
   stop for one particle (gen.dynamic): its scores are -Inf where sigma is
   impossible, and its projections the forward proposal's. Its structure is
   then one for every assignment, so that of the run whose choices it was
   given (an assignment among them), and its choices need no check; a run
   that falls back is run again one assignment at a time."
  [g args sigma others free seen limit batched?]
  (let [base (cm/merge sigma others)
        cs (assignments free seen limit)
        n (+ (count (cm/addresses base)) (count free))
        unconstrained (sel/complement (sel/from-choicemap sigma))
        ps (when batched? (one-batch g args base cs))]
    (if (and ps (batched/batched? ps))
      {:assignments cs :s (batched/scores ps) :r (batched/project ps unconstrained)}
      (one-run-each g args base sigma free cs n))))

(defn- check-retval!
  "Throw unless the return value is the same under every assignment with
   positive probability (an opt-in check: it reads values on the host)."
  [cs s retvals]
  (let [possible (keep-indexed (fn [i ok] (when ok i)) (array/->clj (array/isfinite s)))]
    (when-let [i (some #(when (not= (retvals (first possible)) (retvals %)) %) (rest possible))]
      (throw (ex-info (str "with-exact-proposal: the return value depends on the enumerated choices ("
                           (pr-str (cs (first possible))) " and " (pr-str (cs i)) " give different ones)."
                           " A combinator passing it on (an Unfold's next state) conditions on the choices drawn,"
                           " so its weight is an unbiased estimate, not exact")
                      {:assignments [(cs (first possible)) (cs i)]})))))

(defn- others-of
  "The choices other than sigma's and the free paths'."
  [choices sigma free]
  (sel/get-selected choices (sel/complement (sel/from-paths (concat (cm/addresses sigma) free)))))

(defn- free-paths [paths sigma]
  (let [given (set (cm/addresses sigma))]
    (vec (remove given paths))))

(defn- batch-enumeration?
  "Does the exact proposal egf enumerate in one batched run? With
   :batched?, but not with :check-retval?, which reads every assignment's
   return value."
  [{:keys [batched? check-retval?]}]
  (and batched? (not check-retval?)))

(defn- exact-generate
  "generate under the exact proposal: gf's forward proposal for the choices
   other than the enumerated ones, then the enumerated ones from their exact
   conditional. The weight, log p(τ) - log q(τ ∖ σ | σ), is
   logsumexp(s) - logsumexp(r) (enumerated); it does not depend on the
   enumerated values drawn. In a revise, the forward run's calls drop the
   old choices they leave, and it repeats without them
   (gfi/rerun-without-drops); the enumeration then runs on the final
   constraints, outside the revise."
  [{:keys [gf paths limit check-retval?] :as egf} args sigma key]
  (let [[k-forward k-pick k-final] (rng/split key 3)
        ;; the forward proposal's run, sigma unweighed, so that it does not
        ;; stop where sigma is impossible under the choices it draws
        forward (fn [sigma] (let [seen (volatile! {})]
                              {:trace (:trace (gfi/-generate (unweighing (watching gf (set paths) seen) sigma (volatile! 0.0))
                                                             args sigma k-forward))
                               :seen seen}))
        [{t0 :trace seen :seen} sigma] (if gfi/*revising* (gfi/rerun-without-drops forward sigma) [(forward sigma) sigma])
        g (watching gf (set paths) seen)
        free (free-paths paths sigma)
        _ (check-visited! free seen (:choices t0))
        others (others-of (:choices t0) sigma free)
        {cs :assignments s :s r :r rvs :retvals} (enumerated g args sigma others free seen limit (batch-enumeration? egf))
        _ (when check-retval? (check-retval! cs s rvs))
        idx (rng/categorical k-pick s)
        chosen (reduce (fn [m p] (cm/set-choice m p (array/take (array/array (mapv #(get % p) cs)) idx)))
                       (cm/merge sigma others) free)
        tr (:trace (gfi/generate gf args chosen k-final))]
    {:trace (trace/as-made-by tr egf) :weight (- (array/logsumexp s) (array/logsumexp r))}))

(defn- exact-weight
  "The weight exact-generate gives for producing the trace tr of gf on args
   from sigma, tr's choices at some addresses: -Inf for a trace outside the
   support, whose run stopped at a choice there (gfi/impossible?, DESIGN.md
   §9), so that it holds its choices only up to that one."
  [{:keys [gf paths limit] :as egf} args sigma tr]
  (if (gfi/impossible? (:score tr))
    ##-Inf
    (let [seen (volatile! {})
          g (watching gf (set paths) seen)
          ;; the supports, from a run with every choice given
          _ (gfi/with-exact-density "with-exact-proposal" (gfi/generate g args (:choices tr) (rng/key 0)))
          free (free-paths paths sigma)
          _ (check-visited! free seen (:choices tr))
          {s :s r :r} (enumerated g args sigma (others-of (:choices tr) sigma free) free seen limit (batch-enumeration? egf))]
      (- (array/logsumexp s) (array/logsumexp r)))))

(gfi/defgf ExactProposalGF [gf paths limit check-retval? batched?]
  gfi/IGenerativeFunction
  (-simulate [this args key] (trace/as-made-by (gfi/-simulate gf args key) this))
  (-generate [this args constraints key] (exact-generate this args constraints key))
  (-update [this tr args argdiffs constraints key]
    (proposal/update-under "with-exact-proposal" this #(exact-generate this %1 %2 %3) tr args argdiffs constraints key))
  (-regenerate [this tr args argdiffs selection key]
    (proposal/regenerate-under #(exact-generate this %1 %2 %3) #(exact-weight this %1 %2 %3)
                               tr args argdiffs selection key))
  ;; as its generate, it needs exact densities
  (-assess [_ args choices key] (gfi/with-exact-density "with-exact-proposal" (gfi/-assess gf args choices key)))
  (-propose [_ args key] (gfi/-propose gf args key))
  (-project [this tr selection]
    (exact-weight this (:args tr) (sel/get-selected (:choices tr) selection) tr))
  ;; its traces, density and parameter stores are the model's
  trace/ICalls
  (-call [_ tr p] (trace/-call gf tr p))
  gfi/IStores
  (-stores [_] (gfi/stores gf))
  (-with-stores [this stores] (assoc this :gf (gfi/with-stores gf stores))))

(defn with-exact-proposal
  "gf with the exact conditional as its internal proposal for the finite
   choices at addrs (addresses or paths of gf's own body): the same
   distribution and traces, but generate draws those choices given every
   other choice, enumerating their supports (from the distributions). Its
   weight is then log p(observations) exactly when the other choices are
   all observed; in an Unfold or a Map, only if gf's return value does not
   depend on those choices (see the namespace doc). gf is a dynamic
   generative function whose internal proposal is forward sampling (every
   one is, except this wrapper's). The enumerated choices may not change
   the model's structure or each other's supports, which is a loud error.
   Options:
   - :check-retval? (false): throw from generate when the return value
     depends on the enumerated choices, for uses that need the weight to be
     exact (a known answer), not an estimate;
   - :max-assignments (100000);
   - :batched? (false): enumerate in one batched run (gen.batched), every
     assignment a particle constrained by it, so the handler's cost is paid
     once for all of them, at the price of a few scalar runs that check it
     (gen.batched's spot checks): for many assignments. The weights are the
     same. Inside a batched run, and with :check-retval?, it enumerates one
     assignment at a time, as without the option."
  ([gf addrs] (with-exact-proposal gf addrs nil))
  ([gf addrs {:keys [max-assignments check-retval? batched?] :or {max-assignments 100000}}]
   (when-not (dyn/dynamic-gf? gf)
     (throw (ex-info "with-exact-proposal needs a dynamic generative function" {:gen-fn (type gf)})))
   (when (empty? addrs)
     (throw (ex-info "with-exact-proposal needs the addresses of the choices to enumerate" {})))
   (let [paths (mapv #(if (vector? %) % [%]) addrs)]
     (when-not (apply distinct? paths)
       (throw (ex-info "with-exact-proposal: an address appears twice" {:addresses addrs})))
     (->ExactProposalGF gf paths max-assignments (boolean check-retval?) (boolean batched?)))))

(defmethod print-method ExactProposalGF [g ^java.io.Writer wr]
  (.write wr (str "#gen/with-exact-proposal[" (pr-str (:gf g)) " " (pr-str (:paths g)) "]")))
