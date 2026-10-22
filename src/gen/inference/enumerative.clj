(ns gen.inference.enumerative
  "Exact inference by enumeration: Gen.jl's enumerative_inference and
   choice_vol_grid, and generative functions whose internal proposal draws
   finite choices from their exact conditional.

     (choice-vol-grid [addr vals] [addr ends :continuous] … {:anchor :midpoint})
       → [[choices log-volume] …]
     (enumerative-inference model args observations grid)
       → {:traces :log-normalized-weights :log-ml-estimate}
     (marginal result addr)              → {value probability}
     (with-exact-proposal gf addrs)      → a generative function

   enumerative-inference generates the model once per grid point,
   constrained by the observations and the point's choices, and weights each
   trace by generate's weight plus the point's log-volume. Normalized, the
   weights are the posterior probabilities of the points (of the regions
   they stand for, on continuous grids). log-ml-estimate is log Σ exp(w):
   exact when the grid is discrete and covers every unobserved choice.
   Choices the grid leaves out are drawn by the model's internal proposal.

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
            [gen.choicemap :as cm]
            [gen.diff :as diff]
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

(defn enumerative-inference
  "Gen.jl's enumerative_inference: model generated once per point of grid
   (choice-vol-grid, or any collection of [choices log-volume] pairs),
   constrained by the observations and the point's choices, which must not
   overlap them. Returns the traces in grid order, their normalized log
   weights (generate's weight plus the log-volume) and log-ml-estimate,
   log Σ exp(weight). Options: :key, for choices the grid leaves out."
  ([model args observations grid] (enumerative-inference model args observations grid nil))
  ([model args observations grid {:keys [key]}]
   (let [points (vec grid)
         observed (set (cm/addresses observations))]
     (when (empty? points)
       (throw (ex-info "enumerative-inference: the grid has no points" {})))
     (let [results (mapv (fn [[choices log-volume] k]
                           (when-let [p (some observed (cm/addresses choices))]
                             (throw (ex-info (str "enumerative-inference: the grid sets the observed choice at " p)
                                             {:path p})))
                           (let [{:keys [trace weight]} (gfi/generate model args (cm/merge observations choices) k)]
                             (gfi/eval-trace! trace weight)
                             [trace (+ (double (array/item weight)) (double log-volume))]))
                         points (rng/split (or key (gfi/next-key)) (count points)))
           {:keys [log-total log-weights]} (w/normalize (mapv second results))]
       {:traces (mapv first results) :log-normalized-weights log-weights :log-ml-estimate log-total}))))

(defn marginal
  "The posterior probability of each value of the choice at addr, from an
   enumerative-inference result (or any {:traces :log-normalized-weights}):
   a map from the value (on the host) to its probability."
  [{:keys [traces log-normalized-weights]} addr]
  (reduce (fn [m [tr lw]] (update m (array/->clj (cm/get-choice (:choices tr) addr)) (fnil + 0.0) (Math/exp lw)))
          {} (map vector traces log-normalized-weights)))

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

(defn- enumerated
  "For each assignment c of the free paths, with sigma (the constraints) and
   others (the other choices): s_c, the log density of the trace, and r_c,
   the log density of gf's forward proposal for others and c given sigma
   (project on the unconstrained addresses; finite where s_c is -Inf
   because sigma is impossible under c). Every choice is given, so the
   runs are deterministic. {:assignments :s :r :retvals}, with s and r
   stacked."
  [g args sigma others free seen limit]
  (let [base (cm/merge sigma others)
        cs (assignments free seen limit)
        n-choices (count (cm/addresses base))
        unconstrained (sel/complement (sel/from-choicemap sigma))
        runs (mapv (fn [c]
                     (let [constraints (reduce (fn [m [p v]] (cm/set-choice m p v)) base c)
                           structure-error (fn [cause]
                                             (ex-info (str "with-exact-proposal: the model's structure depends on the"
                                                           " enumerated choices (under " (pr-str c) ")")
                                                      {:assignment c} cause))
                           tr (try (:trace (gfi/with-exact-density "with-exact-proposal" (gfi/generate g args constraints (rng/key 0))))
                                   (catch clojure.lang.ExceptionInfo e
                                     ;; a choice of the other run goes unused under c
                                     (throw (if (:unvisited (ex-data e)) (structure-error e) e))))]
                       (when-not (= (+ n-choices (count c)) (count (cm/addresses (:choices tr))))
                         (throw (structure-error nil)))
                       [(:score tr) (gfi/project tr unconstrained) (:retval tr)]))
                   cs)]
    {:assignments cs :s (array/stack (mapv first runs)) :r (array/stack (mapv second runs))
     :retvals (mapv #(nth % 2) runs)}))

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
        forward (fn [sigma] (let [seen (volatile! {})]
                              {:trace (:trace (gfi/-generate (watching gf (set paths) seen) args sigma k-forward))
                               :seen seen}))
        [{t0 :trace seen :seen} sigma] (if gfi/*revising* (gfi/rerun-without-drops forward sigma) [(forward sigma) sigma])
        g (watching gf (set paths) seen)
        free (free-paths paths sigma)
        _ (check-visited! free seen (:choices t0))
        others (others-of (:choices t0) sigma free)
        {cs :assignments s :s r :r rvs :retvals} (enumerated g args sigma others free seen limit)
        _ (when check-retval? (check-retval! cs s rvs))
        idx (rng/categorical k-pick s)
        chosen (reduce (fn [m p] (cm/set-choice m p (array/take (array/array (mapv #(get % p) cs)) idx)))
                       (cm/merge sigma others) free)
        tr (:trace (gfi/generate gf args chosen k-final))]
    {:trace (trace/as-made-by tr egf) :weight (- (array/logsumexp s) (array/logsumexp r))}))

(defn- exact-weight
  "The weight exact-generate gives for producing the trace tr of gf on args
   from sigma, tr's choices at some addresses."
  [{:keys [gf paths limit]} args sigma tr]
  (let [seen (volatile! {})
        g (watching gf (set paths) seen)
        ;; the supports, from a run with every choice given
        _ (gfi/with-exact-density "with-exact-proposal" (gfi/generate g args (:choices tr) (rng/key 0)))
        free (free-paths paths sigma)
        _ (check-visited! free seen (:choices tr))
        {s :s r :r} (enumerated g args sigma (others-of (:choices tr) sigma free) free seen limit)]
    (- (array/logsumexp s) (array/logsumexp r))))

(defn- same-addresses? [x y] (= (set (cm/addresses x)) (set (cm/addresses y))))

(gfi/defgf ExactProposalGF [gf paths limit check-retval?]
  gfi/IGenerativeFunction
  (-simulate [this args key] (trace/as-made-by (gfi/-simulate gf args key) this))
  (-generate [this args constraints key] (exact-generate this args constraints key))
  (-update [this tr args argdiffs constraints key]
    ;; gf's update keeps, overwrites and removes choices; fresh ones are
    ;; then drawn by the exact proposal given the kept ones
    (let [[k-update k-fresh] (rng/split key)
          {t2 :trace :as r} (gfi/-update gf (trace/as-made-by tr gf) args argdiffs constraints k-update)
          before (set (cm/addresses (:choices tr)))
          given (set (cm/addresses constraints))
          fresh (vec (remove #(or (before %) (given %)) (cm/addresses (:choices t2))))]
      (if (empty? fresh)
        (assoc r :trace (trace/as-made-by t2 this))
        (let [kept (sel/get-selected (:choices t2) (sel/complement (sel/from-paths fresh)))
              {t3 :trace w :weight} (exact-generate this args kept k-fresh)]
          (when-not (same-addresses? (:choices t2) (:choices t3))
            (throw (ex-info "with-exact-proposal: the fresh choices of this update change the model's structure" {})))
          {:trace t3 :weight (- w (:score tr)) :retdiff diff/unknown-change :discard (:discard r)}))))
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
   - :max-assignments (100000)."
  ([gf addrs] (with-exact-proposal gf addrs nil))
  ([gf addrs {:keys [max-assignments check-retval?] :or {max-assignments 100000}}]
   (when-not (dyn/dynamic-gf? gf)
     (throw (ex-info "with-exact-proposal needs a dynamic generative function" {:gen-fn (type gf)})))
   (when (empty? addrs)
     (throw (ex-info "with-exact-proposal needs the addresses of the choices to enumerate" {})))
   (let [paths (mapv #(if (vector? %) % [%]) addrs)]
     (when-not (apply distinct? paths)
       (throw (ex-info "with-exact-proposal: an address appears twice" {:addresses addrs})))
     (->ExactProposalGF gf paths max-assignments (boolean check-retval?)))))

(defmethod print-method ExactProposalGF [g ^java.io.Writer wr]
  (.write wr (str "#gen/with-exact-proposal[" (pr-str (:gf g)) " " (pr-str (:paths g)) "]")))
