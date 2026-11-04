(ns gen.inference.proposal
  "The internal-proposal override (thesis §4.4, Alg. 16, p. 152): a model's
   distribution and traces with another internal proposal.

     (with-internal-proposal gf q)

   q is a dynamic generative function called as (q constraints & args): it
   proposes some of the choices the constraints leave open, at gf's own
   addresses; gf's forward sampling makes the rest. generate then draws q's
   choices and fills in the rest, with weight

     log p(τ) - log q(τ_q | σ) - log f(τ_f | σ, τ_q),

   σ the constraints, τ_q q's choices and τ_f the forward-sampled ones, so
   importance sampling stays unbiased for p(σ) and regenerate stays
   stationary for the posterior (Eq. 4.1), whatever q is, as long as it
   proposes every value the posterior allows. The distribution, simulate
   and assess are gf's: only the internal proposal changes.

   regenerate follows Eq. 4.1 (thesis §4.1.2). The trace's unselected
   choices σ = τ|_{S^c} are the constraints: q proposes given all of them,
   and the new run τ' = υ ⊕ σ|_B keeps those it visits and drops the rest
   (B ⊆ A_σ), as when a selected choice decides that another one no longer
   exists. The reverse move's constraints are σ' = τ'|_{S^c}, which may
   hold choices the old trace lacks, and υ' is the rest of τ. With q(υ; σ)
   the internal proposal's density (q's choices' and the forward-sampled
   ones'), the weight is

     log p(τ'; x') - log q(υ; x', σ) - log p(τ; x) + log q(υ'; x, σ').

   project replays q against a trace: q runs with each choice taken from the
   trace, which gives its log density and the addresses it would propose. A
   q that calls another generative function at an address cannot be
   replayed, which is a loud error (trace those choices in q itself).
   with-exact-proposal (gen.inference.enumerative) is the special case of the
   exact conditional of finite choices."
  {:api :stable}
  (:refer-clojure :exclude [+ -])
  (:require [gen.array :as array :refer [+ -]]
            [gen.choicemap :as cm]
            [gen.diff :as diff]
            [gen.dynamic :as dyn]
            [gen.gfi :as gfi]
            [gen.random :as rng]
            [gen.selection :as sel]
            [gen.trace :as trace]))

(set! *warn-on-reflection* true)

(defn- replay
  "[log q, paths]: q run on qargs with every choice it makes read from
   choices, its log density of them and the paths it visited; nil when q
   visits a path choices lack (q cannot propose them)."
  [q qargs choices]
  (let [visited (volatile! [])
        missing? (volatile! false)
        generate (dyn/site-transitions :generate)
        simulate (dyn/site-transitions :simulate)
        replaying (dyn/with-transitions q
                    (assoc (:transitions q) :simulate
                           (fn [state p dist key]
                             (vswap! visited conj p)
                             (if (cm/has-value? choices p)
                               ;; the trace's value, as a constraint would be taken
                               (generate (assoc state :constraints choices) p dist key)
                               (do (vreset! missing? true) (simulate state p dist key))))))
        tr (gfi/with-exact-density "with-internal-proposal (its proposal's density)" (gfi/simulate replaying qargs (rng/key 0)))]
    (when-not (= (set @visited) (set (cm/addresses (:choices tr))))
      (throw (ex-info (str "with-internal-proposal: the proposal calls a generative function at an address,"
                           " which cannot be replayed against a trace: trace those choices in the proposal")
                      {:visited @visited})))
    (when-not @missing? [(:score tr) @visited])))

(defn choices-of
  "The part of choices that q, run on qargs, would propose, found by
   replaying q against them. A choice q makes that choices lack is an error
   (q could not have proposed it)."
  [q qargs choices]
  (if-let [[_ paths] (replay q qargs choices)]
    (sel/get-selected choices (sel/from-paths paths))
    (throw (ex-info "choices-of: the proposal makes a choice the given choices lack" {:qargs qargs}))))

(defn- check-open [sigma paths]
  (let [given (set (cm/addresses sigma))]
    (when-let [p (some given paths)]
      (throw (ex-info (str "with-internal-proposal: the proposal makes the choice at " p
                           ", which the constraints give: it must propose only open choices")
                      {:path p})))))

(defn- proposal-generate
  "generate under q: q's choices given the constraints sigma, then gf's
   forward sampling for the rest. With drop?, regenerate's (Eq. 4.1),
   sigma is a trace's unselected choices: q proposes given every one of
   them, and gf's run drops those it leaves unvisited, at its own sites
   and in its calls (gfi/generate-dropping), never q's. In a revise,
   gf's calls drop the old choices they leave, never q's
   (gfi/constraining), and the whole run repeats without them, so that q
   proposes given the final constraints (gfi/rerun-without-drops)."
  [{:keys [gf q] :as pgf} args sigma key drop?]
  (let [[k-q k-gen] (rng/split key)
        attempt (fn [sigma]
                  (let [{q-choices :choices q-weight :weight} (gfi/propose q (into [sigma] args) k-q)
                        _ (check-open sigma (cm/addresses q-choices))
                        constraints (cm/merge sigma q-choices)
                        {tr :trace w :weight}
                        (if drop?
                          ;; q's choices are the run's constraints; sigma's, old choices it may drop
                          (binding [gfi/*revising* {:constraints q-choices :prefix []}]
                            (first (gfi/generate-dropping #(gfi/-generate gf args % k-gen) constraints [])))
                          (gfi/constraining q-choices #(gfi/-generate gf args constraints k-gen)))]
                    {:trace (trace/as-made-by tr pgf) :weight (- w q-weight)}))]
    (if (and gfi/*revising* (not drop?)) (first (gfi/rerun-without-drops attempt sigma)) (attempt sigma))))

(defn- proposal-weight
  "The weight proposal-generate gives for producing the trace tr of gf on
   args from sigma, which agrees with tr's choices where both have one:
   +Inf when q could not have proposed tr's choices. q proposes given all
   of sigma; a choice of sigma that tr lacks is one the run drops
   (regenerate's reverse move, Eq. 4.1), outside the projection."
  [{:keys [gf q]} args sigma tr]
  (if-let [[lq q-paths] (replay q (into [sigma] args) (:choices tr))]
    (do (check-open sigma q-paths)
        ;; the model's projection: tr's maker is the wrapper, whose own
        ;; projection this is
        (- (gfi/-project gf tr (sel/from-paths (concat (cm/addresses sigma) q-paths))) lq))
    ##Inf))

(defn- same-addresses? [x y] (= (set (cm/addresses x)) (set (cm/addresses y))))

(defn update-under
  "update for this, a wrapper of the generative function (:gf this) with
   an internal proposal: gf's update keeps, overwrites and removes choices,
   a change in the arguments' number included, and the fresh choices it
   makes are then drawn by the internal proposal given the kept ones,
   (generate args kept key), weighed against tr's score. Fresh choices that
   change the model's structure are a loud error naming the wrapper. Shared
   by with-internal-proposal and gen.inference.enumerative/with-exact-proposal,
   as regenerate-under is."
  [wrapper-name this generate tr args argdiffs constraints key]
  (let [[k-update k-fresh] (rng/split key)
        {t2 :trace :as r} (gfi/-update (:gf this) (trace/as-made-by tr (:gf this)) args argdiffs constraints k-update)
        before (set (cm/addresses (:choices tr)))
        given (set (cm/addresses constraints))
        fresh (vec (remove #(or (before %) (given %)) (cm/addresses (:choices t2))))]
    (if (empty? fresh)
      (assoc r :trace (trace/as-made-by t2 this))
      (let [kept (sel/get-selected (:choices t2) (sel/complement (sel/from-paths fresh)))
            {t3 :trace w :weight} (generate args kept k-fresh)]
        (when-not (same-addresses? (:choices t2) (:choices t3))
          (throw (ex-info (str wrapper-name ": the fresh choices of this update change the model's structure")
                          {:wrapper wrapper-name})))
        {:trace t3 :weight (- w (:score tr)) :retdiff diff/unknown-change :discard (:discard r)}))))

(defn regenerate-under
  "regenerate for a generative function with an internal proposal
   (Eq. 4.1): the new trace τ' from (generate args sigma key), sigma the
   unselected choices of tr, and the old one weighed by
   (weight (:args tr) sigma' tr), sigma' the unselected choices of τ',
   both returning the internal proposal's generate weight. A generate
   that drops the unselected choices its run leaves unvisited follows
   Eq. 4.1 (B ⊆ A_σ), and sigma' then lacks them; for one that throws on
   them (with-exact-proposal's, with-conjugate-proposal's), a selection or
   new arguments that would leave an unselected choice unvisited is a loud
   error. tr is kept when nothing is selected and no argument changed, nor
   their number. (A trace the generative function did not make, or a call
   whose store changed, is gfi/revise's.) Shared by with-internal-proposal,
   gen.inference.enumerative/with-exact-proposal and
   gen.inference.conjugate/with-conjugate-proposal."
  [generate weight tr args argdiffs selection key]
  (if (and (cm/empty? (sel/get-selected (:choices tr) selection))
           (== (count args) (count (:args tr))) (every? diff/no-change? argdiffs))
    {:trace tr :weight 0.0 :retdiff diff/no-change}
    (let [unselected #(sel/get-selected (:choices %) (sel/complement selection))
          {t2 :trace w :weight}
          (try (generate args (unselected tr) key)
               (catch clojure.lang.ExceptionInfo e
                 (if (:unvisited (ex-data e))
                   (throw (ex-info (str "regenerate leaves an unselected choice unvisited (" (ex-message e)
                                        "); select the choices the selection or the new arguments decide")
                                   {:selection selection} e))
                   (throw e))))]
      ;; Eq. 4.1: the reverse move keeps σ' = τ'|_{S^c}, and draws
      ;; υ' = τ|_{((A_τ ∩ A_τ') ∖ S)^c} from q(·; x, σ')
      {:trace t2 :weight (- w (weight (:args tr) (unselected t2) tr)) :retdiff diff/unknown-change})))

(gfi/defgf InternalProposalGF [gf q]
  gfi/IGenerativeFunction
  (-simulate [this args key] (trace/as-made-by (gfi/-simulate gf args key) this))
  (-generate [this args constraints key] (proposal-generate this args constraints key false))
  (-update [this tr args argdiffs constraints key]
    (update-under "with-internal-proposal" this #(proposal-generate this %1 %2 %3 false) tr args argdiffs constraints key))
  ;; Eq. 4.1: the run drops the unselected choices it leaves unvisited
  (-regenerate [this tr args argdiffs selection key]
    (regenerate-under #(proposal-generate this %1 %2 %3 true) #(proposal-weight this %1 %2 %3)
                      tr args argdiffs selection key))
  ;; generate gives the model the second of its key's split: assess, the same
  (-assess [_ args choices key] (gfi/-assess gf args choices (second (rng/split key))))
  (-propose [_ args key] (gfi/-propose gf args key))
  (-project [this tr selection]
    (proposal-weight this (:args tr) (sel/get-selected (:choices tr) selection) tr))
  ;; its traces, density and parameter stores are the model's
  trace/ICalls
  (-call [_ tr p] (trace/-call gf tr p))
  gfi/IStores
  (-stores [_] (gfi/stores gf))
  (-with-stores [this stores] (assoc this :gf (gfi/with-stores gf stores))))

(defn with-internal-proposal
  "gf with q as its internal proposal for the choices q makes (see the
   namespace doc): the same distribution and traces. gf and q are dynamic
   generative functions; q is called as (q constraints & args), on gf's
   arguments, and makes choices at gf's addresses that the constraints
   leave open (one the constraints give is a loud error)."
  [gf q]
  (doseq [[role g] [["model" gf] ["proposal" q]]]
    (when-not (dyn/dynamic-gf? g)
      (throw (ex-info (str "with-internal-proposal needs a dynamic generative function as the " role)
                      {:gen-fn (type g)}))))
  (->InternalProposalGF gf q))

(defmethod print-method InternalProposalGF [g ^java.io.Writer wr]
  (.write wr (str "#gen/with-internal-proposal[" (pr-str (:gf g)) " " (pr-str (:q g)) "]")))
