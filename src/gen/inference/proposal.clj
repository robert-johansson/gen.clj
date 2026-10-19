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

   project replays q against a trace: q runs with each choice taken from the
   trace, which gives its log density and the addresses it would propose. A
   q that calls another generative function at an address cannot be
   replayed, which is a loud error (trace those choices in q itself).
   with-exact-proposal (gen.inference.enumerative) is the special case of the
   exact conditional of finite choices."
  (:refer-clojure :exclude [+ -])
  (:require [gen.array :as array :refer [+ -]]
            [gen.choicemap :as cm]
            [gen.diff :as diff]
            [gen.dynamic :as dyn]
            [gen.gfi :as gfi]
            [gen.random :as rng]
            [gen.selection :as sel]))

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
                             (if (cm/has-value? (cm/get-submap choices p))
                               ;; the trace's value, as a constraint would be taken
                               (generate (assoc state :constraints choices) p dist key)
                               (do (vreset! missing? true) (simulate state p dist key))))))
        tr (gfi/simulate replaying qargs (rng/key 0))]
    (when-not (= (set @visited) (set (cm/addresses (:choices tr))))
      (throw (ex-info (str "with-internal-proposal: the proposal calls a generative function at an address,"
                           " which cannot be replayed against a trace: trace those choices in the proposal")
                      {:visited @visited})))
    (when-not @missing? [(:score tr) @visited])))

(defn choices-of
  "The choices of choices that q, run on qargs, would make: found by
   replaying q against them. A choice q makes that choices lack is an error
   (q could not have proposed them)."
  [q qargs choices]
  (if-let [[_ paths] (replay q qargs choices)]
    (sel/get-selected choices (sel/from-paths paths))
    (throw (ex-info "choices-of: the proposal makes a choice the given choices lack" {:qargs qargs}))))

(defn- overlap! [sigma paths]
  (let [given (set (cm/addresses sigma))]
    (when-let [p (some given paths)]
      (throw (ex-info (str "with-internal-proposal: the proposal makes the choice at " p
                           ", which the constraints give: it must propose only open choices")
                      {:path p})))))

(defn- proposal-generate
  "generate under q: q's choices given the constraints sigma, then gf's
   forward sampling for the rest."
  [{:keys [gf q] :as pgf} args sigma key]
  (let [[k-q k-gen] (rng/split key)
        {q-choices :choices q-weight :weight} (gfi/propose q (into [sigma] args) k-q)
        _ (overlap! sigma (cm/addresses q-choices))
        {tr :trace w :weight} (gfi/generate gf args (cm/merge sigma q-choices) k-gen)]
    {:trace (assoc tr :gen-fn pgf) :weight (- w q-weight)}))

(defn- proposal-weight
  "The weight proposal-generate gives for producing the trace tr of gf on
   args from sigma, tr's choices at some addresses: +Inf when q could not
   have proposed tr's choices."
  [{:keys [gf q]} args sigma tr]
  (if-let [[lq q-paths] (replay q (into [sigma] args) (:choices tr))]
    (do (overlap! sigma q-paths)
        (- (gfi/project (assoc tr :gen-fn gf) (sel/from-paths (concat (cm/addresses sigma) q-paths))) lq))
    ##Inf))

(defn- same-addresses? [x y] (= (set (cm/addresses x)) (set (cm/addresses y))))

(defrecord InternalProposalGF [gf q]
  gfi/IGenerativeFunction
  (-simulate [this args key] (assoc (gfi/-simulate gf args key) :gen-fn this))
  (-generate [this args constraints key] (proposal-generate this args constraints key))
  (-update [this tr args argdiffs constraints key]
    ;; gf's update keeps, overwrites and removes choices; fresh ones are then
    ;; drawn by the proposal given the kept ones
    (let [[k-update k-fresh] (rng/split key)
          {t2 :trace :as r} (gfi/-update gf (assoc tr :gen-fn gf) args argdiffs constraints k-update)
          before (set (cm/addresses (:choices tr)))
          given (set (cm/addresses constraints))
          fresh (vec (remove #(or (before %) (given %)) (cm/addresses (:choices t2))))]
      (if (empty? fresh)
        (assoc r :trace (assoc t2 :gen-fn this))
        (let [kept (sel/get-selected (:choices t2) (sel/complement (sel/from-paths fresh)))
              {t3 :trace w :weight} (proposal-generate this args kept k-fresh)]
          (when-not (same-addresses? (:choices t2) (:choices t3))
            (throw (ex-info "with-internal-proposal: the fresh choices of this update change the model's structure" {})))
          {:trace t3 :weight (- w (:score tr)) :retdiff diff/unknown-change :discard (:discard r)}))))
  (-regenerate [this tr args argdiffs selection key]
    (let [selected (set (cm/addresses (sel/get-selected (:choices tr) selection)))]
      (if (and (empty? selected) (every? diff/no-change? argdiffs))
        {:trace tr :weight 0.0 :retdiff diff/no-change}
        ;; Eq 4.1: the retained choices are the constraints of both moves
        (let [[k-regenerate k-generate] (rng/split key)
              t1 (:trace (gfi/-regenerate gf (assoc tr :gen-fn gf) args argdiffs selection k-regenerate))
              now (set (cm/addresses (:choices t1)))
              retained (filter #(and (now %) (not (selected %))) (cm/addresses (:choices tr)))
              sigma (sel/get-selected (:choices tr) (sel/from-paths retained))
              {t2 :trace w :weight} (proposal-generate this args sigma k-generate)]
          {:trace t2 :weight (- w (proposal-weight this (:args tr) sigma tr)) :retdiff diff/unknown-change}))))
  (-assess [_ args choices] (gfi/-assess gf args choices))
  (-propose [_ args key] (gfi/-propose gf args key))
  (-project [this tr selection]
    (proposal-weight this (:args tr) (sel/get-selected (:choices tr) selection) tr)))

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
