(ns gen.inference.conjugate
  "Conjugate proposals: a model's distribution and traces, with one latent
   choice drawn from its exact posterior given its observed children (the
   thesis's optimal internal proposal, §4.1, pp. 145-146).

     (with-conjugate-proposal gf latent children)       → a generative function
     (with-conjugate-proposal gf latent children opts)  ; :check-exact?, :check-retval?

   latent is the address or path of a choice in gf's own body; children are
   the choices whose distributions take it as a parameter, given as
   addresses or paths, or as a selection ((gen.selection/select :y) for
   every choice under :y). The pairs, in Gen.jl's parameterizations:

     the latent               its children
     normal(mu, std)          normal(latent, std)
     mvnormal(mu, cov)        mvnormal(latent, cov)
     beta(a, b)               bernoulli(latent), binom(n, latent)
     gamma(shape, scale)      poisson(latent)
     dirichlet(alpha)         categorical(latent)

   A child's parameter must be the latent itself, the value its choice
   returned: (normal mu 1.0) for the latent mu, not (normal (+ mu 0) 1.0).
   An array latent is elementwise (a batch of them, for mvnormal and
   dirichlet), and a child may observe each element several times: for a
   scalar mu, (normal mu [1.0 2.0]) is two observations.

   generate draws the latent from its posterior given its observed
   (constrained) children and every other choice, then draws its unobserved
   children given it; gf's forward sampling makes the other open choices.
   The weight is log p(τ) - log q(τ ∖ σ | σ) for this internal proposal q,
   so importance sampling stays unbiased and regenerate stationary for the
   posterior (Eq. 4.1) whatever the model. When nothing but its children
   depends on the latent (a choice that depends on an unobserved child
   depends on it too), the weight is exact: log p(observations) when every
   other choice is observed, the same under every key; and regenerate on
   the latent is a Gibbs move, of weight 0. It satisfies the general GFI
   laws, not the forward-sampling ones.

   How: a reference run puts the latent at its prior mean and each open
   child at a value its distribution fixes (its mean, or near it), which
   gives the latent's prior, its children's distributions and gf's forward
   draws of the other choices, whose density is exact whatever they depend
   on, as no value they depend on is a draw it discards. A final run puts
   the latent at a draw from the posterior those give, keeps every other
   choice, and draws the open children given the latent. The checks, each
   a loud error:
   - always: the latent is a choice of gf's own body, of a family above,
     and at least one child is visited, each in gf's own body, of a family
     conjugate to it, with the latent itself as its parameter, at both
     values of the latent; and both runs make the same choices (the
     structure does not depend on the latent);
   - read from the values, on an eager backend (as distributions validate
     their parameters; a lazy one reads nothing here), unless :check-exact?
     is false: the children's other parameters (std, cov, n) and the density
     of every choice but the latent and its children are the same at both
     values, so nothing else depends on the latent. With {:check-exact?
     false} something else may (a choice depending on an unobserved child,
     say): the weight is then a correct importance weight, but not exact,
     and regenerate not a Gibbs move;
   - with :check-retval? (on any backend), the return value is the same at
     both: inside an Unfold or a Map, each step's weight is exact given its
     arguments, and the total only when the return value (an Unfold's next
     state) does not depend on the latent, as for gen.inference.enumerative's
     with-exact-proposal.
   A dependence the two values cannot show (a branch on the latent that
   both take) is not caught; the weight is then still a correct importance
   weight, but not exact. gf's densities must be exact (no encapsulated
   randomness). regenerate keeps every unselected choice
   (gen.inference.proposal)."
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.string :as str]
            [gen.array :as array :refer [+ - * /]]
            [gen.array.linalg :as la]
            [gen.array.shape :as sh]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :as dyn]
            [gen.gfi :as gfi]
            [gen.inference.proposal :as proposal]
            [gen.random :as rng]
            [gen.selection :as sel]
            [gen.trace :as trace]))

(set! *warn-on-reflection* true)

(def ^:private op-name "with-conjugate-proposal")

(def ^:private children-of
  "Each latent family's children: their families, each with the parameter
   that is the latent."
  {::d/normal {::d/normal :mu}
   ::d/mvnormal {::d/mvnormal :mu}
   ::d/beta {::d/bernoulli :p ::d/binom :p}
   ::d/gamma {::d/poisson :lam}
   ::d/dirichlet {::d/categorical :probs}})

(defn- family [dist] (name (:type dist)))

;; ---------------------------------------------------------------------------
;; Posteriors
;; ---------------------------------------------------------------------------

(defn- like
  "x broadcast to y's shape (a child's parameter to its value's)."
  [x y]
  (let [s (array/shape y)] (if (= s (array/shape x)) x (array/broadcast-to x s))))

(defn- sum-to
  "x, a child's statistic per element of its value, summed to the latent's
   shape s: over the axes the value has beyond the latent's, which hold
   several observations of one element."
  [x s]
  (let [xs (sh/broadcast (array/shape x) s)
        x (if (= xs (array/shape x)) x (array/broadcast-to x xs))
        lead (clojure.core/- (count xs) (count s))
        x (if (pos? lead) (array/sum x {:axis (vec (range lead))}) x)
        repeated (vec (keep-indexed (fn [i n] (when (and (== 1 n) (not= 1 (nth xs (clojure.core/+ lead i)))) i)) s))]
    (if (seq repeated) (array/sum x {:axis repeated :keep-dims? true}) x)))

(defn- matvec [m v] (array/squeeze (array/matmul m (array/expand-dims v -1)) -1))

(defn- symmetric
  "a's symmetric part, (a + aᵀ)/2 over its last two axes: a product of
   symmetric matrices is symmetric only up to rounding."
  [a]
  (let [r (array/ndim a)
        perm (-> (vec (range r)) (assoc (clojure.core/- r 2) (dec r)) (assoc (dec r) (clojure.core/- r 2)))]
    (* 0.5 (+ a (array/transpose a perm)))))

(defn- normal-posterior
  "normal(mu, std) given observations y of normal(latent, s): precision
   1/std² + Σ 1/s², mean (mu/std² + Σ y/s²) / precision."
  [{{mu :mu std :std} :params} observed s]
  (let [prec0 (/ 1.0 (array/square std))
        [prec weighted] (reduce (fn [[prec weighted] [{{sd :std} :params} y]]
                                  (let [w (like (/ 1.0 (array/square sd)) y)]
                                    [(+ prec (sum-to w s)) (+ weighted (sum-to (* w y) s))]))
                                [prec0 (* prec0 mu)]
                                observed)]
    (d/normal (/ weighted prec) (/ 1.0 (array/sqrt prec)))))

(defn- mvnormal-posterior
  "mvnormal(mu, cov) given observations y of mvnormal(latent, c), in
   information form: precision cov⁻¹ + Σ c⁻¹, and mean precision⁻¹ ·
   (cov⁻¹·mu + Σ c⁻¹·y)."
  [{{mu :mu cov :cov} :params} observed s]
  (let [prec0 (la/inv cov)
        prec-shape (conj s (peek s))
        [prec info] (reduce (fn [[prec info] [{{c :cov} :params} y]]
                              (let [p (la/inv c)]
                                [(+ prec (sum-to p prec-shape)) (+ info (sum-to (matvec p y) s))]))
                            [prec0 (matvec prec0 mu)]
                            observed)
        post (symmetric (la/inv prec))]
    (d/mvnormal (matvec post info) post)))

(defn- beta-posterior
  "beta(a, b) given observations of bernoulli(latent) and binom(n, latent):
   a plus the successes, b plus the failures."
  [{{a :a b :b} :params} observed s]
  (let [[a b] (reduce (fn [[a b] [child y]]
                        (let [[hits misses] (if (= ::d/binom (:type child))
                                              [y (- (like (:n (:params child)) y) y)]
                                              [(array/where y 1.0 0.0) (array/where y 0.0 1.0)])]
                          [(+ a (sum-to hits s)) (+ b (sum-to misses s))]))
                      [a b]
                      observed)]
    (d/beta a b)))

(defn- gamma-posterior
  "gamma(shape, scale) given observations of poisson(latent): shape plus
   their sum, and rate 1/scale plus their number."
  [{{k :k theta :theta} :params} observed s]
  (let [[k rate] (reduce (fn [[k rate] [_ y]] [(+ k (sum-to y s)) (+ rate (sum-to (array/ones-like y) s))])
                         [k (/ 1.0 theta)]
                         observed)]
    (d/gamma k (/ 1.0 rate))))

(defn- dirichlet-posterior
  "dirichlet(alpha) given observations of categorical(latent): alpha plus
   each category's count."
  [{{alpha :alpha} :params} observed s]
  (let [categories (array/arange (peek s))]
    (d/dirichlet (reduce (fn [alpha [_ y]]
                           (+ alpha (sum-to (array/where (array/== (array/expand-dims y -1) categories) 1.0 0.0) s)))
                         alpha
                         observed))))

(def ^:private posteriors
  {::d/normal normal-posterior
   ::d/mvnormal mvnormal-posterior
   ::d/beta beta-posterior
   ::d/gamma gamma-posterior
   ::d/dirichlet dirichlet-posterior})

(defn- reference-value
  "The latent's value in the reference run: its prior mean, at its shape. A
   value made here, so that a child whose parameter is the prior's own mean
   cannot pass for one whose parameter is the latent."
  [{:keys [type params] :as prior}]
  (let [{:keys [mu a b k theta alpha]} params
        m (case type
            (::d/normal ::d/mvnormal) (+ mu 0.0)
            ::d/beta (/ a (+ a b))
            ::d/gamma (* k theta)
            ::d/dirichlet (/ alpha (array/sum alpha {:axis -1 :keep-dims? true})))
        s (dc/value-shape prior)]
    (if (= s (array/shape m)) m (array/broadcast-to m s))))

(defn- child-reference
  "An open child's value in the reference run: one in its support, fixed by
   its distribution (its mean, or a value near it), not a draw. The forward
   draws of the other choices there may depend on it, and their density is
   still exact."
  [{:keys [type params] :as dist}]
  (let [{:keys [mu p n lam probs]} params
        ints #(array/astype % (dc/int-dtype (array/backend %)))
        v (case type
            (::d/normal ::d/mvnormal) (+ mu 0.0)
            ::d/bernoulli (array/>= p 0.5)
            ::d/binom (ints (array/round (* n p)))
            ::d/poisson (ints (array/floor lam))
            ::d/categorical (array/argmax probs {:axis -1}))
        s (dc/value-shape dist)]
    (if (= s (array/shape v)) v (array/broadcast-to v s))))

(defn- site-log-prob
  "A site's log density, summed over a value's elements (as the handler's)."
  [dist v]
  (let [lp (dc/log-prob dist v)]
    (if (pos? (array/ndim lp)) (array/sum lp) lp)))

;; ---------------------------------------------------------------------------
;; The runs
;; ---------------------------------------------------------------------------

(defn- check-latent-family! [p dist]
  (when-not (children-of (:type dist))
    (throw (ex-info (str op-name ": the latent at " p " is a " (family dist) ", which has no conjugate children here"
                         " (the latents are normal, mvnormal, beta, gamma and dirichlet)")
                    {:latent p :family (:type dist)}))))

(defn- check-child!
  "The child at p, of distribution dist, must be of a family conjugate to
   the latent's, with the latent itself as its parameter. site is the
   latent's {:dist :value}, nil when the run has not reached it."
  [latent site p dist]
  (let [prior (:dist site)
        param (get-in children-of [(:type prior) (:type dist)])]
    (cond
      (nil? site)
      (throw (ex-info (str op-name ": the child at " p " comes before the latent at " latent
                           ", so none of its parameters is the latent")
                      {:child p :latent latent}))
      (nil? param)
      (throw (ex-info (str op-name ": the child at " p " is a " (family dist) ", not conjugate to the latent's "
                           (family prior) " (its children are " (str/join ", " (map name (keys (children-of (:type prior))))) ")")
                      {:child p :family (:type dist) :latent-family (:type prior)}))
      (not (identical? (:value site) (get-in dist [:params param])))
      (throw (ex-info (str op-name ": the " (name param) " of the child at " p " is not the latent at " latent
                           " itself (its value, passed unchanged)")
                      {:child p :param param})))))

(defn- watched
  "gf whose generate records in seen (a volatile map) the distribution and
   value at the latent and at each child, and whether it was given. With
   reference?, an open latent and open children take their reference
   values, not draws."
  [{:keys [gf latent children]} seen reference?]
  (let [t (or (get (:transitions gf) :generate) (dyn/site-transitions :generate))]
    (dyn/with-transitions gf
      (assoc (:transitions gf) :generate
             (fn [state p dist key]
               (if-let [role (cond (= p latent) :latent (sel/selected? children p) :child)]
                 (let [given? (cm/has-value? (:constraints state) p)
                       state (if (and reference? (not given?))
                               (update state :constraints cm/set-choice p
                                       (if (= role :latent)
                                         (do (check-latent-family! p dist) (reference-value dist))
                                         (do (check-child! latent (get @seen latent) p dist) (child-reference dist))))
                               state)
                       [v state] (t state p dist key)]
                   (vswap! seen assoc p {:dist dist :value v :observed? given?})
                   [v state])
                 (t state p dist key)))))))

(defn- run
  "(generate g args constraints key), g being gf watched: {:trace :weight
   :seen}. Densities must be exact."
  [generate cgf args constraints key reference?]
  (let [seen (volatile! {})
        r (gfi/with-exact-density op-name (generate (watched cgf seen reference?) args constraints key))]
    (assoc r :seen @seen)))

(defn- sites
  "The latent's site and its children's in a run, checked: the latent a
   choice of gf's own body, of a family with conjugate children; at least
   one child, every one in gf's own body, of a family conjugate to the
   latent, with the latent itself as its parameter. {:prior :children},
   the children as {path {:dist :value :observed?}}."
  [{:keys [latent children]} {tr :trace seen :seen}]
  (let [{prior :dist :as site} (get seen latent)
        kids (dissoc seen latent)]
    (when-not site
      (throw (ex-info (str op-name ": the latent at " latent
                           (if (cm/has-value? (:choices tr) latent)
                             " is made by a generative function the model calls at an address; wrap that one"
                             " was not visited"))
                      {:latent latent})))
    (check-latent-family! latent prior)
    (when-let [p (first (remove kids (cm/addresses (sel/get-selected (:choices tr) children))))]
      (throw (ex-info (str op-name ": the child at " p " is made by a generative function the model calls at an"
                           " address; wrap that one")
                      {:child p})))
    (when (empty? kids)
      (throw (ex-info (str op-name ": no child of the latent at " latent " was visited") {:latent latent})))
    (doseq [[p {:keys [dist]}] kids]
      (check-child! latent site p dist))
    {:prior prior :children kids}))

(defn- posterior
  "The latent's posterior given its observed children."
  [{:keys [prior children]}]
  ((posteriors (:type prior)) prior
                              (keep (fn [[_ {:keys [dist value observed?]}]] (when observed? [dist value])) children)
                              (dc/value-shape prior)))

(defn- others-of
  "The choices other than sigma's, the latent and its children."
  [{:keys [latent children]} choices sigma]
  (-> choices
      (sel/get-selected (sel/complement (sel/from-paths (cons latent (cm/addresses sigma)))))
      (sel/get-selected (sel/complement children))))

(defn- differ?
  "Do the arrays a and b differ, as far as an eager backend shows (a lazy
   one's values are not read)?"
  [a b]
  (and (not (identical? a b)) (array/array? a) (array/array? b) (array/concrete? a) (array/concrete? b)
       (not (and (= (array/shape a) (array/shape b)) (array/all-true? (array/== a b))))))

(defn- host-tree [t] (array/tree-map #(if (array/array? %) (array/->clj (array/all-particles %)) %) t))

(defn- depends-error [what p]
  (ex-info (str op-name ": " what " depends on the latent, so the weight would not be exact; declare it a child"
                " if it is one, or leave the latent to the model's own proposal")
           {:path p}))

(defn- check-same!
  "The final run (fin, its sites fs) against the reference run (ref, rs),
   at the latent's other value: the same choices; with check-exact?, on an
   eager backend, the children's other parameters and the density of every
   other choice the same; with check-retval?, the return value the same."
  [{:keys [latent check-exact? check-retval?]} ref rs fin fs]
  (let [t0 (:trace ref)
        t1 (:trace fin)]
    (when-not (= (set (cm/addresses (:choices t0))) (set (cm/addresses (:choices t1))))
      (throw (ex-info (str op-name ": the model's structure depends on the latent at " latent
                           " (it makes other choices at another value of it)")
                      {:latent latent})))
    (doseq [[p {d0 :dist}] (when check-exact? (:children rs))
            :let [d1 (get-in fs [:children p :dist])
                  param (get-in children-of [(:type (:prior rs)) (:type d0)])]
            [k x] (:params d0)
            :when (and (not= k param) (differ? x (get-in d1 [:params k])))]
      (throw (depends-error (str "the " (name k) " of the child at " p) p)))
    (let [paths (cons latent (keys (:children rs)))
          others (sel/complement (sel/from-paths paths))]
      (when (and check-exact? (differ? (gfi/project t0 others) (gfi/project t1 others)))
        (let [p (first (filter #(differ? (gfi/project t0 (sel/select %)) (gfi/project t1 (sel/select %)))
                               (cm/addresses (sel/get-selected (:choices t0) others))))]
          (throw (depends-error (if p (str "the choice at " p) "a call's density") p)))))
    (when (and check-retval? (not= (host-tree (:retval t0)) (host-tree (:retval t1))))
      (throw (ex-info (str op-name ": the return value depends on the latent at " latent ". A combinator passing it"
                           " on (an Unfold's next state) conditions on the latent drawn, so its weight is an"
                           " unbiased estimate, not exact")
                      {:latent latent})))))

(defn- others-density
  "log f(others | sigma, the latent's reference value): the density of
   gf's proposal of the other choices in the reference run, its score
   less its projection on sigma, the latent and the children (a callee's
   proposal of its own choices included)."
  [{:keys [latent]} {t0 :trace} rs sigma]
  (- (:score t0) (gfi/project t0 (sel/from-paths (concat [latent] (keys (:children rs)) (cm/addresses sigma))))))

(defn- reference
  "The reference run on sigma ({:trace :weight :seen}, with its sites
   under :sites): through generate (gf's own -generate in a revise, whose
   calls drop the old choices they leave, rerun until sigma lacks none:
   gfi/rerun-without-drops). [run sigma]."
  [cgf args sigma key]
  (let [attempt (fn [sigma] (run gfi/-generate cgf args sigma key true))
        [r sigma] (if gfi/*revising* (gfi/rerun-without-drops attempt sigma) [(attempt sigma) sigma])]
    [(assoc r :sites (sites cgf r)) sigma]))

(defn- latent-given? [{:keys [latent]} sigma] (cm/has-value? sigma latent))

(defn- conjugate-generate
  "generate under the conjugate proposal: the reference run, the latent
   from the posterior it gives, and the final run with every other choice
   of the reference run but the unobserved children kept. The weight is the
   final run's (log p of its choices but the unobserved children's) less
   log f(others | sigma) and log q(latent). With the latent given, gf's own
   generate."
  [{:keys [gf latent] :as cgf} args sigma key]
  (if (latent-given? cgf sigma)
    (update (gfi/with-exact-density op-name (gfi/-generate gf args sigma key)) :trace trace/as-made-by cgf)
    (let [[k-ref k-draw k-final] (rng/split key 3)
          [ref sigma] (reference cgf args sigma k-ref)
          rs (:sites ref)
          q (posterior rs)
          theta (dc/sample q k-draw)
          chosen (cm/set-choice (cm/merge sigma (others-of cgf (:choices (:trace ref)) sigma)) latent theta)
          fin (try (run gfi/generate cgf args chosen k-final false)
                   (catch clojure.lang.ExceptionInfo e
                     (throw (if (:unvisited (ex-data e))
                              (ex-info (str op-name ": the model's structure depends on the latent at " latent
                                            " (" (ex-message e) ")")
                                       {:latent latent} e)
                              e))))
          fs (sites cgf fin)]
      (check-same! cgf ref rs fin fs)
      {:trace (trace/as-made-by (:trace fin) cgf)
       :weight (- (:weight fin) (others-density cgf ref rs sigma) (site-log-prob q theta))})))

(defn- conjugate-weight
  "The weight conjugate-generate gives for producing the trace tr on args
   from sigma, tr's choices at some addresses: tr's score less the density
   of its unobserved children, of the proposal of the other choices and of
   the posterior at its latent. tr is run again for the checks generate
   makes."
  [{:keys [gf latent] :as cgf} args sigma tr]
  (if (latent-given? cgf sigma)
    (gfi/with-exact-density op-name (gfi/-project gf tr (sel/from-paths (cm/addresses sigma))))
    (let [[ref] (reference cgf args (cm/merge sigma (others-of cgf (:choices tr) sigma)) (rng/key 0))
          rs (:sites ref)
          replay (run gfi/generate cgf args (:choices tr) (rng/key 0) false)
          fs (sites cgf replay)
          unobserved (remove #(cm/has-value? sigma %) (keys (:children fs)))]
      (check-same! cgf ref rs replay fs)
      ;; the model's projection: tr's maker is the wrapper, whose own
      ;; projection this is
      (- (:score tr)
         (gfi/-project gf tr (sel/from-paths unobserved))
         (others-density cgf ref rs sigma)
         (site-log-prob (posterior rs) (cm/get-choice (:choices tr) latent))))))

(gfi/defgf ConjugateProposalGF [gf latent children check-exact? check-retval?]
  gfi/IGenerativeFunction
  (-simulate [this args key] (trace/as-made-by (gfi/-simulate gf args key) this))
  (-generate [this args constraints key] (conjugate-generate this args constraints key))
  (-update [this tr args argdiffs constraints key]
    (proposal/update-under op-name this #(conjugate-generate this %1 %2 %3) tr args argdiffs constraints key))
  (-regenerate [this tr args argdiffs selection key]
    (proposal/regenerate-under #(conjugate-generate this %1 %2 %3) #(conjugate-weight this %1 %2 %3)
                               tr args argdiffs selection key))
  ;; as its generate, it needs exact densities
  (-assess [_ args choices key] (gfi/with-exact-density op-name (gfi/-assess gf args choices key)))
  (-propose [_ args key] (gfi/-propose gf args key))
  (-project [this tr selection]
    (conjugate-weight this (:args tr) (sel/get-selected (:choices tr) selection) tr))
  ;; its traces, density and parameter stores are the model's
  trace/ICalls
  (-call [_ tr p] (trace/-call gf tr p))
  gfi/IStores
  (-stores [_] (gfi/stores gf))
  (-with-stores [this stores] (assoc this :gf (gfi/with-stores gf stores))))

(defn with-conjugate-proposal
  "gf with the exact posterior of the choice at latent, given its children,
   as its internal proposal for it (see the namespace doc): the same
   distribution and traces. gf is a dynamic generative function; latent an
   address or path of its body; children addresses or paths, or a
   selection. Options:
   - :check-exact? (true): throw when something but the children depends
     on the latent (read on an eager backend); false allows it, and the
     weight is then a correct importance weight, not exact;
   - :check-retval? (false): throw from generate when the return value
     depends on the latent, for uses in a combinator that need the total
     weight exact (it reads values, on any backend)."
  ([gf latent children] (with-conjugate-proposal gf latent children nil))
  ([gf latent children {:keys [check-exact? check-retval?] :or {check-exact? true}}]
   (when-not (dyn/dynamic-gf? gf)
     (throw (ex-info (str op-name " needs a dynamic generative function") {:gen-fn (type gf)})))
   (let [latent (cm/path latent)
         children (cond (sel/selection? children) children
                        (and (coll? children) (seq children)) (sel/from-paths (map cm/path children))
                        :else (throw (ex-info (str op-name " needs the children's addresses or a selection of them")
                                              {:children children})))]
     (when (sel/selected? children latent)
       (throw (ex-info (str op-name ": the latent at " latent " is among its children") {:latent latent})))
     (->ConjugateProposalGF gf latent children (boolean check-exact?) (boolean check-retval?)))))

(defmethod print-method ConjugateProposalGF [g ^java.io.Writer wr]
  (.write wr (str "#gen/with-conjugate-proposal[" (pr-str (:gf g)) " " (pr-str (:latent g)) "]")))
