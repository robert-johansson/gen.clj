(ns gen.inference.transform
  "Gen.jl's trace transform DSL (@transform) and trace translators (thesis
   §3.6–3.7).

   A transform maps a model's choices and auxiliary choices to new model
   choices and new auxiliary choices. Its body reads from the two input
   spaces and writes to the two output spaces, each read and write tagged
   :discrete or :continuous:

     (deftransform split-merge [model-in aux-in model-out aux-out]
       (if (= 1 (read model-in :k :discrete))
         (let [mu (read model-in :mu :continuous)
               u (read aux-in :u :continuous)]
           (write model-out :k 2 :discrete)
           (write model-out :mu1 (- mu u) :continuous)
           (write model-out :mu2 (+ mu u) :continuous))
         (let [mu1 (read model-in :mu1 :continuous)
               mu2 (read model-in :mu2 :continuous)]
           (write model-out :k 1 :discrete)
           (write model-out :mu (/ (+ mu1 mu2) 2.0) :continuous)
           (write aux-out :u (/ (- mu2 mu1) 2.0) :continuous))))

   (copy from-space from-addr to-space to-addr) moves a value or a whole
   subtree unchanged, (retval space) is the return value of what made an
   input space's choices (the trace's, the proposal's), for structure such
   as a path, and (tcall t) runs another transform's body on the same
   spaces. The body is ordinary code over gen.array's operators; its
   control flow may depend on discrete reads, and its continuous part must
   be a bijection. The log |det J| of that part comes from automatic
   differentiation: inputs are the continuous reads, outputs the continuous
   writes, the copies of continuous reads, and, where a translator keeps
   the input trace (update), the continuous reads it retains unwritten.
   Inputs and outputs must have the same total dimension, and a continuous
   input the move consumes (an auxiliary choice, or a model choice the
   update overwrites or discards, less the observations :new-observations
   replace) must be read as continuous or copied: one the body never reads
   is lost. The general translator, which does not know the old trace's
   observations, checks only its auxiliary choices. (run t model-in
   aux-in) applies a transform to choice maps, {:model :aux :log-abs-det};
   :retained says which model choices the move keeps.

   Translators are functions (translator trace key) → {:trace :weight
   :log-abs-det}, the weight the log importance weight of the new trace,
   log |det J| in it and beside it:
   - (translator {:p-new :p-new-args :new-observations :q-forward
     :q-forward-args :q-backward :q-backward-args :transform}): Gen.jl's
     GeneralTraceTranslator, and DeterministicTraceTranslator without the
     proposals ((deterministic-translator {:p-new :p-new-args
     :new-observations :transform}));
   - (extending-translator {:p-new-args :p-argdiffs :new-observations
     :q-forward :q-forward-args :transform}): SimpleExtendingTraceTranslator
     (GenParticleFilters' ExtendingTraceTranslator with a :transform);
   - (updating-translator {… :q-backward :q-backward-args :transform}):
     GenParticleFilters' UpdatingTraceTranslator, an SMC-samplers step
     (Del Moral) without a transform and SMCP3's (Lew et al. 2023) with one;
   - (involution t): Gen.jl's SymmetricTraceTranslator, as the involution
     that gen.inference.mh/involutive-mh and gen.inference.kernel/involutive-mh
     take (and gen.inference.particle-filter/reweighting's :involution):
     (h trace fwd-choices fwd-retval proposal-args key) → {:trace
     :bwd-choices :weight :log-abs-det}, or (h trace fwd-choices key)
     called directly. gen.inference.mh/move with :fwd-choices gives a
     move's log α for given auxiliary choices.
   A translator is a move-reweight kernel too: (pf/move-reweight state
   translator) translates every particle (GenParticleFilters' updating
   translators). A move outside the support is not weighed: its weight is
   -Inf, without :log-abs-det.

   (pair f g) makes two transforms each other's inverses and (involutive t)
   its own; :check? on a translator runs the inverse on the result, and on
   an involution the involution again, and throws unless it gives back the
   input."
  {:api :stable}
  (:refer-clojure :exclude [read + - * /])
  (:require [gen.array :as array :refer [+ - * /]]
            [gen.array.dtype :as dt]
            [gen.choicemap :as cm]
            [gen.diff :as diff]
            [gen.gfi :as gfi]
            [gen.gradients :as grad]
            [gen.random :as rng]
            [gen.selection :as sel]))

(set! *warn-on-reflection* true)

;; ---------------------------------------------------------------------------
;; The body's operations
;; ---------------------------------------------------------------------------

(def ^:private ^:dynamic *run*
  "The running transform's state, threaded through one volatile for the
   duration of one execution (as gen.dynamic's handler)."
  nil)

(defrecord Transform [name body inverse])

(defn transform? [x] (instance? Transform x))

(defn- state! [op]
  (or *run* (throw (ex-info (str (name op) " runs inside a transform's body") {:op op}))))

(defn- path [addr] (if (vector? addr) addr [addr]))

(defn- check-tag [op tag]
  (when-not (#{:discrete :continuous} tag)
    (throw (ex-info (str (name op) ": the tag is :discrete or :continuous, not " (pr-str tag)) {:tag tag}))))

(defn- check-space [op space spaces]
  (when-not (spaces space)
    (throw (ex-info (str (name op) ": " (pr-str space) " is not one of " (pr-str (sort spaces))) {:space space}))))

(defn read
  "The value at addr of an input space (model-in or aux-in), tagged
   :discrete or :continuous. A discrete value comes back as a host value
   (a boolean or a number), so the body may branch on it with if: on a lazy
   backend it is an array, which would always be truthy."
  [space addr tag]
  (check-space :read space #{:model-in :aux-in})
  (check-tag :read tag)
  (let [st (state! :read)
        s @st
        p (path addr)
        sub (cm/get-submap (get-in s [:inputs space]) p cm/EMPTY)]
    (when-not (cm/has-value? sub)
      (throw (ex-info (str "read: no choice at " p " of " (name space)) {:space space :path p})))
    (if-let [tag' (get-in s [:read-tags [space p]])]
      (when-not (= tag tag')
        (throw (ex-info (str "read: " p " of " (name space) " read as " tag' " and as " tag) {:path p})))
      (vswap! st #(-> % (update :reads conj [space p tag]) (assoc-in [:read-tags [space p]] tag))))
    (if (= tag :continuous)
      (get (:subst s) [space p] (cm/get-value sub))
      (let [v (cm/get-value sub)] (if (array/array? v) (array/->clj v) v)))))

(defn write
  "Set addr of an output space (model-out or aux-out) to value, tagged
   :discrete or :continuous."
  [space addr value tag]
  (check-space :write space #{:model-out :aux-out})
  (check-tag :write tag)
  (let [st (state! :write)
        p (path addr)]
    (when (contains? (:written-set @st) [space p])
      (throw (ex-info (str "write: " p " of " (name space) " written twice") {:space space :path p})))
    (vswap! st #(-> %
                    (update-in [:outputs space] cm/set-choice p value)
                    (update :writes conj [space p tag])
                    (update :written-set conj [space p])))
    nil))

(defn copy
  "Copy the value or subtree at from-addr of an input space to to-addr of an
   output space, unchanged."
  [from-space from-addr to-space to-addr]
  (check-space :copy from-space #{:model-in :aux-in})
  (check-space :copy to-space #{:model-out :aux-out})
  (let [st (state! :copy)
        fp (path from-addr)
        tp (path to-addr)
        sub (cm/get-submap (get-in @st [:inputs from-space]) fp cm/EMPTY)]
    (when (cm/empty? sub)
      (throw (ex-info (str "copy: nothing at " fp " of " (name from-space)) {:space from-space :path fp})))
    (when (contains? (:written-set @st) [to-space tp])
      (throw (ex-info (str "copy: " tp " of " (name to-space) " written twice") {:space to-space :path tp})))
    (vswap! st #(-> %
                    (update-in [:outputs to-space] cm/set-choice tp (if (cm/has-value? sub) (cm/get-value sub) sub))
                    (update :copies conj [from-space fp to-space tp])
                    (update :written-set conj [to-space tp])))
    nil))

(def ^:private none
  "An input space's return value where it has none."
  (Object.))

(defn- host-values
  "x with every backend array in it (nested maps, vectors and seqs) read
   on the host."
  [x]
  (cond (array/backend-array? x) (array/->clj x)
        (record? x) x
        (map? x) (into (empty x) (map (fn [[k v]] [k (host-values v)])) x)
        (vector? x) (mapv host-values x)
        (seq? x) (doall (map host-values x))
        :else x))

(defn retval
  "The return value of the generative function that made an input space's
   choices: model-in's, the trace's (a choice map has none); aux-in's, the
   proposal's that drew the auxiliary choices (involutive MH's forward
   proposal, a translator's q-forward; in a check, the backward one's;
   tf/run's :aux-retval). It is for the move's structure, such as a path
   or a count, and comes back on the host, as a discrete read does: the
   Jacobian holds it constant, so a value the move maps is read with read.
   Loud where the space has none."
  [space]
  (check-space :retval space #{:model-in :aux-in})
  (let [r (get-in @(state! :retval) [:retvals space] none)]
    (when (identical? r none)
      (throw (ex-info (str "retval: " (name space) " has no return value here"
                           (if (= space :model-in)
                               " (a trace has one, a choice map none)"
                               " (an involution's is its fwd-retval, a translator's its q-forward's, tf/run's :aux-retval)"))
                      {:space space})))
    (host-values (force r))))

(declare check-transform)

(defn tcall
  "Run transform t's body on the current spaces (Gen.jl's @tcall)."
  [t]
  (state! :tcall)
  (check-transform :tcall t)
  ((:body t)))

(def ^:private lazy-fns
  '#{clojure.core/map clojure.core/map-indexed clojure.core/mapcat clojure.core/keep
     clojure.core/keep-indexed clojure.core/filter clojure.core/remove clojure.core/for})

(defn- var-name [env sym]
  (when (symbol? sym)
    (let [v (try (resolve env sym) (catch Exception _ nil))]
      (when (var? v) (symbol v)))))

(defn- check-eager
  "A lazy call (for, map, …) holding a write, copy or tcall might never run:
   a loud error when the transform is defined (Gen.jl's loops are eager)."
  [env body]
  (doseq [form (tree-seq coll? seq body)
          :when (and (seq? form) (lazy-fns (var-name env (first form)))
                     (some #(#{`write `copy `tcall} (var-name env %)) (tree-seq coll? seq (rest form))))]
    (throw (ex-info (str "transform: " (pr-str (first form)) " is lazy, so the writes in it might never run;"
                         " use doseq, run! or mapv")
                    {:form form}))))

(defmacro transform
  "A transform: (transform [model-in aux-in model-out aux-out] body …), the
   four names bound to the spaces read, write and copy take (see the
   namespace doc). Loops that write are doseq, run! or mapv: a lazy one is
   an error."
  [spaces & body]
  (when-not (and (vector? spaces) (= 4 (count spaces)) (every? symbol? spaces))
    (throw (ex-info "transform: the spaces are a vector of four names, [model-in aux-in model-out aux-out]"
                    {:spaces spaces})))
  (check-eager &env body)
  (let [[mi ai mo ao] spaces]
    `(->Transform nil (fn [] (let [~mi :model-in ~ai :aux-in ~mo :model-out ~ao :aux-out] ~@body nil)) nil)))

(defmacro deftransform
  "(deftransform name doc? [model-in aux-in model-out aux-out] body …)."
  [tname & more]
  (let [[doc more] (if (string? (first more)) [(first more) (rest more)] [nil more])]
    `(def ~(cond-> tname doc (vary-meta assoc :doc doc))
       (assoc (transform ~@more) :name '~tname))))

(defn- check-transform [op t]
  (when-not (transform? t)
    (throw (ex-info (str (name op) " needs a transform (gen.inference.transform/transform), not " (pr-str t))
                    {:op op}))))

(defn pair
  "[f' g']: transforms f and g, each with the other as its inverse, both
   ways (Gen.jl's pair_bijections!)."
  [f g]
  (check-transform :pair f)
  (check-transform :pair g)
  [(assoc f :inverse (:body g)) (assoc g :inverse (:body f))])

(defn involutive
  "Transform t as its own inverse (Gen.jl's is_involution!)."
  [t]
  (check-transform :involutive t)
  (assoc t :inverse (:body t)))

(defn inverse
  "t's inverse, as pair or involutive set it; the inverse's inverse is t."
  [t]
  (check-transform :inverse t)
  (if-let [body (:inverse t)]
    (assoc t :body body :inverse (:body t))
    (throw (ex-info (str (if-let [n (:name t)] (str "The transform " n) "The anonymous transform (tf/transform)")
                         " has no inverse: pair it with tf/pair, or mark it tf/involutive")
                    {:transform (:name t)}))))

;; ---------------------------------------------------------------------------
;; Running a transform, and its Jacobian
;; ---------------------------------------------------------------------------

(defn- input-choices [x]
  (cond (nil? x) cm/EMPTY
        (cm/choicemap? x) x
        (contains? x :choices) (:choices x)
        :else (throw (ex-info (str "A transform reads a trace or a choice map, not " (pr-str x)
                                   " (convert data with gen.choicemap/from-map)")
                              {:value x}))))

(defn- trace-retvals
  "The return values of the inputs that are traces, by space."
  [model-in aux-in]
  (into {} (for [[space x] [[:model-in model-in] [:aux-in aux-in]]
                 :when (and (map? x) (not (cm/choicemap? x)) (contains? x :choices))]
             [space (:retval x)])))

(defn- execute
  "One run of t's body on the inputs: the continuous reads replaced by
   subst ({[space path] value}) when given, retvals the input spaces'
   return values (space to a value, a delay of one, or none) over a
   trace's own. The run's state."
  ([t model-in aux-in] (execute t model-in aux-in nil))
  ([t model-in aux-in {:keys [subst retvals]}]
   (let [st (volatile! {:inputs {:model-in (input-choices model-in) :aux-in (input-choices aux-in)}
                        :subst subst :retvals (merge (trace-retvals model-in aux-in) retvals)
                        :reads [] :read-tags {}
                        :outputs {:model-out cm/EMPTY :aux-out cm/EMPTY}
                        :writes [] :written-set #{} :copies []})]
     (binding [*run* st] ((:body t)))
     @st)))

(defn- under? [prefix p]
  (and (<= (count prefix) (count p)) (= prefix (subvec p 0 (count prefix)))))

(defn- output-specs
  "The continuous outputs of run s, in order: the writes, the continuous
   reads at or under a copied path (copied unchanged), and the continuous
   model-in reads that retained? keeps and nothing overwrites."
  [s retained?]
  (let [continuous-reads (for [[sp p tag] (:reads s) :when (= tag :continuous)] [sp p])
        copied? (fn [[sp p]] (some (fn [[fs fp]] (and (= fs sp) (under? fp p))) (:copies s)))
        written-model (set (for [[sp p] (:written-set s) :when (= sp :model-out)] p))]
    (vec (concat (for [[sp p tag] (:writes s) :when (= tag :continuous)] [:write sp p])
                 (for [r continuous-reads :when (copied? r)] (into [:copy] r))
                 (for [[sp p :as r] continuous-reads
                       :when (and (= sp :model-in) (not (written-model p)) (not (copied? r)) (retained? p))]
                   [:retain sp p])))))

(defn- output-value [s [kind sp p]]
  (case kind
    :write (cm/get-choice (get-in s [:outputs sp]) p)
    (:copy :retain) (get (:subst s) [sp p] (cm/get-choice (get-in s [:inputs sp]) p))))

(defn- log-abs-det
  "log |det J| of the continuous part of run s of t on model-in and aux-in
   (see the namespace doc): retained? says which model-in addresses the
   result keeps unwritten."
  [t model-in aux-in s retained?]
  (let [ins (filterv #(= :continuous (nth % 2)) (:reads s))
        in-values (mapv (fn [[sp p]] (cm/get-choice (get-in s [:inputs sp]) p)) ins)
        outs (output-specs s retained?)
        n (reduce + 0 (map array/size in-values))
        m (reduce + 0 (map #(array/size (output-value s %)) outs))]
    (when-not (= n m)
      (let [inputs (mapv #(vec (take 2 %)) ins) outputs (mapv #(vec (rest %)) outs)]
        (throw (ex-info (str "The transform's continuous part maps " n " dimensions to " m
                             "; a bijection keeps them equal (inputs " inputs ", outputs " outputs ")")
                        {:inputs inputs :outputs outputs}))))
    (if (zero? n)
      0.0
      (let [[x unflatten] (grad/flattened in-values)
            g (fn [x] (let [run (execute t model-in aux-in {:subst (zipmap (map #(vec (take 2 %)) ins) (unflatten x))
                                                            :retvals (:retvals s)})]
                        (array/concatenate (mapv #(array/flatten (output-value run %)) outs))))
            logabs (grad/log-abs-det-jacobian g x)]
        (when (== ##-Inf (double (array/item logabs)))
          (throw (ex-info "The transform's Jacobian is singular here: its continuous part is not a bijection"
                          {:transform (:name t)})))
        logabs))))

(defn- check-consumed!
  "Every continuous input of run s that the move consumes, (consumed? space
   path), is read as continuous or copied: one the body never reads is lost,
   so the move is not a bijection, and the Jacobian would not see it (the
   post-M5 review's P3-07)."
  [s consumed?]
  (let [copied? (fn [sp p] (some (fn [[fs fp]] (and (= fs sp) (under? fp p))) (:copies s)))
        lost (vec (for [sp [:model-in :aux-in]
                        :let [choices (get-in s [:inputs sp])]
                        p (cm/addresses choices)
                        :when (and (grad/continuous? (cm/get-choice choices p)) (consumed? sp p)
                                   (not= :continuous (get-in s [:read-tags [sp p]])) (not (copied? sp p)))]
                    [sp p]))]
    (when (seq lost)
      (throw (ex-info (str "The transform's move consumes continuous inputs its body never reads, " lost
                           ", so its continuous part is not a bijection")
                      {:inputs lost})))))

(defn run
  "Transform t of model choices (or a trace) and auxiliary choices:
   {:model the model-out choices, :aux the aux-out choices, :log-abs-det}.
   Every continuous read is consumed by the move, unless opts names the
   model choices it keeps: :retained, a selection of model-in addresses
   whose continuous reads, neither written nor copied, are unchanged
   outputs of the Jacobian, as an involution's update keeps them (a move
   that reads x to move y, and keeps x). :aux-retval is aux-in's return
   value for retval (a trace's model-in has its own)."
  ([t model-in aux-in] (run t model-in aux-in nil))
  ([t model-in aux-in {:keys [retained aux-retval] :as opts}]
   (when-let [unknown (seq (remove #{:retained :aux-retval} (keys opts)))]
     (throw (ex-info (str "tf/run takes :retained and :aux-retval, not " (vec unknown)) {:options (vec unknown)})))
   (when-not (or (nil? retained) (sel/selection? retained))
     (throw (ex-info (str "tf/run: :retained is a selection of model-in addresses (gen.selection/select), not "
                          (pr-str retained))
                     {:retained retained})))
   (let [s (execute t model-in aux-in {:retvals (when (contains? opts :aux-retval) {:aux-in aux-retval})})]
     {:model (get-in s [:outputs :model-out])
      :aux (get-in s [:outputs :aux-out])
      :log-abs-det (log-abs-det t model-in aux-in s (if retained #(sel/selected? retained %) (constantly false)))})))

;; ---------------------------------------------------------------------------
;; Checks
;; ---------------------------------------------------------------------------

(defn- close-value? [x y]
  (cond (not (and (array/array? x) (array/array? y))) (= x y)
        (not= (array/shape x) (array/shape y)) false
        (every? #(= :float (dt/kind (array/dtype %))) [x y])
        ;; about √eps of the dtype: a round trip loses a few ulps
        (array/allclose? x y (if (some #(= :float32 (array/dtype %)) [x y]) {:rtol 3e-4 :atol 1e-5} {:rtol 1e-7 :atol 1e-9}))
        :else (= (array/->clj x) (array/->clj y))))

(defn same-choices?
  "Do two choice maps hold the same addresses and (within float tolerance)
   the same values?"
  [a b]
  (let [as (cm/addresses a)]
    (and (= (set as) (set (cm/addresses b)))
         (every? #(close-value? (cm/get-choice a %) (cm/get-choice b %)) as))))

(defn- check-round-trip!
  "Loud unless the round trip (a check, named by how it ran) gave back the
   choices before as after, what they are."
  [check what before after]
  (when-not (same-choices? before after)
    (throw (ex-info (str check " does not give back the " what)
                    {:before (cm/to-map before) :after (cm/to-map after)}))))

(def ^:private inverse-check
  "A translator's check: its transform's inverse."
  "The inverse transform")

(defn unconstrained-additions
  "The addresses of new-choices (an updated trace's) that neither the
   constraints nor the old trace's choices that the update kept (not in
   discard) account for: choices the update sampled."
  [old-choices constraints discard new-choices]
  (remove (fn [p] (or (cm/has-value? constraints p)
                      (and (cm/has-value? old-choices p)
                           (not (cm/has-value? discard p)))))
          (cm/addresses new-choices)))

;; ---------------------------------------------------------------------------
;; Translators
;; ---------------------------------------------------------------------------

(defn- propose
  "(q trace args…)'s choices, weight and return value; without q, none of
   them (no return value)."
  [q args trace key]
  (if q
    (gfi/propose q (into [trace] args) key)
    {:choices cm/EMPTY :weight 0.0 :retval none}))

(defn- outside-support
  "The result of a move to t', a trace outside the support: its run stopped
   at a choice there (gfi/impossible?, DESIGN.md §9), so t' holds the
   choices up to that one. Weight -Inf, rejected, without weighing its
   backward choices, its Jacobian or checking the move, which would read
   choices t' does not hold."
  [t']
  {:trace t' :weight ##-Inf})

(defn translator
  "Gen.jl's GeneralTraceTranslator: (translator trace key) → {:trace :weight
   :log-abs-det}. Auxiliary choices u come from (q-forward trace
   q-forward-args…) (none without it), the transform maps (trace, u) to
   the new model's choices and auxiliary choices u', and p-new generates on
   p-new-args with those and :new-observations, every choice constrained.
   The weight is log p-new(t') - log p(t) + log q-backward(u'; t', args…) -
   log q-forward(u; t) + log |det J|, the last also :log-abs-det. The
   transform's aux-in retval is q-forward's return value, and its
   inverse's q-backward's. :check? runs the inverse transform on (t', u')
   and throws unless it gives back (t, u), the old model's observations
   (:prev-observations) merged into what it gives. A q-backward that gives
   u' density 0 at a possible t' is a loud error (gfi/assess-backward)."
  [{:keys [p-new p-new-args new-observations q-forward q-forward-args q-backward q-backward-args check?
           prev-observations]
    t :transform}]
  (check-transform :translator t)
  (fn [trace key]
    (let [[k-propose k-generate] (rng/split key)
          {u :choices fwd :weight fwd-retval :retval} (propose q-forward q-forward-args trace k-propose)
          s (execute t trace u {:retvals {:aux-in fwd-retval}})
          model-out (get-in s [:outputs :model-out])
          aux-out (get-in s [:outputs :aux-out])
          constraints (cm/merge model-out new-observations)
          {t' :trace} (gfi/generate p-new (vec p-new-args) constraints k-generate)]
      (if (gfi/impossible? (:score t'))
        (outside-support t')
        (let [_ (when-let [extra (seq (unconstrained-additions cm/EMPTY constraints cm/EMPTY (:choices t')))]
                  (throw (ex-info (str "translator: the new trace has choices the transform did not write: " (vec extra))
                                  {:addresses (vec extra)})))
              backward (cond q-backward (gfi/assess-backward trace t' q-backward q-backward-args aux-out k-generate)
                             (cm/empty? aux-out) {:weight 0.0 :retval none}
                             :else (throw (ex-info "translator: the transform writes auxiliary choices but there is no :q-backward"
                                                   {:aux (cm/to-map aux-out)})))
              ;; the old trace's observations are unknown here, so only u is checked
              _ (check-consumed! s (fn [sp _] (= sp :aux-in)))
              ldj (log-abs-det t trace u s (constantly false))]
          (when check?
            (let [back (execute (inverse t) t' aux-out {:retvals {:aux-in (:retval backward)}})]
              (check-round-trip! inverse-check "model choices" (:choices trace)
                                 (cm/merge (get-in back [:outputs :model-out]) prev-observations))
              (check-round-trip! inverse-check "auxiliary choices" u (get-in back [:outputs :aux-out]))))
          {:trace t' :weight (+ (- (:score t') (:score trace)) (- (:weight backward) fwd) ldj) :log-abs-det ldj})))))

(defn deterministic-translator
  "Gen.jl's DeterministicTraceTranslator: a translator whose transform maps
   the trace's choices to p-new's with no auxiliary choices. Options as for
   translator, without the proposals: {:p-new :p-new-args :new-observations
   :transform}, and :check? with :prev-observations."
  [{:keys [q-forward q-backward] :as opts}]
  (when (or q-forward q-backward)
    (throw (ex-info "A deterministic translator has no auxiliary choices; translator takes :q-forward and :q-backward"
                    {:q-forward q-forward :q-backward q-backward})))
  (translator opts))

(defn new-arguments
  "[args argdiffs] for a translator's update: p-new-args (the trace's own
   by default) and p-argdiffs (unknown changes by default)."
  [trace p-new-args p-argdiffs]
  (let [args (if p-new-args (vec p-new-args) (:args trace))]
    [args (if p-argdiffs (vec p-argdiffs) (vec (repeat (count args) diff/unknown-change)))]))

(defn extending-translator
  "Gen.jl's SimpleExtendingTraceTranslator: the trace updated to
   p-new-args (with p-argdiffs, unknown changes by default) with
   :new-observations and the choices of (q-forward trace q-forward-args…),
   mapped by :transform first when given (its model-in is the proposal's
   choices, as GenParticleFilters' ExtendingTraceTranslator, and its
   model-in retval the proposal's return value); the weight is the
   update's minus the proposal's, plus log |det J| (:log-abs-det, 0
   without a transform). The update may only add choices, and the
   transform may write no auxiliary choices (there is no backward proposal
   to weigh them)."
  [{:keys [p-new-args p-argdiffs new-observations q-forward q-forward-args] t :transform}]
  (some->> t (check-transform :extending-translator))
  (fn [trace key]
    (let [[k-propose k-update] (rng/split key)
          {u :choices fwd :weight fwd-retval :retval} (propose q-forward q-forward-args trace k-propose)
          s (when t (execute t u nil {:retvals {:model-in fwd-retval}}))
          _ (when (and s (not (cm/empty? (get-in s [:outputs :aux-out]))))
              (throw (ex-info "extending-translator: the transform writes auxiliary choices, which nothing weighs"
                              {:aux (cm/to-map (get-in s [:outputs :aux-out]))})))
          [args argdiffs] (new-arguments trace p-new-args p-argdiffs)
          {t' :trace w :weight discard :discard}
          (gfi/update trace args argdiffs (cm/merge new-observations (if t (get-in s [:outputs :model-out]) u)) k-update)]
      ;; a stopped update discards the old choices it did not reach
      (if (gfi/impossible? w)
        (outside-support t')
        (do (when-not (cm/empty? discard)
              (throw (ex-info "extending-translator: the update changed or removed choices; it may only add them"
                              {:discard (cm/to-map discard)})))
            (some-> s (check-consumed! (constantly true)))
            (let [ldj (if t (log-abs-det t u nil s (constantly false)) 0.0)]
              {:trace t' :weight (+ (- w fwd) ldj) :log-abs-det ldj}))))))

(defn updating-translator
  "GenParticleFilters' UpdatingTraceTranslator. Forward choices u come from
   (q-forward trace q-forward-args…). Without a transform, the trace is
   updated to p-new-args (unknown changes unless :p-argdiffs) with u and
   :new-observations, and the backward choices are the discard: Del Moral's
   SMC samplers. With :transform (SMCP3), the transform maps (trace, u) to
   the update's constraints and the backward choices u' (its aux-in retval
   q-forward's return value). Either way the weight is the update's + log
   q-backward(u'; t', args…) - log q-forward(u) + log |det J| (the
   continuous reads the update keeps count as unchanged outputs;
   :log-abs-det, 0 without a transform), and a q-backward that gives u'
   density 0 at a possible t' is a loud error (gfi/assess-backward).
   :check? runs the inverse (the transform's inverse, its aux-in retval
   q-backward's, the proposals swapped, back to the trace's arguments with
   :prev-observations) and throws unless it gives back the trace and u."
  [{:keys [p-new-args p-argdiffs new-observations q-forward q-forward-args q-backward q-backward-args
           check? prev-observations]
    t :transform}]
  (some->> t (check-transform :updating-translator))
  (fn [trace key]
    (let [[k-propose k-update k-check] (rng/split key 3)
          {u :choices fwd :weight fwd-retval :retval} (propose q-forward q-forward-args trace k-propose)
          [args argdiffs] (new-arguments trace p-new-args p-argdiffs)
          s (when t (execute t trace u {:retvals {:aux-in fwd-retval}}))
          {t' :trace w :weight discard :discard}
          (gfi/update trace args argdiffs (cm/merge (if t (get-in s [:outputs :model-out]) u) new-observations) k-update)]
      (if (gfi/impossible? w)
        (outside-support t')
        (let [;; an observation the new ones replace is data, not part of the
              ;; move: neither consumed nor an input that is lost
              left-alone? (fn [p] (or (not (cm/has-value? discard p)) (cm/has-value? new-observations p)))
              _ (some-> s (check-consumed! (fn [sp p] (or (= sp :aux-in) (not (left-alone? p))))))
              ldj (if t (log-abs-det t trace u s left-alone?) 0.0)
              u' (if t (get-in s [:outputs :aux-out]) discard)
              backward (cond q-backward (gfi/assess-backward trace t' q-backward q-backward-args u' k-update)
                             (cm/empty? u') {:weight 0.0 :retval none}
                             :else (throw (ex-info "updating-translator: there are backward choices but no :q-backward"
                                                   {:backward (cm/to-map u')})))]
          (when check?
            (let [s-back (when t (execute (inverse t) t' u' {:retvals {:aux-in (:retval backward)}}))
                  back-args (:args trace)
                  {t'' :trace} (gfi/update t' back-args (vec (repeat (count back-args) diff/unknown-change))
                                         (cm/merge (if t (get-in s-back [:outputs :model-out]) u') prev-observations)
                                         k-check)]
              (check-round-trip! inverse-check "trace" (:choices trace) (:choices t''))
              (when t (check-round-trip! inverse-check "forward choices" u (get-in s-back [:outputs :aux-out])))))
          {:trace t' :weight (+ w (- (:weight backward) fwd) ldj) :log-abs-det ldj})))))

(def ^:private involution-check
  "An involution's check: the involution applied again."
  "The involution applied twice")

(defn involution
  "Transform t as the involution of involutive MCMC (Gen.jl's
   SymmetricTraceTranslator), for gen.inference.mh/involutive-mh:
   (h trace fwd-choices fwd-retval proposal-args key) → {:trace
   :bwd-choices :weight :log-abs-det}, and called directly (h trace
   fwd-choices key), or (h trace fwd-choices) with a key from the implicit
   source, with no forward return value. t's model-out choices update the
   trace with key (every new choice must be written, so the update samples
   none; the key draws a model's encapsulated randomness afresh), its
   aux-out are the backward choices, and the weight is the update's plus
   log |det J|, which is also :log-abs-det: the trace's unwritten
   continuous reads that update keeps count as unchanged outputs. The
   forward return value is aux-in's retval. A move outside the support has
   weight -Inf and no :log-abs-det. With {:check? true}, the involution
   applied again must give back the trace and the forward choices; when t
   reads aux-in's retval, the check needs the backward proposal's, at the
   new trace and the backward choices: :proposal, the proposal, called on
   (trace' proposal-args…)."
  ([t] (involution t {}))
  ([t {:keys [check? proposal]}]
   (check-transform :involution t)
   (letfn [(move [trace fwd-choices fwd-retval proposal-args key]
             (let [s (execute t trace fwd-choices {:retvals {:aux-in fwd-retval}})
                   model-out (get-in s [:outputs :model-out])
                   ;; every new choice is constrained; the key draws a model's
                   ;; encapsulated randomness afresh, a pseudo-marginal refresh
                   {t' :trace w :weight discard :discard} (gfi/update trace model-out key)
                   bwd (get-in s [:outputs :aux-out])]
               (if (gfi/impossible? w)
                 (assoc (outside-support t') :bwd-choices bwd)
                 (let [_ (when-let [extra (seq (unconstrained-additions (:choices trace) model-out discard (:choices t')))]
                           (throw (ex-info (str "involution: the updated trace has new choices the transform did not write: "
                                                (vec extra))
                                           {:addresses (vec extra)})))
                       retained? (fn [p] (not (cm/has-value? discard p)))
                       _ (check-consumed! s (fn [sp p] (or (= sp :aux-in) (not (retained? p)))))
                       ldj (log-abs-det t trace fwd-choices s retained?)]
                   (when check?
                     (let [bwd-retval (delay (if proposal
                                               (:retval (gfi/assess proposal (into [t'] proposal-args) bwd (rng/key 0)))
                                               (throw (ex-info (str "involution: the check applies the transform again to the"
                                                                    " backward choices, where it reads aux-in's retval, the"
                                                                    " backward proposal's return value: give the proposal,"
                                                                    " (tf/involution t {:check? true :proposal q})")
                                                               {:transform (:name t)}))))
                           back (execute t t' bwd {:retvals {:aux-in bwd-retval}})
                           {t'' :trace} (gfi/update t' (get-in back [:outputs :model-out]) (rng/key 0))]
                       (check-round-trip! involution-check "trace" (:choices trace) (:choices t''))
                       (check-round-trip! involution-check "forward choices" fwd-choices (get-in back [:outputs :aux-out]))))
                   {:trace t' :bwd-choices bwd :weight (+ w ldj) :log-abs-det ldj}))))]
     (fn
       ([trace fwd-choices] (move trace fwd-choices none [] (gfi/next-key)))
       ([trace fwd-choices key] (move trace fwd-choices none [] key))
       ([trace fwd-choices fwd-retval proposal-args key] (move trace fwd-choices fwd-retval proposal-args key))))))
