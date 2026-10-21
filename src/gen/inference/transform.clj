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
   subtree unchanged, and (tcall t) runs another transform's body on the
   same spaces. The body is ordinary code over gen.array's operators; its
   control flow may depend on discrete reads, and its continuous part must
   be a bijection. The log |det J| of that part comes from automatic
   differentiation: inputs are the continuous reads, outputs the continuous
   writes, the copies of continuous reads, and, where a translator keeps
   the input trace (update), the continuous reads it retains unwritten.
   Inputs and outputs must have the same total dimension.

   Translators are functions (translator trace key) → {:trace :weight}, the
   weight the log importance weight of the new trace:
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
     take (and gen.inference.particle-filter/reweighting's :involution).
   A translator is a move-reweight kernel too: (pf/move-reweight state
   translator) translates every particle (GenParticleFilters' updating
   translators).

   (pair f g) makes two transforms each other's inverses and (involutive t)
   its own; :check? on a translator or an involution runs the inverse on the
   result and throws unless it gives back the input."
  (:refer-clojure :exclude [read + - * /])
  (:require [gen.array :as array :refer [+ - * /]]
            [gen.array.dtype :as dt]
            [gen.choicemap :as cm]
            [gen.diff :as diff]
            [gen.gfi :as gfi]
            [gen.gradients :as grad]
            [gen.random :as rng]))

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
    (throw (ex-info (str "The transform " (:name t) " has no inverse: pair it, or mark it involutive")
                    {:transform (:name t)}))))

;; ---------------------------------------------------------------------------
;; Running a transform, and its Jacobian
;; ---------------------------------------------------------------------------

(defn- choices-of [x]
  (cond (nil? x) cm/EMPTY
        (cm/choicemap? x) x
        (contains? x :choices) (:choices x)
        :else (throw (ex-info (str "A transform reads a trace or a choice map, not " (pr-str x)
                                   " (convert data with gen.choicemap/from-map)")
                              {:value x}))))

(defn- execute
  "One run of t's body on the inputs, the continuous reads replaced by subst
   ({[space path] value}) when given: the run's state."
  [t model-in aux-in subst]
  (let [st (volatile! {:inputs {:model-in (choices-of model-in) :aux-in (choices-of aux-in)}
                       :subst subst :reads [] :read-tags {}
                       :outputs {:model-out cm/EMPTY :aux-out cm/EMPTY}
                       :writes [] :written-set #{} :copies []})]
    (binding [*run* st] ((:body t)))
    @st))

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
        offsets (vec (reductions clojure.core/+ 0 (map array/size in-values)))
        n (peek offsets)
        m (reduce clojure.core/+ 0 (map #(array/size (output-value s %)) outs))]
    (when-not (= n m)
      (let [inputs (mapv #(vec (take 2 %)) ins) outputs (mapv #(vec (rest %)) outs)]
        (throw (ex-info (str "The transform's continuous part maps " n " dimensions to " m
                             "; a bijection keeps them equal (inputs " inputs ", outputs " outputs ")")
                        {:inputs inputs :outputs outputs}))))
    (if (zero? n)
      0.0
      (let [unflatten (fn [x] (into {} (map-indexed (fn [i [sp p]]
                                                      [[sp p] (array/reshape (array/slice x 0 (offsets i) (offsets (inc i)))
                                                                             (array/shape (in-values i)))])
                                                    ins)))
            g (fn [x] (let [run (execute t model-in aux-in (unflatten x))]
                        (array/concatenate (mapv #(array/flatten (output-value run %)) outs))))
            logabs (grad/log-abs-det-jacobian g (array/concatenate (mapv array/flatten in-values)))]
        (when (== ##-Inf (double (array/item logabs)))
          (throw (ex-info "The transform's Jacobian is singular here: its continuous part is not a bijection"
                          {:transform (:name t)})))
        logabs))))

(defn run
  "Transform t of model choices (or a trace) and auxiliary choices:
   {:model the model-out choices, :aux the aux-out choices, :log-abs-det},
   no input retained (every continuous read is consumed)."
  [t model-in aux-in]
  (let [s (execute t model-in aux-in nil)]
    {:model (get-in s [:outputs :model-out])
     :aux (get-in s [:outputs :aux-out])
     :log-abs-det (log-abs-det t model-in aux-in s (constantly false))}))

;; ---------------------------------------------------------------------------
;; Checks
;; ---------------------------------------------------------------------------

(defn- same-value? [x y]
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
         (every? #(same-value? (cm/get-choice a %) (cm/get-choice b %)) as))))

(defn- check-round-trip! [what before after]
  (when-not (same-choices? before after)
    (throw (ex-info (str "The inverse transform does not give back the " what)
                    {:before (cm/to-map before) :after (cm/to-map after)}))))

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

(defn- propose [q args trace key]
  (if q
    (gfi/propose q (into [trace] args) key)
    {:choices cm/EMPTY :weight 0.0}))

(defn translator
  "Gen.jl's GeneralTraceTranslator: (translator trace key) → {:trace :weight}.
   Auxiliary choices u come from (q-forward trace q-forward-args…) (none
   without it), the transform maps (trace, u) to the new model's choices and
   auxiliary choices u', and p-new generates on p-new-args with those and
   :new-observations, every choice constrained. The weight is
   log p-new(t') - log p(t) + log q-backward(u'; t', args…) - log q-forward(u; t)
   + log |det J|. :check? runs the inverse transform on (t', u') and throws
   unless it gives back (t, u), the old model's observations
   (:prev-observations) merged into what it gives."
  [{:keys [p-new p-new-args new-observations q-forward q-forward-args q-backward q-backward-args check?
           prev-observations]
    t :transform}]
  (check-transform :translator t)
  (fn [trace key]
    (let [[k-propose k-generate] (rng/split key)
          {u :choices fwd :weight} (propose q-forward q-forward-args trace k-propose)
          s (execute t trace u nil)
          model-out (get-in s [:outputs :model-out])
          aux-out (get-in s [:outputs :aux-out])
          constraints (cm/merge model-out new-observations)
          {t' :trace} (gfi/generate p-new (vec p-new-args) constraints k-generate)
          _ (when-let [extra (seq (unconstrained-additions cm/EMPTY constraints cm/EMPTY (:choices t')))]
              (throw (ex-info (str "translator: the new trace has choices the transform did not write: " (vec extra))
                              {:addresses (vec extra)})))
          bwd (if q-backward
                (:weight (gfi/assess q-backward (into [t'] q-backward-args) aux-out (gfi/backward-key k-generate)))
                (if (cm/empty? aux-out)
                  0.0
                  (throw (ex-info "translator: the transform writes auxiliary choices but there is no :q-backward"
                                  {:aux (cm/to-map aux-out)}))))
          ldj (log-abs-det t trace u s (constantly false))]
      (when check?
        (let [back (execute (inverse t) t' aux-out nil)]
          (check-round-trip! "model choices" (:choices trace)
                             (cm/merge (get-in back [:outputs :model-out]) prev-observations))
          (check-round-trip! "auxiliary choices" u (get-in back [:outputs :aux-out]))))
      {:trace t' :weight (+ (- (:score t') (:score trace)) (- bwd fwd) ldj)})))

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
   choices, as GenParticleFilters' ExtendingTraceTranslator); the weight is
   the update's minus the proposal's, plus log |det J|. The update may only
   add choices, and the transform may write no auxiliary choices (there is
   no backward proposal to weigh them)."
  [{:keys [p-new-args p-argdiffs new-observations q-forward q-forward-args] t :transform}]
  (some->> t (check-transform :extending-translator))
  (fn [trace key]
    (let [[k-propose k-update] (rng/split key)
          {u :choices fwd :weight} (propose q-forward q-forward-args trace k-propose)
          s (when t (execute t u nil nil))
          _ (when (and s (not (cm/empty? (get-in s [:outputs :aux-out]))))
              (throw (ex-info "extending-translator: the transform writes auxiliary choices, which nothing weighs"
                              {:aux (cm/to-map (get-in s [:outputs :aux-out]))})))
          [args argdiffs] (new-arguments trace p-new-args p-argdiffs)
          {t' :trace w :weight discard :discard}
          (gfi/update trace args argdiffs (cm/merge new-observations (if t (get-in s [:outputs :model-out]) u)) k-update)]
      (when-not (cm/empty? discard)
        (throw (ex-info "extending-translator: the update changed or removed choices; it may only add them"
                        {:discard (cm/to-map discard)})))
      {:trace t' :weight (+ (- w fwd) (if t (log-abs-det t u nil s (constantly false)) 0.0))})))

(defn updating-translator
  "GenParticleFilters' UpdatingTraceTranslator. Forward choices u come from
   (q-forward trace q-forward-args…). Without a transform, the trace is
   updated to p-new-args (unknown changes unless :p-argdiffs) with u and
   :new-observations, and the backward choices are the discard: Del Moral's
   SMC samplers. With :transform (SMCP3), the transform maps (trace, u) to
   the update's constraints and the backward choices u'. Either way the
   weight is the update's + log q-backward(u'; t', args…) - log q-forward(u)
   + log |det J| (the continuous reads the update keeps count as unchanged
   outputs). :check? runs the inverse (the transform's inverse, the
   proposals swapped, back to the trace's arguments with
   :prev-observations) and throws unless it gives back the trace and u."
  [{:keys [p-new-args p-argdiffs new-observations q-forward q-forward-args q-backward q-backward-args
           check? prev-observations]
    t :transform}]
  (some->> t (check-transform :updating-translator))
  (fn [trace key]
    (let [[k-propose k-update k-check] (rng/split key 3)
          {u :choices fwd :weight} (propose q-forward q-forward-args trace k-propose)
          [args argdiffs] (new-arguments trace p-new-args p-argdiffs)
          s (when t (execute t trace u nil))
          {t' :trace w :weight discard :discard}
          (gfi/update trace args argdiffs (cm/merge (if t (get-in s [:outputs :model-out]) u) new-observations) k-update)
          ldj (if t (log-abs-det t trace u s (fn [p] (not (cm/has-value? discard p)))) 0.0)
          u' (if t (get-in s [:outputs :aux-out]) discard)
          bwd (cond q-backward (:weight (gfi/assess q-backward (into [t'] q-backward-args) u' (gfi/backward-key k-update)))
                    (cm/empty? u') 0.0
                    :else (throw (ex-info "updating-translator: there are backward choices but no :q-backward"
                                          {:backward (cm/to-map u')})))]
      (when check?
        (let [s-back (when t (execute (inverse t) t' u' nil))
              back-args (:args trace)
              {t'' :trace} (gfi/update t' back-args (vec (repeat (count back-args) diff/unknown-change))
                                     (cm/merge (if t (get-in s-back [:outputs :model-out]) u') prev-observations)
                                     k-check)]
          (check-round-trip! "trace" (:choices trace) (:choices t''))
          (when t (check-round-trip! "forward choices" u (get-in s-back [:outputs :aux-out])))))
      {:trace t' :weight (+ w (- bwd fwd) ldj)})))

(defn involution
  "Transform t as the involution of involutive MCMC (Gen.jl's
   SymmetricTraceTranslator): (fn [trace fwd-choices fwd-retval
   proposal-args key]) → {:trace :bwd-choices :weight}, for
   gen.inference.mh/involutive-mh. t's model-out choices update the trace
   with key (every new choice must be written, so the update samples none;
   the key draws a model's encapsulated randomness afresh), its
   aux-out are the backward choices, and the weight is the update's plus
   log |det J|, the trace's unwritten continuous reads that update keeps
   counting as unchanged outputs. With {:check? true}, t applied again must
   give back the trace and the forward choices."
  ([t] (involution t {}))
  ([t {:keys [check?]}]
   (check-transform :involution t)
   (fn [trace fwd-choices _ _ key]
     (let [s (execute t trace fwd-choices nil)
           model-out (get-in s [:outputs :model-out])
           ;; every new choice is constrained; the key draws a model's
           ;; encapsulated randomness afresh, a pseudo-marginal refresh
           {t' :trace w :weight discard :discard} (gfi/update trace model-out key)
           _ (when-let [extra (seq (unconstrained-additions (:choices trace) model-out discard (:choices t')))]
               (throw (ex-info (str "involution: the updated trace has new choices the transform did not write: "
                                    (vec extra))
                               {:addresses (vec extra)})))
           retained? (fn [p] (not (cm/has-value? discard p)))
           ldj (log-abs-det t trace fwd-choices s retained?)
           bwd (get-in s [:outputs :aux-out])]
       (when check?
         (let [back (execute t t' bwd nil)
               {t'' :trace} (gfi/update t' (get-in back [:outputs :model-out]) (rng/key 0))]
           (check-round-trip! "trace" (:choices trace) (:choices t''))
           (check-round-trip! "forward choices" fwd-choices (get-in back [:outputs :aux-out]))))
       {:trace t' :bwd-choices bwd :weight (+ w ldj)}))))
