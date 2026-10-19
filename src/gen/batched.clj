(ns gen.batched
  "N particles of a generative function at once. A particle collection runs
   the model's code once for all n particles, under the batch tracer
   (gen.array.batch): the code sees per-particle values, every op acts per
   particle and every draw is independent per particle, so the per-site
   handler cost is paid once rather than n times.

   When the code cannot treat the particles alike (it branches on a value
   that differs across them, compares or reads one on the host, or does host
   math on it), the batched run fails and the collection falls back to one
   scalar trace per particle, which gives the scalar semantics; fallback-reason
   says why. Three layers raise the fallback:
   - Inside gen bodies the gen macro guards every if and case, and
     clojure.core's truthiness tests and lookups (not, filter, every?, get, …;
     gen.dynamic.core).
   - Anywhere, the tracer raises it for a host read, comparison or hash of a
     batched value, and for a batched comparison that nothing read by the end
     of the run, nor the trace holds: what a raw if on it in a helper
     function leaves, when nothing else uses it.
   - Spot checks. After a batched simulate or generate, and on a schedule of
     updates and regenerates (*verified-updates*), *spot-checks* evenly spaced
     particles are re-run through the scalar handler (the ground truth) from
     their choices, and a different score or array return value, or a
     re-run that fails, means the fallback (or, after unchecked updates, a
     loud error). Results are checked when read (check): the accessors
     below, and a fallback, check a collection whose last operations went
     unchecked.

   What no layer sees for certain is a plain helper function (outside gen
   bodies) that branches on a batched value in other ways: a raw if or case
   on a choice value (say, a bernoulli's), a vector lookup keyed by one, or a
   raw if on a comparison that is also used elsewhere. Spot checks catch it
   only with high probability, since a branch few particles take can escape
   them. Define such helpers with gen.dynamic/defn-guarded, which guards
   them as gen bodies are; or bind *spot-checks* to :all to check every
   particle (a scalar run each) when validating a model's batched form.
   docs/array-api.md §8a–§8b.

     (simulate gf args n key)                    → particles
     (generate gf args constraints n key)        → {:particles :weights}
     (update ps args argdiffs constraints key)   → {:particles :weights :discards}
     (regenerate ps args argdiffs selection key) → {:particles :weights}
     (take ps indices)        → particles: the given ones (resampling)
     (select mask ps qs)      → particles: ps's where mask is true, qs's elsewhere
     (traces ps)              → the n scalar traces
     (choice ps addr)         → an [n …] array of the choice across particles
     (scores ps)              → an [n] array

   Constraints are shared by every particle (observations). Weights are [n]
   arrays; :discards is a vector of n choice maps."
  (:refer-clojure :exclude [take update])
  (:require [gen.array :as a]
            [gen.array.batch :as batch]
            [gen.choicemap :as cm]
            [gen.gfi :as gfi]
            [gen.random :as rng]))

(set! *warn-on-reflection* true)

(defrecord Particles [n gen-fn args trace traces])

(defn particles? [x] (instance? Particles x))

(defn batched?
  "Does ps still hold one batched trace (rather than the per-particle
   fallback)?"
  [ps]
  (some? (:trace ps)))

(defn size [ps] (:n ps))

(def ^:dynamic *spot-checks*
  "How many evenly spaced particles of a batched run are verified against
   the scalar handler: a number (8), :all, or 0 for none."
  8)

(def ^:dynamic *verified-updates*
  "Updates and regenerates of a collection are spot-checked on a schedule:
   the first *verified-updates* (2), then the 4th, 8th, 16th and so on. A
   check re-runs whole particles, which for a growing trace (a particle
   filter) costs the trace's size, so checking every step would make a long
   filter quadratic; the doubling schedule costs about two checks of the
   final trace. A mismatch found after unchecked updates is a loud error, not
   a fallback: those updates' results may be wrong too."
  2)

(defn fallback-reason
  "Why ps fell back to per-particle traces: {:message :data} of the failure
   of the batched run, or nil."
  [ps]
  (:fallback-reason ps))

(defn- fallen-back
  "The fallback result r (particles, or a map holding them) marked with the
   failure e of the batched run."
  [r ^Exception e]
  (let [reason {:message (.getMessage e) :data (ex-data e)}]
    (if (particles? r)
      (assoc r :fallback-reason reason)
      (clojure.core/update r :particles assoc :fallback-reason reason))))

(defmacro ^:private batched-or
  "batched, or fallback if the batched run fails. The usual cause is code
   that cannot treat the particles alike (the batched-branch error, host
   math or a host collection lookup on a batched value, a failed spot
   check); whatever the cause, the per-particle run gives the scalar
   semantics, and a genuine error surfaces there."
  [batched fallback]
  `(try ~batched
        (catch Exception e#
          (if (::diverged (ex-data e#)) (throw e#) (fallen-back ~fallback e#)))))

(defn- handed-back
  "The batched values a batched run's result hands back, anywhere in it (its
   trace's choices, return value, sub-traces and their return values, such
   as an Unfold's states), as an identity set."
  [r]
  (let [s (java.util.Collections/newSetFromMap (java.util.IdentityHashMap.))
        walk (fn walk [x]
               (cond (batch/batched? x) (.add s x)
                     (a/array? x) nil
                     (cm/choicemap? x) (walk (cm/to-map x))
                     (map? x) (run! walk (vals x))
                     ;; a realized prefix of a lazy return value, as realize! forced it
                     (seq? x) (run! walk (clojure.core/take 10000 x))
                     (coll? x) (run! walk x)))]
    (walk r)
    s))

(defn- run
  "op on gf batched over n particles. A batched comparison that nothing
   read, and that the run does not hand back, raises the fallback: it is
   what a raw if on a comparison in a helper function leaves (a batch is
   true to Clojure's if, so one branch was taken for every particle)."
  [gf op args n opts]
  (let [[[r bools] branch] (a/logging-branches
                            #(batch/unread-bools (fn [] (gfi/run-op gf op args (assoc opts :batch-shape [n])))))]
    ;; a batched-branch error the code caught still means the particles differ
    (when branch (throw branch))
    (when (seq bools)
      (let [kept ^java.util.Set (handed-back r)]
        (when-not (every? #(.contains kept %) bools)
          (throw (ex-info (str "A batched comparison was never read. Either a raw if, when or case on it (in a"
                               " helper function outside gen) took one branch for every particle, or it is"
                               " unused")
                          {:gen.array/batched-branch true ::unread-comparison true})))))
    r))

(defn- per-particle [ps traces] (assoc ps :trace nil :traces (vec traces)))

(defn- stack-weights [ws n] (if (empty? ws) (a/zeros [0]) (a/broadcast-to (a/stack (vec ws)) [n])))

(defn- plain
  "A batched run's value as an array with the leading particle axis (a
   shared, unbatched value broadcast to every particle)."
  [x n]
  (batch/unwrap x [n]))

(defn- map-leaves [f choices] (cm/from-map (a/tree-map f (cm/to-map choices))))

(defn- plain-choices [choices n] (map-leaves #(plain % n) choices))

(defn- rebuild
  "The batched trace of gf on args holding exactly the given per-particle
   choices (arrays with the leading particle axis)."
  [gf args n choices]
  (:trace (run gf :generate args n {:constraints (map-leaves #(batch/wrap % [n]) choices) :key (rng/key 0)})))

(declare check)

(defn traces
  "The n scalar traces, each rebuilt exactly from its particle's choices
   (after check)."
  [ps]
  (if-let [t (:trace (check ps))]
    (mapv #(:trace (gfi/generate (:gen-fn ps) (:args ps) % (rng/key 0)))
          (cm/unstack (plain-choices (:choices t) (:n ps)) (:n ps)))
    (:traces ps)))

(defn- unbatched [ps] (if (batched? ps) (per-particle ps (traces ps)) ps))

;; ---------------------------------------------------------------------------
;; Spot checks against the scalar handler
;; ---------------------------------------------------------------------------

(defn- particle-value
  "Particle i's value of x (a shared value is every particle's)."
  [x i]
  (if (batch/batched? x) (a/take (batch/value x) i 0) x))

(defn- spot-indices [n]
  (let [m *spot-checks*]
    (cond (= :all m) (range n)
          (>= m n) (range n)
          :else (map #(quot (* % n) m) (range m)))))

(defn- same-value?
  "Do a scalar run's value x and a batched run's particle value y agree?
   Only arrays are compared."
  [x y]
  (or (not (a/array? x))
      (and (a/array? y)
           (= (a/shape x) (a/shape y))
           (a/allclose? (a/astype x :float64) (a/astype y :float64) {:rtol 1e-8 :atol 1e-8 :equal-nan? true}))))

(defn- spot-check!
  "Throw unless the batched trace t of gf on args agrees with the scalar
   handler re-run on each spot-checked particle's choices."
  [gf args n t]
  (doseq [i (spot-indices n)]
    (let [st (try (:trace (gfi/generate gf args (map-leaves #(particle-value % i) (:choices t)) (rng/key 0)))
                  (catch Exception e
                    ;; the scalar run of the particle's choices takes another path
                    (throw (ex-info (str "The batched run disagrees with the scalar handler at particle " i
                                         ": the scalar run of its choices fails (" (ex-message e) ")")
                                    {:particle i ::mismatch true} e))))
          score (particle-value (:score t) i)
          retval (particle-value (:retval t) i)]
      (when-not (and (same-value? (:score st) score) (same-value? (:retval st) retval))
        (throw (ex-info (str "The batched run disagrees with the scalar handler at particle " i
                             ": the code branches on a batched value where the tracer cannot see it")
                        {:particle i :scalar-score (:score st) :batched-score score ::mismatch true})))))
  t)

(defn- diverged
  "The loud error of a spot-check mismatch e after unchecked operations."
  [e]
  (ex-info (str (ex-message e) ". Earlier operations on these particles were not checked,"
                " so their results may be wrong too: branch with gen.array's where or truthy?"
                " (or define the helper with gen.dynamic/defn-guarded), or run unbatched")
           (assoc (ex-data e) ::diverged true)))

(defn check
  "ps, verified: when operations since its last spot check went unchecked
   (*verified-updates*), the trace they left is spot-checked now, and a
   mismatch is the loud divergence error. The accessors (traces, scores,
   choice) and the particle filter's results call it, so results are not
   read from unchecked particles; so does a fallback, which starts from
   them."
  [ps]
  (when (and (batched? ps) (::unchecked? ps))
    (try (spot-check! (:gen-fn ps) (:args ps) (:n ps) (:trace ps))
         (catch clojure.lang.ExceptionInfo e
           (throw (if (::mismatch (ex-data e)) (diverged e) e)))))
  ps)

(defn- check-update?
  "Is the update of index k (from 0) on the spot-check schedule?"
  [k]
  (or (< k *verified-updates*) (zero? (bit-and (inc k) k))))

(defn- spot-check-update!
  "ps holding t, the batched trace of an update or regenerate on args,
   spot-checked on the schedule (*verified-updates*)."
  [ps args t]
  (let [k (:updates ps 0)
        check? (check-update? k)]
    (when check?
      (try (spot-check! (:gen-fn ps) args (:n ps) t)
           (catch clojure.lang.ExceptionInfo e
             (throw (if (and (::unchecked? ps) (::mismatch (ex-data e))) (diverged e) e)))))
    (assoc ps :trace t :args (vec args) :updates (inc k) ::unchecked? (or (::unchecked? ps) (not check?)))))

;; ---------------------------------------------------------------------------
;; GFI operations
;; ---------------------------------------------------------------------------

(defn simulate
  "n particles of gf on args."
  [gf args n key]
  (let [ps (->Particles n gf (gfi/check-args :simulate args) nil nil)]
    (batched-or
     (assoc ps :trace (spot-check! gf args n (:trace (run gf :simulate args n {:key key}))))
     (per-particle ps (mapv #(gfi/simulate gf args %) (rng/split key n))))))

(defn generate
  "n particles of gf on args with the shared constraints, and their weights."
  [gf args constraints n key]
  (let [constraints (gfi/check-choice-map :generate "the constraints" constraints)
        ps (->Particles n gf (gfi/check-args :generate args) nil nil)]
    (batched-or
     (let [{t :trace w :weight} (run gf :generate args n {:key key :constraints constraints})]
       {:particles (assoc ps :trace (spot-check! gf args n t)) :weights (plain w n)})
     (let [rs (mapv #(gfi/generate gf args constraints %) (rng/split key n))]
       {:particles (per-particle ps (map :trace rs)) :weights (stack-weights (map :weight rs) n)}))))

(defn update
  "Every particle updated to args with the shared constraints."
  [ps args argdiffs constraints key]
  (gfi/check-argdiffs :update (gfi/check-args :update args) argdiffs)
  (let [{:keys [n gen-fn]} ps
        constraints (gfi/check-choice-map :update "the constraints" constraints)
        fallback (fn [ps]
                   (let [rs (mapv #(gfi/update %1 args argdiffs constraints %2) (traces ps) (rng/split key n))]
                     {:particles (per-particle (assoc ps :args (vec args)) (map :trace rs))
                      :weights (stack-weights (map :weight rs) n)
                      :discards (mapv :discard rs)}))]
    (if (batched? ps)
      (batched-or
       (let [{t :trace w :weight d :discard}
             (run gen-fn :update args n {:prev (:trace ps) :argdiffs argdiffs :constraints constraints :key key})]
         {:particles (spot-check-update! ps args t)
          :weights (plain w n)
          :discards (if (cm/empty? d) (vec (repeat n cm/EMPTY)) (cm/unstack (plain-choices d n) n))})
       (fallback (unbatched (check ps))))
      (fallback ps))))

(defn regenerate
  "Every particle with the selection regenerated, at args."
  [ps args argdiffs selection key]
  (gfi/check-argdiffs :regenerate (gfi/check-args :regenerate args) argdiffs)
  (gfi/check-selection :regenerate selection)
  (let [{:keys [n gen-fn]} ps
        fallback (fn [ps]
                   (let [rs (mapv #(gfi/regenerate %1 args argdiffs selection %2) (traces ps) (rng/split key n))]
                     {:particles (per-particle (assoc ps :args (vec args)) (map :trace rs))
                      :weights (stack-weights (map :weight rs) n)}))]
    (if (batched? ps)
      (batched-or
       (let [{t :trace w :weight}
             (run gen-fn :regenerate args n {:prev (:trace ps) :argdiffs argdiffs :selection selection :key key})]
         {:particles (spot-check-update! ps args t) :weights (plain w n)})
       (fallback (unbatched (check ps))))
      (fallback ps))))

;; ---------------------------------------------------------------------------
;; Particles as a population
;; ---------------------------------------------------------------------------

(defn- gather
  "x with every batched value's particles gathered at idx, as a new batch of
   n' particles: through traces, choice maps, records, maps and vectors, so
   a resampled trace is made without re-running the model."
  [x idx n']
  (cond (batch/batched? x) (batch/gather x idx [n'])
        (record? x) (reduce-kv (fn [m k v] (assoc m k (gather v idx n'))) x x)
        (map? x) (into (empty x) (map (fn [[k v]] [k (gather v idx n')])) x)
        (vector? x) (mapv #(gather % idx n') x)
        :else x))

(defn take
  "The particles at indices (a seq of n' integers), e.g. after resampling."
  [ps indices]
  (let [indices (vec indices)
        n' (count indices)]
    (if (batched? ps)
      (assoc ps :n n' :trace (gather (:trace ps) (a/array indices {:dtype :int64}) n'))
      (assoc ps :n n' :traces (mapv (:traces ps) indices)))))

(defn select
  "ps's particle where mask (n booleans, or a bool array) is true and qs's
   elsewhere: the per-particle accept step of n MH chains."
  [mask ps qs]
  (let [n (:n ps)
        keep (vec (if (a/array? mask) (a/->clj mask) mask))
        fallback #(per-particle ps (mapv (fn [keep? p q] (if keep? p q)) keep (traces ps) (traces qs)))]
    (if (and (batched? ps) (batched? qs)
             (= (cm/addresses (:choices (:trace ps))) (cm/addresses (:choices (:trace qs)))))
      (let [m (a/array keep)
            cp (:choices (:trace ps))
            cq (:choices (:trace qs))
            pick (fn pick [p q]
                   (if (vector? p)
                     (mapv pick p q)
                     (a/where (a/reshape m (into [n] (repeat (dec (a/ndim p)) 1))) p q)))
            merged (reduce (fn [c path] (cm/set-choice c path (pick (a/tree-map #(plain % n) (cm/get-choice cp path))
                                                                     (a/tree-map #(plain % n) (cm/get-choice cq path)))))
                           cm/EMPTY (cm/addresses cp))]
        ;; the rebuild is not checked
        (batched-or (assoc ps :trace (rebuild (:gen-fn ps) (:args ps) n merged) ::unchecked? true) (fallback)))
      (fallback))))

(defn choice
  "The choice at addr across the particles, as an [n …] array."
  [ps addr]
  (if (batched? ps)
    (a/tree-map #(plain % (:n ps)) ((:trace (check ps)) addr))
    (a/stack (mapv #(% addr) (:traces ps)))))

(defn scores
  "The particles' log densities, an [n] array."
  [ps]
  (if (batched? ps)
    (plain (:score (:trace (check ps))) (:n ps))
    (stack-weights (map :score (:traces ps)) (:n ps))))
