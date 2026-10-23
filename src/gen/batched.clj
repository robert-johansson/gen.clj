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

     (simulate gf args n key)                         → particles
     (generate gf args constraints n key opts?)       → {:particles :weights}
     (update ps args argdiffs constraints key opts?)  → {:particles :weights :discards}
     (regenerate ps args argdiffs selection key)      → {:particles :weights}
     (take ps indices)        → particles: the given ones (resampling)
     (select mask ps qs)      → particles: ps's where mask is true, qs's elsewhere
     (traces ps)              → the n scalar traces
     (choice ps addr)         → an [n …] array of the choice across particles
     (scores ps)              → an [n] array
     (project ps selection)   → an [n] array of the particles' projections

   Constraints are shared by every particle (observations). generate and
   update also take per-particle constraints, {:per-particle cs}: cs holds
   n choice maps, and particle i is constrained by the shared ones and by
   (cs i), which must not overlap them, as its scalar operation would be.
   Which constraints are per particle is said so, never read from a shape:
   a shared [n] array is one value of shape [n] for every particle. Weights
   are [n] arrays; :discards is a vector of n choice maps.

   A Switch whose index differs across the particles stays batched: each
   branch runs on its own particles (by partition, gen.partition), and the
   batched trace holds that call's choices by part (a
   gen.array.batch/Parts value), which the accessors read per particle.
   choice reads through it where every particle makes the choice; select
   joins traces whose particles took other branches."
  (:refer-clojure :exclude [take update])
  (:require [gen.array :as array]
            [gen.array.batch :as batch]
            [gen.choicemap :as cm]
            [gen.gfi :as gfi]
            [gen.partition :as part]
            [gen.random :as rng]
            [gen.trace :as trace]))

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
                     (array/array? x) nil
                     (cm/choicemap? x) (walk (cm/to-map x))
                     (map? x) (run! walk (vals x))
                     ;; a realized prefix of a lazy return value, as realize! forced it
                     (seq? x) (run! walk (clojure.core/take gfi/realize-limit x))
                     (coll? x) (run! walk x)))]
    (walk r)
    s))

(defn- run
  "op on gf batched over n particles. A batched comparison that nothing
   read, and that the run does not hand back, raises the fallback: it is
   what a raw if on a comparison in a helper function leaves (a batch is
   true to Clojure's if, so one branch was taken for every particle)."
  [gf op args n opts]
  (let [[[r bools] branch] (array/logging-branches
                            #(batch/unread-bools (fn [] (gfi/outside-revise (gfi/run-op gf op args (assoc opts :batch-shape [n]))))))]
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

(defn- stack-weights [ws n] (if (empty? ws) (array/zeros [0]) (array/broadcast-to (array/stack (vec ws)) [n])))

(defn- plain
  "A batched run's value as an array with the leading particle axis (a
   shared, unbatched value broadcast to every particle)."
  [x n]
  (batch/unwrap x [n]))

(defn- map-leaves [f choices] (cm/from-map (array/tree-map f (cm/to-map choices))))

(defn- particle-choices
  "The n particles' own choice maps of a batched run's choices, from each
   leaf's n values (each v n); under a Switch that ran by partition (a
   gen.array.batch/Parts value), each group's particles' own."
  [choices n each]
  (reduce (fn [cms [p v]]
            (if (batch/parts? v)
              (reduce (fn [cms {rows :rows data :data}]
                        (reduce (fn [cms [r c]] (if (cm/empty? c) cms (clojure.core/update cms r cm/set-choice p c)))
                                cms (map vector rows (particle-choices data (count rows) each))))
                      cms (:groups v))
              (reduce (fn [cms [i x]] (clojure.core/update cms i cm/set-choice p x)) cms (map-indexed vector (each v n)))))
          (vec (repeat n cm/EMPTY))
          (cm/leaves choices)))

(defn- rows-of
  "The n particles' values of a batched run's value (a tree of them, a
   product distribution's), arrays of the wrapped backend: a shared one
   broadcast to every particle."
  [v n]
  (let [x (array/tree-map #(plain % n) v)]
    (mapv (fn [i] (array/tree-map #(array/take % i) x)) (range n))))

(defn- unstacked
  "The n particles' own choice maps of a batched run's choices."
  [choices n]
  (particle-choices choices n rows-of))

(defn- by-part?
  "Does a batched trace hold choices by part (a Switch that ran by
   partition)?"
  [t]
  (boolean (some batch/parts? (cm/values (:choices t)))))

(defn- batched-choices
  "choices whose values have the leading particle axis, as batched values:
   what a batched run reads as one value per particle."
  [choices n]
  (map-leaves #(batch/wrap % [n]) choices))

(defn- rebuild
  "The batched trace of gf on args holding exactly the given per-particle
   choices (arrays with the leading particle axis)."
  [gf args n choices]
  (:trace (run gf :generate args n {:constraints (batched-choices choices n) :key (rng/key 0)})))

(declare check)

(defn traces
  "The n scalar traces, each rebuilt exactly from its particle's choices
   (after check) by the generative function that made them, and given the
   particles' :gen-fn: particles given another gen-fn, (assoc ps :gen-fn g),
   hold traces given g, which an update or regenerate revises (gfi/revise),
   per particle as when batched (review 6, R6-2)."
  [ps]
  (let [gf (:gen-fn ps)
        given (fn [t] (if (identical? (:gen-fn t) gf) t (assoc t :gen-fn gf)))]
    (if-let [t (:trace (check ps))]
      (mapv #(given (:trace (gfi/generate (trace/maker t) (:args ps) % (rng/key 0))))
            (unstacked (:choices t) (:n ps)))
      (mapv given (:traces ps)))))

(defn- unbatched [ps] (if (batched? ps) (per-particle ps (traces ps)) ps))

;; ---------------------------------------------------------------------------
;; Spot checks against the scalar handler
;; ---------------------------------------------------------------------------

(defn- spot-indices [n]
  (let [m *spot-checks*]
    (cond (= :all m) (range n)
          (>= m n) (range n)
          :else (map #(quot (* % n) m) (range m)))))

(defn- tolerance-of
  "The relative tolerance for comparing a value of dtype dt: its float
   precision (a scalar run of host parameters computes in float64 on the
   host, a batched run in the backend's float)."
  [dt]
  (case dt (:float16 :bfloat16) 1e-2 :float32 1e-5 1e-8))

(defn- close-host-values?
  "Are the host values u and v (numbers, booleans, nested vectors) equal
   within rtol (and as much absolutely), NaN equal to NaN?"
  [u v rtol]
  (cond (and (sequential? u) (sequential? v))
        (and (= (count u) (count v)) (every? true? (map #(close-host-values? %1 %2 rtol) u v)))
        (and (number? u) (number? v))
        (let [u (double u) v (double v)]
          (or (== u v)
              (and (Double/isNaN u) (Double/isNaN v))
              (<= (Math/abs (- u v)) (* rtol (max 1.0 (Math/abs u) (Math/abs v))))))
        :else (= u v)))

;; A value of the batched trace at the checked particles: a batched one's
;; rows there (an [k …] array) or a shared one as it is, its dtype (nil for
;; a choice that is no array: a label, gen.dist.dsl), and its host values
;; once read.
(deftype ^:private Read [batched? v dtype host])

(defn- read-of
  "x, a value of the batched trace gathered at the checked particles, not
   yet read."
  [x]
  (if (batch/batched? x)
    (let [a (batch/value x)] (Read. true a (array/dtype a) nil))
    (Read. false x (when (array/array? x) (array/dtype x)) nil)))

(defn- on-host
  "A read with its host values: an array's, read; a value that is no array
   as it is."
  [^Read r]
  (let [v (.-v r)]
    (Read. (.-batched? r) v (.-dtype r) (if (array/array? v) (array/->clj v) v))))

(defn- particle-host
  "The host value of a read at the j-th checked particle."
  [^Read r j]
  (if (.-batched? r) (nth (.-host r) j) (.-host r)))

(defn- arrays-of
  "The arrays of a tree of values: batched ones' arrays, shared ones as
   they are (through a Switch's choices by part too)."
  [t]
  (keep #(cond (batch/batched? %) (batch/value %) (array/array? %) %) (array/tree-leaves t)))

(declare choice-at)

(defn- reads-of
  "The batched trace t's choices, score and array return value at the
   checked particles idx, read on the host; the choices as a function of
   reference? (choice-at), giving each checked particle's choice map. On a
   lazy backend (lazy?) the whole trace is evaluated at once (the caller
   reads it next: weights, scores), then the checked rows, and only the
   rows move: two evaluations, where reading each value would cost one
   each."
  [t idx lazy?]
  (let [array-retval? (array/array? (:retval t))
        at (part/gather (cond-> {:choices (:choices t) :score (:score t)} array-retval? (assoc :retval (:retval t))) idx)]
    (when lazy?
      (array/eval! (vec (arrays-of [(cm/to-map (:choices t)) (:score t) (when array-retval? (:retval t))])))
      (array/eval! (vec (arrays-of [(cm/to-map (:choices at)) (:score at) (:retval at)]))))
    (cond-> {:score (on-host (read-of (:score at)))
             :choices (fn [reference?]
                        (particle-choices (:choices at) (count idx)
                                          (fn [v m]
                                            (let [rs (array/tree-map #(on-host (read-of %)) v)]
                                              (mapv (fn [j] (array/tree-map #(choice-at % j reference?) rs)) (range m))))))}
      array-retval? (assoc :retval (on-host (read-of (:retval at)))))))

(defn- agrees?
  "Does a scalar run's value x agree with the batched run's read r at the
   j-th checked particle? Only an array x is compared, on the host (no
   dtype need exist on the backend: Metal has no float64), within the
   precision of the narrower float; the batched run must then have given
   an array too."
  [x ^Read r j]
  (or (not (array/array? x))
      (and (some? r)
           (close-host-values? (array/->clj x) (particle-host r j)
                               (max (tolerance-of (array/dtype x)) (tolerance-of (.-dtype r)))))))

(defn- mismatch
  "A spot check's error: the batched run disagrees with the scalar handler
   at particle i, because why."
  ([i why data] (mismatch i why data nil))
  ([i why data cause]
   (ex-info (str "The batched run disagrees with the scalar handler at particle " i ": " why)
            (assoc data :particle i ::mismatch true) cause)))

(defn- choice-at
  "The j-th checked particle's choice from its read r, for a re-run on the
   JVM backend (reference?: its host value, an array for nested vectors) or
   on the batch's own (a batched one's host value made again in its dtype,
   a shared one as it is)."
  [^Read r j reference?]
  (cond reference? (let [v (particle-host r j)] (if (sequential? v) (array/array v) v))
        (.-batched? r) (array/array (particle-host r j) {:dtype (.-dtype r)})
        :else (.-v r)))

(defn- re-run!
  "Throw unless each checked particle (idx), re-run through the scalar
   handler from its choices (choice-at), gives the batched run's score and
   array return value."
  [gf args idx reads reference?]
  (doseq [[[j i] cs] (map vector (map-indexed vector idx) ((:choices reads) reference?))]
    (let [st (try (:trace (gfi/generate gf args cs (rng/key 0)))
                  (catch Exception e
                    ;; the scalar run of the particle's choices takes another path
                    (throw (mismatch i (str "the scalar run of its choices fails (" (ex-message e) ")") {} e))))]
      (when-not (and (agrees? (:score st) (:score reads) j) (agrees? (:retval st) (:retval reads) j))
        (throw (mismatch i "the code branches on a batched value where the tracer cannot see it"
                         {:scalar-score (:score st) :batched-score (particle-host (:score reads) j)}))))))

(def ^:private reference (delay (array/backend-named :jvm)))

(defn- on-reference
  "x as the JVM backend holds it: a backend array read on the host and made
   there; anything else as it is."
  [x]
  (if (and (array/backend-array? x) (not (batch/batched? x)))
    (array/with-backend @reference (array/array (array/->clj x)))
    x))

(defn- mixed-backends? [^Throwable e]
  (loop [e e]
    (cond (nil? e) false
          (and (instance? clojure.lang.ExceptionInfo e) (:backends (ex-data e))) true
          :else (recur (.getCause e)))))

(defn- lazy-batch?
  "Is t, a batched trace, on a lazy backend (where each scalar re-run would
   cost evaluations)?"
  [t]
  (let [s (:score t)]
    (and (batch/batched? s) (not (array/concrete? (batch/value s))))))

(defn- spot-check!
  "Throw unless the batched trace t of gf on args agrees with the scalar
   handler re-run on each spot-checked particle's choices, read on the host
   for all of them at once. On a lazy backend the re-runs are on the JVM
   backend (microseconds, where a scalar run on the lazy backend costs
   evaluations), unless the model reaches arrays of the lazy backend that
   the JVM run cannot mix with (it closes over them): then, as on other
   backends, on the batch's own. With no particles to check (*spot-checks*
   0) it reads nothing."
  [gf args n t]
  (let [idx (vec (spot-indices n))]
    (when (seq idx)
      (let [lazy? (lazy-batch? t)
            reads (reads-of t idx lazy?)]
        (when-not (and lazy?
                       (try (array/with-backend @reference
                              (re-run! gf (array/tree-map on-reference args) idx reads true))
                            true
                            (catch Exception e (if (mixed-backends? e) false (throw e)))))
          (re-run! gf args idx reads false))))
    t))

(defn- diverged
  "The loud error of a spot-check mismatch e after unchecked operations."
  [e]
  (ex-info (str (ex-message e) ". Earlier operations on these particles were not checked,"
                " so their results may be wrong too: branch with gen.array's where or truthy?"
                " (or define the helper with gen.dynamic/defn-guarded), or run unbatched")
           (assoc (ex-data e) ::diverged true)))

(defn check
  "ps, verified: when operations since its last spot check went unchecked
   (*verified-updates*), the trace they left is spot-checked now, against
   the generative function that made it (particles may since have been
   given another, review 7, R7-3), and a mismatch is the loud divergence
   error. The accessors (traces, scores, choice) and the particle filter's
   results call it, so results are not read from unchecked particles; so
   does a fallback, which starts from them."
  [ps]
  (when (and (batched? ps) (::unchecked? ps))
    (try (spot-check! (trace/maker (:trace ps)) (:args ps) (:n ps) (:trace ps))
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
;; Per-particle constraints
;; ---------------------------------------------------------------------------

;; In a batched run a constraint is per particle when its value is a batched
;; value, and shared otherwise: the handler reads constraints as it reads
;; every value (gen.dynamic). A batched run of per-particle constraints
;; stacks them into one choice map of batched values (batched-constraints);
;; code that runs part of a batched run's particles (a Switch whose index
;; differs across them) takes their rows of each batched constraint with
;; them (gen.array.batch/gather).

(defn- own-constraints
  "The per-particle constraints of op, (:per-particle opts): one choice map
   for each of the n particles, as a vector, or nil."
  [op opts n]
  (when-let [cs (:per-particle opts)]
    (when-not (and (sequential? cs) (= n (count cs)))
      (throw (ex-info (str (name op) ": :per-particle takes one choice map for each of the " n " particles")
                      {:op op :n n :given (if (sequential? cs) (count cs) (type cs))})))
    (mapv #(gfi/check-choice-map op "a particle's constraints" %) cs)))

(defn- constraints-of
  "Particle i's constraints: the shared ones, with its own when there are
   per-particle ones (own), which must not overlap them."
  [op shared own i]
  (if own
    (try (cm/merge shared (own i))
         (catch clojure.lang.ExceptionInfo e
           (throw (ex-info (str (name op) ": the constraints of particle " i " overlap the shared ones")
                           {:op op :particle i} e))))
    shared))

(defn- batched-constraints
  "The constraints of a batched run of the n particles: the shared ones,
   with the per-particle ones (own) stacked into batched values. Choice maps
   that differ in their addresses, or values that do not stack, cannot share
   a run (and an overlap with the shared ones fails here too, then loudly
   in the per-particle fallback)."
  [op shared own n]
  (if own
    (let [stacked (try (cm/stack own)
                       (catch Exception e
                         (throw (ex-info (str (name op) ": the per-particle constraints do not stack into one batch ("
                                              (ex-message e) ")")
                                         {:op op} e))))]
      (cm/merge shared (batched-choices stacked n)))
    shared))

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
  "n particles of gf on args with the shared constraints, and their weights.
   Options: :per-particle, n choice maps, each particle's own constraints
   beside the shared ones."
  ([gf args constraints n key] (generate gf args constraints n key nil))
  ([gf args constraints n key opts]
   (let [constraints (gfi/check-choice-map :generate "the constraints" constraints)
         own (own-constraints :generate opts n)
         ps (->Particles n gf (gfi/check-args :generate args) nil nil)]
     (batched-or
      (let [{t :trace w :weight}
            (run gf :generate args n {:key key :constraints (batched-constraints :generate constraints own n)})]
        {:particles (assoc ps :trace (spot-check! gf args n t)) :weights (plain w n)})
      (let [rs (mapv #(gfi/generate gf args (constraints-of :generate constraints own %1) %2) (range n) (rng/split key n))]
        {:particles (per-particle ps (map :trace rs)) :weights (stack-weights (map :weight rs) n)})))))

(defn update
  "Every particle updated to args with the shared constraints. Options:
   :per-particle, as generate's."
  ([ps args argdiffs constraints key] (update ps args argdiffs constraints key nil))
  ([ps args argdiffs constraints key opts]
   (gfi/check-argdiffs :update (gfi/check-args :update args) argdiffs)
   (let [{:keys [n gen-fn]} ps
         constraints (gfi/check-choice-map :update "the constraints" constraints)
         own (own-constraints :update opts n)
         fallback (fn [ps]
                    (let [rs (mapv #(gfi/update %1 args argdiffs (constraints-of :update constraints own %2) %3)
                                   (traces ps) (range n) (rng/split key n))]
                      {:particles (per-particle (assoc ps :args (vec args)) (map :trace rs))
                       :weights (stack-weights (map :weight rs) n)
                       :discards (mapv :discard rs)}))]
     (if (batched? ps)
       (batched-or
        (let [{t :trace w :weight d :discard}
              (run gen-fn :update args n {:prev (:trace ps) :argdiffs argdiffs :key key
                                          :constraints (batched-constraints :update constraints own n)})]
          {:particles (spot-check-update! ps args t)
           :weights (plain w n)
           :discards (if (cm/empty? d) (vec (repeat n cm/EMPTY)) (unstacked d n))})
        (fallback (unbatched (check ps))))
       (fallback ps)))))

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

(defn take
  "The particles at indices (a seq of n' integers), e.g. after resampling:
   every batched value of the trace gathered there, lazily, so a resampled
   trace is made without re-running the model (gen.partition/gather)."
  [ps indices]
  (let [indices (vec indices)
        n' (count indices)]
    (if (batched? ps)
      (assoc ps :n n' :trace (part/gather (:trace ps) indices))
      (assoc ps :n n' :traces (mapv (:traces ps) indices)))))

(defn- joined-select
  "select's batched trace by joining: the two traces as one of 2n particles
   (gen.partition/join), each particle taken from its own, with no run."
  [keep tp tq n]
  (part/gather (part/join [tp tq] [n n] (batch/backend array/*backend* [(* 2 n)]))
               (map-indexed (fn [i keep?] (if keep? i (+ n i))) keep)))

(defn select
  "ps's particle where mask (n booleans, or a bool array) is true and qs's
   elsewhere: the per-particle accept step of n MH chains. Each particle
   keeps its trace's maker, given ps's :gen-fn, as traces gives it: so
   particles of different makers fall back (review 7, R7-5). Traces of the
   same addresses are merged and rebuilt by the makers' generative function;
   others (a Switch whose particles took other branches) are joined, and
   each particle taken from its own (gen.partition/join)."
  [mask ps qs]
  (let [n (:n ps)
        keep (vec (if (array/array? mask) (array/->clj mask) mask))
        fallback #(per-particle ps (mapv (fn [keep? p q] (if keep? p q)) keep (traces ps) (traces qs)))
        maker (some-> (:trace ps) trace/maker)
        tp (:trace ps)
        tq (:trace qs)]
    (when-not (= n (:n qs) (count keep))
      (throw (ex-info (str "select: a mask of one boolean for each of the " n " particles of both collections")
                      {:particles [n (:n qs)] :mask (count keep)})))
    (cond
      (not (and (batched? ps) (batched? qs) (gfi/same-gen-fn? maker (trace/maker tq))))
      (fallback)

      (and (not (by-part? tp)) (not (by-part? tq)) (= (cm/addresses (:choices tp)) (cm/addresses (:choices tq))))
      (let [m (array/array keep)
            cp (:choices tp)
            cq (:choices tq)
            pick (fn pick [p q]
                   (if (vector? p)
                     (mapv pick p q)
                     (array/where (array/reshape m (into [n] (repeat (dec (array/ndim p)) 1))) p q)))
            merged (reduce (fn [c path] (cm/set-choice c path (pick (array/tree-map #(plain % n) (cm/get-choice cp path))
                                                                     (array/tree-map #(plain % n) (cm/get-choice cq path)))))
                           cm/EMPTY (cm/addresses cp))]
        ;; the rebuild is not checked
        (batched-or (assoc ps :trace (assoc (rebuild maker (:args ps) n merged) :gen-fn (:gen-fn ps)) ::unchecked? true)
                    (fallback)))

      :else
      (batched-or (assoc ps :trace (assoc (joined-select keep tp tq n) :gen-fn (:gen-fn ps))
                         ::unchecked? (boolean (or (::unchecked? ps) (::unchecked? qs))))
                  (fallback)))))

(defn- by-part-at
  "[the choices by part (a gen.array.batch/Parts) at a prefix of the path p
   of choices, the rest of p], or nil."
  [choices p]
  (some (fn [i] (let [v (cm/get-choice choices (subvec p 0 i) nil)] (when (batch/parts? v) [v (subvec p i)])))
        (range (inc (count p)))))

(defn- scattered-choice
  "The choice at the path more under the choices by part parts, an [m …]
   array over their m particles: each group's, scattered back. Loud when a
   group makes none there (p, the whole path, names it)."
  [parts more p]
  (let [gs (:groups parts)
        rows (mapv :rows gs)
        m (reduce + (map count rows))
        vs (mapv (fn [{data :data rs :rows}]
                   (if-let [[inner further] (by-part-at data more)]
                     (scattered-choice inner further p)
                     (let [v (cm/get-choice data more ::none)]
                       (when (identical? ::none v)
                         (throw (ex-info (str "choice: not every particle has a choice at " p
                                              " (the branches of a Switch whose index differs across them make different ones)")
                                         {:path p})))
                       (array/tree-map #(plain % (count rs)) v))))
                 gs)
        scatter (fn scatter [vs]
                  (if (vector? (first vs))
                    (apply mapv (fn [& xs] (scatter (vec xs))) vs)
                    (plain (batch/scatter (mapv #(batch/wrap %1 [(count %2)]) vs rows)
                                          rows (batch/backend (array/backend (first vs)) [m]))
                           m)))]
    (scatter vs)))

(defn choice
  "The choice at addr across the particles, as an [n …] array: through a
   Switch whose index differs across them when every particle makes it."
  [ps addr]
  (if (batched? ps)
    (let [t (:trace (check ps))]
      (if-let [[parts more] (by-part-at (:choices t) (cm/path addr))]
        (scattered-choice parts more (cm/path addr))
        (array/tree-map #(plain % (:n ps)) (t addr))))
    (array/stack (mapv #(% addr) (:traces ps)))))

(defn scores
  "The particles' log densities, an [n] array."
  [ps]
  (if (batched? ps)
    (plain (:score (:trace (check ps))) (:n ps))
    (stack-weights (map :score (:traces ps)) (:n ps))))

(defn project
  "The particles' log densities of the selected choices (gfi/project), an
   [n] array: the batched trace's projection under the batch tracer, or
   each particle's when that fails (a projection that runs code which
   cannot treat the particles alike)."
  [ps selection]
  (gfi/check-selection :project selection)
  (let [n (:n ps)
        each #(stack-weights (map (fn [t] (gfi/project t selection)) (traces ps)) n)]
    (if (batched? ps)
      (try (plain (batch/batched [n] (gfi/project (:trace (check ps)) selection)) n)
           (catch Exception _ (each)))
      (each))))
