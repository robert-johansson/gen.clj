(ns gen.batched
  "N particles of a generative function at once. A particle collection runs
   the model's code once for all n particles, under the batch tracer
   (gen.array.batch): the code sees per-particle values, every op acts per
   particle and every draw is independent per particle, so the per-site
   handler cost is paid once rather than n times.

   When the code cannot treat the particles alike (it branches on a value
   that differs across them, compares or reads one on the host, or does host
   math on it), the batched run fails and the collection falls back to one
   scalar trace per particle, which gives the scalar semantics. Batching is an
   optimization. It does not change meaning for code inside gen bodies that
   works through gen.array. The limits are a raw if or case on a batched
   value in a helper function outside gen, which no macro guards, and a
   vector get or contains? with a batched key (docs/array-api.md §8a).

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

(defmacro ^:private batched-or
  "batched, or fallback if the batched run fails. The usual cause is code
   that cannot treat the particles alike (the batched-branch error, host
   math or a host collection lookup on a batched value); whatever the cause,
   the per-particle run gives the scalar semantics, and a genuine error
   surfaces there."
  [batched fallback]
  `(try ~batched
        (catch Exception _# ~fallback)))

(defn- run [gf op args n opts] (gfi/run-op gf op args (assoc opts :batch-shape [n])))

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

(defn traces
  "The n scalar traces, each rebuilt exactly from its particle's choices."
  [ps]
  (if-let [t (:trace ps)]
    (mapv #(:trace (gfi/generate (:gen-fn ps) (:args ps) % (rng/key 0)))
          (cm/unstack (plain-choices (:choices t) (:n ps)) (:n ps)))
    (:traces ps)))

(defn- unbatched [ps] (if (batched? ps) (per-particle ps (traces ps)) ps))

;; ---------------------------------------------------------------------------
;; GFI operations
;; ---------------------------------------------------------------------------

(defn simulate
  "n particles of gf on args."
  [gf args n key]
  (let [ps (->Particles n gf (vec args) nil nil)]
    (batched-or
     (assoc ps :trace (:trace (run gf :simulate args n {:key key})))
     (per-particle ps (mapv #(gfi/simulate gf args %) (rng/split key n))))))

(defn generate
  "n particles of gf on args with the shared constraints, and their weights."
  [gf args constraints n key]
  (let [ps (->Particles n gf (vec args) nil nil)]
    (batched-or
     (let [{t :trace w :weight} (run gf :generate args n {:key key :constraints constraints})]
       {:particles (assoc ps :trace t) :weights (plain w n)})
     (let [rs (mapv #(gfi/generate gf args constraints %) (rng/split key n))]
       {:particles (per-particle ps (map :trace rs)) :weights (stack-weights (map :weight rs) n)}))))

(defn update
  "Every particle updated to args with the shared constraints."
  [ps args argdiffs constraints key]
  (let [{:keys [n gen-fn]} ps
        fallback (fn [ps]
                   (let [rs (mapv #(gfi/update %1 args argdiffs constraints %2) (traces ps) (rng/split key n))]
                     {:particles (per-particle (assoc ps :args (vec args)) (map :trace rs))
                      :weights (stack-weights (map :weight rs) n)
                      :discards (mapv :discard rs)}))]
    (if (batched? ps)
      (batched-or
       (let [{t :trace w :weight d :discard}
             (run gen-fn :update args n {:prev (:trace ps) :argdiffs argdiffs :constraints constraints :key key})]
         {:particles (assoc ps :trace t :args (vec args))
          :weights (plain w n)
          :discards (if (cm/empty? d) (vec (repeat n cm/EMPTY)) (cm/unstack (plain-choices d n) n))})
       (fallback (unbatched ps)))
      (fallback ps))))

(defn regenerate
  "Every particle with the selection regenerated, at args."
  [ps args argdiffs selection key]
  (let [{:keys [n gen-fn]} ps
        fallback (fn [ps]
                   (let [rs (mapv #(gfi/regenerate %1 args argdiffs selection %2) (traces ps) (rng/split key n))]
                     {:particles (per-particle (assoc ps :args (vec args)) (map :trace rs))
                      :weights (stack-weights (map :weight rs) n)}))]
    (if (batched? ps)
      (batched-or
       (let [{t :trace w :weight}
             (run gen-fn :regenerate args n {:prev (:trace ps) :argdiffs argdiffs :selection selection :key key})]
         {:particles (assoc ps :trace t :args (vec args)) :weights (plain w n)})
       (fallback (unbatched ps)))
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
        (batched-or (assoc ps :trace (rebuild (:gen-fn ps) (:args ps) n merged)) (fallback)))
      (fallback))))

(defn choice
  "The choice at addr across the particles, as an [n …] array."
  [ps addr]
  (if (batched? ps)
    (a/tree-map #(plain % (:n ps)) ((:trace ps) addr))
    (a/stack (mapv #(% addr) (:traces ps)))))

(defn scores
  "The particles' log densities, an [n] array."
  [ps]
  (if (batched? ps)
    (plain (:score (:trace ps)) (:n ps))
    (stack-weights (map :score (:traces ps)) (:n ps))))
