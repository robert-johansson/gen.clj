(ns gen.batched
  "N particles of a generative function at once. A particle collection runs
   the model's body once for all n particles, through the same handler
   transitions with a batch shape [n] (DESIGN.md §3.3): every choice, score
   and weight gains a leading axis of size n, and the per-site handler cost is
   paid once rather than n times.

   When the model branches on a value that differs across particles (the
   gen.array/batched-branch error, docs/array-api.md §8a), the collection falls
   back to one scalar trace per particle and carries on: batching is an
   optimization, never a change of meaning.

     (simulate gf args n key)                    → particles
     (generate gf args constraints n key)        → {:particles :weights}
     (update ps args argdiffs constraints key)   → {:particles :weights :discards}
     (regenerate ps args argdiffs selection key) → {:particles :weights}
     (take ps indices)        → particles: the given ones (resampling)
     (select mask ps qs)      → particles: ps's where mask is true, qs's elsewhere
     (traces ps)              → the n scalar traces
     (choice ps addr)         → an [n] array of the choice across particles
     (scores ps)              → an [n] array

   Constraints are shared by every particle (observations). Weights are [n]
   arrays; :discards is a vector of n choice maps."
  (:refer-clojure :exclude [take update])
  (:require [gen.array :as a]
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

(defn- branch-error? [e] (boolean (:gen.array/batched-branch (ex-data e))))

(defmacro ^:private batched-or
  "batched, or fallback if it throws the batched-branch error."
  [batched fallback]
  `(try ~batched
        (catch clojure.lang.ExceptionInfo e#
          (if (branch-error? e#) ~fallback (throw e#)))))

(defn- run
  "A batched op; constraints are shared by every particle unless :shared? is
   false (per-particle choices with a leading batch axis)."
  ([gf op args n opts] (run gf op args n opts true))
  ([gf op args n opts shared?]
   (gfi/run-op gf op args (assoc opts :batch-shape [n] :shared-constraints? shared?))))

(defn- per-particle [ps traces] (assoc ps :trace nil :traces (vec traces)))

(defn- stack-weights [ws n] (if (empty? ws) (a/zeros [0]) (a/broadcast-to (a/stack (vec ws)) [n])))

(defn- full-weights [w n] (a/broadcast-to w [n]))

(defn- rebuild
  "The batched trace of gf on args holding exactly the given batched choices."
  [gf args n choices]
  (:trace (run gf :generate args n {:constraints choices :key (rng/key 0)} false)))

(defn traces
  "The n scalar traces, each rebuilt exactly from its particle's choices."
  [ps]
  (if-let [t (:trace ps)]
    (mapv #(:trace (gfi/generate (:gen-fn ps) (:args ps) % (rng/key 0))) (cm/unstack (:choices t) (:n ps)))
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
       {:particles (assoc ps :trace t) :weights (full-weights w n)})
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
          :weights (full-weights w n)
          :discards (if (cm/empty? d) (vec (repeat n cm/EMPTY)) (cm/unstack d n))})
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
         {:particles (assoc ps :trace t :args (vec args)) :weights (full-weights w n)})
       (fallback (unbatched ps)))
      (fallback ps))))

;; ---------------------------------------------------------------------------
;; Particles as a population
;; ---------------------------------------------------------------------------

(defn- map-leaves [f choices] (cm/from-map (a/tree-map f (cm/to-map choices))))

(defn take
  "The particles at indices (a seq of n' integers), e.g. after resampling."
  [ps indices]
  (let [indices (vec indices)
        n' (count indices)]
    (if (batched? ps)
      (let [idx (a/array indices {:dtype :int64})]
        (assoc ps :n n' :trace (rebuild (:gen-fn ps) (:args ps) n' (map-leaves #(a/take % idx 0) (:choices (:trace ps))))))
      (assoc ps :n n' :traces (mapv (:traces ps) indices)))))

(defn select
  "ps's particle where mask (n booleans, or a bool array) is true and qs's
   elsewhere: the per-particle accept step of n MH chains."
  [mask ps qs]
  (let [n (:n ps)
        m (a/array (vec (if (a/array? mask) (a/->clj mask) mask)))]
    (if (and (batched? ps) (batched? qs)
             (= (cm/addresses (:choices (:trace ps))) (cm/addresses (:choices (:trace qs)))))
      (let [pick (fn [p q] (a/where (a/reshape m (into [n] (repeat (dec (a/ndim p)) 1))) p q))
            paths (cm/addresses (:choices (:trace ps)))
            merged (reduce (fn [c path] (cm/set-choice c path (pick (cm/get-choice (:choices (:trace ps)) path)
                                                                     (cm/get-choice (:choices (:trace qs)) path))))
                           cm/EMPTY paths)]
        (assoc ps :trace (rebuild (:gen-fn ps) (:args ps) n merged)))
      (per-particle ps (mapv (fn [keep? p q] (if keep? p q)) (a/->clj m) (traces ps) (traces qs))))))

(defn choice
  "The choice at addr across the particles, as an [n] (or [n …]) array."
  [ps addr]
  (if (batched? ps)
    ((:trace ps) addr)
    (a/stack (mapv #(% addr) (:traces ps)))))

(defn scores
  "The particles' log densities, an [n] array."
  [ps]
  (if (batched? ps)
    (full-weights (:score (:trace ps)) (:n ps))
    (stack-weights (map :score (:traces ps)) (:n ps))))
