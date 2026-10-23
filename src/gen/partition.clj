(ns gen.partition
  "A batched run's particles in groups: a Switch whose index differs across
   the particles of a batch runs each branch batched over its own particles
   (DESIGN.md §9, docs/array-api.md §8b). Each group runs under a part's
   backend (gen.array.batch/part-backend), which draws for a particle what
   the batch's own run would; the batch's values go in by the group's rows
   (gather-into) and the results come back (scatter). A batched trace holds
   such a call's data by group (gen.array.batch/Parts), each group's data a
   batch of its own particles.

     (groups ks)            → [[key rows] …]: the particles by key
     (gather x idx)         → x for the particles idx (resampling: gen.batched/take)
     (gather-into x idx b)  → x's particles idx as values of the part's backend b
     (join xs ns b)         → consecutive parts' data as one batch
     (consecutive gss ns)   → consecutive parts' groups, rows in the batch's order
     (merge-groups gs b)    → a batch's groups merged by key
     (scatter xs rows b)    → parts' values back where their particles are
     (trace-backend tr m b) → the batch backend a group's trace is on

   The walks go through maps and records, vectors, sets, lists and seqs.
   They keep a generative function as it is, and anything that holds no
   batched value, its identity too (a static update checks a generative
   function's, and a store's)."
  (:require [gen.array :as array]
            [gen.array.batch :as batch]
            [gen.array.protocols :as p]
            [gen.choicemap :as cm]
            [gen.gfi :as gfi]
            [gen.trace :as trace]))

(set! *warn-on-reflection* true)

(defn groups
  "The particles by key: [[key rows] …] for ks, a key per particle, in the
   order of each key's first particle, rows ascending."
  [ks]
  (vec (sort-by (comp first val) (group-by ks (range (count ks))))))

;; ---------------------------------------------------------------------------
;; Gathering: a batch's data for some of its particles
;; ---------------------------------------------------------------------------

(declare gather)

(defn- regroup
  "The Parts ps for the particles at idx (host integers: the particles of
   ps's batch the new ones are): each group's particles among them, in the
   new order, its data gathered for them; a group none is left in is gone."
  [ps idx]
  (let [gs (:groups ps)
        n (reduce + (map (comp count :rows) gs))
        group-of (int-array n)
        place-of (int-array n)]
    (doseq [[g {rows :rows}] (map-indexed vector gs)
            [i r] (map-indexed vector rows)]
      (aset group-of (int r) (int g))
      (aset place-of (int r) (int i)))
    (let [picked (reduce (fn [acc [j r]]
                           (let [g (aget group-of (int r))]
                             (assoc acc g (-> (get acc g [[] []])
                                              (update 0 conj j)
                                              (update 1 conj (aget place-of (int r)))))))
                         {}
                         (map-indexed vector idx))]
      (batch/->Parts (into [] (keep-indexed (fn [g group]
                                              (when-let [[rows places] (picked g)]
                                                (assoc group :rows rows :data (gather (:data group) places)))))
                           gs)))))

(defn- walk
  "x with each batched value v outside Parts as (leaf v), and each Parts
   regrouped for the particles idx (unchanged when idx is nil, every
   particle in order)."
  [x leaf idx]
  (let [same? (fn [xs ys] (every? true? (map identical? xs ys)))
        walk #(walk % leaf idx)]
    (cond (batch/batched? x) (leaf x)
          (batch/parts? x) (if idx (regroup x idx) x)
          (gfi/generative-function? x) x
          (map? x) (reduce-kv (fn [m k v] (let [v' (walk v)] (if (identical? v v') m (assoc m k v')))) x x)
          (vector? x) (let [v (mapv walk x)] (if (same? v x) x v))
          (set? x) (let [v (map walk x)] (if (same? v x) x (into (empty x) v)))
          (list? x) (let [v (map walk x)] (if (same? v x) x (apply list v)))
          ;; lazily, as a seq may be infinite
          (seq? x) (map walk x)
          :else x)))

(defn gather
  "x, a batch's data, for the particles idx (host integers, the particles
   of x's batch the new ones are): every batched value's particles there,
   lazily, as a batch of their number on its interned backend
   (gen.array.batch/gather); every Parts regrouped. Resampling a trace so
   costs nothing for the values never read again."
  [x idx]
  (let [a (array/array (vec idx) {:dtype :int64})
        n (count idx)]
    (walk x #(batch/gather % a [n]) idx)))

(defn gather-into
  "x, data of the batch in effect, for the particles idx (host integers; nil
   for all of them, in order) as values of the batch backend b, a part's
   (gen.array.batch/part-backend), lazily: what a part's run reads. Parts
   are regrouped, their data kept on their own backends."
  [x idx b]
  (let [a (when idx (batch/index-array b idx))]
    (walk x #(batch/gather-into % a b) idx)))

;; ---------------------------------------------------------------------------
;; Joining: parts' data as one batch
;; ---------------------------------------------------------------------------

(defprotocol IJoin
  "Generative functions whose traces of consecutive parts of a batch join
   other than field by field: a Switch's, whose traces hold one branch's
   subtrace or several groups'."
  (-join [gf traces ns b]
    "The traces (of gf) of consecutive parts of a batch, (traces j) of (ns
     j) particles, as one trace over all their particles in order, on the
     batch backend b."))

(defn- differ!
  [xs]
  (throw (array/batched-branch (str "Data that differ across the parts of a batch cannot be one batched value: "
                                    (pr-str (mapv type xs)))
                               {:types (mapv type xs)})))

(defn- holds-batched?
  "Does x hold a batched value where the walks reach (a seq as far as a run
   realizes one)?"
  [x]
  (cond (batch/batched? x) true
        (gfi/generative-function? x) false
        (map? x) (boolean (some holds-batched? (vals x)))
        (seq? x) (boolean (some holds-batched? (take gfi/realize-limit x)))
        (coll? x) (boolean (some holds-batched? x))
        :else false))

(defn- same-structure?
  "Are the choice maps cs alike at their top: all values, or all nodes of
   the same addresses?"
  [cs]
  (or (every? cm/has-value? cs)
      (and (not-any? cm/has-value? cs) (apply = (map #(set (keys (:m %))) cs)))))

(defn- joint
  "The data xs of parts walked in step, as one datum, by the fns of f:
   arrays by (arrays xs), Parts by (parts xs), traces whose maker joins
   them by (traces maker xs), choice maps that differ in structure by
   (by-part xs), and what every part shares (the same value, holding no
   batched one) as it is. A batched value is each part's own, though it be
   the same object in several (a choice an update kept)."
  [xs {:keys [arrays parts traces by-part] :as f}]
  (let [x (first xs)
        each (fn [k] (joint (mapv #(get % k) xs) f))]
    (cond
      (every? array/array? xs) (if (and (not-any? batch/batched? xs) (every? #(gfi/same-value? x %) xs)) x (arrays xs))
      (every? batch/parts? xs) (parts xs)
      (some batch/parts? xs) (differ! xs)
      (and (every? trace/trace? xs) (instance? gen.partition.IJoin (trace/maker x))
           (every? #(gfi/same-gen-fn? (trace/maker x) (trace/maker %)) xs))
      (traces (trace/maker x) xs)
      (every? gfi/generative-function? xs) (if (every? #(gfi/same-gen-fn? x %) xs) x (differ! xs))
      ;; the particles' choices differ in structure there (a Switch's
      ;; branches that differ across the parts)
      (and (every? cm/choicemap? xs) (not (same-structure? xs))) (by-part xs)
      (and (every? map? xs) (apply = (map class xs)) (apply = (map (comp set keys) xs)))
      (reduce (fn [m k] (let [v (each k)] (if (identical? v (get m k)) m (assoc m k v)))) x (keys x))
      (and (every? vector? xs) (apply = (map count xs)))
      (let [v (mapv each (range (count x)))] (if (every? true? (map identical? v x)) x v))
      (and (every? #(gfi/same-value? x %) xs) (not (holds-batched? x))) x
      ;; a seq as a run realizes it (gen.dynamic's realize!), as a list
      (and (every? seq? xs) (every? #(< (bounded-count gfi/realize-limit %) gfi/realize-limit) xs) (apply = (map count xs)))
      (apply list (map #(joint % f) (apply map vector xs)))
      :else (differ! xs))))

(declare merge-groups)

(defn consecutive
  "The groups of consecutive parts of a batch, (groupss j) those of the
   (ns j) particles of part j, their rows in the batch's order."
  [groupss ns]
  (mapcat (fn [gs o] (map (fn [g] (update g :rows (fn [rows] (mapv #(+ (long %) (long o)) rows)))) gs))
          groupss (reductions + 0 ns)))

(defn- by-part
  "The choice maps cs of consecutive parts of a batch, (cs j) of (ns j)
   particles, held by part: a Parts value, each part a group."
  [cs ns]
  (cm/->Value (batch/->Parts (vec (consecutive (map-indexed (fn [j c] [{:key [::part j] :rows (vec (range (nth ns j))) :data c}]) cs)
                                               ns)))))

(defn join
  "The data xs of consecutive parts of a batch, (xs j) of (ns j)
   particles, as one datum over all of them in order, its batched values on
   the batch backend b (gen.array.batch/join): Parts merged by key
   (merge-groups), traces walked field by field or joined by their maker
   (IJoin), choice maps that differ in structure held by part (each part a
   group, a Parts value), and a value every part shares kept. Data that
   differ otherwise (in structure, in a value that is no array, in an
   array's shape or dtype) cannot be one batch: the batched-branch error."
  [xs ns b]
  (joint xs {:arrays #(batch/join % ns b)
             :parts (fn [pss] (batch/->Parts (merge-groups (consecutive (map :groups pss) ns) b)))
             :traces (fn [gf trs] (-join gf trs ns b))
             :by-part (fn [cs] (by-part cs ns))}))

(defn merge-groups
  "groups ({:key :rows :data}) of parts of a batch, their rows in the
   batch's order, merged by key in order of each key's first group: a key's
   data joined (join), a batch of its particles on the interned batch
   backend of their number over the one b wraps."
  [groups b]
  (let [by (group-by :key groups)]
    (mapv (fn [k]
            (let [gs (by k)]
              (if (== 1 (count gs))
                (first gs)
                (let [ns (mapv (comp count :rows) gs)]
                  {:key k
                   :rows (into [] (mapcat :rows) gs)
                   :data (join (mapv :data gs) ns (batch/backend (p/-inner b) [(reduce + ns)]))}))))
          (distinct (map :key groups)))))

(defn trace-backend
  "The batch backend of a group's trace of m particles of the batch backend
   b, for code that runs on its values (a projection): its score's, or,
   when the score is shared, the interned one of m particles."
  [tr m b]
  (let [s (:score tr)]
    (if (batch/batched? s) (array/backend s) (batch/backend (p/-inner b) [m]))))

(defn scatter
  "The data xs of parts that partition the particles of a batch, (xs j)
   that of the particles at (rows j), as one datum of the batch backend b,
   each particle's from its part (gen.array.batch/scatter), walked in step
   as join's. A part's return value, say."
  [xs rows b]
  (joint xs {:arrays #(batch/scatter % rows b) :parts differ! :traces (fn [_ trs] (differ! trs)) :by-part differ!}))
