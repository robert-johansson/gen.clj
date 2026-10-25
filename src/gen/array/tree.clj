(ns gen.array.tree
  "Trees of arrays: nested maps, vectors and seqs whose leaves are arrays or
   numbers. gen.array's tree-leaves and tree-map are these, and compile and
   the MLX backend take trees apart and refill them with them. Below
   gen.array and the backends, so a backend requires it statically."
  {:api :internal})

(set! *warn-on-reflection* true)

(defn leaves
  "The leaves of a tree of nested maps, vectors and seqs, depth first."
  [t]
  (cond (map? t) (mapcat leaves (vals t))
        (or (vector? t) (seq? t)) (mapcat leaves t)
        :else [t]))

(defn map-leaves
  "The tree t with f applied to every leaf (maps, including records, keep their
   type)."
  [f t]
  (cond (map? t) (reduce-kv (fn [m k v] (assoc m k (map-leaves f v))) t t)
        (vector? t) (mapv #(map-leaves f %) t)
        (seq? t) (doall (map #(map-leaves f %) t))
        :else (f t)))

(defn skeleton
  "t with every leaf nil: its structure, to refill."
  [t]
  (map-leaves (fn [_] nil) t))

(defn refill
  "The tree t (or its skeleton) with its leaves, in leaves' order, replaced
   by xs."
  [t xs]
  (let [it (volatile! (seq xs))]
    (map-leaves (fn [_] (let [x (first @it)] (vswap! it next) x)) t)))

(defn leaves-in
  "t's leaves in the order of skel, a skeleton equal to t's. Equal maps may
   list their keys in different orders, so a cached trace's arguments
   follow its own skeleton, not the call's."
  [skel t]
  (cond (map? skel) (mapcat (fn [k] (leaves-in (get skel k) (get t k))) (keys skel))
        (or (vector? skel) (seq? skel)) (mapcat leaves-in skel t)
        :else [t]))
