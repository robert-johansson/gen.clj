(ns gen.choicemap
  "Choice maps: hierarchical maps from addresses to the values of random
   choices. A choice map is a Node (a persistent map from address to child) or
   a Value (a leaf holding one choice); they are plain Clojure data with a thin
   protocol, so equality, hashing and structural sharing come for free.

(set! *warn-on-reflection* true)

   An address is any value except a vector; a vector is a path of addresses
   ([:y 3] is :y, then 3), playing the role of Gen.jl's :y => 3. Leaf order is
   canonical (addresses sorted), so to-array and from-array agree for any two
   choice maps with the same addresses."
  (:refer-clojure :exclude [empty? merge to-array])
  (:require [gen.array :as a]))

(defprotocol IChoiceMap
  (-has-value? [cm])
  (-get-value [cm])
  (-get-submap [cm addr])
  (-submaps [cm]))

(defrecord Value [v]
  IChoiceMap
  (-has-value? [_] true)
  (-get-value [_] v)
  (-get-submap [_ _] nil)
  (-submaps [_] nil))

(defrecord Node [m]
  IChoiceMap
  (-has-value? [_] false)
  (-get-value [_]
    (throw (ex-info (str "This choice map is a node, not a value. Addresses: " (vec (keys m)))
                    {:addresses (vec (keys m))})))
  (-get-submap [_ addr] (get m addr))
  (-submaps [_] (seq m)))

(def EMPTY (->Node {}))

(defn choicemap? [x] (satisfies? IChoiceMap x))

(declare from-map set-choice)

(defn choicemap
  "A choice map from address-value pairs. An address may be a path vector;
   a value that is a plain map becomes a nested choice map.
   (choicemap :slope 2.0 [:y 0] 1.5 :params {:a 1 :b 2})"
  [& kvs]
  (when (odd? (count kvs))
    (throw (ex-info "choicemap takes address-value pairs (use from-map for a single map)"
                    {:arg-count (count kvs)})))
  (reduce (fn [cm [addr v]] (set-choice cm addr (if (map? v) (from-map v) v)))
          EMPTY
          (partition 2 kvs)))

(defn from-map
  "A choice map from a nested Clojure map (maps become nodes, anything else a
   leaf)."
  [m]
  (if (map? m)
    (->Node (into {} (map (fn [[k v]] [k (from-map v)])) m))
    (->Value m)))

;; ---------------------------------------------------------------------------
;; Access
;; ---------------------------------------------------------------------------

(defn has-value?
  "Does this choice map hold a leaf value?"
  [cm]
  (and (choicemap? cm) (-has-value? cm)))

(defn get-value
  "The leaf value of a Value choice map (nil for a node or nil)."
  [cm]
  (when (has-value? cm) (-get-value cm)))

(defn- path [addr] (if (vector? addr) addr [addr]))

(defn get-submap
  "The choice map at an address or path (EMPTY when absent)."
  [cm addr]
  (reduce (fn [cm a] (or (when (choicemap? cm) (-get-submap cm a)) (reduced EMPTY)))
          cm
          (path addr)))

(defn get-choice
  "The value at an address or path, or nil when absent or a node."
  [cm addr]
  (get-value (get-submap cm addr)))

(defn empty?
  "Does the choice map hold no values?"
  [cm]
  (or (nil? cm)
      (and (not (has-value? cm))
           (every? (fn [[_ sub]] (empty? sub)) (-submaps cm)))))

(defn has-submap?
  "Is there a non-empty choice map at the address or path?"
  [cm addr]
  (not (empty? (get-submap cm addr))))

(def ^:private address-order
  "A total order on addresses: comparable values of one class by value,
   otherwise by class name."
  (fn [x y]
    (if (and (= (class x) (class y)) (instance? Comparable x))
      (compare x y)
      (let [nx (and (number? x) (number? y))]
        (if nx (compare x y) (compare (.getName ^Class (class x)) (.getName ^Class (class y))))))))

(defn submaps
  "The [address child] pairs of a node, in canonical address order."
  [cm]
  (if (has-value? cm) [] (sort-by first address-order (-submaps cm))))

(defn addresses
  "Every leaf path, in canonical order."
  [cm]
  (cond (not (choicemap? cm)) []
        (has-value? cm) [[]]
        :else (into [] (mapcat (fn [[addr sub]] (map #(into [addr] %) (addresses sub))))
                    (submaps cm))))

(defn values
  "Every leaf value, in the canonical order of addresses."
  [cm]
  (map #(get-choice cm %) (addresses cm)))

(defn to-map
  "The choice map as a nested Clojure map of values."
  [cm]
  (cond (not (choicemap? cm)) {}
        (has-value? cm) (-get-value cm)
        :else (into {} (map (fn [[k v]] [k (to-map v)])) (-submaps cm))))

;; ---------------------------------------------------------------------------
;; Functional update
;; ---------------------------------------------------------------------------

(defn set-value
  "cm with the leaf value v at a single address (fast path for handlers)."
  [cm addr v]
  (cond (instance? Node cm) (->Node (assoc (:m cm) addr (->Value v)))
        (nil? cm) (->Node {addr (->Value v)})
        :else (throw (ex-info "set-value needs a node" {:addr addr}))))

(defn set-submap
  "cm with the choice map sub at a single address."
  [cm addr sub]
  (->Node (assoc (:m (or cm EMPTY)) addr sub)))

(defn set-choice
  "cm with value v (a leaf value or a choice map) at an address or path.
   Descending through an existing leaf throws rather than discarding it."
  [cm addr v]
  (let [[a & more] (path addr)]
    (when (nil? v)
      (throw (ex-info (str "A choice value cannot be nil (at " (path addr) ")") {:path (path addr)})))
    (when (has-value? cm)
      (throw (ex-info "set-choice: the path descends through a leaf value"
                      {:path (path addr) :leaf (get-value cm)})))
    (let [m (:m (or cm EMPTY))]
      (->Node (assoc m a (if more
                           (set-choice (get m a EMPTY) (vec more) v)
                           (if (choicemap? v) v (->Value v))))))))

(defn merge
  "The union of two choice maps whose addresses are disjoint (Gen.jl's merge).
   Throws when both hold a value at an address, or a value in one sits on a
   path the other descends through."
  [x y]
  (cond (empty? x) y
        (empty? y) x
        (or (has-value? x) (has-value? y))
        (throw (ex-info "merge: the choice maps overlap" {:left (to-map x) :right (to-map y)}))
        :else (->Node (merge-with merge (:m x) (:m y)))))

(defn override
  "y laid over x: where both have entries, y's replaces x's whole subtree."
  [x y]
  (cond (nil? y) x
        (nil? x) y
        (or (has-value? x) (has-value? y)) y
        :else (->Node (merge-with override (:m x) (:m y)))))

;; ---------------------------------------------------------------------------
;; Arrays: flat views and batches
;; ---------------------------------------------------------------------------

(defn to-array
  "Every leaf, flattened and concatenated in canonical address order, as one
   1-d array (Gen.jl's to_array)."
  [cm]
  (let [leaves (values cm)]
    (if (clojure.core/empty? leaves)
      (a/zeros [0])
      (a/concatenate (mapv a/flatten leaves)))))

(defn from-array
  "A choice map with the addresses and leaf shapes of proto, filled in order
   from the 1-d array arr (the inverse of to-array)."
  [proto arr]
  (let [paths (addresses proto)
        shapes (map #(a/shape (get-choice proto %)) paths)
        sizes (map #(reduce * 1 %) shapes)
        needed (reduce + sizes)]
    (when-not (== needed (a/size arr))
      (throw (ex-info (str "from-array: proto has " needed " elements, the array " (a/size arr))
                      {:expected needed :got (a/size arr)})))
    (first
     (reduce (fn [[cm offset] [p shape n]]
               [(set-choice cm p (a/reshape (a/slice arr 0 offset (+ offset n)) shape))
                (+ offset n)])
             [EMPTY 0]
             (map vector paths shapes sizes)))))

(defn stack
  "Stack choice maps with identical addresses into one whose leaves have a new
   leading axis (the batch of particles)."
  [cms]
  (let [paths (addresses (first cms))]
    (doseq [cm (rest cms)]
      (when-not (= paths (addresses cm))
        (throw (ex-info "stack: the choice maps have different addresses"
                        {:expected paths :got (addresses cm)}))))
    (reduce (fn [acc p] (set-choice acc p (a/stack (mapv #(get-choice % p) cms))))
            EMPTY
            paths)))

(defn unstack
  "Split a choice map whose leaves have a leading batch axis of size n into n
   choice maps (the inverse of stack)."
  [cm n]
  (mapv (fn [i]
          (reduce (fn [acc p] (set-choice acc p (a/take (get-choice cm p) i)))
                  EMPTY
                  (addresses cm)))
        (range n)))

;; ---------------------------------------------------------------------------
;; Printing
;; ---------------------------------------------------------------------------

(defmethod print-method Node [cm ^java.io.Writer w]
  (.write w "#gen/choicemap ")
  (print-method (to-map cm) w))

(defmethod print-method Value [cm ^java.io.Writer w]
  (.write w "#gen/value ")
  (print-method (:v cm) w))
