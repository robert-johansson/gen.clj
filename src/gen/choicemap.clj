(ns gen.choicemap
  "Choice maps: hierarchical maps from addresses to the values of random
   choices. A choice map is a Node (a persistent map from address to child) or
   a Value (a leaf holding one choice); they are plain Clojure data with a thin
   protocol, so equality, hashing and structural sharing come for free.

   An address is any value except a vector; a vector is a path of addresses
   ([:y 3] is :y, then 3), playing the role of Gen.jl's :y => 3. Leaf order is
   canonical (addresses sorted), so to-array and from-array agree for any two
   choice maps with the same addresses.

   Lookups are loud where Gen.jl's throw KeyError: get-choice where no value
   is, get-value of a node, get-submap on a path through a value. Each takes
   a not-found value to give there instead, as nth does:
   (get-choice c :x nil). The questions has-value? and has-submap? never
   throw. Clojure's get on a choice map reads the record's fields, not
   addresses."
  (:refer-clojure :exclude [empty? merge to-array])
  (:require [gen.array :as array]))

(set! *warn-on-reflection* true)

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

(defn choicemap?
  "Is x a choice map (a Value or a Node)? An instance check: satisfies? is
   slow when the answer is no, which it is for every leaf value."
  [x]
  (instance? gen.choicemap.IChoiceMap x))

(declare from-map set-choice merge)

(defn choicemap
  "A choice map from address-value pairs. An address may be a path vector;
   a value that is a choice map is set as itself, and one that is a plain
   map becomes a nested choice map.
   (choicemap :slope 2.0 [:y 0] 1.5 :params {:a 1 :b 2})

   A value that is a Clojure vector is a product value (one value of a
   product distribution, one element per factor). An array-valued choice,
   such as an mvnormal's, is given as an array: (gen.array/array [0.0 1.0])."
  [& kvs]
  (when (odd? (count kvs))
    (throw (ex-info "choicemap takes address-value pairs (use from-map for a single map)"
                    {:arg-count (count kvs)})))
  (reduce (fn [cm [addr v]] (set-choice cm addr (if (map? v) (from-map v) v)))
          EMPTY
          (partition 2 kvs)))

(defn from-map
  "A choice map from a nested Clojure map: plain maps become nodes, a choice
   map stays itself, anything else is a leaf. A vector key is a path, and
   keys whose paths overlap are an error (merge's). Nodes and values are
   records, so map? is true of them: they are told apart first, or their
   fields would be read as addresses."
  [m]
  (cond (choicemap? m) m
        (not (map? m)) (->Value m)
        (some vector? (keys m)) (reduce merge EMPTY (map (fn [[k v]] (set-choice EMPTY k (from-map v))) m))
        :else (->Node (into {} (map (fn [[k v]] [k (from-map v)])) m))))

;; ---------------------------------------------------------------------------
;; Access
;; ---------------------------------------------------------------------------

(defn path
  "The path of an address or a path: [addr] for an address, a vector
   itself. A vector inside a path is an error, since a vector is a path and
   never an address."
  [addr]
  (if (vector? addr)
    (do (dotimes [i (count addr)]
          (when (vector? (nth addr i))
            (throw (ex-info (str "A path's addresses cannot be vectors: " (pr-str addr)) {:path addr}))))
        addr)
    [addr]))

(defn ^:no-doc path*
  "path without its check, for paths built by code (a handler's per-site
   path): [addr] for an address, a vector itself."
  [addr]
  (if (vector? addr) addr [addr]))

(defn- single-address
  "addr, which f takes as one address: a vector, a path, is an error."
  [f addr]
  (when (vector? addr)
    (throw (ex-info (str f " takes a single address, not the path " (pr-str addr)) {:addr addr})))
  addr)

(declare get-submap)

(defn has-value?
  "Does this choice map hold a leaf value, or hold one at the address or
   path? A question, which never throws: false where nothing is, at a node,
   and on a path through a value."
  ([cm] (and (choicemap? cm) (-has-value? cm)))
  ([cm addr] (has-value? (get-submap cm addr nil))))

(defn empty?
  "Does the choice map hold no values?"
  [cm]
  (or (nil? cm)
      (and (not (has-value? cm))
           (every? (fn [[_ sub]] (empty? sub)) (-submaps cm)))))

(defn- walk
  "[the choice map at path p of cm (EMPTY when nothing is there), nil], or
   [nil the prefix of p that holds a value] when p goes on through one."
  [cm p]
  (loop [cm cm i 0]
    (cond (== i (count p)) [cm nil]
          (has-value? cm) [nil (subvec p 0 i)]
          :else (if-let [sub (when (choicemap? cm) (-get-submap cm (nth p i)))]
                  (recur sub (inc i))
                  [EMPTY nil]))))

(defn ^:no-doc lookup-error
  "The error of a loud lookup that found nothing it could return at path p
   of cm (what names the lookup and what it missed): an ex-info whose :reason
   is what cm holds at p (:absent, :value, :node or :through-value), with
   :path p and :at, the prefix of p that holds a value when p goes through
   one."
  [what cm p]
  (let [[sub at] (walk cm p)
        reason (cond at :through-value (has-value? sub) :value (empty? sub) :absent :else :node)]
    (ex-info (str what " at " (pr-str p) ": " (case reason
                                                 :absent "nothing is there"
                                                 :value "a value is there"
                                                 :node "nested choices are there"
                                                 :through-value (str "the path goes through the value at " (pr-str at))))
             {:reason reason :path p :at at})))

(def ^:private none (Object.))

(defn get-submap
  "The choice map at an address or path: a node, a value (a choice map
   too), or EMPTY when nothing is there. A path through a value is loud, as
   Gen.jl's KeyError; with not-found, it gives not-found there."
  ([cm addr]
   (let [sub (get-submap cm addr none)]
     (if (identical? sub none) (throw (lookup-error "get-submap: no choice map" cm (path addr))) sub)))
  ([cm addr not-found]
   (let [[sub at] (walk cm (path addr))] (if at not-found sub))))

(defn get-choice
  "The value at an address or path. Loud where there is none (nothing,
   nested choices, a path through a value), as Gen.jl's KeyError; with
   not-found, it gives not-found there, as nth does."
  ([cm addr]
   (let [v (get-choice cm addr none)]
     (if (identical? v none) (throw (lookup-error "get-choice: no value" cm (path addr))) v)))
  ([cm addr not-found]
   (let [[sub] (walk cm (path addr))] (if (has-value? sub) (-get-value sub) not-found))))

(defn get-value
  "The value of a value choice map (a leaf). Loud for a node or nothing;
   with not-found, it gives not-found there."
  ([cm] (if (has-value? cm) (-get-value cm) (throw (lookup-error "get-value: no value" cm []))))
  ([cm not-found] (if (has-value? cm) (-get-value cm) not-found)))

(defn has-submap?
  "Are there choices under the address or path: a node that holds values
   (Gen.jl's has_submap, false at a value)? A question, which never throws:
   false on a path through a value."
  [cm addr]
  (let [sub (get-submap cm addr nil)]
    (and (not (has-value? sub)) (not (empty? sub)))))

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

(defn values-shallow
  "The [address value] pairs directly at the top of a node (Gen.jl's
   get_values_shallow), in canonical order."
  [cm]
  (keep (fn [[addr sub]] (when (has-value? sub) [addr (get-value sub)])) (submaps cm)))

(defn submaps-shallow
  "The [address choice-map] pairs of the nodes directly under the top of cm
   (Gen.jl's get_submaps_shallow), in canonical order."
  [cm]
  (remove (fn [[_ sub]] (has-value? sub)) (submaps cm)))

(defn leaves
  "Every leaf as [path value], in canonical address order: one depth-first
   walk."
  [cm]
  (letfn [(add [acc prefix cm]
            (if (has-value? cm)
              (conj! acc [prefix (-get-value cm)])
              (reduce (fn [acc [addr sub]] (add acc (conj prefix addr) sub)) acc (submaps cm))))]
    (if (choicemap? cm) (persistent! (add (transient []) [] cm)) [])))

(defn addresses
  "Every leaf path, in canonical order."
  [cm]
  (mapv first (leaves cm)))

(defn values
  "Every leaf value, in the canonical order of addresses."
  [cm]
  (mapv second (leaves cm)))

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
  "cm with the leaf value v at a single address, of a node (or nil); a path
   is an error: set-choice takes paths."
  [cm addr v]
  (single-address "set-value" addr)
  (cond (instance? Node cm) (->Node (assoc (:m cm) addr (->Value v)))
        (nil? cm) (->Node {addr (->Value v)})
        :else (throw (ex-info "set-value needs a node" {:addr addr}))))

(defn remove-submap
  "cm without anything at a single address (a path is an error)."
  [cm addr]
  (single-address "remove-submap" addr)
  (if (instance? Node cm) (->Node (dissoc (:m cm) addr)) cm))

(defn- put-at
  "cm with leaf at path p, from p's i-th address on."
  [cm p ^long i leaf]
  (if (== i (count p))
    leaf
    (do (when (has-value? cm)
          (throw (ex-info "set-choice: the path descends through a leaf value" {:path p :leaf (get-value cm)})))
        (let [m (:m (or cm EMPTY))
              a (nth p i)]
          (->Node (assoc m a (put-at (get m a EMPTY) p (inc i) leaf)))))))

(defn set-choice
  "cm with value v (a leaf value or a choice map) at an address or path; at
   the empty path, v is the whole choice map. Descending through an existing
   leaf throws rather than discarding it."
  [cm addr v]
  (let [p (path addr)]
    (when (nil? v)
      (throw (ex-info (str "A choice value cannot be nil (at " p ")") {:path p})))
    (put-at cm p 0 (if (choicemap? v) v (->Value v)))))

(defn set-submap
  "cm with the choice map sub at an address or path, as get-submap reads
   it (Gen.jl's set_submap!). Descending through an existing leaf throws
   rather than discarding it."
  [cm addr sub]
  (when-not (choicemap? sub)
    (throw (ex-info (str "set-submap takes a choice map, not a " (type sub) " (set-choice takes values)")
                    {:addr addr})))
  (set-choice cm addr sub))

(defn remove-path
  "cm without anything at an address or path, and without the nodes that
   leaves empty (at the empty path, EMPTY)."
  [cm addr]
  (let [p (path addr)]
    (case (count p)
      0 EMPTY
      1 (remove-submap cm (first p))
      (let [sub (remove-path (get-submap cm [(first p)] EMPTY) (subvec p 1))]
        (if (empty? sub) (remove-submap cm (first p)) (set-submap cm (first p) sub))))))

(defn merge
  "The union of choice maps whose addresses are disjoint (Gen.jl's merge;
   EMPTY of none). Throws when two hold a value at an address, or a value
   in one sits on a path another descends through."
  ([] EMPTY)
  ([x] x)
  ([x y]
   (cond (empty? x) y
         (empty? y) x
         (or (has-value? x) (has-value? y))
         (throw (ex-info "merge: the choice maps overlap" {:left (to-map x) :right (to-map y)}))
         :else (->Node (merge-with merge (:m x) (:m y)))))
  ([x y & more] (reduce merge (merge x y) more)))

(defn override
  "y laid over x: where both have entries, y's replaces x's whole subtree;
   an empty y leaves x (a value too) as it is."
  [x y]
  (cond (or (nil? y) (empty? y)) x
        (nil? x) y
        (or (has-value? x) (has-value? y)) y
        :else (->Node (merge-with override (:m x) (:m y)))))

;; ---------------------------------------------------------------------------
;; Arrays: flat views and batches
;; ---------------------------------------------------------------------------

(defn pair
  "Gen.jl's pair: a choice map holding a under key-a and b under key-b."
  [a b key-a key-b]
  (when (= key-a key-b)
    (throw (ex-info "pair: the two keys must differ" {:key key-a})))
  (when (some vector? [key-a key-b])
    (throw (ex-info "pair takes two single addresses, not paths" {:keys [key-a key-b]})))
  (-> EMPTY (set-submap key-a a) (set-submap key-b b)))

(defn unpair
  "Gen.jl's unpair: [the submap at key-a, the submap at key-b] of c, which
   must hold nothing else."
  [c key-a key-b]
  (when (some vector? [key-a key-b])
    (throw (ex-info "unpair takes two single addresses, not paths" {:keys [key-a key-b]})))
  (let [others (remove #{key-a key-b} (map first (submaps c)))]
    (when (seq others)
      (throw (ex-info (str "unpair: the choice map holds more than " (pr-str key-a) " and " (pr-str key-b))
                      {:keys (vec others)})))
    [(get-submap c key-a) (get-submap c key-b)]))

(defn to-array
  "Every leaf, flattened and concatenated in canonical address order, as one
   1-d array (Gen.jl's to_array)."
  [cm]
  (let [vs (values cm)]
    (if (clojure.core/empty? vs)
      (array/zeros [0])
      (array/concatenate (mapv array/flatten vs)))))

(defn from-array
  "A choice map with the addresses, leaf shapes and leaf dtypes of proto,
   filled in order from the 1-d array arr (the inverse of to-array, which
   concatenates into one dtype: a bool choice comes back a bool)."
  [proto arr]
  (let [ls (leaves proto)
        paths (map first ls)
        ;; a host value's dtype is weak: its kind's default on arr's backend
        dtypes (mapv #(array/weak-dtype (second %) (array/backend arr)) ls)
        shapes (map #(array/shape (second %)) ls)
        sizes (map #(reduce * 1 %) shapes)
        needed (reduce + sizes)]
    (when-not (== needed (array/size arr))
      (throw (ex-info (str "from-array: proto has " needed " elements, the array " (array/size arr))
                      {:expected needed :got (array/size arr)})))
    (first
     (reduce (fn [[cm offset] [p shape n dtype]]
               (let [v (array/reshape (array/slice arr 0 offset (+ offset n)) shape)]
                 [(set-choice cm p (if (= dtype (array/dtype v)) v (array/astype v dtype)))
                  (+ offset n)]))
             [EMPTY 0]
             (map vector paths shapes sizes dtypes)))))

(defn- stack-values
  "Values stacked along a new leading axis; a vector value (a product
   distribution's) is stacked element by element."
  [vs]
  (if (vector? (first vs))
    (apply mapv (fn [& xs] (stack-values (vec xs))) vs)
    (array/stack vs)))

(defn stack
  "Stack choice maps with identical addresses into one whose leaves have a new
   leading axis (the batch of particles)."
  [cms]
  (let [lss (mapv leaves cms)
        paths (mapv first (first lss))]
    (doseq [ls (rest lss)
            :let [got (mapv first ls)]]
      (when-not (= paths got)
        (throw (ex-info "stack: the choice maps have different addresses" {:expected paths :got got}))))
    (reduce (fn [acc [i p]] (set-choice acc p (stack-values (mapv #(second (nth % i)) lss))))
            EMPTY
            (map-indexed vector paths))))

(defn unstack
  "Split a choice map whose leaves have a leading batch axis of size n into n
   choice maps (the inverse of stack)."
  [cm n]
  (let [ls (leaves cm)]
    ;; a value at the root (a distribution's choice map) is at the path []
    (mapv (fn [i] (reduce (fn [acc [p v]] (set-choice acc p (array/tree-map #(array/take % i) v))) EMPTY ls))
          (range n))))

;; ---------------------------------------------------------------------------
;; Printing
;; ---------------------------------------------------------------------------

(defmethod print-method Node [cm ^java.io.Writer w]
  (.write w "#gen/choicemap ")
  (print-method (to-map cm) w))

(defmethod print-method Value [cm ^java.io.Writer w]
  (.write w "#gen/value ")
  (print-method (:v cm) w))
