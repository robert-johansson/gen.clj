(ns gen.device.stage
  "A staging backend: gen.array's operations recorded as a graph instead of
   computed, so that code written against gen.array (a model's density, a
   kernel's move, gen.random's draws, the tape's gradients) becomes a
   program for another compiler: the device program of bean genclj-a1hu.
   Research prototype, outside the core (ext/device).

   (stage f arg-specs) runs f once on staged parameters, one per leaf of
   arg-specs (trees of [shape dtype] pairs, or of staged keys' specs from
   key-spec), and returns the program: {:graph :inputs :outputs}, where
   the outputs are f's result's leaves. Every gen.array op on a staged value
   adds a node; the tape (gen.array.tape) wraps this backend as it wraps
   the JVM's, so value-and-grad records the gradient's nodes in the same
   graph, and gen.random's draws on a staged key record Threefry's.

   Nodes: {:id :op :args :shape :dtype :attrs}. :args are node ids or host
   numbers as {:const x}. Ops: gen.array's elementwise op keywords; :param, :constant,
   :iota, :convert, :reduce, :reshape, :transpose, :broadcast,
   :concatenate, :slice, :matmul; the bit ops :bit-xor, :bit-or,
   :shift-left, :shift-right-logical and :bitcast (the draws' Threefry);
   :scan, a loop whose body is a graph of its own, and :result, one of a
   scan's results.

   Reading a staged value on the host (truthy?, item, ->clj) is a loud
   error, as in array/compile: the program is staged once and run many
   times."
  {:api :internal}
  (:refer-clojure :exclude [+ - * / < > <= >= ==])
  (:require [gen.array :as array :refer [+ - * / < > <= >= ==]]
            [gen.array.dtype :as dt]
            [gen.array.protocols :as p]
            [gen.array.shape :as sh]
            [gen.array.tree :as tree]
            [gen.choicemap :as cm]
            [gen.random.threefry :as threefry]))

(set! *warn-on-reflection* true)

;; ---------------------------------------------------------------------------
;; Graphs and staged values
;; ---------------------------------------------------------------------------

(deftype Graph [parent ^java.util.ArrayList nodes ^java.util.LinkedHashMap captures])
;; nodes: the nodes in order; captures: an outer value (by identity) → the
;; :param node standing for it here (a scan body's free variables)

(defn- graph [parent] (Graph. parent (java.util.ArrayList.) (java.util.LinkedHashMap.)))

(def ^:dynamic *graph*
  "The graph that ops record into (the innermost one being staged)."
  nil)

(declare backend)

(deftype Staged [^Graph g ^long id shape dtype]
  p/IArray
  (-backend [_] backend)
  Object
  (toString [_] (str "#staged[" id " " dtype " " shape "]")))

(defmethod print-method Staged [^Staged x ^java.io.Writer w] (.write w (.toString x)))

(defn staged? [x] (instance? Staged x))

(defn- add-node!
  "A new node of op in graph g, returning its staged value."
  [^Graph g op args shape dtype attrs]
  (let [id (.size ^java.util.ArrayList (.-nodes g))]
    (.add ^java.util.ArrayList (.-nodes g) {:id id :op op :args args :shape (vec shape) :dtype dtype :attrs attrs})
    (Staged. g id (vec shape) dtype)))

(defn- current ^Graph []
  (or *graph* (throw (ex-info "A staged op outside stage: the staging backend runs inside gen.device.stage/stage" {}))))

(defn- ancestor? [^Graph a ^Graph g]
  (loop [^Graph g (.-parent g)] (and g (or (identical? a g) (recur (.-parent g))))))

(defn- capture
  "The id in graph g of x, a staged value of g or of an enclosing graph
   (captured as a :param of g, through every graph between)."
  ^long [^Graph g ^Staged x]
  (cond (identical? (.-g x) g) (.-id x)
        (ancestor? (.-g x) g)
        (let [^java.util.LinkedHashMap caps (.-captures g)]
          (or (.get caps x)
              (let [outer (capture (.-parent g) x)
                    _ outer
                    ^Staged p (add-node! g :param [] (.-shape x) (.-dtype x) {:capture true})]
                (.put caps x (.-id p))
                (.-id p))))
        :else (throw (ex-info "A staged value used outside the graph that made it (escaped from a scan body?)" {:value (str x)}))))

(defn- operand
  "An operand of a node of graph g: a node id, or a host scalar as {:const x}."
  [^Graph g x]
  (cond (staged? x) (capture g x)
        (array/host? x) {:const x}
        :else (throw (ex-info (str "Not a staged value or host scalar: " (pr-str x)) {:value x}))))

(defn- op! [op args shape dtype & [attrs]]
  (let [g (current)]
    (add-node! g op (mapv #(operand g %) args) shape dtype attrs)))

(defn- shape-of [x] (if (staged? x) (.-shape ^Staged x) []))

;; ---------------------------------------------------------------------------
;; Draws: Threefry-2x32-20 on staged keys, as gen.random.threefry on host ones
;; ---------------------------------------------------------------------------

(declare threefry-words)

(deftype StagedKey [words]
  ;; words: a staged uint32 [2]
  p/IStagedKey
  (-split-key [_ n] (let [ws (threefry-words words (array/arange 0 n 1 {:dtype :uint32}))]
                      (mapv #(StagedKey. (array/reshape (array/slice ws 0 % (inc %)) [2])) (range n))))
  (-fold-in-key [_ data] (StagedKey. (array/reshape (threefry-words words (array/array [data] {:dtype :uint32})) [2])))
  (-key-words [_] words))

(defn- bit [op x y] (op! op [x y] (sh/broadcast (shape-of x) (shape-of y)) :uint32))

(defn- add32 [x y] (array/+ x y))

(defn- rotl32 [x r] (bit :bit-or (bit :shift-left x (long r)) (bit :shift-right-logical x (long (- 32 r)))))

(defn- rounds [[x0 x1] rs]
  (reduce (fn [[x0 x1] r] (let [x0 (add32 x0 x1)] [x0 (bit :bit-xor (rotl32 x1 r) x0)])) [x0 x1] rs))

(defn threefry-words
  "Threefry-2x32-20 of the counters [0 c] (c a uint32 array) under the
   staged key words (uint32 [2]): the words, [.. 2] (the last axis w0, w1),
   for counters below 2^32, as gen.random.threefry/split's."
  [words c]
  (let [k0 (array/reshape (array/slice words 0 0 1) [])
        k1 (array/reshape (array/slice words 0 1 2) [])
        u (fn [v] (array/array v {:dtype :uint32}))
        k2 (bit :bit-xor (bit :bit-xor k0 k1) (u 0x1BD11BDA))
        x0 (add32 (array/zeros-like c) k0)
        x1 (add32 c k1)
        [x0 x1] (rounds [x0 x1] [13 15 26 6])
        [x0 x1] (rounds [(add32 x0 k1) (add32 x1 (add32 k2 (u 1)))] [17 29 16 24])
        [x0 x1] (rounds [(add32 x0 k2) (add32 x1 (add32 k0 (u 2)))] [13 15 26 6])
        [x0 x1] (rounds [(add32 x0 k0) (add32 x1 (add32 k1 (u 3)))] [17 29 16 24])
        [x0 x1] (rounds [(add32 x0 k1) (add32 x1 (add32 k2 (u 4)))] [13 15 26 6])
        x0 (add32 x0 k2)
        x1 (add32 x1 (add32 k0 (u 5)))]
    (array/stack [x0 x1] -1)))

(defn- uniform-floats
  "float32 uniforms in [0, 1) of shape from the staged key words: 23 random
   mantissa bits of w0 ^ w1, as gen.random.threefry/uniform-floats."
  [words shape]
  (let [n (sh/size shape)
        ws (threefry-words words (array/arange 0 n 1 {:dtype :uint32}))
        w0 (array/reshape (array/slice ws 1 0 1) [n])
        w1 (array/reshape (array/slice ws 1 1 2) [n])
        bits (bit :bit-or (bit :shift-right-logical (bit :bit-xor w0 w1) 9) (array/array 0x3F800000 {:dtype :uint32}))
        f (op! :bitcast [bits] [n] :float32)]
    (array/reshape (- f 1.0) shape)))

(defn key-spec
  "A staged key's place among stage's argument specs."
  []
  ::key)

;; ---------------------------------------------------------------------------
;; The backend
;; ---------------------------------------------------------------------------

(defn- elementwise-shape [args] (sh/broadcast-all (map shape-of args)))

(defn- reduced-shape [shape axes keep-dims?]
  (let [axes (set axes)]
    (vec (keep-indexed (fn [i d] (cond (not (axes i)) d keep-dims? 1)) shape))))

(defn- host-read! [what]
  (throw (ex-info (str "Cannot read a staged value on the host (" what "): the program is staged once and run many"
                       " times, so a value read now would be the same in every run")
                  {:op :host-read})))

(defn- node-of [^Staged x] (.get ^java.util.ArrayList (.-nodes ^Graph (.-g x)) (.-id x)))

(defn- const-arg [a] (when (map? a) (:const a)))

(defn- strided-range
  "[start step] when indices (staged) are start + step·j along axis, j the
   position, broadcast over every other axis (the tape's indices-along: a
   broadcast of a reshape of an affine iota), else nil."
  [indices axis]
  (when (staged? indices)
    (let [^Graph g (.-g ^Staged indices)
          node (fn [id] (.get ^java.util.ArrayList (.-nodes g) id))
          strip (fn [{:keys [op args] :as n}]
                  (if (and (#{:broadcast :reshape :convert} op) (number? (first args))) (recur (node (first args))) n))
          affine (fn affine [{:keys [op args] :as n}]
                   (case op
                     :iota [0 1]
                     :multiply (let [[a b] args
                                     [c other] (if (const-arg a) [(const-arg a) b] [(const-arg b) a])]
                                 (when (and c (number? other))
                                   (when-let [[s0 s1] (affine (strip (node other)))] [(* c s0) (* c s1)])))
                     :add (let [[a b] args
                                [c other] (if (const-arg a) [(const-arg a) b] [(const-arg b) a])]
                            (when (and c (number? other))
                              (when-let [[s0 s1] (affine (strip (node other)))] [(+ c s0) s1])))
                     nil))
          b (node-of indices)
          r (when (and (= :broadcast (:op b)) (number? (first (:args b)))) (node (first (:args b))))
          rank (count (:shape b))
          ;; the iota must run along axis: the reshape puts its length there
          ;; and 1 on every other axis
          along? (and r (= :reshape (:op r)) (= rank (count (:shape r)))
                      (every? (fn [k] (or (= k axis) (= 1 (nth (:shape r) k)))) (range rank)))]
      (when along?
        (when-let [[start step] (affine (strip r))]
          (when (and (integer? start) (integer? step) (pos? step) (>= start 0)) [(long start) (long step)]))))))

(deftype StageBackend []
  p/IBackend
  (-name [_] :stage)
  (-defaults [_] {:float :float32 :int :int32})
  (-lazy? [_] true)
  (-native-ops [_] #{})

  (-from-host [_ values shape dtype]
    (op! :constant [] shape dtype {:values (if (instance? double/1 values) (vec values) (vec values))}))
  (-to-host [_ _] (host-read! "a host read"))
  (-eval! [_ _] nil)

  (-shape [_ x] (shape-of x))
  (-dtype [_ x] (if (staged? x) (.-dtype ^Staged x) (dt/host-dtype x)))

  (-full [_ shape value dtype] (op! :broadcast [(dt/host-cast value dtype)] shape dtype {:shape (vec shape)}))
  (-arange [_ start stop step dtype]
    (let [n (max 0 (long (Math/ceil (/ (- (double stop) (double start)) (double step)))))
          i (op! :iota [] [n] dtype)]
      (if (and (zero? (double start)) (= 1.0 (double step)))
        i
        (array/+ (array/* i (dt/host-cast step dtype)) (dt/host-cast start dtype)))))
  (-astype [_ x dtype] (op! :convert [x] (shape-of x) dtype))

  (-elementwise [_ op args dtype]
    (op! op args (elementwise-shape args) (dt/result-dtype op dtype) {:in dtype}))
  (-reduce [_ op x axes keep-dims?]
    (op! :reduce [x] (reduced-shape (shape-of x) axes keep-dims?)
         (dt/reduce-result-dtype op (.-dtype ^Staged x) {:float :float32 :int :int32})
         {:op op :axes (vec axes) :keep-dims? keep-dims?}))
  (-scan [_ op _ _ _] (throw (ex-info (str "The staging backend has no " op " yet") {:op op})))

  (-reshape [_ x shape] (op! :reshape [x] shape (array/dtype x)))
  (-transpose [_ x perm] (op! :transpose [x] (mapv #(nth (shape-of x) %) perm) (array/dtype x) {:perm (vec perm)}))
  (-broadcast-to [_ x shape] (op! :broadcast [x] shape (array/dtype x) {:shape (vec shape)}))
  (-concatenate [_ xs axis] (op! :concatenate xs (sh/concatenate-shape (map shape-of xs) axis) (array/dtype (first xs)) {:axis axis}))
  (-slice [_ x starts stops steps]
    (op! :slice [x] (mapv (fn [a b s] (max 0 (long (Math/ceil (/ (- (double b) (double a)) (double s)))))) starts stops steps)
         (array/dtype x) {:starts (vec starts) :stops (vec stops) :steps (vec steps)}))

  (-take [_ _ _ _] (throw (ex-info "The staging backend has no take yet" {})))
  (-take-along-axis [_ _ _ _] (throw (ex-info "The staging backend has no take-along-axis yet" {})))
  (-scatter-add [_ x indices updates axis]
    (if-let [[start step] (strided-range indices axis)]
      ;; a strided slice's gradient (the tape's unslice): updates padded into
      ;; place, which a compiler fuses, where a scatter is a kernel of its own
      (let [n (nth (shape-of x) axis)
            m (nth (shape-of updates) axis)
            high (- n (+ start (* step (dec m))) 1)]
        (array/+ x (op! :pad [updates] (shape-of x) (array/dtype x) {:axis axis :low start :high high :interior (dec step)})))
      (op! :scatter-add [x indices updates] (shape-of x) (array/dtype x) {:axis axis})))

  (-matmul [_ x y] (op! :matmul [x y] (sh/matmul-shape (shape-of x) (shape-of y)) (array/dtype x)))
  (-linalg [_ op _] (throw (ex-info (str "The staging backend has no " op " yet") {:op op})))

  (-random-uniform [_ key shape dtype]
    (when-not (= dtype :float32)
      (throw (ex-info (str "Staged draws are float32 only (asked " dtype ")") {:dtype dtype})))
    (if (instance? StagedKey key)
      (uniform-floats (.-words ^StagedKey key) (vec shape))
      ;; a host key: its draws are constants, the JVM backend's
      (op! :constant [] shape :float32 {:values (vec (threefry/uniform-floats key (sh/size shape)))})))
  (-value-and-grad [b f args argnums] ((requiring-resolve 'gen.array.tape/value-and-grad) b f args argnums)))

(def backend (StageBackend.))

;; ---------------------------------------------------------------------------
;; Staging a function, and loops
;; ---------------------------------------------------------------------------

(defn- spec-param
  "A :param for the spec s of input i: a staged array, or a staged key."
  [^Graph g s i]
  (if (= s ::key)
    (StagedKey. (add-node! g :param [] [2] :uint32 {:index i}))
    (let [[shape dtype] s] (add-node! g :param [] shape dtype {:index i}))))

(defn- spec-leaf? [s] (or (= s ::key) (and (vector? s) (= 2 (count s)) (vector? (first s)) (keyword? (second s)))))

(defn- spec-leaves [specs] (filter spec-leaf? (tree-seq (complement spec-leaf?) #(if (map? %) (vals %) (seq %)) specs)))

(defn- fill-specs
  "specs with each leaf replaced by (f leaf i), i counting leaves in order."
  [specs f]
  (let [i (volatile! -1)
        walk (fn walk [s] (cond (spec-leaf? s) (f s (vswap! i inc))
                                (map? s) (into (empty s) (map (fn [[k v]] [k (walk v)])) s)
                                (sequential? s) (mapv walk s)
                                :else s))]
    (walk specs)))

(defn- as-node
  "x as a node id of g: a staged value (captured if outer), or a constant
   for a host number."
  [^Graph g x]
  (cond (staged? x) (capture g x)
        (instance? StagedKey x) (capture g (.-words ^StagedKey x))
        (array/host? x) (.-id ^Staged (add-node! g :constant [] [] (array/dtype x) {:values [x]}))
        :else (throw (ex-info (str "A staged function returned " (pr-str x) ", not an array") {:value x}))))

(defn constants
  "x with every array of another backend in it (a tree, or a choice map's
   leaves) made a constant of the graph being staged: a model's arguments
   and observations, made on the JVM backend."
  [x]
  (let [c (fn [a] (if (and (array/backend-array? a) (not (staged? a)))
                    (op! :constant [] (array/shape a) (if (= :float (dt/kind (array/dtype a))) :float32 (array/dtype a))
                         {:values (vec (flatten [(array/->clj a)]))})
                    a))]
    (if (cm/choicemap? x)
      (reduce (fn [m [path v]] (cm/set-choice m path (c v))) x (cm/leaves x))
      (array/tree-map c x))))

(defn- leaves [t] (if (instance? StagedKey t) [(.-words ^StagedKey t)] (array/tree-leaves t)))

(defn stage
  "The program of f on staged parameters for the argument specs (one
   argument per spec; a spec is a tree of [shape dtype] leaves and
   (key-spec) keys): {:graph (the nodes), :inputs (the params' [shape
   dtype] in order), :outputs (node ids of f's result's leaves),
   :out-tree (its skeleton)}."
  [f & specs]
  (let [g (graph nil)
        args (binding [*graph* g] (vec (fill-specs (vec specs) #(spec-param g %1 %2))))
        out (binding [*graph* g array/*backend* backend] (apply f args))
        outs (binding [*graph* g] (mapv #(as-node g %) (leaves out)))]
    {:graph (vec (.-nodes ^Graph g))
     :inputs (mapv (fn [s] (if (= s ::key) [[2] :uint32] s)) (spec-leaves (vec specs)))
     :outputs outs
     :out-tree (tree/skeleton out)}))

(defn- scan*
  "scan over xs (or over nothing, xs nil, f's x nil)."
  [f carry xs n]
  (let [outer (current)
        g (graph outer)
        carry-leaves (vec (leaves carry))
        x-leaves (if (nil? xs) [] (vec (leaves xs)))
        carry-leaves (mapv #(if (array/host? %) (array/array % {:dtype (array/weak-dtype % backend)}) %) carry-leaves)
        cparams (binding [*graph* g] (mapv (fn [x i] (add-node! g :param [] (array/shape x) (array/dtype x) {:index i}))
                                           carry-leaves (range)))
        xparams (binding [*graph* g] (mapv (fn [x i] (add-node! g :param [] (vec (rest (array/shape x))) (array/dtype x)
                                                                {:index (+ (count carry-leaves) i)}))
                                           x-leaves (range)))
        carry-in (tree/refill (tree/skeleton carry) cparams)
        x-in (when xs (tree/refill (tree/skeleton xs) xparams))
        [carry' y] (binding [*graph* g array/*backend* backend] (f carry-in x-in))
        c-out (binding [*graph* g] (mapv #(as-node g %) (leaves carry')))
        y-leaves (leaves y)
        y-out (binding [*graph* g] (mapv #(as-node g %) y-leaves))
        nodes (vec (.-nodes ^Graph g))]
    (doseq [[c c'] (map vector cparams c-out)]
      (let [{:keys [shape dtype]} (nth nodes c')]
        (when-not (and (= shape (array/shape c)) (= dtype (array/dtype c)))
          (throw (ex-info (str "scan: the carry changed from " (array/dtype c) (array/shape c) " to " dtype shape)
                          {:in [(array/shape c) (array/dtype c)] :out [shape dtype]})))))
    (let [^java.util.LinkedHashMap captures (.-captures ^Graph g)
          caps (vec (.keySet captures))
          ;; the body's params: carries, xs, then captures in capture order
          body {:graph nodes :carries (mapv #(.-id ^Staged %) cparams) :xs (mapv #(.-id ^Staged %) xparams)
                :captures (mapv #(.get captures %) caps)
                :carry-out c-out :y-out y-out}
          ^Staged s (add-node! outer :scan (into (mapv #(capture outer %) carry-leaves)
                                                 (concat (map #(capture outer %) x-leaves) (map #(capture outer %) caps)))
                               [] nil {:body body :length n :n-carry (count carry-leaves) :n-x (count x-leaves)})
          results (mapv (fn [i [shape dtype]] (add-node! outer :result [(.-id s)] shape dtype {:index i}))
                        (range)
                        (concat (map (fn [c] [(array/shape c) (array/dtype c)]) carry-leaves)
                                (map (fn [id] (let [{:keys [shape dtype]} (nth nodes id)] [(into [n] shape) dtype])) y-out)))
          [cr yr] (split-at (count carry-leaves) results)]
      [(tree/refill (tree/skeleton carry') (vec cr))
       (tree/refill (tree/skeleton y) (vec yr))])))

(defn scan
  "lax.scan on the staging backend: (f carry x) → [carry' y], run once on
   staged values for a loop of n steps over xs (a tree of staged arrays
   with a leading axis of n); returns [the last carry, the ys stacked on a
   leading axis of n]. With xs nil, f's x is the loop counter, an int32
   scalar from 0 to n - 1 (carried beside carry). carry is a tree of
   staged values or host numbers (made constants), and carry' must keep
   its shapes and dtypes. Values of enclosing graphs that f reads are
   captured."
  [f carry xs n]
  (if (some? xs)
    (scan* f carry xs n)
    (let [[[c _] y] (scan* (fn [[c t] _] (let [[c' y] (f c t)] [[c' (array/+ t 1)] y]))
                           [carry (array/array 0 {:dtype :int32})] nil n)]
      [c y])))
