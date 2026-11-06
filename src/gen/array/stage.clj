(ns gen.array.stage
  "A staging backend: gen.array's operations recorded as a graph instead of
   computed, so that code written against gen.array (a model's density, a
   kernel's move, gen.random's draws, the tape's gradients) becomes a
   program another compiler runs: the device programs (DESIGN §9; the
   lowerings, to JVM code, Metal and CUDA C, are the extension ext/device).
   Experimental: outside the 1.x promise.

   (stage f & arg-specs) runs f once on staged parameters, one per leaf of
   arg-specs (trees of [shape dtype] pairs, or (key-spec) for a key), and
   returns the program: {:graph :inputs :outputs :out-tree :f}, the outputs
   f's result's leaves. Every gen.array op on a staged value adds a node;
   the tape (gen.array.tape) wraps this backend as it wraps the JVM's, so
   value-and-grad records the gradient's nodes in the same graph; the batch
   tracer wraps it as it wraps MLX; gen.random's draws on a staged key (a
   value too, rule 2) record Threefry's, so a program draws the host's
   bits for the same key. *float* stages float32 (every device) or float64
   programs. gen.array/scan is one loop node here.

   (evaluate program inputs) replays a program on an eager backend, by the
   same protocol calls staging recorded (Threefry's uint32 words on the
   host): the reference for every lowering, and the proof that staging is
   observational (on the JVM backend in float64, a replay equals the run of
   f itself, bit for bit).

   Nodes: {:id :op :args :shape :dtype :attrs}. :args are node ids or host
   numbers as {:const x}. Ops: gen.array's elementwise op keywords; :param,
   :constant, :iota, :convert, :reduce, :reshape, :transpose, :broadcast,
   :concatenate, :slice, :pad, :matmul, :take, :take-along-axis,
   :scatter-add; the bit ops :bit-xor, :bit-or, :shift-left,
   :shift-right-logical and :bitcast (Threefry's); :scan, a loop whose body
   is a graph of its own, and :result, one of a scan's results. A where
   whose branches are one staged value of its result's shape and dtype
   records no node: where(c, x, x) is x.

   Reading a staged value on the host (truthy?, item, ->clj, = with a
   number or boolean) is a loud error: the program is staged once and run
   many times."
  {:api :experimental}
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

(def ^:dynamic *float*
  "The staging backend's float dtype: its default (so a model's draws and
   arguments), its reductions' and its constants'. :float32 (every
   device), or :float64 (the JVM lowering: the JVM backend's default, whose
   draws it then makes)."
  :float32)

(def ^:dynamic *graph*
  "The graph that ops record into (the innermost one being staged)."
  nil)

(declare backend)

(declare host-read!)

(deftype Staged [^Graph g ^long id shape dtype]
  p/IArray
  (-backend [_] backend)
  Object
  (toString [_] (str "#staged[" id " " dtype " " shape "]"))
  ;; clojure.core/= on a 0-d staged value and a host number or boolean is
  ;; a host read in disguise: (= k 0) in plain Clojure would answer false
  ;; for every run (genclj-ryg0's census found one), so it is loud, as for
  ;; a 0-d MLX array (DESIGN §9, genclj-8rdm). Every other comparison is
  ;; false; two staged values compare by identity, as the graph's maps need.
  (equals [this o]
    (cond (identical? this o) true
          (and (= [] shape) (or (instance? Number o) (instance? Boolean o)))
          (host-read! "clojure.core/= on a staged value")
          :else false))
  (hashCode [this] (System/identityHashCode this))

  ;; get, nth and count are loud (rule 6; genclj-fzez): an array is no
  ;; Clojure collection, and get would answer nil
  clojure.lang.ILookup
  (valAt [this _] (throw (p/not-a-collection "get" this)))
  (valAt [this _ _] (throw (p/not-a-collection "get" this)))
  clojure.lang.Indexed
  (nth [this _] (throw (p/not-a-collection "nth" this)))
  (nth [this _ _] (throw (p/not-a-collection "nth" this)))
  (count [this] (throw (p/not-a-collection "count" this))))

(defmethod print-method Staged [^Staged x ^java.io.Writer w] (.write w (.toString x)))

(defn staged? [x] (instance? Staged x))

(defn- add-node!
  "A new node of op in graph g, returning its staged value."
  [^Graph g op args shape dtype attrs]
  (let [id (.size ^java.util.ArrayList (.-nodes g))]
    (.add ^java.util.ArrayList (.-nodes g) {:id id :op op :args args :shape (vec shape) :dtype dtype :attrs attrs})
    (Staged. g id (vec shape) dtype)))

(defn- current ^Graph []
  (or *graph* (throw (ex-info "A staged op outside stage: the staging backend runs inside gen.array.stage/stage" {}))))

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

(defn- same-branches
  "where's [c x y] branch x when y is the same staged value and the result
   is x's shape and dtype: where(c, x, x) is x for every c, so no node is
   recorded, and a value a select passes on stays the value it was (a
   chain's observations, which kernel/run's device path then does not
   write out every step)."
  [[_ x y] shape dtype]
  (when (and (staged? x) (identical? x y)
             (= shape (.-shape ^Staged x)) (= dtype (.-dtype ^Staged x)))
    x))

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

(defn threefry-words
  "Threefry-2x32-20 of the counters [0 c] (c a uint32 array) under the
   staged key words (uint32 [2]): the words, [.. 2] (the last axis w0, w1),
   for counters below 2^32, as gen.random.threefry/split's."
  [words c]
  (let [[w0 w1] (threefry/hash-words
                 {:add array/+
                  :xor #(bit :bit-xor %1 %2)
                  :rotl (fn [x r] (bit :bit-or (bit :shift-left x (long r)) (bit :shift-right-logical x (long (- 32 r)))))
                  :word #(array/array % {:dtype :uint32})
                  :zeros array/zeros-like}
                 (array/reshape (array/slice words 0 0 1) [])
                 (array/reshape (array/slice words 0 1 2) [])
                 c)]
    (array/stack [w0 w1] -1)))

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

(defn- uniform-doubles
  "float64 uniforms in [0, 1) of shape from the staged key words: the top 52
   bits of the 64-bit word w0·2^32 + w1, as gen.random.threefry/
   uniform-doubles, built exactly in float64 as (w0·2^20 + (w1 >>> 12))·2^-52
   (no 64-bit integers on the staging backend)."
  [words shape]
  (let [n (sh/size shape)
        ws (threefry-words words (array/arange 0 n 1 {:dtype :uint32}))
        w0 (array/reshape (array/slice ws 1 0 1) [n])
        w1 (array/reshape (array/slice ws 1 1 2) [n])
        hi (op! :convert [w0] [n] :float64)
        lo (op! :convert [(bit :shift-right-logical w1 12)] [n] :float64)]
    (array/reshape (* (+ (* hi (Math/pow 2.0 20)) lo) (Math/pow 2.0 -52)) shape)))

(defn key-at
  "(rng/split key n)'s key at the staged index t (an int32 scalar below n):
   Threefry of the counter [0 t] under key's words, as split's, so a loop's
   step t draws what (nth (rng/split key n) t) draws on the host."
  [key t]
  (->StagedKey (array/reshape (threefry-words (p/-key-words key) (array/reshape (array/astype t :uint32) [1])) [2])))

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

(defn- staged-dtype
  "The staging backend's dtype for another backend's: its float (*float*)
   for a float, int32 for int64 (the JVM backend's default int)."
  [dtype]
  (cond (= :float (dt/kind dtype)) *float*
        (= :int64 dtype) :int32
        :else dtype))

(defn- staged
  "x as a value of the graph being staged: a staged value or host number as
   it is, an array of another backend (a constant table, indices made on
   the host) a constant node."
  [x]
  (if (or (staged? x) (array/host? x))
    x
    (op! :constant [] (array/shape x) (staged-dtype (array/dtype x)) {:values (vec (flatten [(array/->clj x)]))})))

(declare scan)

(deftype StageBackend []
  p/ILooping
  (-scan-loop [_ f carry xs n] (scan f carry xs n))

  p/IBackend
  (-name [_] :stage)
  (-defaults [_] {:float *float* :int :int32})
  (-lazy? [_] true)
  (-native-ops [_] #{})

  (-from-host [_ values shape dtype]
    (op! :constant [] shape dtype {:values (if (instance? double/1 values) (vec values) (vec values))}))
  (-to-host [_ _] (host-read! "a host read"))
  (-eval! [_ _] nil)

  (-shape [_ x] (shape-of x))
  (-dtype [_ x] (if (staged? x) (.-dtype ^Staged x) (dt/host-dtype x)))

  (-full [b shape value dtype]
    ;; a 0-d array fills too (docs/array-api.md §4), converted to the dtype:
    ;; host-cast took only host numbers (genclj-kdau)
    (let [v (if (array/host? value)
              (dt/host-cast value dtype)
              (let [v (staged value)] (if (= dtype (array/dtype v)) v (p/-astype b v dtype))))]
      (op! :broadcast [v] shape dtype {:shape (vec shape)})))
  (-arange [_ start stop step dtype]
    (let [n (max 0 (long (Math/ceil (/ (- (double stop) (double start)) (double step)))))
          i (op! :iota [] [n] dtype)
          ;; an int dtype's values are NumPy's, as the backends' (genclj-06ba)
          [v0 dv] (if (= :int (dt/kind dtype))
                    (dt/int-arange start step dtype)
                    [(dt/host-cast start dtype) (dt/host-cast step dtype)])]
      (if (and (zero? (double v0)) (clojure.core/== 1.0 (double dv)))
        i
        (array/+ (array/* i dv) v0))))
  (-astype [_ x dtype] (op! :convert [x] (shape-of x) dtype))

  (-elementwise [_ op args dtype]
    (let [shape (elementwise-shape args)
          out (dt/result-dtype op dtype)]
      (or (when (= :where op) (same-branches args shape out))
          (op! op args shape out {:in dtype}))))
  (-reduce [_ op x axes keep-dims?]
    (op! :reduce [x] (reduced-shape (shape-of x) axes keep-dims?)
         (dt/reduce-result-dtype op (.-dtype ^Staged x) {:float *float* :int :int32})
         {:op op :axes (vec axes) :keep-dims? keep-dims?}))
  (-scan [_ op _ _ _] (throw (ex-info (str "The staging backend has no " op " yet") {:op op})))

  (-reshape [_ x shape] (op! :reshape [x] shape (array/dtype x)))
  (-transpose [_ x perm] (op! :transpose [x] (mapv #(nth (shape-of x) %) perm) (array/dtype x) {:perm (vec perm)}))
  (-broadcast-to [_ x shape] (op! :broadcast [x] shape (array/dtype x) {:shape (vec shape)}))
  (-concatenate [_ xs axis] (op! :concatenate xs (sh/concatenate-shape (map shape-of xs) axis) (array/dtype (first xs)) {:axis axis}))
  (-slice [_ x starts stops steps]
    (op! :slice [x] (mapv (fn [a b s] (max 0 (long (Math/ceil (/ (- (double b) (double a)) (double s)))))) starts stops steps)
         (array/dtype x) {:starts (vec starts) :stops (vec stops) :steps (vec steps)}))

  (-take [_ x indices axis]
    (let [x (staged x) indices (staged indices)
          sx (shape-of x)]
      (op! :take [x indices] (vec (concat (subvec sx 0 axis) (shape-of indices) (subvec sx (inc axis)))) (array/dtype x) {:axis axis})))
  (-take-along-axis [_ x indices axis]
    (let [x (staged x) indices (staged indices)]
      (op! :take-along-axis [x indices] (mapv (fn [k a b] (if (= k axis) b (max (long a) (long b)))) (range) (shape-of x) (shape-of indices))
           (array/dtype x) {:axis axis})))
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
    (when-not (#{:float32 :float64} dtype)
      (throw (ex-info (str "Staged draws are float32 or float64 (asked " dtype ")") {:dtype dtype})))
    (cond (not (instance? StagedKey key))
          ;; a host key: its draws are constants, the JVM backend's
          (op! :constant [] shape dtype {:values (vec ((if (= dtype :float64) threefry/uniform-doubles threefry/uniform-floats)
                                                        key (sh/size shape)))})
          (= dtype :float64) (uniform-doubles (.-words ^StagedKey key) (vec shape))
          :else (uniform-floats (.-words ^StagedKey key) (vec shape))))
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
                    (op! :constant [] (array/shape a) (staged-dtype (array/dtype a))
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
   :out-tree (its skeleton), :f (f)}."
  [f & specs]
  (let [g (graph nil)
        args (binding [*graph* g] (vec (fill-specs (vec specs) #(spec-param g %1 %2))))
        out (binding [*graph* g array/*backend* backend] (apply f args))
        outs (binding [*graph* g] (mapv #(as-node g %) (leaves out)))]
    {:graph (vec (.-nodes ^Graph g))
     :inputs (mapv (fn [s] (if (= s ::key) [[2] :uint32] s)) (spec-leaves (vec specs)))
     :outputs outs
     :out-tree (tree/skeleton out)
     ;; the staged function itself: an entry point runs it on the handler
     ;; to raise a run's error (DESIGN §9, genclj-i2c4)
     :f f}))

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
    ;; as the host loop's checks (gen.array/scan): the pairs below would
    ;; drop a leaf of another structure
    (when-not (= (tree/skeleton carry) (tree/skeleton carry'))
      (throw (ex-info "scan: the step changed the carry's structure" {:reason :invalid-carry :changed "structure"})))
    (doseq [[c c'] (map vector cparams c-out)]
      (let [{:keys [shape dtype]} (nth nodes c')]
        (when-not (and (= shape (array/shape c)) (= dtype (array/dtype c)))
          (throw (ex-info (str "scan: the carry changed from " (array/dtype c) (array/shape c) " to " dtype shape)
                          {:reason :invalid-carry :in [(array/shape c) (array/dtype c)] :out [shape dtype]})))))
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

;; ---------------------------------------------------------------------------
;; Replay: a program on an eager backend, by the protocol calls staging
;; recorded (the reference every lowering is held to). uint32 values,
;; Threefry's words, are computed on the host: eager backends need not
;; hold uint32.
;; ---------------------------------------------------------------------------

(defrecord ^:private U32 [shape v])

(defn- u32? [x] (instance? U32 x))

(def ^:private mask32 0xFFFFFFFF)

(defn- strides [shape] (vec (reverse (reductions clojure.core/* 1 (reverse (rest shape))))))

(defn- unravel [^long i shape] (mapv (fn [s d] (mod (quot i (long s)) (long d))) (strides shape) shape))

(defn- ravel ^long [idx shape] (reduce clojure.core/+ 0 (map clojure.core/* idx (strides shape))))

(defn- u32-of
  "x (a host number or a backend array) as host uint32 words."
  [x]
  (if (u32? x)
    x
    (->U32 (vec (array/shape x)) (mapv #(bit-and (long %) mask32) (flatten [(array/->clj x)])))))

(defn- u32->array
  "x's words as an array of dtype on the current backend."
  [x dtype]
  (array/reshape (array/array (:v x) {:dtype dtype}) (:shape x)))

(defn- u32-index
  "A uint32 value of shape whose element at each index is x's at (f idx)."
  [x shape f]
  (->U32 (vec shape) (mapv #(nth (:v x) (ravel (f (unravel % shape)) (:shape x))) (range (sh/size shape)))))

(defn- u32-binary [op a b]
  (case op
    :add (bit-and (clojure.core/+ a b) mask32)
    :subtract (bit-and (clojure.core/- a b) mask32)
    :multiply (bit-and (unchecked-multiply (long a) (long b)) mask32)
    :bit-xor (bit-xor a b)
    :bit-or (bit-or a b)
    :bit-and (bit-and a b)
    :shift-left (if (clojure.core/< b 32) (bit-and (bit-shift-left a b) mask32) 0)
    :shift-right-logical (if (clojure.core/< b 32) (unsigned-bit-shift-right a b) 0)
    (throw (ex-info (str "evaluate: no uint32 " op) {:op op}))))

(defn- u32-elementwise [op shape args]
  (let [args (mapv u32-of args)
        shape (vec shape)
        n (sh/size shape)
        k (count shape)
        at (fn [x i] (let [s (:shape x) o (unravel i shape) off (clojure.core/- k (count s))]
                       (nth (:v x) (ravel (map-indexed (fn [j d] (if (== 1 d) 0 (nth o (clojure.core/+ off j)))) s) s))))]
    (when-not (= 2 (count args)) (throw (ex-info (str "evaluate: no uint32 " op) {:op op})))
    (->U32 shape (mapv (fn [i] (u32-binary op (at (first args) i) (at (second args) i))) (range n)))))

(defn- constant-array [values shape dtype]
  (if (and (= 1 (count values)) (not= [] shape) (clojure.core/> (sh/size shape) 1))
    (array/full shape (first values) {:dtype dtype})
    (array/reshape (array/array (vec values) {:dtype dtype}) shape)))

(declare run-graph)

(defn- slice-at [x i]
  (if (u32? x)
    (let [inner (vec (rest (:shape x))) k (sh/size inner)]
      (->U32 inner (subvec (:v x) (clojure.core/* i k) (clojure.core/* (inc i) k))))
    (array/take x i 0)))

(defn- stack-steps [ys]
  (if (u32? (first ys))
    (->U32 (into [(count ys)] (:shape (first ys))) (vec (mapcat :v ys)))
    (array/stack (vec ys) 0)))

(defn- eval-scan [b env {:keys [args attrs]}]
  (let [{:keys [body length n-carry n-x]} attrs
        {:keys [graph carries xs captures carry-out y-out]} body
        vals (mapv #(if (map? %) (:const %) (env %)) args)
        cinit (subvec vals 0 n-carry)
        xvals (subvec vals n-carry (clojure.core/+ n-carry n-x))
        caps (subvec vals (clojure.core/+ n-carry n-x))]
    (loop [i 0 cs cinit ys []]
      (if (clojure.core/< i length)
        (let [benv (run-graph b graph (merge (zipmap carries cs) (zipmap xs (map #(slice-at % i) xvals)) (zipmap captures caps)))]
          (recur (inc i) (mapv benv carry-out) (conj ys (mapv benv y-out))))
        {:results (into cs (map #(stack-steps (map (fn [y] (nth y %)) ys)) (range (count y-out))))}))))

(defn- eval-node [b env {:keys [id op args shape dtype attrs] :as node}]
  (let [arg (fn [i] (let [a (nth args i)] (if (map? a) (:const a) (env a))))
        all-args (fn [] (mapv #(if (map? %) (:const %) (env %)) args))
        in (:in attrs)
        v (case op
            :param (if (contains? env id) (env id) (throw (ex-info "evaluate: an unbound param" {:node (dissoc node :attrs)})))
            :constant (if (= dtype :uint32)
                        (let [vs (mapv #(bit-and (long %) mask32) (:values attrs)) n (sh/size shape)]
                          (->U32 (vec shape) (if (= 1 (count vs)) (vec (repeat n (first vs))) vs)))
                        (constant-array (:values attrs) shape dtype))
            :iota (if (= dtype :uint32) (->U32 (vec shape) (vec (range (first shape)))) (array/arange 0 (first shape) 1 {:dtype dtype}))
            :broadcast (let [x (arg 0)]
                         (cond (= dtype :uint32) (let [x (u32-of x) k (clojure.core/- (count shape) (count (:shape x)))]
                                                   (u32-index x shape (fn [o] (map-indexed (fn [j d] (if (== 1 d) 0 (nth o (clojure.core/+ k j)))) (:shape x)))))
                               (array/host? x) (array/full shape x {:dtype dtype})
                               :else (p/-broadcast-to b x shape)))
            :convert (let [x (arg 0)]
                       (cond (= dtype :uint32) (u32-of x)
                             (u32? x) (u32->array x dtype)
                             :else (p/-astype b x dtype)))
            :bitcast (let [x (arg 0)]
                       (case [(if (u32? x) :uint32 :float32) dtype]
                         [:uint32 :float32] (array/reshape (array/array (mapv #(double (Float/intBitsToFloat (unchecked-int (long %)))) (:v x)) {:dtype :float32}) (:shape x))
                         [:float32 :uint32] (->U32 (vec (array/shape x)) (mapv #(bit-and (long (Float/floatToRawIntBits (float %))) mask32) (flatten [(array/->clj x)])))
                         (throw (ex-info (str "evaluate: no bitcast to " dtype) {:node (dissoc node :attrs)}))))
            :stop-gradient (arg 0)
            :reduce (p/-reduce b (:op attrs) (arg 0) (:axes attrs) (:keep-dims? attrs))
            :reshape (let [x (arg 0)] (if (u32? x) (assoc x :shape (vec shape)) (p/-reshape b x shape)))
            :transpose (let [x (arg 0) perm (:perm attrs)]
                         (if (u32? x)
                           (u32-index x shape (fn [o] (reduce (fn [v k] (assoc v (nth perm k) (nth o k))) (vec (repeat (count shape) 0)) (range (count shape)))))
                           (p/-transpose b x perm)))
            :concatenate (let [xs (all-args) ax (:axis attrs)]
                           (if (some u32? xs)
                             (let [xs (mapv u32-of xs)
                                   offs (vec (reductions clojure.core/+ 0 (map #(nth (:shape %) ax) xs)))]
                               (->U32 (vec shape) (mapv (fn [i] (let [o (unravel i shape) p (nth o ax)
                                                                      k (count (take-while #(clojure.core/<= % p) (rest offs)))
                                                                      x (nth xs k)]
                                                                  (nth (:v x) (ravel (assoc o ax (clojure.core/- p (nth offs k))) (:shape x)))))
                                                        (range (sh/size shape)))))
                             (p/-concatenate b xs ax)))
            :slice (let [x (arg 0) {:keys [starts stops steps]} attrs]
                     (if (u32? x)
                       (u32-index x shape (fn [o] (map (fn [s k c] (clojure.core/+ s (clojure.core/* k c))) starts steps o)))
                       (p/-slice b x starts stops steps)))
            :pad (let [x (arg 0) {:keys [axis low interior]} attrs
                       m (nth (array/shape x) axis)]
                   (p/-scatter-add b (array/zeros shape {:dtype dtype})
                                   (array/array (mapv #(clojure.core/+ low (clojure.core/* % (inc interior))) (range m)) {:dtype :int64})
                                   x axis))
            :scatter-add (p/-scatter-add b (arg 0) (arg 1) (arg 2) (:axis attrs))
            :matmul (p/-matmul b (arg 0) (arg 1))
            :take (p/-take b (arg 0) (arg 1) (:axis attrs))
            :take-along-axis (p/-take-along-axis b (arg 0) (arg 1) (:axis attrs))
            :scan (eval-scan b env node)
            :result (nth (:results (env (first args))) (:index attrs))
            ;; elementwise: gen.array's op keyword on the computing dtype
            (let [xs (all-args)]
              (cond (= dtype :uint32) (u32-elementwise op shape xs)
                    (some u32? xs) (p/-elementwise b op (mapv #(if (u32? %) (u32->array % :int64) %) xs) in)
                    :else (p/-elementwise b op xs in))))]
    (assoc env id v)))

(defn- run-graph [b nodes env]
  (reduce (fn [env node] (eval-node b env node)) env nodes))

(defn evaluate
  "program (stage's) replayed on backend b (default array/*backend*, an
   eager one) from inputs, the program's input leaves in order: a key
   (gen.random's) for each key spec, arrays or host numbers for the rest.
   Returns f's result's structure. The same protocol calls run as staging
   recorded, so on the JVM backend a float64 program's replay equals f run
   directly, bit for bit (uint32 outputs come back as int64 arrays)."
  ([program inputs] (evaluate program inputs array/*backend*))
  ([{:keys [graph outputs out-tree]} inputs b]
   (let [params (filter #(and (= :param (:op %)) (not (:capture (:attrs %)))) graph)
         env0 (into {} (for [{:keys [id attrs]} params
                             :let [v (nth inputs (:index attrs))]]
                         [id (if (instance? gen.random.threefry.Key v)
                               (let [^gen.random.threefry.Key k v] (->U32 [2] [(.-k0 k) (.-k1 k)]))
                               v)]))
         env (run-graph b graph env0)]
     (tree/refill out-tree (mapv #(let [v (env %)] (if (u32? v) (u32->array v :int64) v)) outputs)))))
