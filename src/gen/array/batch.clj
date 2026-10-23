(ns gen.array.batch
  "Batching by tracing: N particles run through code written for one, with
   vmap semantics (as JAX's vmap). A batched value wraps an array of the
   wrapped backend whose leading axes are the batch; to the code it flows
   through, it has its per-particle shape (gen.array/shape) and every op acts
   per particle. Batched and unbatched operands align by their per-particle
   axes, so a per-particle scalar meets a shared vector correctly, and random
   draws on the batch backend gain the batch axes, one independent draw per
   particle.

   A batch backend wraps a real one for a batch shape (one instance per pair,
   so that values of one batched run join). Running scalar code under
   (array/with-backend (batch/backend inner batch-shape) …) runs it batched:
   creation functions and constants stay unbatched (shared), draws are
   batched. Reading a batched value on the host (array/item, array/->clj, truthy? of
   a boolean, = in either order, condp, hashing, counting) cannot give one
   answer for every particle: it throws ex-info with
   :gen.array/batched-branch true, the signal to fall back to one particle at
   a time.

   What cannot be intercepted is a raw if or case on a batched value outside
   a gen body, or a vector get or contains? with a batched key
   (docs/array-api.md §8a). For the commonest of these, unread-bools reports
   the batched comparisons that no op read: every op marks the values it
   reads, and a raw if reads nothing.

   Gradients are vmap's: array/value-and-grad of batched arguments, or
   anywhere in a batched run, gives each particle the value and gradient of
   its own scalar run, a shared argument's gradient included (summed over
   the particles, it is the gradient of their total). The tape
   (gen.array.tape) wraps the tracer:
   - its primals are batched or shared values, and each op's primal runs
     through the tracer, per particle;
   - so does the backward pass, whose rules are written against gen.array
     (the linear-algebra ones against the tape's inner backend, the
     tracer), so particle i's cotangents are its own scalar pass's;
   - neither knows the other: gen.array joins their values (a wrapping
     backend wraps what its inner backend wraps), and the tracer passes an
     op with no batched operand to the backend it wraps;
   - custom-vjp is one node whose function and rule run through the tracer
     (one that reads a value on the host raises the batched-branch error),
     and stop-gradient an op without a rule.
   A native transformation (MLX's, p/INativeGrad) cannot wrap the tracer,
   so the tracer runs inside it (value-and-grad-inside). A batched run
   inside a differentiated function, the tape outside the tracer, is a loud
   error: differentiate inside the run, and sum.

   Some of a batch's particles can run apart, in parts (a Switch whose index
   differs across them, gen.partition): a part's backend (part-backend)
   draws what the batch's own run would for its particles, gather-into
   takes their rows of the batch's values into it, and join and scatter
   put parts' values back together. Parts holds a batch's data by part."
  (:require [gen.array :as array]
            [gen.array.dtype :as dt]
            [gen.array.protocols :as p]
            [gen.array.shape :as sh]
            [gen.array.tape :as tape]
            [gen.array.tree :as tree]))

(set! *warn-on-reflection* true)

(defn- branch!
  "The batched-branch error: one answer cannot serve every particle."
  [what]
  (throw (array/batched-branch (str "Cannot " what " a batched value: it differs across particles") {})))

(declare value)

(definterface IRead
  (markRead [])
  (wasRead []))

;; v is the wrapped array, or a delay of it (a lazily gathered value: see
;; gather); value derefs it once and caches it. read? records whether any op
;; has read the value (see unread-bools).
(deftype Batched [v backend ^:unsynchronized-mutable read?]
  IRead
  (markRead [_] (set! read? true) nil)
  (wasRead [_] read?)

  p/IArray
  (-backend [_] backend)

  p/IBatchedValue
  (-all-particles [this] (value this))

  ;; A batch of values equals no single value, and hashing it would pick one.
  ;; As a persistent collection, it is also what Clojure's = consults when the
  ;; batch is on the right, (= 0 x), and what condp, case and = inside vectors
  ;; reach: every one of them raises the batched-branch error.
  Object
  (equals [this o] (or (identical? this o) (branch! "compare")))
  (hashCode [_] (branch! "hash"))

  clojure.lang.IHashEq
  (hasheq [_] (branch! "hash"))

  clojure.lang.IPersistentCollection
  (count [_] (branch! "count"))
  (cons [_ _] (branch! "extend"))
  (empty [_] (branch! "empty"))
  (equiv [this o] (or (identical? this o) (branch! "compare")))
  (seq [_] (branch! "iterate")))

(defmethod print-method Batched [x ^java.io.Writer w]
  (.write w "#gen/batched[") (print-method (value x) w) (.write w "]"))

(defn batched?
  "Is x a batched value, one value per particle?"
  [x]
  (instance? Batched x))

(defn value
  "The wrapped array of a batched value (leading batch axes). Every op reads
   its operands through value, which marks them read."
  [^Batched x]
  (.markRead x)
  (let [v (.-v x)] (if (delay? v) @v v)))

(declare backend lift full-batch made value-and-grad-inside by-kind)

;; draws is nil, or [shape rows] for a part of a batch (part-backend): the
;; draws of the batch of shape at rows, an int array of the wrapped backend.
(deftype BatchBackend [inner batch-shape draws]
  p/IWrappingBackend
  (-inner [_] inner)

  p/IBackend
  (-name [_] :batch)
  (-defaults [_] (p/-defaults inner))
  ;; batched values are as concrete as the wrapped backend's: validation reads
  ;; every particle (gen.array/all-true?)
  (-lazy? [_] (p/-lazy? inner))
  ;; the wrapped backend's, since -elementwise passes any op on lifted: a
  ;; batched op takes the implementation its particles would (rule 1;
  ;; genclj-bk62: MLX's logaddexp unbatched, the composite batched, whose
  ;; gradient at ties is 0 under MLX's autodiff where the op's is 1/2)
  (-native-ops [_] (p/-native-ops inner))

  (-from-host [_ values shape dtype] (p/-from-host inner values shape dtype))
  (-to-host [_ x] (if (batched? x) (branch! "read on the host") (p/-to-host inner x)))
  (-eval! [_ xs] (p/-eval! inner (keep #(if (batched? %) (value %) (when-not (array/host? %) %)) xs)))

  (-shape [_ x] (if (batched? x)
                  (subvec (p/-shape inner (value x)) (count batch-shape))
                  (array/shape x)))
  (-dtype [_ x] (if (batched? x) (p/-dtype inner (value x)) (array/dtype x)))

  (-full [b shape v dtype]
    (if (batched? v)
      ;; a per-particle fill value: each particle's array is filled with its own
      (made (p/-broadcast-to inner (lift b (made (p/-astype inner (value v) dtype) b) (count shape))
                                  (into batch-shape shape))
                 b)
      (p/-full inner shape v dtype)))
  (-arange [_ start stop step dtype] (p/-arange inner start stop step dtype))
  (-astype [b x dtype]
    (if (batched? x)
      (made (p/-astype inner (value x) dtype) b)
      (if (array/host? x) (dt/host-cast x dtype) (p/-astype inner x dtype))))

  (-elementwise [b op args dtype]
    (if (some batched? args)
      (let [;; the per-particle rank, from the operands' per-particle
            ;; broadcast: checked before they are lifted, so a mismatch names
            ;; the model's shapes (unbatched, the backend checks as it
            ;; broadcasts; genclj-q45d)
            r (count (reduce (fn [s x] (sh/broadcast s (p/-shape b x))) [] args))
            lifted (mapv #(lift b % r) args)]
        (made (p/-elementwise inner op lifted dtype) b))
      ;; shared operands only, which the tape over the tracer passes on
      ;; (gen.array never does: they are not this backend's)
      (p/-elementwise inner op args dtype)))

  (-reduce [b op x axes keep-dims?]
    (if (batched? x)
      (made (p/-reduce inner op (value x) (mapv #(+ % (count batch-shape)) axes) keep-dims?) b)
      (p/-reduce inner op x axes keep-dims?)))

  (-scan [b op x axis reverse?]
    (if (batched? x)
      (made (p/-scan inner op (value x) (+ axis (count batch-shape)) reverse?) b)
      (p/-scan inner op x axis reverse?)))

  (-reshape [b x shape]
    (if (batched? x)
      (made (p/-reshape inner (value x) (into batch-shape shape)) b)
      (p/-reshape inner x shape)))

  (-transpose [b x perm]
    (if (batched? x)
      (let [nb (count batch-shape)]
        (made (p/-transpose inner (value x) (into (vec (range nb)) (map #(+ % nb)) perm)) b))
      (p/-transpose inner x perm)))

  (-broadcast-to [b x shape]
    (if (batched? x)
      (made (p/-broadcast-to inner (lift b x (count shape)) (into batch-shape shape)) b)
      (p/-broadcast-to inner x shape)))

  (-concatenate [b xs axis]
    (if (some batched? xs)
      (made (p/-concatenate inner (mapv #(full-batch b %) xs) (+ axis (count batch-shape))) b)
      (p/-concatenate inner xs axis)))

  (-slice [b x starts stops steps]
    (if (batched? x)
      (let [nb (count batch-shape)]
        (made (p/-slice inner (value x) (into (vec (repeat nb 0)) starts)
                             (into batch-shape stops) (into (vec (repeat nb 1)) steps))
                   b))
      (p/-slice inner x starts stops steps)))

  (-take [b x indices axis]
    (cond
      (batched? indices)
      ;; per-particle indices: a gather along the axis, as take-along-axis
      (let [nb (count batch-shape)
            sx (p/-shape b x)
            si (p/-shape b indices)
            m (reduce * 1 si)
            xb (full-batch b x)
            idx (p/-reshape inner (value indices)
                            (-> batch-shape (into (repeat axis 1)) (conj m) (into (repeat (- (count sx) axis 1) 1))))
            out (p/-take-along-axis inner xb idx (+ nb axis))]
        (made (p/-reshape inner out (-> batch-shape (into (subvec sx 0 axis)) (into si) (into (subvec sx (inc axis)))))
                   b))
      (batched? x) (made (p/-take inner (value x) indices (+ axis (count batch-shape))) b)
      :else (p/-take inner x indices axis)))

  (-take-along-axis [b x indices axis]
    (if (or (batched? x) (batched? indices))
      (let [r (count (p/-shape b x))]
        (made (p/-take-along-axis inner (lift b x r) (lift b indices r) (+ axis (count batch-shape))) b))
      (p/-take-along-axis inner x indices axis)))

  (-scatter-add [b x indices updates axis]
    (if (some batched? [x indices updates])
      (made (p/-scatter-add inner (full-batch b x) (full-batch b indices) (full-batch b updates)
                                 (+ axis (count batch-shape)))
                 b)
      (p/-scatter-add inner x indices updates axis)))

  (-matmul [b x y]
    (if (or (batched? x) (batched? y))
      ;; per-particle NumPy matmul: a per-particle vector becomes a matrix,
      ;; and its added axis is removed from the product
      (let [sx (p/-shape b x)
            sy (p/-shape b y)
            xv? (== 1 (count sx))
            yv? (== 1 (count sy))
            x2 (if xv? (p/-reshape b x (into [1] sx)) x)
            y2 (if yv? (p/-reshape b y (conj sy 1)) y)
            r (max (count (p/-shape b x2)) (count (p/-shape b y2)))
            out (made (p/-matmul inner (lift b x2 r) (lift b y2 r)) b)
            t (p/-shape b out)
            n (count t)
            t' (cond (and xv? yv?) (subvec t 0 (- n 2))
                     yv? (pop t)
                     xv? (conj (subvec t 0 (- n 2)) (peek t))
                     :else t)]
        (if (= t t') out (p/-reshape b out t')))
      (p/-matmul inner x y)))

  (-linalg [b op args]
    (let [arrays (vec (remove map? args))
          opts (first (filter map? args))]
      (if (some batched? arrays)
        (let [r (reduce max 0 (map #(count (p/-shape b %)) arrays))
              ;; the operands aligned on their per-particle axes; the wrapped
              ;; backend broadcasts the batch axes
              vs (mapv #(lift b % r) arrays)
              out (p/-linalg inner op (cond-> vs opts (conj opts)))]
          (if (= op :slogdet) (mapv #(made % b) out) (made out b)))
        (p/-linalg inner op args))))

  (-random-uniform [b key shape dtype]
    (made (if-let [[full rows] draws]
            (p/-take inner (p/-random-uniform inner key (into full shape) dtype) rows 0)
            (p/-random-uniform inner key (into batch-shape shape) dtype))
          b))

  (-value-and-grad [b f args argnums]
    (let [[v grads] (if (and (instance? gen.array.protocols.INativeGrad inner) (p/-native-grad? inner))
                      (value-and-grad-inside b f args argnums)
                      ;; the tape over the tracer: vmap's gradient (see the namespace doc)
                      (tape/value-and-grad b f args argnums))]
      [v (by-kind b v args argnums grads)]))

  p/ICustomVjp
  (-custom-vjp [b f vjp args]
    (cond (not (instance? gen.array.protocols.ICustomVjp inner)) (apply f args)
          (not-any? batched? args) (p/-custom-vjp inner f vjp args)
          :else
          ;; the wrapped backend differentiates through f by the rule (MLX's
          ;; native gradients, value-and-grad-inside): both lifted to its
          ;; whole batches, each particle's cotangents its own
          (let [particles (fn [vs] (mapv #(made % b) vs))]
            (made (p/-custom-vjp inner
                                 (fn [& vs] (full-batch b (apply f (particles vs))))
                                 (fn [vs out g]
                                   (mapv #(some->> % (full-batch b)) (vjp (particles vs) (made out b) (made g b))))
                                 (mapv #(full-batch b %) args))
                  b)))))

(def ^:dynamic ^:private *bools*
  "The batched bool values made in the run in effect (unread-bools), or nil."
  nil)

(defn- made
  "The batched value of v, an op's result, registered when it is a bool."
  [v b]
  (let [x (Batched. v b false)]
    (when-let [bools *bools*]
      (when (= :bool (p/-dtype (.-inner ^gen.array.batch.BatchBackend b) v))
        (vswap! bools conj x)))
    x))

(defn unread-bools
  "[(f) bools]: f's result, and the batched bool values made during the
   call that no op ever read. Branching on a batched comparison with a raw
   if (or when, case, …) outside a gen body reads nothing, since Clojure's if
   tests the object itself, which is true, for every particle. A comparison
   whose result nothing read is therefore the trace of such a branch."
  [f]
  (binding [*bools* (volatile! [])]
    (let [r (f)]
      [r (filterv #(not (.wasRead ^Batched %)) @*bools*)])))

(defn- lift
  "x as an array of the wrapped backend (or a host scalar) ready to
   broadcast per particle with operands of per-particle rank r: the batch
   axes (batch-shape, or 1s for an unbatched value), then 1s, then its
   per-particle shape."
  [^BatchBackend b x r]
  (let [bs (.-batch-shape b)
        inner (.-inner b)]
    (cond
      (batched? x) (let [v (value x)
                         s (subvec (p/-shape inner v) (count bs))]
                     (if (== (count s) r) v (p/-reshape inner v (-> bs (into (repeat (- r (count s)) 1)) (into s)))))
      (array/host? x) x
      :else (let [s (p/-shape inner x)]
              (p/-reshape inner x (-> [] (into (repeat (+ (count bs) (- r (count s))) 1)) (into s)))))))

(defn- with-batch-axes
  "x as an array of backend inner with the leading batch axes of
   batch-shape: a batched value's array, or an unbatched value broadcast to
   every particle (a host scalar weakly typed, as everywhere)."
  [inner x batch-shape]
  (if (batched? x)
    (value x)
    (let [s (array/shape x)
          v (if (array/host? x) (p/-full inner [] x (dt/weak-host-dtype x (p/-defaults inner))) x)]
      (p/-broadcast-to inner (p/-reshape inner v (into (vec (repeat (count batch-shape) 1)) s)) (into (vec batch-shape) s)))))

(defn- full-batch
  "x as an array of the wrapped backend with the full batch axes."
  [^BatchBackend b x]
  (with-batch-axes (.-inner b) x (.-batch-shape b)))

;; ---------------------------------------------------------------------------
;; Gradients inside a native transformation
;; ---------------------------------------------------------------------------

(defn- by-kind
  "grads, f's gradients with respect to args at argnums, each batched
   exactly when its argument or f's value v is (a shared one broadcast to
   every particle): so that the tape and a native transformation, which
   compute the same values, also give them in the same kind."
  [b v args argnums grads]
  (mapv (fn [i g]
          (tree/refill g (map (fn [leaf gl]
                                (if (and (or (batched? leaf) (batched? v)) (not (batched? gl)))
                                  (made (full-batch b gl) b)
                                  gl))
                              (tree/leaves (nth args i)) (tree/leaves g))))
        argnums grads))

(defn- value-and-grad-inside
  "-value-and-grad by the wrapped backend's own transformation (MLX's
   native gradients, p/INativeGrad), which calls the function it
   differentiates with arrays of its own: so the tracer runs inside it. It
   differentiates g at the differentiated leaves, a batched one as its
   whole batch: g is f where f's value is shared (nothing batched reached
   it), and the sum over the particles of f where it is batched. Particle
   i's value depends on its own values only, so the gradient of the sum
   with respect to particle i's slice is particle i's gradient. A shared
   leaf's gradient is then per particle only if the leaf is broadcast to
   every particle first: when the value turns out batched with a shared
   leaf, f runs again, on the broadcast leaves. f's value is kept from the
   call."
  [^BatchBackend b f args argnums]
  (let [inner (.-inner b)
        leaves (fn [as i] (tree/leaves (nth as i)))
        shared? (some #(not (batched? %)) (mapcat #(leaves args %) argnums))
        attempt (fn [copies?]
                  ;; a leaf as the transformation takes it, its whole batch
                  ;; where it is batched (or broadcast, copies?), and back
                  (let [whole? #(or copies? (batched? %))
                        out (volatile! nil)
                        in (reduce (fn [as i] (update as i #(array/tree-map (fn [l] (if (whole? l) (full-batch b l) l)) %)))
                                   (vec args) argnums)
                        g (fn [& as]
                            (let [as (reduce (fn [as i]
                                               (assoc as i (tree/refill (nth as i)
                                                                        (map #(if (whole? %1) (made %2 b) %2) (leaves args i) (leaves as i)))))
                                             (vec as) argnums)
                                  y (apply f as)]
                              (when-not (and (array/array? y) (= [] (array/shape y)))
                                (throw (ex-info "value-and-grad needs a function returning a scalar"
                                                {:shape (when (array/array? y) (array/shape y)) :value y})))
                              (vreset! out y)
                              (cond (not (batched? y)) y
                                    (and shared? (not copies?)) (throw (ex-info "A shared leaf of a batched value" {::copies true}))
                                    :else (p/-reduce inner :sum (value y) (vec (range (count (.-batch-shape b)))) false))))
                        [_ grads] (p/-value-and-grad inner g in argnums)]
                    [@out (mapv (fn [i gr] (tree/refill gr (map #(if (whole? %1) (made %2 b) %2) (leaves args i) (tree/leaves gr))))
                                  argnums grads)]))]
    (try (attempt false)
         (catch clojure.lang.ExceptionInfo e
           (if (::copies (ex-data e)) (attempt true) (throw e))))))

(def ^:private backends (atom {}))

(defn backend
  "The batch backend over inner for batch-shape (one instance per pair, so
   that the values of a batched run join)."
  [inner batch-shape]
  (let [k [inner (vec batch-shape)]]
    (or (get @backends k)
        (get (swap! backends #(if (contains? % k) % (assoc % k (->BatchBackend inner (vec batch-shape) nil)))) k))))

(defn batch-backend?
  "Is b a batch backend (the tracer)?"
  [b]
  (instance? BatchBackend b))

(defn inner-backend
  "The backend a batched value's particles are arrays of. It reads nothing,
   so it marks no read (unread-bools)."
  [^Batched x]
  (.-inner ^BatchBackend (.-backend x)))

(defn backend-for
  "The batch backend for batch-shape over the backend in effect (which may
   already be that batch backend, in a nested call)."
  [batch-shape]
  (let [b array/*backend*
        bs (vec batch-shape)]
    (if (instance? BatchBackend b)
      ;; a batched collection inside a batched run would tie its particles to
      ;; the outer ones: run the outer one per particle instead
      (throw (array/batched-branch "A batched run inside a batched run" {:outer (.-batch-shape ^BatchBackend b) :inner bs}))
      (backend b bs))))

(defmacro batched
  "Run body batched over batch-shape: under the batch backend."
  [batch-shape & body]
  `(array/with-backend (backend-for ~batch-shape) ~@body))

(defn wrap
  "The batched value of an array of the wrapped backend whose leading axes
   are batch-shape."
  [x batch-shape]
  (Batched. x (backend (array/backend x) batch-shape) false))

(defn gather
  "x's particles at the indices idx (an int array), as a batch of
   batch-shape, lazily: the gather runs the first time the value is read, so
   gathering a whole trace (resampling) costs nothing for the values that are
   never read again."
  [x idx batch-shape]
  (Batched. (delay (array/take (value x) idx 0)) (backend (.-inner ^BatchBackend (.-backend ^Batched x)) batch-shape) false))

(defn unwrap
  "x as an array of the wrapped backend with the leading batch axes of
   batch-shape: a batched value's array, or an unbatched value broadcast to
   every particle."
  [x batch-shape]
  (with-batch-axes (array/backend x) x batch-shape))

;; ---------------------------------------------------------------------------
;; Parts of a batch
;; ---------------------------------------------------------------------------

;; Code that runs some of a batch's particles apart (a Switch whose index
;; differs across them runs each branch on its own: DESIGN.md §9) runs under
;; a part's backend (part-backend), takes the part's rows of the batch's
;; values into it (gather-into), and puts its results back (scatter). Such a
;; batch has one axis.

(defrecord Parts [groups])
;; A batch's data that differs in structure across its particles, by part:
;; groups is a vector of {:key :rows :data}, the data of the particles at
;; rows (host integers, in the order the part holds them), itself a batch of
;; their number; the particles of a group share key (a Switch's branch).

(defn parts?
  "Is x a batch's data held by parts (Parts)?"
  [x]
  (instance? Parts x))

(defn- one-axis
  "b's batch shape, which must have one axis."
  [^BatchBackend b]
  (let [bs (.-batch-shape b)]
    (when-not (== 1 (count bs))
      (throw (array/batched-branch "Parts of a batch of more than one axis" {:batch-shape bs})))
    bs))

(defn size
  "The number of particles of b, a batch backend of one axis."
  [b]
  (long (first (one-axis b))))

(defn index-array
  "rows (host integers) as an int array of the batch backend b's wrapped
   backend: gather-into's indices."
  [^BatchBackend b rows]
  (p/-from-host (.-inner b) (vec rows) [(count rows)] :int64))

(defn part-backend
  "A batch backend for the particles of the batch backend b at rows (host
   integers, in the part's order): a batch of their number, whose draws are
   b's draws at rows, so that a particle draws in a part what it would in
   b's own run. A new backend at each call, as a part's values must join no
   other part's."
  [^BatchBackend b rows]
  (let [inner (.-inner b)
        idx (index-array b rows)
        [full outer] (or (.-draws b) [(one-axis b) nil])]
    (BatchBackend. inner [(count rows)] [full (if outer (p/-take inner outer idx 0) idx)])))

(defn gather-into
  "x's particles at idx (index-array's int array), or all of them in order
   when idx is nil, as a value of the batch backend b, lazily (as gather);
   an unbatched value as it is, every particle's."
  [x idx b]
  (if (batched? x)
    (Batched. (delay (cond-> (value x) idx (array/take idx 0))) b false)
    x))

(defn- joined
  "join's array, of the wrapped backend inner."
  [inner xs ns]
  (let [shape (array/shape (first xs))
        strong (remove array/host? xs)
        dtype (if (seq strong) (array/dtype (first strong)) (dt/weak-host-dtype (first xs) (p/-defaults inner)))]
    (when-not (and (every? #(= shape (array/shape %)) xs)
                   (every? #(= dtype (array/dtype %)) strong)
                   (every? #(= (dt/kind dtype) (dt/kind (dt/host-dtype %))) (filter array/host? xs)))
      (throw (array/batched-branch "Values that differ in shape or dtype across the parts of a batch cannot be one batched value"
                                   {:shapes (mapv array/shape xs) :dtypes (mapv array/dtype xs)})))
    (p/-concatenate inner
                    (mapv (fn [x n]
                            (if (batched? x)
                              (let [v (value x)]
                                (when-not (= n (first (p/-shape inner v)))
                                  (throw (ex-info "A part's value is not a batch of the part's particles"
                                                  {:particles n :shape (p/-shape inner v)})))
                                v)
                              (p/-broadcast-to inner
                                               (p/-reshape inner (if (array/host? x) (p/-full inner [] x dtype) x) (into [1] shape))
                                               (into [n] shape))))
                          xs ns)
                    0)))

(defn join
  "The values xs of consecutive parts of a batch, (xs j) one of (ns j)
   particles (a batched value of that many, or a value they all share), as
   one value of the batch backend b over all their particles in order. The
   parts must agree in per-particle shape and dtype, a host scalar taking
   the others' dtype of its kind; otherwise no one value can hold them: the
   batched-branch error."
  [xs ns ^BatchBackend b]
  (made (joined (.-inner b) xs ns) b))

(defn scatter
  "The values xs of parts that partition the particles of a batch, (xs j)
   those of the particles at (rows j), as one value of the batch backend b:
   join's, each particle's value from its part."
  [xs rows ^BatchBackend b]
  (let [order (into [] cat rows)
        inv (long-array (count order))]
    (dotimes [j (count order)] (aset inv (long (order j)) (long j)))
    (made (p/-take (.-inner b) (joined (.-inner b) xs (mapv count rows)) (index-array b inv) 0) b)))
