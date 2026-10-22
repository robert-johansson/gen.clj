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
   reads, and a raw if reads nothing."
  (:require [gen.array :as array]
            [gen.array.dtype :as dt]
            [gen.array.protocols :as p]))

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

(defn batched? [x] (instance? Batched x))

(defn value
  "The wrapped array of a batched value (leading batch axes). Every op reads
   its operands through value, which marks them read."
  [^Batched x]
  (.markRead x)
  (let [v (.-v x)] (if (delay? v) @v v)))

(declare backend lift full-batch made)

(deftype BatchBackend [inner batch-shape]
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
    (let [r (reduce max 0 (map #(count (p/-shape b %)) args))
          lifted (mapv #(lift b % r) args)
          out (p/-elementwise inner op lifted dtype)]
      (if (some batched? args) (made out b) out)))

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
    (made (p/-random-uniform inner key (into batch-shape shape) dtype) b))

  (-value-and-grad [_ f args argnums]
    (if (some batched? (mapcat #(array/tree-leaves (nth args %)) argnums))
      (throw (array/batched-branch "Differentiating with respect to batched values is not supported" {}))
      ;; unbatched leaves: an ordinary gradient on the wrapped backend
      (p/-value-and-grad inner f args argnums))))

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

(defn- full-batch
  "x as an array of the wrapped backend with the full batch axes (an
   unbatched value broadcast to every particle)."
  [^BatchBackend b x]
  (let [bs (.-batch-shape b)
        inner (.-inner b)]
    (if (batched? x)
      (value x)
      (let [s (array/shape x)
            v (if (array/host? x) (p/-full inner [] x (array/dtype x)) x)]
        (p/-broadcast-to inner (p/-reshape inner v (into (vec (repeat (count bs) 1)) s)) (into bs s))))))

(def ^:private backends (atom {}))

(defn backend
  "The batch backend over inner for batch-shape (one instance per pair, so
   that the values of a batched run join)."
  [inner batch-shape]
  (let [k [inner (vec batch-shape)]]
    (or (get @backends k)
        (get (swap! backends #(if (contains? % k) % (assoc % k (->BatchBackend inner (vec batch-shape))))) k))))

(defn batch-backend? [b] (instance? BatchBackend b))

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
  (if (batched? x)
    (value x)
    (let [s (array/shape x)
          v (if (array/host? x) (array/array x) x)]
      (array/broadcast-to (array/reshape v (into (vec (repeat (count batch-shape) 1)) s)) (into (vec batch-shape) s)))))
