(ns gen.array.tape
  "Reverse-mode automatic differentiation for any backend
   (docs/array-api.md §8).

   value-and-grad wraps the leaves of the differentiated arguments as tracked
   values. A tracked value is immutable: its primal (a value of the wrapped
   backend, or a host scalar), the tracked values it was computed from, and a
   VJP closure mapping a cotangent of the result to cotangents of those
   inputs. Tracked values belong to a tape backend that wraps the real one,
   created afresh for each value-and-grad call: it computes every primitive
   op's primal on the wrapped backend and records the op's VJP. There is no
   global tape. The backward pass sorts the graph reachable from the result
   topologically and accumulates cotangents in a local identity map.

   Composite ops need no rules: the tape backend declares no native ops, so
   gen.array builds composites from primitives. Results that are not floats,
   ops whose derivative is zero almost everywhere (comparisons, floor, sign,
   floor-divide, …) and stop-gradient give untracked values. cumprod's VJP
   divides by the input (NaN at zeros). First order only: differentiating
   inside a differentiated function is an error (bean genclj-hors)."
  (:refer-clojure :exclude [+ - * / < > <= >= ==])
  (:require [gen.array :as array :refer [+ - * / < > <= >= ==]]
            [gen.array.dtype :as dt]
            [gen.array.protocols :as p]))

(set! *warn-on-reflection* true)

(declare ->TapeBackend)

(deftype Tracked [primal parents vjp tape]
  p/IArray
  (-backend [_] tape))

(defmethod print-method Tracked [x ^java.io.Writer w]
  (.write w "#gen/tracked[") (print-method (.-primal ^Tracked x) w) (.write w "]"))

(defn tracked? [x] (instance? Tracked x))

(defn- primal [x] (if (tracked? x) (.-primal ^Tracked x) x))

(defn- primal-dtype
  "The dtype of a primal on backend inner: a host scalar's is weak, its
   kind's default there (float64 only where that is the default)."
  [inner v]
  (if (array/host? v)
    (dt/with-weak nil #{(dt/kind (dt/host-dtype v))} (p/-defaults inner))
    (p/-dtype inner v)))

(defn- differentiable? [x] (= :float (dt/kind (array/dtype x))))

(defn- record
  "The result out of an op on args: a tracked value whose VJP is
   (vjp cotangent need), need saying which args are tracked, when out is a
   float and some arg is tracked; out itself otherwise."
  [tape out args vjp]
  (if (and (some tracked? args) (differentiable? out))
    (let [need (mapv tracked? args)]
      (->Tracked out (mapv #(when (tracked? %) %) args) (fn [g] (vjp g need)) tape))
    out))

(defmacro ^:private grads
  "A vector of cotangents, the i-th computed only when (need i)."
  [need & exprs]
  `[~@(map-indexed (fn [i e] `(when (~need ~i) ~e)) exprs)])

;; ---------------------------------------------------------------------------
;; Shape helpers for cotangents
;; ---------------------------------------------------------------------------

(defn- unbroadcast
  "g summed down to shape: the cotangent of an operand broadcast to g's shape."
  [g shape]
  (let [gs (array/shape g)]
    (if (= gs shape)
      g
      (let [lead (clojure.core/- (count gs) (count shape))
            g (if (pos? lead) (array/sum g {:axis (vec (range lead))}) g)
            gs (array/shape g)
            axes (vec (keep-indexed (fn [i d] (when (and (= 1 d) (not= 1 (nth gs i))) i)) shape))
            g (if (seq axes) (array/sum g {:axis axes :keep-dims? true}) g)]
        (array/reshape g shape)))))

(defn- kept-shape
  "shape with the reduced axes set to 1 (an empty axes vector reduces none)."
  [shape axes]
  (let [ax (set axes)]
    (vec (map-indexed (fn [i d] (if (ax i) 1 d)) shape))))

(defn- spread
  "A cotangent of a reduction over axes, broadcast back to shape."
  [g shape axes]
  (let [g (array/reshape g (kept-shape shape axes))]
    (if (= (array/shape g) shape) g (array/broadcast-to g shape))))

(defn- inverse-perm [perm]
  (reduce-kv (fn [inv i p] (assoc inv p i)) (vec perm) (vec perm)))

(defn- swap-last-two [x]
  (let [r (array/ndim x)]
    (array/transpose x (-> (vec (range r)) (assoc (clojure.core/- r 2) (dec r)) (assoc (dec r) (clojure.core/- r 2))))))

(defn- tril
  "The lower (or, with upper?, upper) triangle of the last two axes."
  [x upper?]
  (let [n (peek (array/shape x))
        i (array/reshape (array/arange n) [-1 1])
        j (array/reshape (array/arange n) [1 -1])]
    (array/where (if upper? (<= i j) (>= i j)) x 0.0)))

(defn- product-of-others
  "For each element of x, the product of the other elements in its
   reduction group over axes, computed without division (exact at zeros)."
  [x axes]
  (let [r (array/ndim x)
        axes (vec (sort axes))
        keep (vec (remove (set axes) (range r)))
        perm (into keep axes)
        xt (array/transpose x perm)
        s (array/shape xt)
        lead (subvec s 0 (count keep))
        n (reduce clojure.core/* 1 (subvec s (count keep)))]
    (if (zero? n)
      (array/zeros-like x)
      (let [flat (array/reshape xt (conj lead n))
            ones (array/ones (conj lead 1) {:dtype (array/dtype x)})
            left (array/concatenate [ones (array/slice (array/cumprod flat {:axis -1}) -1 0 (dec n))] -1)
            right (array/concatenate [(array/slice (array/cumprod flat {:axis -1 :reverse? true}) -1 1 nil) ones] -1)]
        (array/transpose (array/reshape (* left right) s) (inverse-perm perm))))))

(defn- indices-along
  "An int array of shape s holding start + step·j along axis (j the position)."
  [s axis start step]
  (let [n (nth s axis)
        idx (+ start (* step (array/arange n)))]
    (array/broadcast-to (array/reshape idx (assoc (vec (repeat (count s) 1)) axis n)) s)))

(defn- unslice
  "The cotangent of x from the cotangent g of a strided slice of x."
  [g shape starts steps]
  (reduce (fn [g k]
            (let [gs (array/shape g)]
              (if (and (= (nth gs k) (nth shape k)) (zero? (nth starts k)) (= 1 (nth steps k)))
                g
                (array/scatter-add (array/zeros (assoc gs k (nth shape k)) {:dtype (array/dtype g)})
                                   (indices-along gs k (nth starts k) (nth steps k))
                                   g k))))
          g
          (range (count shape))))

;; ---------------------------------------------------------------------------
;; VJP rules
;; ---------------------------------------------------------------------------

(def ^:private sqrt-pi (Math/sqrt Math/PI))

(def ^:private elementwise-rules
  "op → (fn [g primals out need] cotangents), before unbroadcasting."
  {:negative (fn [g _ _ need] (grads need (array/negate g)))
   :add (fn [g _ _ need] (grads need g g))
   :subtract (fn [g _ _ need] (grads need g (array/negate g)))
   :multiply (fn [g [x y] _ need] (grads need (* g y) (* g x)))
   :divide (fn [g [_ y] out need] (grads need (/ g y) (array/negate (/ (* g out) y))))
   :power (fn [g [x y] out need]
            (grads need (* g (* y (array/pow x (- y 1))))
                   (* g (array/where (== x 0) 0.0 (* out (array/log x))))))
   ;; ties split the cotangent evenly, as JAX
   :maximum (fn [g [x y] _ need]
              (let [wx (array/where (> x y) 1.0 (array/where (== x y) 0.5 0.0))]
                (grads need (* g wx) (* g (- 1.0 wx)))))
   :minimum (fn [g [x y] _ need]
              (let [wx (array/where (< x y) 1.0 (array/where (== x y) 0.5 0.0))]
                (grads need (* g wx) (* g (- 1.0 wx)))))
   :remainder (fn [g [x y] _ need] (grads need g (* (array/negate g) (array/floor (/ x y)))))
   :abs (fn [g [x] _ need] (grads need (* g (array/sign x))))
   :sqrt (fn [g _ out need] (grads need (/ (* 0.5 g) out)))
   :exp (fn [g _ out need] (grads need (* g out)))
   :log (fn [g [x] _ need] (grads need (/ g x)))
   :expm1 (fn [g _ out need] (grads need (* g (+ out 1.0))))
   :log1p (fn [g [x] _ need] (grads need (/ g (+ x 1.0))))
   :sin (fn [g [x] _ need] (grads need (* g (array/cos x))))
   :cos (fn [g [x] _ need] (grads need (array/negate (* g (array/sin x)))))
   :tan (fn [g _ out need] (grads need (* g (+ 1.0 (* out out)))))
   :tanh (fn [g _ out need] (grads need (* g (- 1.0 (* out out)))))
   :erf (fn [g [x] _ need] (grads need (* g (* (/ 2.0 sqrt-pi) (array/exp (array/negate (* x x)))))))
   :erfinv (fn [g _ out need] (grads need (* g (* (/ sqrt-pi 2.0) (array/exp (* out out))))))
   :where (fn [g [c] _ need] (grads need nil (array/where c g 0.0) (array/where c 0.0 g)))})

(defn- matmul-vjp
  "Cotangents of x and y from g, for NumPy matmul (1-d operands are vectors)."
  [g x y need]
  (let [xv? (== 1 (array/ndim x)) yv? (== 1 (array/ndim y))
        x2 (if xv? (array/reshape x [1 -1]) x)
        y2 (if yv? (array/reshape y [-1 1]) y)
        g2 (cond-> g yv? (array/expand-dims -1) xv? (array/expand-dims -2))]
    (grads need
           (array/reshape (unbroadcast (array/matmul g2 (swap-last-two y2)) (array/shape x2)) (array/shape x))
           (array/reshape (unbroadcast (array/matmul (swap-last-two x2) g2) (array/shape y2)) (array/shape y)))))

(defn- cholesky-vjp
  "The cotangent of a from the cotangent g of its lower Cholesky factor l.
   Only a's lower triangle is read, so the cotangent is lower triangular:
   S + Sᵀ below the diagonal and S on it, for S = l⁻ᵀ Φ(lᵀ g) l⁻¹, Φ the lower
   triangle with its diagonal halved (Murray 2016)."
  [inner l g]
  (let [n (peek (array/shape l))
        eye (array/eye n)
        phi (fn [x] (- (tril x false) (* 0.5 (* x eye))))
        p (phi (array/matmul (swap-last-two l) (tril g false)))
        ;; S = l⁻ᵀ p l⁻¹, by two triangular solves
        x (p/-linalg inner :solve-triangular [l p {:lower? true :transpose? true}])
        s (swap-last-two (p/-linalg inner :solve-triangular [l (swap-last-two x) {:lower? true :transpose? true}]))]
    (- (tril (+ s (swap-last-two s)) false) (* s eye))))

(defn linalg-vjp
  "The cotangents of a linear-algebra op's array arguments (where need, a
   vector of booleans, says) from the cotangent g of its result out (of
   logabs, for :slogdet), by inner's ops, args being the op's arguments as
   values. The tape's rules; the MLX backend gives them to MLX as custom
   VJPs in its native mode."
  [inner op args out g need]
  (case op
    :cholesky (grads need (cholesky-vjp inner out g))
    ;; x = A⁻¹ rhs with A = m or mᵀ: rhs̄ = A⁻ᵀ ḡ, Ā = -rhs̄ xᵀ
    :solve-triangular
    (let [[m rhs {:keys [lower? transpose?]}] args
          rhs-bar (p/-linalg inner :solve-triangular [m g {:lower? lower? :transpose? (not transpose?)}])
          a-bar (array/negate (array/matmul rhs-bar (swap-last-two out)))]
      (grads need
             (unbroadcast (tril (if transpose? (swap-last-two a-bar) a-bar) (not lower?)) (array/shape m))
             (unbroadcast rhs-bar (array/shape rhs))))
    :solve
    (let [[m rhs] args
          rhs-bar (p/-linalg inner :solve [(swap-last-two m) g])]
      (grads need
             (unbroadcast (array/negate (array/matmul rhs-bar (swap-last-two out))) (array/shape m))
             (unbroadcast rhs-bar (array/shape rhs))))
    ;; d log|det m| = tr(m⁻¹ dm): m̄ = ḡ m⁻ᵀ
    :slogdet
    (let [[m] args
          n (peek (array/shape m))
          inv (p/-linalg inner :solve [m (array/broadcast-to (array/eye n) (array/shape m))])]
      (grads need (* (array/expand-dims (array/expand-dims g -1) -1) (swap-last-two inv))))))

;; ---------------------------------------------------------------------------
;; The tape backend
;; ---------------------------------------------------------------------------

(defn- nested! []
  (throw (ex-info "Differentiating inside a differentiated function is not supported yet (bean genclj-hors)" {})))

(deftype TapeBackend [inner]
  p/IWrappingBackend
  (-inner [_] inner)

  p/IBackend
  (-name [_] :tape)
  (-defaults [_] (p/-defaults inner))
  (-lazy? [_] (p/-lazy? inner))
  (-native-ops [_] #{})

  (-from-host [_ data dtype] (p/-from-host inner data dtype))
  (-to-host [_ x] (let [v (primal x)] (if (array/host? v) v (p/-to-host inner v))))
  (-eval! [_ xs] (p/-eval! inner (remove array/host? (map primal xs))))

  (-shape [_ x] (let [v (primal x)] (if (array/host? v) [] (p/-shape inner v))))
  (-dtype [_ x] (primal-dtype inner (primal x)))

  (-full [_ shape value dtype] (p/-full inner shape value dtype))
  (-arange [_ start stop step dtype] (p/-arange inner start stop step dtype))
  (-astype [t x dtype]
    (let [v (primal x)
          out (if (array/host? v) (dt/host-cast v dtype) (p/-astype inner v dtype))]
      (record t out [x] (fn [g _] [(array/astype g (primal-dtype inner v))]))))

  (-elementwise [t op args dtype]
    (let [ps (mapv primal args)
          out (p/-elementwise inner op ps dtype)]
      (if-let [rule (elementwise-rules op)]
        (record t out args (fn [g need]
                             (mapv (fn [ct x] (when ct (unbroadcast ct (array/shape x)))) (rule g ps out need) ps)))
        out)))

  (-reduce [t op x axes keep-dims?]
    (let [v (primal x)
          s (array/shape v)
          out (p/-reduce inner op v axes keep-dims?)]
      (case op
        :sum (record t out [x] (fn [g _] [(spread g s axes)]))
        (:max :min) (record t out [x]
                            (fn [g _]
                              (let [hit (array/astype (== v (spread out s axes)) (array/dtype v))
                                    ties (array/sum hit {:axis axes :keep-dims? true})]
                                [(* (spread g s axes) (/ hit ties))])))
        :prod (record t out [x] (fn [g _] [(* (spread g s axes) (product-of-others v axes))]))
        out)))

  (-scan [t op x axis reverse?]
    (let [v (primal x)
          out (p/-scan inner op v axis reverse?)]
      (case op
        :cumsum (record t out [x] (fn [g _] [(array/cumsum g {:axis axis :reverse? (not reverse?)})]))
        :cumprod (record t out [x] (fn [g _] [(/ (array/cumsum (* g out) {:axis axis :reverse? (not reverse?)}) v)]))
        (if (tracked? x)
          (throw (ex-info (str "The scan " op " is not differentiable") {:op op}))
          out))))

  (-reshape [t x shape]
    (let [v (primal x)
          out (if (and (array/host? v) (empty? shape)) v (p/-reshape inner v shape))]
      (record t out [x] (fn [g _] [(array/reshape g (array/shape v))]))))

  (-transpose [t x perm]
    (record t (p/-transpose inner (primal x) perm) [x]
            (fn [g _] [(array/transpose g (inverse-perm perm))])))

  (-broadcast-to [t x shape]
    (let [v (primal x)]
      (record t (p/-broadcast-to inner v shape) [x] (fn [g _] [(unbroadcast g (array/shape v))]))))

  (-concatenate [t xs axis]
    (let [ps (mapv primal xs)]
      (record t (p/-concatenate inner ps axis) xs
              (fn [g need]
                (let [ends (reductions clojure.core/+ (map #(nth (array/shape %) axis) ps))]
                  (mapv (fn [i start end] (when (need i) (array/slice g axis start end)))
                        (range) (cons 0 ends) ends))))))

  (-slice [t x starts stops steps]
    (let [v (primal x)]
      (record t (p/-slice inner v starts stops steps) [x]
              (fn [g _] [(unslice g (array/shape v) starts steps)]))))

  (-take [t x indices axis]
    (let [v (primal x)
          idx (primal indices)]
      (record t (p/-take inner v idx axis) [x]
              (fn [g _]
                (let [s (array/shape v)
                      k (array/size idx)
                      gk (array/reshape g (-> (subvec s 0 axis) (conj k) (into (subvec s (inc axis)))))
                      ik (array/broadcast-to (array/reshape idx (assoc (vec (repeat (count s) 1)) axis k)) (array/shape gk))]
                  [(array/scatter-add (array/zeros s {:dtype (array/dtype g)}) ik gk axis)])))))

  (-take-along-axis [t x indices axis]
    (let [v (primal x)
          idx (primal indices)]
      (record t (p/-take-along-axis inner v idx axis) [x]
              (fn [g _]
                (let [gs (array/shape g)
                      full (assoc gs axis (nth (array/shape v) axis))]
                  [(unbroadcast (array/scatter-add (array/zeros full {:dtype (array/dtype g)}) (array/broadcast-to idx gs) g axis)
                                (array/shape v))])))))

  (-scatter-add [t x indices updates axis]
    (let [idx (primal indices)]
      (record t (p/-scatter-add inner (primal x) idx (primal updates) axis) [x updates]
              (fn [g need] (grads need g (array/take-along-axis g idx axis))))))

  (-matmul [t x y]
    (let [px (primal x) py (primal y)]
      (record t (p/-matmul inner px py) [x y] (fn [g need] (matmul-vjp g px py need)))))

  (-linalg [t op args]
    (case op
      :cholesky
      (let [[m] args
            pm (primal m)
            l (p/-linalg inner :cholesky [pm])]
        (record t l [m] (fn [g need] (linalg-vjp inner :cholesky [pm] l g need))))

      :solve-triangular
      (let [[m rhs opts] args
            ps [(primal m) (primal rhs) opts]
            x (p/-linalg inner :solve-triangular ps)]
        (record t x [m rhs] (fn [g need] (linalg-vjp inner :solve-triangular ps x g need))))

      :solve
      (let [[m rhs] args
            ps [(primal m) (primal rhs)]
            x (p/-linalg inner :solve ps)]
        (record t x [m rhs] (fn [g need] (linalg-vjp inner :solve ps x g need))))

      :slogdet
      (let [[m] args
            pm (primal m)
            [sign logabs] (p/-linalg inner :slogdet [pm])]
        [sign (record t logabs [m] (fn [g need] (linalg-vjp inner :slogdet [pm] logabs g need)))])))

  (-random-uniform [_ key shape dtype] (p/-random-uniform inner key shape dtype))
  (-value-and-grad [_ _ _ _] (nested!)))

;; ---------------------------------------------------------------------------
;; The backward pass
;; ---------------------------------------------------------------------------

(defn- topological-order
  "The tracked values reachable from root, each after all of its parents
   (iterative, so long chains do not overflow the stack)."
  [root]
  (let [visited (java.util.IdentityHashMap.)
        order (java.util.ArrayList.)
        stack (java.util.ArrayDeque.)]
    (.push stack [root 0])
    (.put visited root true)
    (while (not (.isEmpty stack))
      (let [[^Tracked node i] (.pop stack)
            parents (.-parents node)]
        (if (clojure.core/< (long i) (count parents))
          (let [parent (nth parents i)]
            (.push stack [node (inc (long i))])
            (when (and parent (not (.containsKey visited parent)))
              (.put visited parent true)
              (.push stack [parent 0])))
          (.add order node))))
    order))

(defn- backward
  "The cotangent of every tracked value reachable from root, seeded with 1."
  ^java.util.IdentityHashMap [^Tracked root]
  (let [cts (java.util.IdentityHashMap.)
        order (topological-order root)]
    (.put cts root (array/ones-like (.-primal root)))
    (doseq [^Tracked node (reverse order)]
      (when-let [g (.get cts node)]
        (when-let [vjp (.-vjp node)]
          (doseq [[parent ct] (map vector (.-parents node) (vjp g))]
            (when (and parent ct)
              (.put cts parent (if-let [acc (.get cts parent)] (+ acc ct) ct)))))))
    cts))

(defn- track-leaf [tape leaf]
  (when (tracked? leaf) (nested!))
  (when-not (and (array/array? leaf) (differentiable? leaf))
    (throw (ex-info (str "Cannot differentiate with respect to " (pr-str leaf)
                         " (only float arrays and floating-point numbers)")
                    {:leaf leaf})))
  (->Tracked leaf [] nil tape))

(defn value-and-grad
  "[value grads]: f's value on args and its gradients with respect to the
   positions in argnums, computed on backend inner. The backend contract's
   -value-and-grad, for backends without native autodiff."
  [inner f args argnums]
  (when (instance? TapeBackend inner) (nested!))
  (let [tape (->TapeBackend inner)
        args (reduce (fn [args i] (update args i #(array/tree-map (partial track-leaf tape) %))) (vec args) argnums)
        out (apply f args)]
    (when-not (and (array/array? out) (= [] (array/shape out)))
      (throw (ex-info "value-and-grad needs a function returning a scalar"
                      {:shape (when (array/array? out) (array/shape out)) :value out})))
    (binding [array/*backend* inner]
      (let [cts (if (tracked? out) (backward out) (java.util.IdentityHashMap.))
            grad-of (fn [leaf]
                      (let [v (primal leaf)
                            ct (.get cts leaf)]
                        (if ct (array/astype ct (primal-dtype inner v)) (array/zeros-like v))))]
        [(primal out) (mapv #(array/tree-map grad-of (nth args %)) argnums)]))))
