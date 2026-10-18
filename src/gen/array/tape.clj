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
  (:refer-clojure :exclude [+ - * / < > ==])
  (:require [gen.array :as a :refer [+ - * / < > ==]]
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

(defn- differentiable? [x] (= :float (dt/kind (a/dtype x))))

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
  (let [gs (a/shape g)]
    (if (= gs shape)
      g
      (let [lead (clojure.core/- (count gs) (count shape))
            g (if (pos? lead) (a/sum g {:axis (vec (range lead))}) g)
            gs (a/shape g)
            axes (vec (keep-indexed (fn [i d] (when (and (= 1 d) (not= 1 (nth gs i))) i)) shape))
            g (if (seq axes) (a/sum g {:axis axes :keep-dims? true}) g)]
        (a/reshape g shape)))))

(defn- kept-shape
  "shape with the reduced axes set to 1 (an empty axes vector reduces none)."
  [shape axes]
  (let [ax (set axes)]
    (vec (map-indexed (fn [i d] (if (ax i) 1 d)) shape))))

(defn- spread
  "A cotangent of a reduction over axes, broadcast back to shape."
  [g shape axes]
  (let [g (a/reshape g (kept-shape shape axes))]
    (if (= (a/shape g) shape) g (a/broadcast-to g shape))))

(defn- inverse-perm [perm]
  (reduce-kv (fn [inv i p] (assoc inv p i)) (vec perm) (vec perm)))

(defn- swap-last-two [x]
  (let [r (a/ndim x)]
    (a/transpose x (-> (vec (range r)) (assoc (clojure.core/- r 2) (dec r)) (assoc (dec r) (clojure.core/- r 2))))))

(defn- tril
  "The lower (or, with upper?, upper) triangle of the last two axes."
  [x upper?]
  (let [n (peek (a/shape x))
        i (a/reshape (a/arange n) [-1 1])
        j (a/reshape (a/arange n) [1 -1])]
    (a/where (if upper? (a/<= i j) (a/>= i j)) x 0.0)))

(defn- product-of-others
  "For each element of x, the product of the other elements in its
   reduction group over axes, computed without division (exact at zeros)."
  [x axes]
  (let [r (a/ndim x)
        axes (vec (sort axes))
        keep (vec (remove (set axes) (range r)))
        perm (into keep axes)
        xt (a/transpose x perm)
        s (a/shape xt)
        lead (subvec s 0 (count keep))
        n (reduce clojure.core/* 1 (subvec s (count keep)))]
    (if (zero? n)
      (a/zeros-like x)
      (let [flat (a/reshape xt (conj lead n))
            ones (a/ones (conj lead 1) {:dtype (a/dtype x)})
            left (a/concatenate [ones (a/slice (a/cumprod flat {:axis -1}) -1 0 (dec n))] -1)
            right (a/concatenate [(a/slice (a/cumprod flat {:axis -1 :reverse? true}) -1 1 nil) ones] -1)]
        (a/transpose (a/reshape (* left right) s) (inverse-perm perm))))))

(defn- indices-along
  "An int array of shape s holding start + step·j along axis (j the position)."
  [s axis start step]
  (let [n (nth s axis)
        idx (+ start (* step (a/arange n)))]
    (a/broadcast-to (a/reshape idx (assoc (vec (repeat (count s) 1)) axis n)) s)))

(defn- unslice
  "The cotangent of x from the cotangent g of a strided slice of x."
  [g shape starts steps]
  (reduce (fn [g k]
            (let [gs (a/shape g)]
              (if (and (= (nth gs k) (nth shape k)) (zero? (nth starts k)) (= 1 (nth steps k)))
                g
                (a/scatter-add (a/zeros (assoc gs k (nth shape k)) {:dtype (a/dtype g)})
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
  {:negative (fn [g _ _ need] (grads need (a/negate g)))
   :add (fn [g _ _ need] (grads need g g))
   :subtract (fn [g _ _ need] (grads need g (a/negate g)))
   :multiply (fn [g [x y] _ need] (grads need (* g y) (* g x)))
   :divide (fn [g [_ y] out need] (grads need (/ g y) (a/negate (/ (* g out) y))))
   :power (fn [g [x y] out need]
            (grads need (* g (* y (a/pow x (- y 1))))
                   (* g (a/where (== x 0) 0.0 (* out (a/log x))))))
   ;; ties split the cotangent evenly, as JAX
   :maximum (fn [g [x y] _ need]
              (let [wx (a/where (> x y) 1.0 (a/where (== x y) 0.5 0.0))]
                (grads need (* g wx) (* g (- 1.0 wx)))))
   :minimum (fn [g [x y] _ need]
              (let [wx (a/where (< x y) 1.0 (a/where (== x y) 0.5 0.0))]
                (grads need (* g wx) (* g (- 1.0 wx)))))
   :remainder (fn [g [x y] _ need] (grads need g (* (a/negate g) (a/floor (/ x y)))))
   :abs (fn [g [x] _ need] (grads need (* g (a/sign x))))
   :sqrt (fn [g _ out need] (grads need (/ (* 0.5 g) out)))
   :exp (fn [g _ out need] (grads need (* g out)))
   :log (fn [g [x] _ need] (grads need (/ g x)))
   :expm1 (fn [g _ out need] (grads need (* g (+ out 1.0))))
   :log1p (fn [g [x] _ need] (grads need (/ g (+ x 1.0))))
   :sin (fn [g [x] _ need] (grads need (* g (a/cos x))))
   :cos (fn [g [x] _ need] (grads need (a/negate (* g (a/sin x)))))
   :tan (fn [g _ out need] (grads need (* g (+ 1.0 (* out out)))))
   :tanh (fn [g _ out need] (grads need (* g (- 1.0 (* out out)))))
   :erf (fn [g [x] _ need] (grads need (* g (* (/ 2.0 sqrt-pi) (a/exp (a/negate (* x x)))))))
   :erfinv (fn [g _ out need] (grads need (* g (* (/ sqrt-pi 2.0) (a/exp (* out out))))))
   :where (fn [g [c] _ need] (grads need nil (a/where c g 0.0) (a/where c 0.0 g)))})

(defn- matmul-vjp
  "Cotangents of x and y from g, for NumPy matmul (1-d operands are vectors)."
  [g x y need]
  (let [xv? (== 1 (a/ndim x)) yv? (== 1 (a/ndim y))
        x2 (if xv? (a/reshape x [1 -1]) x)
        y2 (if yv? (a/reshape y [-1 1]) y)
        g2 (cond-> g yv? (a/expand-dims -1) xv? (a/expand-dims -2))]
    (grads need
           (a/reshape (unbroadcast (a/matmul g2 (swap-last-two y2)) (a/shape x2)) (a/shape x))
           (a/reshape (unbroadcast (a/matmul (swap-last-two x2) g2) (a/shape y2)) (a/shape y)))))

(defn- cholesky-vjp
  "The cotangent of a from the cotangent g of its lower Cholesky factor l.
   Only a's lower triangle is read, so the cotangent is lower triangular:
   S + Sᵀ below the diagonal and S on it, for S = l⁻ᵀ Φ(lᵀ g) l⁻¹, Φ the lower
   triangle with its diagonal halved (Murray 2016)."
  [inner l g]
  (let [n (peek (a/shape l))
        eye (a/eye n)
        phi (fn [x] (- (tril x false) (* 0.5 (* x eye))))
        p (phi (a/matmul (swap-last-two l) (tril g false)))
        ;; S = l⁻ᵀ p l⁻¹, by two triangular solves
        x (p/-linalg inner :solve-triangular [l p {:lower? true :transpose? true}])
        s (swap-last-two (p/-linalg inner :solve-triangular [l (swap-last-two x) {:lower? true :transpose? true}]))]
    (- (tril (+ s (swap-last-two s)) false) (* s eye))))

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
  (-to-host [_ x] (let [v (primal x)] (if (a/host? v) v (p/-to-host inner v))))
  (-eval! [_ xs] (p/-eval! inner (remove a/host? (map primal xs))))

  (-shape [_ x] (let [v (primal x)] (if (a/host? v) [] (p/-shape inner v))))
  (-dtype [_ x] (let [v (primal x)] (if (a/host? v) (dt/host-dtype v) (p/-dtype inner v))))

  (-full [_ shape value dtype] (p/-full inner shape value dtype))
  (-arange [_ start stop step dtype] (p/-arange inner start stop step dtype))
  (-astype [t x dtype]
    (let [v (primal x)
          out (if (a/host? v) (dt/host-cast v dtype) (p/-astype inner v dtype))]
      (record t out [x] (fn [g _] [(a/astype g (a/dtype v))]))))

  (-elementwise [t op args dtype]
    (let [ps (mapv primal args)
          out (p/-elementwise inner op ps dtype)]
      (if-let [rule (elementwise-rules op)]
        (record t out args (fn [g need]
                             (mapv (fn [ct x] (when ct (unbroadcast ct (a/shape x)))) (rule g ps out need) ps)))
        out)))

  (-reduce [t op x axes keep-dims?]
    (let [v (primal x)
          s (a/shape v)
          out (p/-reduce inner op v axes keep-dims?)]
      (case op
        :sum (record t out [x] (fn [g _] [(spread g s axes)]))
        (:max :min) (record t out [x]
                            (fn [g _]
                              (let [hit (a/astype (== v (spread out s axes)) (a/dtype v))
                                    ties (a/sum hit {:axis axes :keep-dims? true})]
                                [(* (spread g s axes) (/ hit ties))])))
        :prod (record t out [x] (fn [g _] [(* (spread g s axes) (product-of-others v axes))]))
        out)))

  (-scan [t op x axis reverse?]
    (let [v (primal x)
          out (p/-scan inner op v axis reverse?)]
      (case op
        :cumsum (record t out [x] (fn [g _] [(a/cumsum g {:axis axis :reverse? (not reverse?)})]))
        :cumprod (record t out [x] (fn [g _] [(/ (a/cumsum (* g out) {:axis axis :reverse? (not reverse?)}) v)]))
        (if (tracked? x)
          (throw (ex-info (str "The scan " op " is not differentiable") {:op op}))
          out))))

  (-reshape [t x shape]
    (let [v (primal x)
          out (if (and (a/host? v) (empty? shape)) v (p/-reshape inner v shape))]
      (record t out [x] (fn [g _] [(a/reshape g (a/shape v))]))))

  (-transpose [t x perm]
    (record t (p/-transpose inner (primal x) perm) [x]
            (fn [g _] [(a/transpose g (inverse-perm perm))])))

  (-broadcast-to [t x shape]
    (let [v (primal x)]
      (record t (p/-broadcast-to inner v shape) [x] (fn [g _] [(unbroadcast g (a/shape v))]))))

  (-concatenate [t xs axis]
    (let [ps (mapv primal xs)]
      (record t (p/-concatenate inner ps axis) xs
              (fn [g need]
                (let [ends (reductions clojure.core/+ (map #(nth (a/shape %) axis) ps))]
                  (mapv (fn [i start end] (when (need i) (a/slice g axis start end)))
                        (range) (cons 0 ends) ends))))))

  (-slice [t x starts stops steps]
    (let [v (primal x)]
      (record t (p/-slice inner v starts stops steps) [x]
              (fn [g _] [(unslice g (a/shape v) starts steps)]))))

  (-take [t x indices axis]
    (let [v (primal x)
          idx (primal indices)]
      (record t (p/-take inner v idx axis) [x]
              (fn [g _]
                (let [s (a/shape v)
                      k (a/size idx)
                      gk (a/reshape g (-> (subvec s 0 axis) (conj k) (into (subvec s (inc axis)))))
                      ik (a/broadcast-to (a/reshape idx (assoc (vec (repeat (count s) 1)) axis k)) (a/shape gk))]
                  [(a/scatter-add (a/zeros s {:dtype (a/dtype g)}) ik gk axis)])))))

  (-take-along-axis [t x indices axis]
    (let [v (primal x)
          idx (primal indices)]
      (record t (p/-take-along-axis inner v idx axis) [x]
              (fn [g _]
                (let [gs (a/shape g)
                      full (assoc gs axis (nth (a/shape v) axis))]
                  [(unbroadcast (a/scatter-add (a/zeros full {:dtype (a/dtype g)}) (a/broadcast-to idx gs) g axis)
                                (a/shape v))])))))

  (-scatter-add [t x indices updates axis]
    (let [idx (primal indices)]
      (record t (p/-scatter-add inner (primal x) idx (primal updates) axis) [x updates]
              (fn [g need] (grads need g (a/take-along-axis g idx axis))))))

  (-matmul [t x y]
    (let [px (primal x) py (primal y)]
      (record t (p/-matmul inner px py) [x y] (fn [g need] (matmul-vjp g px py need)))))

  (-linalg [t op args]
    (case op
      :cholesky
      (let [[m] args
            l (p/-linalg inner :cholesky [(primal m)])]
        (record t l [m] (fn [g _] [(cholesky-vjp inner l g)])))

      :solve-triangular
      (let [[m rhs {:keys [lower? transpose?] :as opts}] args
            pm (primal m)
            x (p/-linalg inner :solve-triangular [pm (primal rhs) opts])]
        (record t x [m rhs]
                (fn [g need]
                  ;; x = A⁻¹ rhs with A = m or mᵀ: rhs̄ = A⁻ᵀ ḡ, Ā = -rhs̄ xᵀ
                  (let [rhs-bar (p/-linalg inner :solve-triangular [pm g {:lower? lower? :transpose? (not transpose?)}])
                        a-bar (a/negate (a/matmul rhs-bar (swap-last-two x)))]
                    (grads need
                           (unbroadcast (tril (if transpose? (swap-last-two a-bar) a-bar) (not lower?)) (a/shape pm))
                           (unbroadcast rhs-bar (a/shape (primal rhs))))))))

      :solve
      (let [[m rhs] args
            pm (primal m)
            x (p/-linalg inner :solve [pm (primal rhs)])]
        (record t x [m rhs]
                (fn [g need]
                  (let [rhs-bar (p/-linalg inner :solve [(swap-last-two pm) g])]
                    (grads need
                           (unbroadcast (a/negate (a/matmul rhs-bar (swap-last-two x))) (a/shape pm))
                           (unbroadcast rhs-bar (a/shape (primal rhs))))))))

      :slogdet
      (let [[m] args
            pm (primal m)
            [sign logabs] (p/-linalg inner :slogdet [pm])]
        [sign (record t logabs [m]
                      (fn [g _]
                        (let [n (peek (a/shape pm))
                              inv (p/-linalg inner :solve [pm (a/broadcast-to (a/eye n) (a/shape pm))])]
                          [(* (a/expand-dims (a/expand-dims g -1) -1) (swap-last-two inv))])))])))

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
    (.put cts root (a/ones-like (.-primal root)))
    (doseq [^Tracked node (reverse order)]
      (when-let [g (.get cts node)]
        (when-let [vjp (.-vjp node)]
          (doseq [[parent ct] (map vector (.-parents node) (vjp g))]
            (when (and parent ct)
              (.put cts parent (if-let [acc (.get cts parent)] (+ acc ct) ct)))))))
    cts))

(defn- track-leaf [tape leaf]
  (when (tracked? leaf) (nested!))
  (when-not (and (a/array? leaf) (differentiable? leaf))
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
        args (reduce (fn [args i] (update args i #(a/tree-map (partial track-leaf tape) %))) (vec args) argnums)
        out (apply f args)]
    (when-not (and (a/array? out) (= [] (a/shape out)))
      (throw (ex-info "value-and-grad needs a function returning a scalar"
                      {:shape (when (a/array? out) (a/shape out)) :value out})))
    (binding [a/*backend* inner]
      (let [cts (if (tracked? out) (backward out) (java.util.IdentityHashMap.))
            grad-of (fn [leaf]
                      (let [v (primal leaf)
                            ct (.get cts leaf)]
                        (if ct (a/astype ct (a/dtype v)) (a/zeros-like v))))]
        [(primal out) (mapv #(a/tree-map grad-of (nth args %)) argnums)]))))
