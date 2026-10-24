;; book: ns
(ns book.arrays-backends
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.edn :as edn]
            [gen.array :as array :refer [+ - * /]]
            [gen.array.batch :as batch]
            [gen.array.dtype :as dtype]
            [gen.array.protocols :as p]
            [gen.book.svg :as svg]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.mh :as mh]
            [gen.random :as rng]
            [gen.random.threefry :as threefry]
            [gen.selection :as sel]))

;; book: values
(def m (array/array [[1.0 2.0] [3.0 4.0]]))

[(array/shape m) (array/ndim m) (array/shape 2.5) (array/ndim 2.5)]
;=> [[2 2] 2 [] 0]

[(array/host? 2.5) (array/host? true) (array/host? m) (array/array? m)]
;=> [true true false true]

;; book: host-math
[(+ 1 2) (* 2 3.5) (/ 1 2) (array/exp 0.0)]
;=> [3 7.0 0.5 1.0]

(clojure.core// 1 2)
;=> 1/2

(+ Long/MAX_VALUE 1)
;=> -9223372036854775808

;; book: host-dtypes
(mapv array/dtype [2.5 3 true])
;=> [:float64 :int64 :bool]

;; book: jvm
(def jvm
  "The JVM backend, named: code that shows what it does runs on it
   wherever the book runs."
  (array/backend-named :jvm))

;; book: zero-d
(array/with-backend jvm
  (let [s (array/sum (array/array [1.0 2.0 3.0]))]
    [s (instance? Double s)]))
;=> [6.0 true]

(array/item (array/sum (array/array [1.0 2.0 3.0])))
;=> 6.0

;; book: equality
[(array/equal? (array/array [1 2]) (array/array [1 2]))
 (array/equal? (+ 0.1 0.2) 0.3)
 (array/allclose? (+ 0.1 0.2) 0.3)]
;=> [true false true]

;; book: broadcast
(def column (array/array [[0.0] [10.0] [20.0]]))

(def row (array/array [1.0 2.0 3.0]))

(def table (+ column row))

[(array/shape column) (array/shape row) (array/shape table)]
;=> [[3 1] [3] [3 3]]

table
;=> [[1.0 2.0 3.0] [11.0 12.0 13.0] [21.0 22.0 23.0]]

;; book: broadcast-error
;; error: Shapes [3] and [2] do not broadcast
#_(+ row (array/array [1.0 2.0]))

;; book: reductions
[(array/sum table {:axis 0}) (array/shape (array/sum table {:axis -1}))]
;=> [[33.0 36.0 39.0] [3]]

(array/shape (array/sum table {:axis -1 :keep-dims? true}))
;=> [3 1]

[(array/max row) (array/maximum row 2.0)]
;=> [3.0 [2.0 2.0 3.0]]

;; book: pick
(array/pick (array/log (array/array [[0.1 0.9] [0.5 0.5]]))
            (array/array [1 0]))
;=> [-0.1054 -0.6931]

;; book: defaults
(array/defaults jvm)
;=> {:float :float64 :int :int64}

(let [{:keys [float int]} (array/defaults array/*backend*)]
  [(= float (array/dtype (array/array [1.5 2.0])))
   (= int (array/dtype (array/array [1 2])))])
;=> [true true]

;; book: promotion
(def dtypes [:bool :int32 :int64 :float32 :float64])

(defn promoted
  "The dtype of the sum of two arrays of the dtypes a and b."
  [a b]
  (array/dtype (+ (array/zeros [1] {:dtype a}) (array/zeros [1] {:dtype b}))))

(def promotion
  "The dtype of a sum, by the dtypes of its operands, on the JVM backend."
  (array/with-backend jvm
    (into {} (for [a dtypes] [a (mapv #(promoted a %) dtypes)]))))

(promotion :bool)
;=> [:int64 :int32 :int64 :float32 :float64]

(promotion :int32)
;=> [:int32 :int32 :int64 :float32 :float64]

(promotion :int64)
;=> [:int64 :int64 :int64 :float32 :float64]

(promotion :float32)
;=> [:float32 :float32 :float32 :float32 :float64]

(promotion :float64)
;=> [:float64 :float64 :float64 :float64 :float64]

;; book: weak
(let [f32 (array/array [1.0 2.0] {:dtype :float32})
      i32 (array/array [1 2] {:dtype :int32})]
  [(array/dtype (+ f32 2.5)) (array/dtype (+ i32 2))
   (= (array/dtype (+ i32 2.5)) (:float (array/defaults array/*backend*)))])
;=> [:float32 :int32 true]

;; book: booleans
[(+ true true) (array/negate true) (* true 2.5)]
;=> [2 -1 2.5]

(let [b (array/array [true false true])]
  [(array/sum b) (array/dtype (array/< row 2.5))
   (array/dtype (rng/bernoulli (rng/key 1) 0.5 [3]))])
;=> [2 :bool :bool]

;; book: float32
(= (array/->clj (array/array [0.1] {:dtype :float32})) [(double (float 0.1))])
;=> true

(array/item (array/array 0.1 {:dtype :float32}))
;=> 0.10000000149011612

;; book: storage
(array/with-backend jvm
  (array/->clj (+ (array/array [(long (Math/pow 2 53))]) 1)))
;=> [9007199254740992]

;; book: eval
(def noisy
  (gen []
    (let [mu (trace :mu (d/normal 0 1))]
      (trace :y (d/normal mu 0.5)))))

(defn mh-chain
  "n MH moves of tr's :mu, the trace evaluated after each move."
  [tr n key]
  (reduce (fn [tr k] (gfi/eval-trace! (:trace (mh/mh tr (sel/select :mu) k))))
          tr
          (rng/split key n)))

(def start (:trace (gfi/generate noisy [] (cm/choicemap :y 1.2) (rng/key 2))))

(identical? start (gfi/eval-trace! start))
;=> true

((mh-chain start 100 (rng/key 3)) :mu)
;=> 1.151 ; float64

;; book: concrete
(array/with-backend jvm (array/concrete? (array/array [0.3 0.7])))
;=> true

;; book: validation
;; error: bernoulli: prob must be in [0, 1], got 1.5
#_(array/with-backend jvm (d/bernoulli (array/array 1.5)))

;; book: numpy
(let [x (array/array [-7.0 ##NaN 7.0])]
  [(array/isnan (array/sign x))
   (array/floor-div (array/array [-7 7]) 2)
   (array/argmax x)])
;=> [[false true false] [-4 3] 1]

;; book: index-check
;; error: Index 5 is out of bounds for a dimension of size 2
#_(array/with-backend jvm
  (array/take (array/array [1.0 2.0]) (array/array [5])))

;; book: keys
(def k (rng/key 42))

[(pr-str k) (rng/key-data k) (array/array? k)]
;=> ["#gen/key[0 42]" [0 42] false]

(= (rng/split k 5) (rng/split k 5))
;=> true

(= (rng/fold-in k 3) ((rng/split k 5) 3))
;=> true

;; book: jax
(def jax
  "JAX's keys and draws for the key 42 (test/fixtures/random.edn)."
  (edn/read-string (slurp "test/fixtures/random.edn")))

(= (mapv rng/key-data (rng/split k 3))
   (:out (first (filter #(= 3 (:n %)) (:split jax)))))
;=> true

(= (array/->clj (rng/uniform k [6])) (:out (:uniform-f64 jax)))
;=> true ; float64

(= (array/->clj (rng/uniform k [6] {:dtype :float32}))
   (:out (:uniform-f32 jax)))
;=> true

;; book: index-local
(let [u (array/->clj (rng/uniform k [6]))]
  [(= (array/->clj (rng/uniform k [3])) (subvec u 0 3))
   (= (array/->clj (rng/uniform k [2 3])) [(subvec u 0 3) (subvec u 3 6)])])
;=> [true true]

;; book: bits
(defn uniform-64
  "The float64 uniform of a 64-bit Threefry word w: its top 52 bits."
  [w]
  (* (unsigned-bit-shift-right w 12) (Math/pow 2 -52)))

(defn uniform-32
  "The float32 uniform of a 32-bit Threefry word w: its top 23 bits."
  [w]
  (* (unsigned-bit-shift-right w 9) (Math/pow 2 -23)))

[(= (mapv uniform-64 (threefry/bits64 k 6)) (:out (:uniform-f64 jax)))
 (= (mapv uniform-32 (threefry/bits32 k 6)) (:out (:uniform-f32 jax)))]
;=> [true true]

(first (:out (:uniform-f64 jax)))
;=> 0.4267

(first (:out (:uniform-f32 jax)))
;=> 0.4887

;; book: pairs
(def pairs
  "The float64 and float32 uniforms of the key 7, element by element."
  (array/with-backend jvm
    (let [k (rng/key 7)]
      (mapv vector
            (array/->clj (rng/uniform k [10000]))
            (array/->clj (rng/uniform k [10000] {:dtype :float32}))))))

;; book: uniform-figure
;; figure: figures/041-uniforms.svg
(svg/plot [{:points (take 600 pairs) :r 1.5 :opacity 0.6}]
          {:x-range [0 1] :y-range [0 1]
           :x-label "float64 uniform" :y-label "float32 uniform"})

;; book: correlation
(defn correlation
  "The sample correlation of the pairs [x y]."
  [pairs]
  (let [n (count pairs)
        mean (fn [vs] (/ (reduce + vs) n))
        xs (map first pairs)
        ys (map second pairs)
        mx (mean xs)
        my (mean ys)
        dx (map #(- % mx) xs)
        dy (map #(- % my) ys)
        dot (fn [a b] (reduce + (map * a b)))]
    (/ (dot dx dy) (Math/sqrt (* (dot dx dx) (dot dy dy))))))

(< (abs (correlation pairs)) (/ 5 (Math/sqrt (count pairs))))
;=> true

;; book: normals
(def sqrt-2 (Math/sqrt 2.0))

(let [lo (Math/nextAfter -1.0 0.0)
      u (rng/uniform k [6] {:low lo :high 1.0})]
  (array/allclose? (rng/normal k [6]) (* sqrt-2 (array/erfinv u))))
;=> true ; float64

(array/allclose? (rng/normal k [6]) (array/array (:out (:normal-f64 jax))))
;=> true ; float64

;; book: normal-dtypes
(defn moments
  "The mean and the variance of the values xs."
  [xs]
  (let [n (count xs)
        mean (/ (reduce + xs) n)]
    [mean (/ (reduce + (map #(array/square (- % mean)) xs)) (dec n))]))

(defn standard?
  "Are the mean and the variance of n standard normal draws within five
   standard errors of 0 and 1? (Their errors are 1/√n and √(2/n).)"
  [xs]
  (let [n (count xs)
        [mean variance] (moments xs)]
    (and (< (abs mean) (* 5 (Math/sqrt (/ 1.0 n))))
         (< (abs (- variance 1.0)) (* 5 (Math/sqrt (/ 2.0 n)))))))

(def normals
  "The key 7's float64 and float32 standard normals, 10,000 of each."
  (array/with-backend jvm
    (let [k (rng/key 7)]
      [(array/->clj (rng/normal k [10000]))
       (array/->clj (rng/normal k [10000] {:dtype :float32}))])))

(mapv standard? normals)
;=> [true true]

(let [[f64 f32] normals]
  (< (abs (correlation (map vector f64 f32))) (/ 5 (Math/sqrt 10000))))
;=> true

;; book: grad
(defn log-density
  "The log density of 1.5 under a normal of mean mu and standard
   deviation sigma."
  [mu sigma]
  (dc/log-prob (d/normal mu sigma) 1.5))

((array/grad log-density) 0.5 2.0)
;=> 0.25

((array/grad log-density {:argnums [0 1]}) 0.5 2.0)
;=> [0.25 -0.375]

((array/value-and-grad log-density {:argnums [0 1]}) 0.5 2.0)
;=> [-1.7371 [0.25 -0.375]]

((array/grad (fn [{:keys [mu sigma]}] (log-density mu sigma)))
 {:mu 0.5 :sigma 2.0})
;=> {:mu 0.25 :sigma -0.375}

;; book: tape-backend
(def seen (atom nil))

((array/grad (fn [x]
               (reset! seen (array/backend-name (array/backend x)))
               (* x x)))
 3.0)
;=> 6.0

@seen
;=> :tape

;; book: composites
(let [v (array/array [1.0 2.0 3.0])]
  (array/allclose? ((array/grad array/logsumexp) v) (array/softmax v)))
;=> true

(def xs-lgamma (mapv #(* 0.1 %) (range 1 51)))

(def lgamma-slopes (mapv (array/grad array/lgamma) xs-lgamma))

(array/allclose? (array/array lgamma-slopes)
                 (array/digamma (array/array xs-lgamma))
                 {:atol 1e-6})
;=> true

;; book: lgamma-figure
;; figure: figures/041-lgamma.svg
(svg/plot [{:curves [array/lgamma] :color "gray"}
           {:curves [array/digamma]}
           {:points (map vector xs-lgamma lgamma-slopes) :r 2}]
          {:x-range [0.05 5] :y-range [-6 4]
           :y-label "log-gamma and digamma"})

;; book: branching
(defn kinked
  "|x|, by a branch on x's sign."
  [x]
  (if (array/truthy? (array/< x 0)) (array/negate x) x))

(mapv (array/grad kinked) [-2.0 3.0])
;=> [-1.0 1.0]

;; book: host-read
;; error: Cannot read a value being differentiated on the host
#_((array/grad (fn [x] (* x (array/item x)))) 3.0)

;; book: host-arithmetic
;; error: Host arithmetic (clojure.core's operators, Math/…, double)
#_((array/grad (fn [x] (Math/exp x))) 1.0)

;; book: stop-gradient
[((array/grad (fn [x] (* x x))) 3.0)
 ((array/grad (fn [x] (* x (array/stop-gradient x)))) 3.0)
 ((array/grad (fn [x] (* x (array/item (array/stop-gradient x))))) 3.0)]
;=> [6.0 3.0 3.0]

;; book: model-gradient
(defn joint
  "The log density of noisy's choices, mu and y = 1.2, as a function of
   mu."
  [mu]
  (:weight (gfi/assess noisy [] (cm/choicemap :mu mu :y 1.2) (rng/key 0))))

(array/allclose? ((array/grad joint) 0.5)
                 (+ (- 0.5) (/ (- 1.2 0.5) (* 0.5 0.5))))
;=> true

;; book: custom-vjp
(def cbrt
  (array/custom-vjp (fn [x] (Math/cbrt (array/item x)))
                    (fn [_ out g] [(/ g (* 3.0 out out))])))

[(cbrt 8.0) ((array/grad cbrt) 8.0) ((array/grad #(* % (cbrt %))) 8.0)]
;=> [2.0 0.08333 2.6667]

;; book: custom-vjp-error
;; error: custom-vjp: the VJP returned
#_((array/grad (array/custom-vjp (fn [x] (Math/cbrt (array/item x)))
                               (fn [_ _ g] [g g])))
 8.0)

;; book: nested
;; error: Differentiating inside a differentiated function is not supported
#_((array/grad (fn [x] ((array/grad (fn [y] (* x y y))) x))) 1.0)

;; book: compile
(def log-density-step
  (array/value-and-grad (fn [{:keys [mu sigma]}] (log-density mu sigma))))

(def compiled-step (array/compile log-density-step))

(let [params {:mu 0.5 :sigma 2.0}
      [v g] (log-density-step params)
      [cv cg] (compiled-step params)]
  [(array/allclose? v cv)
   (array/allclose? (:mu g) (:mu cg))
   (array/allclose? (:sigma g) (:sigma cg))])
;=> [true true true]

;; book: surface
[(count (:sigs p/IBackend)) (count dtype/elementwise-ops)]
;=> [27 46]

(let [x (array/with-backend jvm (array/array [0.0 1.0]))]
  (array/equal? (array/exp x)
                (p/-elementwise (array/backend x) :exp [x] :float64)))
;=> true

;; book: batched
(def points (array/with-backend jvm (array/array [0.0 1.0 2.0])))

(def particles (batch/wrap points [3]))

[(array/shape points) (array/shape particles)
 (array/backend-name (array/backend particles))]
;=> [[3] [] :batch]

(defn standard-log-density
  "The log density of x under the standard normal."
  [x]
  (dc/log-prob (d/normal 0 1) x))

(array/allclose? (batch/unwrap (standard-log-density particles) [3])
                 (standard-log-density points))
;=> true

;; book: per-particle
(def shared (array/with-backend jvm (array/array [10.0 20.0 30.0])))

(batch/unwrap (+ particles shared) [3])
;=> [[10.0 20.0 30.0] [11.0 21.0 31.0] [12.0 22.0 32.0]]

;; book: two-batches
;; error: add: arrays from different backends (:batch and :batch)
#_(+ particles (batch/wrap (array/with-backend jvm (array/array [1.0 2.0])) [2]))

;; book: batched-read
;; error: Cannot read on the host a batched value
#_(array/item particles)

;; book: wrapped
(defn wrapped
  "The names of backend b and of the backends it wraps, outermost first."
  [b]
  (cons (array/backend-name b)
        (when (instance? gen.array.protocols.IWrappingBackend b)
          (wrapped (p/-inner b)))))

(def chain-seen (atom nil))

(def slopes
  ((array/grad (fn [x]
                 (reset! chain-seen (wrapped (array/backend x)))
                 (standard-log-density x)))
   particles))

[(batch/unwrap slopes [3]) @chain-seen]
;=> [[-0.0 -1.0 -2.0] (:tape :batch :jvm)]

;; book: solution-1
(array/with-backend jvm
  (let [i64 (array/array [1 2])
        i32 (array/array [1 2] {:dtype :int32})
        f32 (array/array [1.0 2.0] {:dtype :float32})]
    (mapv array/dtype [(+ i64 f32) (* i32 2.5) (array/sum (array/< i64 2))
                       (/ i64 i64)])))
;=> [:float32 :float64 :int64 :float64]

;; book: solution-2
(defn gumbel
  "Standard Gumbel values of shape from key: -log(-log u), u uniform on
   [tiny, 1), tiny the least normal float64."
  [key shape]
  (let [u (rng/uniform key shape {:low Double/MIN_NORMAL :high 1.0})]
    (array/negate (array/log (array/negate (array/log u))))))

(def euler-gamma 0.5772156649015329)

(array/with-backend jvm
  (let [key (rng/key 9)
        g (gumbel key [10000])
        sd (/ Math/PI (Math/sqrt 6.0))]
    [(array/equal? g (rng/gumbel key [10000]))
     (< (abs (- (array/item (array/mean g)) euler-gamma))
        (* 5 (/ sd (Math/sqrt 10000))))]))
;=> [true true]

(array/allclose? (gumbel k [6]) (array/array (:out (:gumbel-f64 jax))))
;=> true ; float64

;; book: solution-3
[((array/grad (fn [x] (* x (array/floor x)))) 2.5)
 ((array/grad (fn [x] (array/where (array/> x 0) (* x x) 0.0))) 2.5)
 ((array/grad (fn [x] (* x (array/item (array/floor x))))) 2.5)]
;=> [2.0 5.0 2.0]

;; book: solution-3-error
;; error: Host arithmetic
#_((array/grad (fn [x] (Math/sqrt x))) 2.5)

;; book: solution-4
(def straight-through-round
  (array/custom-vjp array/round (fn [_ _ g] [g])))

[((array/grad #(array/square (array/round %))) 1.3)
 ((array/grad #(array/square (straight-through-round %))) 1.3)]
;=> [0.0 2.0]

;; book: solution-5
(def readings (array/with-backend jvm (array/array [-1.0 0.5 2.0 3.5])))

(defn reading-density
  "The log density of a reading x under a normal of mean mu and sd 2."
  [mu x]
  (dc/log-prob (d/normal mu 2.0) x))

(def per-reading
  (array/with-backend jvm
    (let [xs (batch/wrap readings [4])
          slopes (batch/batched [4]
                   ((array/grad reading-density) 1.0 xs))]
      (batch/unwrap slopes [4]))))

per-reading
;=> [-0.5 -0.125 0.25 0.625]

(array/with-backend jvm
  (let [one-at-a-time (mapv #((array/grad reading-density) 1.0 %)
                            (array/->clj readings))
        total (array/grad #(array/sum (reading-density % readings)))]
    [(array/allclose? per-reading (array/array one-at-a-time))
     (array/allclose? (array/sum per-reading) (total 1.0))]))
;=> [true true]
