(ns gen.random
  "Purely functional randomness. Keys are immutable values and every sampler is
   a pure function of its key: the same key gives the same draw.

   Keys follow JAX's Threefry scheme (gen.random.threefry), with fold-in
   taking 64-bit data. Every sampler is built from a single backend
   primitive, -random-uniform, using JAX's transforms. On the float64 JVM
   backend, uniform draws are bit-identical to JAX's, and normal, gumbel and
   exponential draws match JAX to rounding."
  (:refer-clojure :exclude [key + - * / < > <= >= ==])
  (:require [gen.array :as array :refer [+ - * / < > <= >= ==]]
            [gen.array.dtype :as dt]
            [gen.array.protocols :as p]
            [gen.random.threefry :as threefry]))

(set! *warn-on-reflection* true)

;; ---------------------------------------------------------------------------
;; Keys
;; ---------------------------------------------------------------------------

(defn key
  "The key for an integer seed (as JAX's PRNGKey)."
  [seed]
  (threefry/seed->key seed))

(defn key? [x] (instance? gen.random.threefry.Key x))

(defn- check-key [k]
  (if (key? k)
    k
    (throw (ex-info (str "Not a gen.random key: " (pr-str k)) {:value k}))))

(defn split
  "n new, independent keys from k (two by default)."
  ([k] (split k 2))
  ([k n] (threefry/split (check-key k) n)))

(defn fold-in
  "A new key from k and an integer (64 bits), e.g. to derive one key per
   address: (split k n)'s key at index data, for every n past it."
  [k data]
  (threefry/fold-in (check-key k) data))

(defn key-data
  "The two uint32 words of a key."
  [k]
  (let [^gen.random.threefry.Key k (check-key k)]
    [(.-k0 k) (.-k1 k)]))

;; ---------------------------------------------------------------------------
;; Samplers
;; ---------------------------------------------------------------------------

(defn- backend-for
  "The backend a sampler draws on: :backend in opts, else that of the first
   backend-array parameter, else gen.array/*backend*."
  [opts & params]
  (array/effective-backend
   (or (:backend opts)
       (some #(when (and (some? %) (not (array/host? %))) (array/backend %)) params)
       array/*backend*)))

(defn- float-dtype
  "The float dtype for a draw: :dtype in opts, else that of a float array
   parameter, else the backend's default float."
  [opts b param]
  (or (:dtype opts)
      (when (and (some? param) (not (array/host? param)) (= :float (dt/kind (array/dtype param))))
        (array/dtype param))
      (:float (p/-defaults b))))

(defn uniform
  "Uniform values in [low, high) of shape (default [], a host number on the JVM
   backend). Options: :low, :high (arrays broadcast), :dtype, :backend."
  ([key] (uniform key [] nil))
  ([key shape] (uniform key shape nil))
  ([key shape {:keys [low high] :as opts}]
   (let [b (backend-for opts low high)
         u (p/-random-uniform b (check-key key) (vec shape) (float-dtype opts b nil))]
     (if (and (nil? low) (nil? high))
       u
       (let [low (or low 0.0) high (or high 1.0)]
         (array/maximum low (+ (* u (- high low)) low)))))))

(def ^:private sqrt-2 (Math/sqrt 2.0))

(defn normal
  "Standard normal values of shape: √2·erfinv(u), u uniform on [nextafter(-1, 0),
   1) (as JAX). Options: :dtype, :backend."
  ([key] (normal key [] nil))
  ([key shape] (normal key shape nil))
  ([key shape opts]
   (let [b (backend-for opts)
         dtype (float-dtype opts b nil)
         lo (if (= dtype :float32)
              (double (Math/nextAfter (float -1.0) 0.0))
              (Math/nextAfter -1.0 0.0))]
     (* sqrt-2 (array/erfinv (uniform key shape {:low lo :high 1.0 :dtype dtype :backend b}))))))

(defn gumbel
  "Standard Gumbel values of shape: -log(-log u), u uniform on [tiny, 1) (as
   JAX). Options: :dtype, :backend."
  ([key] (gumbel key [] nil))
  ([key shape] (gumbel key shape nil))
  ([key shape opts]
   (let [b (backend-for opts)
         dtype (float-dtype opts b nil)
         tiny (if (= dtype :float32) (double Float/MIN_NORMAL) Double/MIN_NORMAL)]
     (- (array/log (- (array/log (uniform key shape {:low tiny :high 1.0 :dtype dtype :backend b}))))))))

(defn exponential
  "Standard exponential values of shape: -log1p(-u) (as JAX). Options: :dtype,
   :backend."
  ([key] (exponential key [] nil))
  ([key shape] (exponential key shape nil))
  ([key shape opts]
   (- (array/log1p (- (uniform key shape opts))))))

(defn bernoulli
  "Bool values, true with probability p (an array or number). shape defaults
   to p's shape. Options: :backend."
  ([key p] (bernoulli key p nil nil))
  ([key p shape] (bernoulli key p shape nil))
  ([key p shape opts]
   (let [b (backend-for opts p)]
     (< (uniform key (or shape (array/shape p)) {:dtype (float-dtype opts b p) :backend b}) p))))

(defn categorical
  "Index values drawn from the unnormalized log-probabilities along the last
   axis of logits (the Gumbel-max trick, as JAX). shape defaults to the batch
   shape of logits (all but the last axis) and must broadcast with it.
   Options: :backend."
  ([key logits] (categorical key logits nil nil))
  ([key logits shape] (categorical key logits shape nil))
  ([key logits shape opts]
   (let [ls (array/shape logits)
         _ (when (empty? ls) (throw (ex-info "categorical needs logits of rank >= 1" {:shape ls})))
         b (backend-for opts logits)
         g (gumbel key (conj (vec (or shape (pop ls))) (peek ls))
                   {:dtype (float-dtype opts b logits) :backend b})]
     (array/argmax (+ g logits) {:axis -1}))))

(defn first-accepted
  "The value of the first accepted of proposals, [accept? value] pairs in
   order, else fallback: the fold of a rejection sampler with a fixed number
   of vectorized rounds (gamma's here, gen.dist's Poisson and binomial), so
   it runs branch-free and batches."
  [fallback proposals]
  (reduce (fn [acc [ok value]] (array/where ok value acc)) fallback (reverse proposals)))

(def ^:private gamma-rounds
  "Proposal rounds per gamma draw. Marsaglia–Tsang accepts with probability
   above 0.95 for alpha >= 1, so twelve rounds all fail with probability below
   1e-15 (and the draw then falls back to the mode-like value d)."
  12)

(defn gamma
  "Gamma(alpha, 1) values of shape (default: alpha's shape), by Marsaglia–Tsang
   with a fixed number of vectorized proposal rounds (no data-dependent loop),
   boosting alpha < 1 as Γ(α) = Γ(α+1)·U^(1/α). Options: :dtype, :backend."
  ([key alpha] (gamma key alpha nil nil))
  ([key alpha shape] (gamma key alpha shape nil))
  ([key alpha shape opts]
   (let [shape (vec (or shape (array/shape alpha)))
         b (backend-for opts alpha)
         o {:dtype (float-dtype opts b alpha) :backend b}
         boost? (< alpha 1)
         a1 (array/where boost? (+ alpha 1) alpha)
         d (- a1 (/ 1.0 3.0))
         c (/ 1.0 (array/sqrt (* 9.0 d)))
         ks (split key (inc (* 2 gamma-rounds)))
         proposal (fn [r]
                    (let [x (normal (ks (+ 1 (* 2 r))) shape o)
                          u (uniform (ks (+ 2 (* 2 r))) shape o)
                          v (array/pow (+ 1.0 (* c x)) 3)
                          ok (array/logical-and (> v 0)
                                                (< (array/log u) (+ (* 0.5 x x) d (- (* d v)) (* d (array/log v)))))]
                      [ok (* d v)]))
         g (first-accepted d (map proposal (range gamma-rounds)))]
     (array/where boost? (* g (array/pow (uniform (ks 0) shape o) (/ 1.0 alpha))) g))))

(defn randint
  "Integers uniform in [low, high) of shape (default []). Options: :backend."
  ([key low high] (randint key low high [] nil))
  ([key low high shape] (randint key low high shape nil))
  ([key low high shape opts]
   (let [b (backend-for opts low high)
         u (uniform key shape {:backend b :dtype (:float (p/-defaults b))})]
     (array/astype (array/floor (+ low (* u (- high low)))) (:int (p/-defaults b))))))
