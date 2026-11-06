(ns gen.random
  "Purely functional randomness. Keys are immutable values and every sampler is
   a pure function of its key: the same key gives the same draw.

   Keys follow JAX's Threefry scheme (gen.random.threefry), with fold-in
   taking 64-bit data. Every sampler is built from a single backend
   primitive, -random-uniform, using JAX's transforms. On the float64 JVM
   backend, uniform draws are bit-identical to JAX's, and normal, gumbel and
   exponential draws match JAX to rounding."
  {:api :stable}
  (:refer-clojure :exclude [key + - * / < > <= >= ==])
  (:require [gen.array :as array :refer [+ - * / < > ==]]
            [gen.array.dtype :as dt]
            [gen.array.protocols :as p]
            [gen.random.threefry :as threefry]))

(set! *warn-on-reflection* true)

;; ---------------------------------------------------------------------------
;; Keys
;; ---------------------------------------------------------------------------

(defn key
  "The key for an integer seed, 64 bits (as JAX's PRNGKey). Other seeds
   are an error: a long cast took 2.5 for 2 (genclj-kdau), as fold-in
   refuses such data."
  [seed]
  (when-not (and (integer? seed) (clojure.core/<= Long/MIN_VALUE seed Long/MAX_VALUE))
    (throw (ex-info (str "A key's seed is a 64-bit integer, not " (pr-str seed))
                    {:reason :invalid-data :seed seed})))
  (threefry/seed->key seed))

(defn key?
  "Is x a key: a host key, or a staged one (a device program's,
   p/IStagedKey)?"
  [x]
  (or (instance? gen.random.threefry.Key x) (instance? gen.array.protocols.IStagedKey x)))

(defn- staged-key? [k] (instance? gen.array.protocols.IStagedKey k))

(defn- check-key [k]
  (if (key? k)
    k
    (throw (ex-info (str "Not a gen.random key: " (pr-str k)) {:value k}))))

(defn split
  "n new, independent keys from k (two by default)."
  ([k] (split k 2))
  ([k n] (if (staged-key? k) (p/-split-key k n) (threefry/split (check-key k) n))))

(defn fold-in
  "A new key from k and a non-negative integer (64 bits), e.g. to derive one
   key per address: (split k n)'s key at index data, for every n past it.
   Other data is an error: as split's index it means nothing, JAX's key for
   a negative integer (data mod 2^32) is not this scheme's, and a long cast
   would take 2.5 for 2 (and fail past Long/MAX_VALUE)."
  [k data]
  (when-not (and (integer? data) (clojure.core/<= 0 data Long/MAX_VALUE))
    (throw (ex-info (str "fold-in takes a non-negative 64-bit integer, not " (pr-str data))
                    {:reason :invalid-data :data data})))
  (if (staged-key? k) (p/-fold-in-key k data) (threefry/fold-in (check-key k) data)))

(defn key-data
  "The two uint32 words of a host key. A staged key's words are a staged
   value, which the host cannot read: an error (genclj-kdau)."
  [k]
  (check-key k)
  (when (staged-key? k)
    (throw (ex-info "key-data: a staged key's words are staged values, not host numbers"
                    {:reason :staged-key})))
  (let [^gen.random.threefry.Key k k]
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

(defn ^:no-doc first-accepted
  "The value of the first accepted of proposals, [accept? value] pairs in
   order, else fallback: the fold of a rejection sampler with a fixed number
   of vectorized rounds, so it runs branch-free and batches. The samplers'
   shared helper (gamma's here, gen.dist's Poisson and binomial), not part
   of gen.random's API (docs/array-api.md §7)."
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

;; randint: low + floor(k·r / 2^(3L)) for r = high - low and k a uniform
;; integer of 3L bits, in three L-bit chunks, from the uniforms of the
;; backend's default float: in float64 one uniform's 52 bits (u·2^54, an
;; integer, in 18-bit chunks), in float32 three uniforms' 23 bits each
;; (u·2^23). Computed in int64 by L-bit limbs of r and columns with
;; carries, so that no intermediate passes 2^53 (the JVM backend's exact
;; int64s) for ranges below 2^53, nor 2^63 on MLX. Exact, so never high:
;; floor(low + u·r) in the default float reached high in float32
;; (genclj-psgi). Each integer has floor or ceil of 2^(3L) / r of the k's,
;; so it is drawn with probability within a relative r / 2^52 (float64) or
;; r / 2^69 (float32) of 1/r. In float64 it is floor(u·r) exactly, the
;; integer the old floor(low + u·r) gave wherever that did not round; the
;; uniforms are bit-identical across backends (docs/array-api.md §7), so a
;; key draws the same integers on every backend of the same default float.

(defn- chunks
  "[[d0 d1 d2] L]: the three L-bit chunks (low first) of a uniform integer
   of shape and 3L bits, from key's uniforms in the default float of b, as
   int64 arrays."
  [key shape b]
  (if (= :float64 (:float (p/-defaults b)))
    ;; u·2^54 (2^54 = 1.8014398509481984E16) is 4k for u's 52 random bits
    ;; k: an integer below 2^54, held exactly by the JVM backend's doubles
    ;; (a multiple of 4)
    (let [k (array/astype (* (uniform key shape {:dtype :float64 :backend b}) 1.8014398509481984E16) :int64)
          l 262144]
      [[(array/mod k l) (array/mod (array/floor-div k l) l) (array/floor-div k (* l l))] 18])
    (let [[k0 k1 k2] (split key 3)
          chunk #(array/astype (* (uniform % shape {:dtype :float32 :backend b}) 8388608.0) :int64)]
      [[(chunk k2) (chunk k1) (chunk k0)] 23])))

(defn- limbs
  "r (an int64 array or host integer, 0 <= r < 2^(3L)) as its L-bit limbs
   [r0 r1 r2], low first; a host r gives host limbs."
  [r ^long l-bits]
  (let [l (bit-shift-left 1 l-bits)]
    (if (array/host? r)
      (let [r (long r) mask (dec l)]
        [(bit-and r mask) (bit-and (bit-shift-right r l-bits) mask) (bit-shift-right r (* 2 l-bits))])
      [(array/mod r l) (array/mod (array/floor-div r l) l) (array/floor-div r (* l l))])))

(defn- top-of-product
  "floor(k·r / 2^(3L)) for k = d2·2^(2L) + d1·2^L + d0 and r's L-bit limbs
   [r0 r1 r2], by columns with carries; a host limb of 0 adds no term."
  [[d0 d1 d2] [r0 r1 r2] ^long l-bits]
  (let [l (bit-shift-left 1 l-bits)
        term (fn [d r] (when-not (and (array/host? r) (zero? (long r))) (* d r)))
        col (fn [& terms] (reduce (fn [acc t] (cond (nil? t) acc (nil? acc) t :else (+ acc t))) nil terms))
        carry (fn [c] (if (nil? c) 0 (array/floor-div c l)))
        plus (fn [a c] (cond (nil? a) c (and (array/host? c) (zero? (long c))) a :else (+ a c)))
        c1 (carry (col (term d0 r0)))
        c2 (carry (plus (col (term d0 r1) (term d1 r0)) c1))
        c3 (carry (plus (col (term d0 r2) (term d1 r1) (term d2 r0)) c2))
        col3 (col (term d1 r2) (term d2 r1))
        col4 (term d2 r2)]
    (cond-> (plus col3 c3)
      col4 (+ (* col4 l)))))

(defn- check-bound
  "A host bound must be an integer; an array's is cast to int64 (and read,
   when concrete, if it holds floats)."
  [what x]
  (if (array/host? x)
    (when-not (or (integer? x) (and (number? x) (Double/isFinite (double x)) (clojure.core/== (double x) (Math/floor (double x)))))
      (throw (ex-info (str "randint: " what " must be an integer, not " (pr-str x)) {:reason :invalid-args what x})))
    (when (and (= :float (dt/kind (array/dtype x))) (array/concrete? x)
               (not (array/all-true? (== x (array/floor x)))))
      (throw (ex-info (str "randint: " what " must hold integers") {:reason :invalid-args what x}))))
  (array/astype x :int64))

(defn randint
  "Integers uniform in [low, high) of shape (default []), low < high
   integers (arrays broadcast), computed exactly from the uniforms of the
   backend's default float: the same integers on every backend of that
   float for a key. Options: :backend."
  ([key low high] (randint key low high [] nil))
  ([key low high shape] (randint key low high shape nil))
  ([key low high shape opts]
   (let [b (backend-for opts low high)
         int-dt (:int (p/-defaults b))
         lo (check-bound :low low)
         hi (check-bound :high high)
         host? (and (array/host? lo) (array/host? hi))
         r (if host? (-' hi lo) (- hi lo))
         _ (cond host? (do (when-not (clojure.core/< 0 r 9223372036854775808N)
                             (throw (ex-info (str "randint: an empty (or too wide) range [" low ", " high ")")
                                             {:reason :empty-range :low low :high high})))
                           (when (and (= :int32 int-dt) (not (clojure.core/<= Integer/MIN_VALUE lo hi 2147483648)))
                             (throw (ex-info (str "randint: [" low ", " high ") holds integers outside int32, "
                                                  "the backend's int dtype")
                                             {:reason :invalid-args :low low :high high :dtype int-dt}))))
                 (every? array/concrete? [lo hi]) (when-not (array/all-true? (< lo hi))
                                                    (throw (ex-info "randint: an empty range (low >= high)"
                                                                    {:reason :empty-range :low low :high high}))))
         r (if host? (long r) r)
         [ds l-bits] (chunks key (vec shape) b)
         j (top-of-product ds (limbs r l-bits) l-bits)
         v (+ lo j)
         ;; exact, so in range, for host bounds whose integers the JVM
         ;; backend's doubles hold (within +-2^53, so r below 2^53 too);
         ;; past them, or for an array range (empty where it was not read),
         ;; clamped into [low, high - 1]: low when empty
         v (if (and host? (clojure.core/< (long r) 9007199254740992)
                    (clojure.core/<= -9007199254740992 lo) (clojure.core/<= hi 9007199254740992))
             v
             (array/maximum (array/minimum v (- hi 1)) lo))]
     (array/astype v int-dt))))
