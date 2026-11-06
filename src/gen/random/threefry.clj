(ns gen.random.threefry
  "Threefry-2x32 with 20 rounds (Salmon et al., Random123), the counter-based
   generator behind JAX and MLX, and the key scheme of JAX's default
   ('partitionable') mode:

     key from seed s  = [hi32(s) lo32(s)]
     split(key, n)[i] = threefry(key, [hi32(i) lo32(i)])   (both output words)
     fold-in(key, d)  = threefry(key, [hi32(d) lo32(d)]) = split(key, ·)[d]
     bits32[i]        = w0 ^ w1   where [w0 w1] = threefry(key, [hi32(i) lo32(i)])
     bits64[i]        = w0 << 32 | w1

   fold-in takes a non-negative 64-bit integer (gen.random rejects other
   data): below 2^32 its key is JAX's, bit for bit;
   beyond, it is distinct where JAX masks typed data to 32 bits, so that
   Recurse's node numbers and the implicit key source's counter never
   repeat a key (DESIGN.md §9).

   Everything is pure and index-local, so element i of any draw depends only on
   the key and i. 32-bit words are held in longs, masked to 32 bits.

   The schedule (rounds, rotations and key injections) is written once, as
   data: hash2x32 unrolls it on host longs, and hash-words runs it in the
   uint32 ops of an array backend (the MLX backend's and the staging
   backend's draws), so that a key draws the same bits everywhere."
  {:api :internal})

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

(deftype Key [^long k0 ^long k1]
  Object
  (equals [_ o]
    (and (instance? Key o)
         (== k0 (.-k0 ^Key o))
         (== k1 (.-k1 ^Key o))))
  (hashCode [_] (unchecked-int (bit-xor (* 31 k0) k1))))

(defmethod print-method Key [^Key k ^java.io.Writer w]
  (.write w (str "#gen/key[" (.-k0 k) " " (.-k1 k) "]")))

(defmacro ^:private add32 [a b] `(bit-and (+ ~a ~b) 0xFFFFFFFF))

(defmacro ^:private rotl32 [x r]
  `(bit-and (bit-or (bit-shift-left ~x ~r) (unsigned-bit-shift-right ~x (- 32 ~r)))
            0xFFFFFFFF))

(def ^:private rotations
  "Threefry-2x32-20's schedule: five groups of four rounds, each round
   x0 += x1, x1 = rotl(x1, r) ^ x0 for its rotation r, and a key injection
   after each group (injection)."
  [[13 15 26 6] [17 29 16 24] [13 15 26 6] [17 29 16 24] [13 15 26 6]])

(def ^:private ^:const parity
  "The key schedule's parity: the third key word is k0 ^ k1 ^ parity."
  0x1BD11BDA)

(defn- injection
  "The key injection after group i (from 1): which key word of [k0 k1 k2]
   is added to x0, and which to x1, with i itself."
  [^long i]
  [(rem i 3) (rem (inc i) 3)])

(defmacro ^:private unrolled
  "The schedule on the long locals x0 and x1 under the key words ks (three
   locals), unrolled, then the output words packed: hash2x32's body."
  [x0 x1 ks]
  `(let [~@(mapcat (fn [i rs]
                     (let [[a b] (injection i)]
                       (concat (mapcat (fn [r] [x0 `(add32 ~x0 ~x1) x1 `(bit-xor (rotl32 ~x1 ~r) ~x0)]) rs)
                               [x0 `(add32 ~x0 ~(ks a)) x1 `(add32 ~x1 (add32 ~(ks b) ~i))])))
                   (iterate inc 1) rotations)]
     (bit-or (bit-shift-left ~x0 32) ~x1)))

(defn hash2x32
  "Threefry-2x32-20 of counter [c0 c1] under key [k0 k1] (uint32 values in
   longs). Returns both output words packed as (w0 << 32) | w1."
  ^long [^long k0 ^long k1 ^long c0 ^long c1]
  (let [k2 (bit-xor k0 k1 parity)
        x0 (add32 c0 k0)
        x1 (add32 c1 k1)]
    (unrolled x0 x1 [k0 k1 k2])))

(defn hash-words
  "Threefry-2x32-20 of the counters [0 c] under the key words k0 and k1, as
   hash2x32, in a vocabulary of uint32 words given as functions: (add x y)
   (wrapping at 2^32) and (xor x y) of words, (rotl x r) by a host count r,
   (word v) of a host uint32 v, and (zeros c) a zero word of c's shape.
   Returns [w0 w1]. The schedule is hash2x32's, op for op, for words that
   are arrays: the MLX backend's handles and the staging backend's values,
   whose counters (split's and the draws') are below 2^32."
  [{:keys [add xor rotl word zeros]} k0 k1 c]
  (let [ks [k0 k1 (xor (xor k0 k1) (word parity))]
        round (fn [[x0 x1] r] (let [x0 (add x0 x1)] [x0 (xor (rotl x1 r) x0)]))]
    (reduce (fn [xs [i rs]]
              (let [[x0 x1] (reduce round xs rs)
                    [a b] (injection i)]
                [(add x0 (ks a)) (add x1 (add (ks b) (word i)))]))
            [(add (zeros c) k0) (add c k1)]
            (map vector (iterate inc 1) rotations))))

(defn- hash-at
  "The packed Threefry output for flat index i under key."
  ^long [^Key key ^long i]
  (hash2x32 (.-k0 key) (.-k1 key) (unsigned-bit-shift-right i 32) (bit-and i 0xFFFFFFFF)))

(defn- unpack ^Key [^long w]
  (Key. (unsigned-bit-shift-right w 32) (bit-and w 0xFFFFFFFF)))

(defn seed->key
  "The key for an integer seed: [hi32 lo32] of the 64-bit seed."
  ^Key [seed]
  (let [s (long seed)]
    (Key. (unsigned-bit-shift-right s 32) (bit-and s 0xFFFFFFFF))))

(defn split
  "n new keys from key."
  [^Key key n]
  (mapv #(unpack (hash-at key %)) (range n)))

(defn fold-in
  "A new key from key and a non-negative integer, all of its 64 bits
   (deterministic, e.g. per address): split's key at index data."
  ^Key [^Key key data]
  (unpack (hash-at key (long data))))

(defn bits32
  "n uint32 values (in longs) from key."
  ^longs [^Key key n]
  (let [n (long n) out (long-array n)]
    (dotimes [i n]
      (let [w (hash-at key i)]
        (aset out i (bit-xor (unsigned-bit-shift-right w 32) (bit-and w 0xFFFFFFFF)))))
    out))

(defn bits64
  "n 64-bit values from key (two's-complement longs)."
  ^longs [^Key key n]
  (let [n (long n) out (long-array n)]
    (dotimes [i n] (aset out i (hash-at key i)))
    out))

(defn uniform-doubles
  "n float64 values in [0, 1): 52 random mantissa bits, as JAX."
  ^doubles [^Key key n]
  (let [n (long n) out (double-array n)]
    (dotimes [i n]
      (let [w (hash-at key i)]
        (aset out i (- (Double/longBitsToDouble
                        (bit-or (unsigned-bit-shift-right w 12) 0x3FF0000000000000))
                       1.0))))
    out))

(defn uniform-floats
  "n float32 values in [0, 1) (held as doubles): 23 random mantissa bits, as
   JAX."
  ^doubles [^Key key n]
  (let [n (long n) ^longs b (bits32 key n) out (double-array n)]
    (dotimes [i n]
      (aset out i (double (- (Float/intBitsToFloat
                              (unchecked-int (bit-or (unsigned-bit-shift-right (aget b i) 9)
                                                     0x3F800000)))
                             (float 1.0)))))
    out))
