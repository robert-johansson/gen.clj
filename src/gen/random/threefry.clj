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
   the key and i. 32-bit words are held in longs, masked to 32 bits."
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

(defmacro ^:private rounds
  "Unrolled Threefry rounds on locals x0, x1 with rotations rs, then body."
  [x0 x1 rs & body]
  (if (empty? rs)
    `(do ~@body)
    `(let [~x0 (add32 ~x0 ~x1)
           ~x1 (bit-xor (rotl32 ~x1 ~(first rs)) ~x0)]
       (rounds ~x0 ~x1 ~(rest rs) ~@body))))

(defn hash2x32
  "Threefry-2x32-20 of counter [c0 c1] under key [k0 k1] (uint32 values in
   longs). Returns both output words packed as (w0 << 32) | w1."
  ^long [^long k0 ^long k1 ^long c0 ^long c1]
  (let [k2 (bit-xor k0 k1 0x1BD11BDA)
        x0 (add32 c0 k0)
        x1 (add32 c1 k1)]
    (rounds x0 x1 [13 15 26 6]
      (let [x0 (add32 x0 k1) x1 (add32 x1 (add32 k2 1))]
        (rounds x0 x1 [17 29 16 24]
          (let [x0 (add32 x0 k2) x1 (add32 x1 (add32 k0 2))]
            (rounds x0 x1 [13 15 26 6]
              (let [x0 (add32 x0 k0) x1 (add32 x1 (add32 k1 3))]
                (rounds x0 x1 [17 29 16 24]
                  (let [x0 (add32 x0 k1) x1 (add32 x1 (add32 k2 4))]
                    (rounds x0 x1 [13 15 26 6]
                      (let [x0 (add32 x0 k2) x1 (add32 x1 (add32 k0 5))]
                        (bit-or (bit-shift-left x0 32) x1)))))))))))))

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
