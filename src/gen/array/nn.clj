(ns gen.array.nn
  "gen.array's neural-network operations (docs/array-api.md §6), the
   bodies of gen.array's vars of the same names, whose docstrings are
   there:
   - rms-norm, rope and sdpa: composites of gen.array's ops, so the tape
     differentiates them, the batch tracer batches them and the staging
     backend stages them; a backend with fused kernels (p/IFused: MLX's
     fast ops) computes them in one op where its arrays are the operands;
   - selective-log-softmax: a composite, its logsumexp over chunks of the
     last axis, of logits given whole or a chunk at a time;
   - quantize, dequantize, quantized-matmul and gather-qmm: primitives of
     p/IQuantized, in MLX's layouts, checked here.
   Written in gen.array's ops, this namespace requires gen.array, which
   loads it at the first call of one. Experimental: outside the 1.x
   promise (DESIGN §9)."
  {:api :internal}
  (:refer-clojure :exclude [+ - * / < > <= >= ==])
  (:require [clojure.core :as c]
            [gen.array :as array :refer [+ - * / >= ==]]
            [gen.array.dtype :as dt]
            [gen.array.protocols :as p]
            [gen.array.shape :as sh]))

(set! *warn-on-reflection* true)

;; ---------------------------------------------------------------------------
;; Operands
;; ---------------------------------------------------------------------------

(defn- invalid
  "The loud error of op's arguments: ex-info with :reason :invalid-args."
  ([op message] (invalid op message {}))
  ([op message data]
   (ex-info (str (name op) ": " message) (merge {:reason :invalid-args :op op} data))))

(defn- backend-of
  "The backend op runs on: its array operands' (nils left out)."
  [op xs]
  (array/backend-of op (remove nil? xs)))

(defn- float-dtype
  "The one float dtype the operands xs (nils left out) take on backend b:
   their common dtype as gen.array's ops promote them, host numbers
   weakly; b's default float for ints and bools."
  [b xs]
  (let [xs (remove nil? xs)
        strong (reduce (fn [s x] (if (array/host? x) s (let [d (array/dtype x)] (if s (dt/promote s d) d)))) nil xs)
        d (dt/with-weak strong (set (keep #(when (array/host? %) (dt/kind (dt/host-dtype %))) xs)) (p/-defaults b))]
    (if (= :float (dt/kind d)) d (:float (p/-defaults b)))))

(defn- as-dtype
  "x (an array, or nil) in dtype d."
  [x d]
  (cond (nil? x) nil
        (and (array/backend-array? x) (= d (array/dtype x))) x
        :else (array/astype x d)))

(defn- as-floats
  "The operands xs (nils kept) in their one float dtype on backend b."
  [b xs]
  (let [d (float-dtype b xs)]
    (mapv #(as-dtype % d) xs)))

(def ^:private half? #{:float16 :bfloat16})

(defn- compute-dtype
  "The dtype a composite computes a d-typed operand's norms, angles and
   softmax in: float32 for the half floats, as MLX's fast ops do, d
   otherwise."
  [d]
  (if (half? d) :float32 d))

(defn- fused
  "op's value on args by b's fused kernel (p/IFused), or nil where b has
   none or declines: then the composite runs."
  [b op args]
  (when (instance? gen.array.protocols.IFused b)
    (p/-fused b op args)))

(defn- check-host-number
  [op what v]
  (when-not (and (number? v) (Double/isFinite (double v)))
    (throw (invalid op (str what " must be a finite host number, not " (pr-str v)) {what v}))))

(defn- swap-last-two [x]
  (let [r (array/ndim x)]
    (array/transpose x (-> (vec (range r)) (assoc (c/- r 2) (dec r)) (assoc (dec r) (c/- r 2))))))

;; ---------------------------------------------------------------------------
;; RMS normalization
;; ---------------------------------------------------------------------------

(defn- rms-norm-composite [x weight eps]
  (let [d (array/dtype x)
        cd (compute-dtype d)
        xc (as-dtype x cd)
        y (* xc (/ 1.0 (array/sqrt (+ (array/mean (array/square xc) {:axis -1 :keep-dims? true}) eps))))
        y (as-dtype y d)]
    (if weight (* y weight) y)))

(defn rms-norm
  "gen.array/rms-norm's body."
  [x weight eps]
  (let [sx (array/shape x)]
    (when (empty? sx)
      (throw (invalid :rms-norm "x must have an axis to normalize over (it is 0-d)" {:shapes [sx]})))
    (when (and weight (not= [(peek sx)] (array/shape weight)))
      (throw (invalid :rms-norm (str "the weight's shape " (array/shape weight) " is not [" (peek sx) "], x's last axis")
                      {:shapes [sx (array/shape weight)]})))
    (check-host-number :rms-norm :eps eps)
    (when (c/< (double eps) 0.0)
      (throw (invalid :rms-norm (str "eps must not be negative, not " eps) {:eps eps})))
    (let [b (backend-of :rms-norm [x weight])
          [x weight] (as-floats b [x weight])]
      (or (fused b :rms-norm [x weight (double eps)])
          (rms-norm-composite x weight (double eps))))))

;; ---------------------------------------------------------------------------
;; Rotary position embeddings
;; ---------------------------------------------------------------------------

(defn- rope-angles
  "cos and sin of the angles of positions (offset + t)·scale, t along x's
   axis -2, times the inverse frequencies: arrays of shape [.. T dims/2]
   that broadcast against x's halves, in x's dtype."
  [x dims {:keys [base scale offset freqs]}]
  (let [s (array/shape x)
        r (count s)
        t (nth s (c/- r 2))
        half (quot (long dims) 2)
        d (array/dtype x)
        cd (compute-dtype d)
        b (array/backend-of :rope [x])
        pos (array/with-backend b (array/arange 0 t 1 {:dtype cd}))
        pos (if (and (array/array? offset) (c/> (array/size offset) 1))
              ;; one offset per row of axis 0: [B 1 .. 1] + [T] is [B 1 .. T]
              (+ (array/reshape (as-dtype offset cd) (into [(array/size offset)] (repeat (c/- r 2) 1))) pos)
              (+ pos (if (array/array? offset) (array/reshape (as-dtype offset cd) []) offset)))
        pos (* pos scale)
        inv (if freqs
              (/ 1.0 (as-dtype freqs cd))
              (array/exp (* (array/with-backend b (array/arange 0 (c/- half) -1 {:dtype cd}))
                            (c// (Math/log (double base)) half))))
        theta (* (array/expand-dims pos -1) inv)]
    [(as-dtype (array/cos theta) d) (as-dtype (array/sin theta) d)]))

(defn- rope-composite [x dims opts]
  (let [s (array/shape x)
        d (peek s)
        half (quot (long dims) 2)
        [cs sn] (rope-angles x dims opts)
        [x1 x2] (if (:traditional? opts)
                  [(array/slice x -1 0 dims 2) (array/slice x -1 1 dims 2)]
                  [(array/slice x -1 0 half) (array/slice x -1 half dims)])
        o1 (- (* x1 cs) (* x2 sn))
        o2 (+ (* x1 sn) (* x2 cs))
        rotated (if (:traditional? opts)
                  ;; the pairs interleaved again: [.. half 2] read as [.. dims]
                  [(array/reshape (array/stack [o1 o2] -1) (conj (pop s) dims))]
                  [o1 o2])]
    (if (c/< (long dims) (long d))
      (array/concatenate (conj rotated (array/slice x -1 dims d)) -1)
      (array/concatenate rotated -1))))

(defn rope
  "gen.array/rope's body."
  [x dims {:keys [traditional? base scale offset freqs] :or {scale 1.0 offset 0}}]
  (let [s (array/shape x)
        r (count s)]
    (when (c/< r 2)
      (throw (invalid :rope (str "x needs a position axis and a feature axis, not shape " s) {:shapes [s]})))
    (when-not (and (integer? dims) (pos? (long dims)) (even? (long dims)) (c/<= (long dims) (long (peek s))))
      (throw (invalid :rope (str "dims must be a positive even integer at most x's last axis (" (peek s) "), not " (pr-str dims))
                      {:dims dims :shapes [s]})))
    (when (and base freqs)
      (throw (invalid :rope "give :base or :freqs, not both" {})))
    (when base (check-host-number :rope :base base))
    (when (and base (not (c/> (double base) 0.0)))
      (throw (invalid :rope (str ":base must be positive, not " base) {:base base})))
    (check-host-number :rope :scale scale)
    (cond (integer? offset) nil
          (and (array/array? offset) (not (array/host? offset)))
          (let [so (array/shape offset)]
            (when-not (= :int (dt/kind (array/dtype offset)))
              (throw (invalid :rope (str "an :offset array must be of ints, not " (array/dtype offset)) {:dtype (array/dtype offset)})))
            (when-not (or (c/<= (array/size offset) 1)
                          (and (= 1 (count so)) (c/>= r 3) (= (first so) (first s))))
              (throw (invalid :rope (str "an :offset array is one offset, or one per row of x's axis 0 (" (first s)
                                         ") where x has 3 axes or more; its shape is " so)
                              {:shapes [s so]}))))
          :else (throw (invalid :rope (str ":offset must be a host integer or an int array, not " (pr-str offset)) {:offset offset})))
    (when freqs
      (when-not (= [(quot (long dims) 2)] (array/shape freqs))
        (throw (invalid :rope (str ":freqs must have shape [" (quot (long dims) 2) "], not " (array/shape freqs))
                        {:shapes [s (array/shape freqs)]}))))
    (let [b (backend-of :rope [x offset freqs])
          [x freqs] (as-floats b [x freqs])
          opts {:traditional? (boolean traditional?)
                :base (when-not freqs (double (or base 10000.0)))
                :scale (double scale)
                :offset offset
                :freqs freqs}]
      (or (fused b :rope [x dims opts])
          (rope-composite x dims opts)))))

;; ---------------------------------------------------------------------------
;; Scaled dot-product attention
;; ---------------------------------------------------------------------------

(defn- lowest
  "The lowest finite value of the float dtype d: a bool mask's score where
   it hides a key, as MLX's, not -Inf: a query that sees no key would have
   a softmax of NaN, and so NaN gradients through the where that zeroes
   its row."
  [d]
  (case d
    :float64 (c/- Double/MAX_VALUE)
    :float32 (double (c/- Float/MAX_VALUE))
    :float16 -65504.0
    :bfloat16 -3.3895313892515355E38))

(defn- causal-mask
  "Query i (of tq) sees key j (of tk) when j <= i + tk - tq: the queries
   are the last tq positions."
  [b tq tk]
  (array/with-backend b
    (let [off (c/- (long tk) (long tq))]
      (>= (array/expand-dims (array/arange off (c/+ off (long tq)) 1) -1) (array/arange 0 tk 1)))))

(defn- sees-no-key?
  "Can a query see no key? Under a bool mask, or a causal one with more
   queries than keys (the first Tq - Tk see none)."
  [mask tq tk]
  (or (and (array/array? mask) (= :bool (array/dtype mask)))
      (and (identical? :causal mask) (c/> (long tq) (long tk)))))

(defn- sdpa-composite [q k v {:keys [scale mask]}]
  (let [[b hq tq d] (array/shape q)
        [_ hk tk _] (array/shape k)
        dv (peek (array/shape v))
        rep (quot (long hq) (long hk))
        dt (array/dtype q)
        bk (array/backend-of :sdpa [q k v])
        q (* q scale)
        [q k v] (if (c/> rep 1)
                  [(array/reshape q [b hk rep tq d]) (array/expand-dims k 2) (array/expand-dims v 2)]
                  [q k v])
        scores (array/matmul q (swap-last-two k))
        m (cond (nil? mask) nil
                (identical? :causal mask) (causal-mask bk tq tk)
                :else mask)
        ;; the mask's head axis as the grouped scores have it
        m (if (and m (c/> rep 1) (c/>= (array/ndim m) 3))
            (let [sm (array/shape m)
                  r (count sm)]
              (if (= 1 (nth sm (c/- r 3)))
                (array/expand-dims m (c/- r 2))
                (array/reshape m (-> (subvec sm 0 (c/- r 3)) (conj hk rep) (into (subvec sm (c/- r 2)))))))
            m)
        scores (cond (nil? m) scores
                     (= :bool (array/dtype m)) (array/where m scores (lowest dt))
                     :else (+ scores m))
        cd (compute-dtype dt)
        w (as-dtype (array/softmax (as-dtype scores cd) {:axis -1}) dt)
        out (array/matmul w v)
        out (if (c/> rep 1) (array/reshape out [b hq tq dv]) out)]
    (if (sees-no-key? mask tq tk)
      ;; a query that sees no key gets zeros, its softmax being of nothing
      (array/where (array/any (if (identical? :causal mask) (causal-mask bk tq tk) mask) {:axis -1 :keep-dims? true}) out 0.0)
      out)))

(defn sdpa
  "gen.array/sdpa's body."
  [q k v {:keys [scale mask] :as opts}]
  (let [[sq sk sv :as shapes] (mapv array/shape [q k v])]
    (doseq [[what s] [[:q sq] [:k sk] [:v sv]]]
      (when-not (= 4 (count s))
        (throw (invalid :sdpa (str (name what) " must have 4 axes [B H T D], not shape " s) {:shapes shapes}))))
    (let [[bq hq tq dq] sq [bk hk tk dk] sk [bv hv tv _] sv]
      (when-not (= bq bk bv)
        (throw (invalid :sdpa (str "q, k and v disagree on the batch axis: " shapes) {:shapes shapes})))
      (when-not (= dq dk)
        (throw (invalid :sdpa (str "q and k disagree on the head size: " shapes) {:shapes shapes})))
      (when-not (and (= hk hv) (= tk tv))
        (throw (invalid :sdpa (str "k and v disagree on heads or positions: " shapes) {:shapes shapes})))
      (when-not (and (pos? (long hk)) (zero? (rem (long hq) (long hk))))
        (throw (invalid :sdpa (str "q's heads (" hq ") must be a multiple of k's (" hk ")") {:shapes shapes})))
      (when scale (check-host-number :sdpa :scale scale))
      (cond (or (nil? mask) (identical? :causal mask)) nil
            (array/array? mask)
            (let [sm (array/shape mask)
                  full [bq hq tq tk]]
              (when (c/> (count sm) 4)
                (throw (invalid :sdpa (str "the mask has at most 4 axes, not shape " sm) {:shapes (conj shapes sm)})))
              (when-not (try (= full (sh/broadcast sm full)) (catch clojure.lang.ExceptionInfo _ false))
                (throw (invalid :sdpa (str "the mask's shape " sm " does not broadcast to [B Hq Tq Tk] " full)
                                {:shapes (conj shapes sm)}))))
            :else (throw (invalid :sdpa (str ":mask must be nil, :causal or an array, not " (pr-str mask)) {:mask mask})))
      (let [b (backend-of :sdpa [q k v (when (array/array? mask) mask)])
            [q k v] (as-floats b [q k v])
            dt (array/dtype q)
            mask (if (and (array/array? mask) (not= :bool (array/dtype mask))) (as-dtype mask dt) mask)
            opts (assoc opts :scale (double (or scale (c// 1.0 (Math/sqrt (double dq))))) :mask mask)]
        (or (fused b :sdpa [q k v opts])
            (sdpa-composite q k v opts))))))

;; ---------------------------------------------------------------------------
;; Log-probabilities of chosen entries, over vocabulary chunks
;; ---------------------------------------------------------------------------

(def ^:private selective-options #{:chunk :vocab :args})

(defn- chunk-part
  "The logsumexp of lc, the logits of the entries [start, stop) over its
   last axis, and the logit of each index that falls among them (0 where
   it does not), stacked on a new axis 0, in the compute dtype."
  [lc indices start stop]
  (let [lc (as-dtype lc (compute-dtype (array/dtype lc)))
        n (c/- (long stop) (long start))
        local (- indices start)
        in? (array/logical-and (>= local 0) (array/< local n))]
    (array/stack [(array/logsumexp lc {:axis -1})
                  (array/where in? (array/pick lc (array/clip local 0 (dec n))) 0.0)]
                 0)))

(defn- chunk-bounds
  "The chunks [start stop) of v entries, step entries each (the last
   fewer)."
  [v step]
  (vec (for [s (range 0 v step)] [s (c/min (long v) (c/+ (long s) (long step)))])))

(defn- check-chunk
  "lc, the logits of the entries [start, stop) the function form gave:
   a float array of the indices' shape and stop - start entries."
  [lc indices start stop]
  (let [want (conj (array/shape indices) (c/- (long stop) (long start)))]
    (when-not (and (array/array? lc) (= :float (dt/kind (array/dtype lc))) (= want (array/shape lc)))
      (throw (invalid :selective-log-softmax
                      (str "the logits of entries [" start ", " stop ") must be a float array of shape " want ", not "
                           (if (array/array? lc) (str (array/dtype lc) " " (array/shape lc)) (pr-str lc)))
                      {:shapes [(array/shape indices) (when (array/array? lc) (array/shape lc))] :start start :stop stop})))
    lc))

(defn selective-log-softmax
  "gen.array/selective-log-softmax's body."
  [logits indices opts]
  (when-not (or (nil? opts) (map? opts))
    (throw (invalid :selective-log-softmax (str "options must be a map, not " (pr-str opts)))))
  (when-let [unknown (seq (remove selective-options (keys opts)))]
    (throw (ex-info (str "selective-log-softmax: unknown options " (vec unknown) "; it takes " (vec (sort selective-options)))
                    {:reason :unknown-options :op :selective-log-softmax :unknown (vec unknown) :accepted selective-options})))
  (let [{:keys [chunk vocab args]} opts
        f? (fn? logits)]
    (when-not (or f? (and (array/backend-array? logits) (seq (array/shape logits)) (= :float (dt/kind (array/dtype logits)))))
      (throw (invalid :selective-log-softmax
                      (str "logits must be a float array [.. V] or a function of a range of entries, not "
                           (if (array/array? logits) (str (array/dtype logits) " " (array/shape logits)) (pr-str logits))))))
    (when-not (and (array/array? indices) (= :int (dt/kind (array/dtype indices))))
      (throw (invalid :selective-log-softmax (str "indices must be an int array, not " (pr-str indices)))))
    (when (and f? (not (and (integer? vocab) (pos? (long vocab)))))
      (throw (invalid :selective-log-softmax (str "a function of the logits needs :vocab, a positive integer, not " (pr-str vocab))
                      {:vocab vocab})))
    (when-not f?
      (let [lead (pop (array/shape logits))]
        (when-not (= lead (array/shape indices))
          (throw (invalid :selective-log-softmax (str "indices must have the logits' leading shape " lead ", not "
                                                      (array/shape indices))
                          {:shapes [(array/shape logits) (array/shape indices)]})))
        (when (and vocab (not= vocab (peek (array/shape logits))))
          (throw (invalid :selective-log-softmax (str ":vocab " vocab " is not the logits' last axis, " (peek (array/shape logits)))
                          {:vocab vocab :shapes [(array/shape logits)]})))
        (when (some? args)
          (throw (invalid :selective-log-softmax ":args go with a function of the logits, not an array" {})))))
    (when-not (or (nil? chunk) (and (integer? chunk) (pos? (long chunk))))
      (throw (invalid :selective-log-softmax (str ":chunk must be a positive integer, not " (pr-str chunk)) {:chunk chunk})))
    (when-not (or (nil? args) (vector? args))
      (throw (invalid :selective-log-softmax (str ":args must be a vector of the function's arguments, not " (pr-str args))
                      {:args args})))
    (let [v (long (if f? vocab (peek (array/shape logits))))
          ;; out of range: loud where the indices can be read, clamped on a
          ;; lazy backend (as take's), so that no index matches no chunk
          indices (if (array/concrete? indices)
                    (do (when-not (array/all-true? (array/logical-and (>= indices 0) (array/< indices v)))
                          (throw (invalid :selective-log-softmax (str "an index is outside the entries [0, " v ")") {:vocab v})))
                        indices)
                    (array/clip indices 0 (dec v)))
          step (long (or chunk v))
          bounds (chunk-bounds v step)
          logsumexp-of (fn [lses] (if (c/== 1 (count lses)) (first lses) (array/logsumexp (array/stack lses -1) {:axis -1})))]
      (if-not f?
        ;; an array: its entries picked whole, its logsumexp by chunks
        (let [x (as-dtype logits (compute-dtype (array/dtype logits)))]
          (- (array/pick x indices)
             (logsumexp-of (mapv (fn [[s e]] (array/logsumexp (array/slice x -1 s e) {:axis -1})) bounds))))
        ;; a function: each chunk's logits made, reduced and dropped, and
        ;; with :args recomputed in the backward pass (checkpoint)
        (let [part (fn [s e lc] (chunk-part (check-chunk lc indices s e) indices s e))
              parts (mapv (fn [[s e]]
                            (if args
                              (apply (array/checkpoint (fn [& as] (part s e (apply logits s e as)))) args)
                              (part s e (logits s e))))
                          bounds)]
          (- (reduce + (map #(array/take % 1 0) parts))
             (logsumexp-of (mapv #(array/take % 0 0) parts))))))))

;; ---------------------------------------------------------------------------
;; Quantized weights
;; ---------------------------------------------------------------------------

(def ^:private affine-bits #{2 3 4 5 6 8})
(def ^:private affine-groups #{32 64 128})

(defn- quantization
  "{:mode :group-size :bits} of op's options, checked: MLX's affine
   quantization (group sizes 32, 64 and 128; 2, 3, 4, 5, 6 or 8 bits;
   defaults 64 and 4). Its other modes (mxfp4, nvfp4, mxfp8) keep scales
   as uint8, a dtype gen.array lacks: a loud error."
  [op {:keys [mode group-size bits] :or {mode :affine group-size 64 bits 4}}]
  (when-not (= :affine mode)
    (throw (ex-info (str (name op) ": the quantization mode " (pr-str mode) " is not supported"
                         (when (#{:mxfp4 :nvfp4 :mxfp8} mode)
                           " (its scales are uint8, a dtype gen.array does not have yet)")
                         "; :affine is")
                    {:reason :unsupported :op op :mode mode})))
  (when-not (affine-groups group-size)
    (throw (invalid op (str ":group-size must be 32, 64 or 128, not " (pr-str group-size)) {:group-size group-size})))
  (when-not (affine-bits bits)
    (throw (invalid op (str ":bits must be 2, 3, 4, 5, 6 or 8, not " (pr-str bits)) {:bits bits})))
  {:mode mode :group-size (long group-size) :bits (long bits)})

(defn- check-quantized
  "The quantized matrix wq, scales and biases (affine: both) agree with
   each other and with q's group size and bits: [the matrix's rows, its
   columns] once dequantized."
  [op wq scales biases {:keys [group-size bits]}]
  (let [sw (array/shape wq)
        ss (array/shape scales)
        shapes [sw ss (when biases (array/shape biases))]]
    (when-not (= :uint32 (array/dtype wq))
      (throw (invalid op (str "the quantized matrix must be uint32, not " (array/dtype wq)) {:dtype (array/dtype wq)})))
    (when (c/< (count sw) 2)
      (throw (invalid op (str "the quantized matrix needs 2 axes or more, not shape " sw) {:shapes shapes})))
    (doseq [[what x] [[:scales scales] [:biases biases]]]
      (when (nil? x)
        (throw (invalid op (str "affine quantization needs " (name what)) {})))
      (when-not (= :float (dt/kind (array/dtype x)))
        (throw (invalid op (str (name what) " must be floats, not " (array/dtype x)) {:dtype (array/dtype x)}))))
    (when-not (and (= ss (array/shape biases)) (= (pop sw) (pop ss))
                   (== (quot (c/* (long (peek sw)) 32) (long bits)) (c/* (long (peek ss)) (long group-size))))
      (throw (invalid op (str "the quantized matrix " sw ", scales " ss " and biases " (array/shape biases)
                              " do not agree for groups of " group-size " at " bits " bits")
                      {:shapes shapes})))
    [(peek (pop sw)) (quot (c/* (long (peek sw)) 32) (long bits))]))

(defn quantize
  "gen.array/quantize's body."
  [w opts]
  (let [q (quantization :quantize opts)
        s (array/shape w)]
    (when-not (= :float (dt/kind (array/dtype w)))
      (throw (invalid :quantize (str "w must be floats, not " (array/dtype w)) {:dtype (array/dtype w)})))
    (when (c/< (count s) 2)
      (throw (invalid :quantize (str "w needs 2 axes or more, not shape " s) {:shapes [s]})))
    (when-not (zero? (rem (long (peek s)) (:group-size q)))
      (throw (invalid :quantize (str "w's last axis (" (peek s) ") is not a multiple of the group size " (:group-size q))
                      {:shapes [s] :group-size (:group-size q)})))
    (let [b (backend-of :quantize [w])]
      (p/quantized b :quantize [w q]))))

(defn dequantize
  "gen.array/dequantize's body."
  [wq scales biases {:keys [dtype] :as opts}]
  (let [q (quantization :dequantize opts)]
    (check-quantized :dequantize wq scales biases q)
    (when (and dtype (not= :float (dt/kind dtype)))
      (throw (invalid :dequantize (str ":dtype must be a float dtype, not " (pr-str dtype)) {:dtype dtype})))
    (let [b (backend-of :dequantize [wq scales biases])
          [scales biases] (as-floats b [scales biases])]
      (p/quantized b :dequantize [wq scales biases (assoc q :dtype (or dtype (array/dtype scales)))]))))

(defn- matmul-dims
  "[inner outer] of the quantized matrix [rows cols] as x multiplies it:
   x·wᵀ (transpose?) or x·w."
  [[rows cols] transpose?]
  (if transpose? [cols rows] [rows cols]))

(defn quantized-matmul
  "gen.array/quantized-matmul's body."
  [x wq scales biases {:keys [transpose?] :or {transpose? true} :as opts}]
  (let [q (quantization :quantized-matmul opts)
        dims (check-quantized :quantized-matmul wq scales biases q)
        [inner _] (matmul-dims dims transpose?)
        sx (array/shape x)]
    (when-not (= 2 (array/ndim wq))
      (throw (invalid :quantized-matmul (str "the quantized matrix must have 2 axes (gather-qmm takes a stack), not shape "
                                             (array/shape wq))
                      {:shapes [sx (array/shape wq)]})))
    (when-not (and (seq sx) (= inner (peek sx)))
      (throw (invalid :quantized-matmul (str "x's last axis must be " inner " to multiply the quantized matrix "
                                             (if transpose? "transposed" "") ", not shape " sx)
                      {:shapes [sx (array/shape wq)]})))
    (let [b (backend-of :quantized-matmul [x wq scales biases])
          [x scales biases] (as-floats b [x scales biases])]
      (p/quantized b :quantized-matmul [x wq scales biases (assoc q :transpose? (boolean transpose?))]))))

(defn- default-indices
  "The flat indices of the matrices of an array of shape s (all but its
   last two axes), shaped as those axes: what gather-qmm gathers by
   default."
  [b s]
  (let [batch (subvec s 0 (c/- (count s) 2))]
    (array/with-backend b
      (array/reshape (array/arange 0 (sh/size batch) 1 {:dtype :int32}) batch))))

(defn gather-qmm
  "gen.array/gather-qmm's body."
  [x wq scales biases {:keys [lhs-indices rhs-indices transpose? sorted?] :or {transpose? true} :as opts}]
  (let [q (quantization :gather-qmm opts)
        dims (check-quantized :gather-qmm wq scales biases q)
        [inner _] (matmul-dims dims transpose?)
        sx (array/shape x)
        sw (array/shape wq)]
    (when-not (and (c/>= (count sx) 2) (= inner (peek sx)))
      (throw (invalid :gather-qmm (str "x must be matrices [.. M " inner "] to multiply the quantized matrices, not shape " sx)
                      {:shapes [sx sw]})))
    (doseq [[what i] [[:lhs-indices lhs-indices] [:rhs-indices rhs-indices]] :when i]
      (when-not (and (array/array? i) (= :int (dt/kind (array/dtype i))))
        (throw (invalid :gather-qmm (str (name what) " must be an int array, not " (pr-str i)) {what i}))))
    (let [b (backend-of :gather-qmm [x wq scales biases lhs-indices rhs-indices])
          lhs (or lhs-indices (default-indices b sx))
          rhs (or rhs-indices (default-indices b sw))
          shape (try (sh/broadcast (array/shape lhs) (array/shape rhs))
                     (catch clojure.lang.ExceptionInfo _
                       (throw (invalid :gather-qmm (str "the indices' shapes " (array/shape lhs) " and " (array/shape rhs)
                                                        " do not broadcast")
                                       {:shapes [(array/shape lhs) (array/shape rhs)]}))))
          lhs (if (= shape (array/shape lhs)) lhs (array/broadcast-to lhs shape))
          rhs (if (= shape (array/shape rhs)) rhs (array/broadcast-to rhs shape))
          [x scales biases] (as-floats b [x scales biases])]
      (p/quantized b :gather-qmm [x wq scales biases lhs rhs
                                  (assoc q :transpose? (boolean transpose?) :sorted? (boolean sorted?))]))))
