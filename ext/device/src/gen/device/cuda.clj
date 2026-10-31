(ns gen.device.cuda
  "A staged program (gen.device.stage) as one CUDA C kernel, one thread
   per chain: the CUDA C target of bean genclj-a1hu (research prototype,
   ext/device), beside StableHLO.

   (source program) is the kernel's C++ text. Thread i runs the program
   on the key (rng/split key n)[i], the program's key derived in the thread
   (Threefry of the counter [0 i] under the root key), so n threads are n
   independent runs of the per-chain program: kernel/run's chains, each
   from its own key. Every node is scalarized at generation time: an
   array is a vector of C expressions, one per element, so reshapes,
   broadcasts, slices, transposes, pads and concatenations cost nothing
   and the compiler sees straight-line scalar code. A scan is a for loop
   (its carries mutable locals, its ys written to the outputs in the
   loop); a counted scan's counter is a carry like any other. Each output
   is laid out [n …] in global memory, bools as bytes.

   (compile src) compiles it with NVRTC for the device (sm_120 by default,
   -Dgen.cuda.arch; no fused multiply-adds, as MLX's op-by-op kernels) and
   loads the module through the driver API by FFM; (buffers k n) makes the
   outputs' device and pinned host buffers, and (launch! k bufs n
   key-words) launches n threads, waits and copies every output to the
   host. The libraries: -Dgen.cuda.nvrtc=<libnvrtc.so>,
   -Dgen.cuda.driver=<libcuda.so.1> (default /usr/lib/x86_64-linux-gnu).

   Arrays past gen.cuda.max-elements (256 by default) and the program's
   data inputs (its inputs after the key: X and y of a regression, held on
   the device and shared by every particle) are arrays in memory, not
   scalarized: a data input its pointer, a computed array a slice of the
   particle's scratch (global memory, (scratch k n) bytes in all), filled
   by a loop. A particle may have several lanes (gen.cuda.lanes, 1 by
   default, a power of two up to 32: a warp per particle): the loops over
   an array are split across its lanes, reductions combined by warp
   shuffles, and the scalar code run by every lane alike; lane 0 writes the
   outputs (bean genclj-4zmv).

   Refused, loudly, by node: scatter-add (pads replace the slices'
   gradients), take, a scan over xs (counted scans only), a nested scan, a
   scan's ys read in the thread, float64, and batched matmul."
  {:api :internal}
  (:refer-clojure :exclude [compile])
  (:require [clojure.string :as str])
  (:import [java.lang.foreign AddressLayout Arena FunctionDescriptor Linker Linker$Option MemoryLayout MemorySegment
            SymbolLookup ValueLayout ValueLayout$OfInt ValueLayout$OfLong]
           [java.lang.invoke MethodHandle]
           [java.nio.charset StandardCharsets]
           [java.nio.file Path]))

(set! *warn-on-reflection* true)

;; ---------------------------------------------------------------------------
;; Shapes
;; ---------------------------------------------------------------------------

(defn- size ^long [shape] (reduce * 1 shape))

(defn- strides [shape]
  (vec (reverse (reductions * 1 (reverse (rest shape))))))

(defn- unravel [^long i shape]
  (mapv (fn [s d] (mod (quot i (long s)) (long d))) (strides shape) shape))

(defn- ravel ^long [idx shape]
  (reduce + 0 (map * idx (strides shape))))

(defn- broadcast-map
  "For each element of out-shape, the flat index of the element of an
   operand of shape in-shape that NumPy's broadcasting pairs with it."
  [out-shape in-shape]
  (let [k (- (count out-shape) (count in-shape))]
    (mapv (fn [i] (let [o (unravel i out-shape)]
                    (ravel (map-indexed (fn [j d] (if (= 1 d) 0 (nth o (+ k j)))) in-shape) in-shape)))
          (range (size out-shape)))))

;; ---------------------------------------------------------------------------
;; C text
;; ---------------------------------------------------------------------------

(def ^:private ctypes {:float32 "float" :uint32 "unsigned int" :int32 "int" :bool "bool"})

(defn- ctype [dtype]
  (or (ctypes dtype) (throw (ex-info (str "CUDA C: no " dtype " (float32, uint32, int32 and bool only)") {:dtype dtype}))))

(def ^:private buffer-types {:float32 "float" :uint32 "unsigned int" :int32 "int" :bool "unsigned char"})
(def ^:private buffer-bytes {:float32 4 :uint32 4 :int32 4 :bool 1})

(defn- lit
  "A host value as a C literal of dtype, exactly."
  [v dtype]
  (case dtype
    :float32 (let [f (unchecked-float (if (boolean? v) (if v 1.0 0.0) (double v)))]
               (if (or (Float/isNaN f) (Float/isInfinite f))
                 (format "__int_as_float(0x%08x)" (Float/floatToRawIntBits f))
                 (str (Float/toString f) "f")))
    :uint32 (str (bit-and (long (if (boolean? v) (if v 1 0) v)) 0xFFFFFFFF) "u")
    :int32 (str (long (if (boolean? v) (if v 1 0) v)))
    :bool (str (if (boolean? v) v (not (zero? (double v)))))
    (ctype dtype)))

(def ^:private prelude
  "Helpers every kernel uses: Threefry for the threads' keys, and the
   ops whose C operators do not have gen.array's semantics."
  "
__device__ __forceinline__ float g_max(float a, float b) { return (a != a || b != b) ? __int_as_float(0x7fc00000) : fmaxf(a, b); }
__device__ __forceinline__ float g_min(float a, float b) { return (a != a || b != b) ? __int_as_float(0x7fc00000) : fminf(a, b); }
__device__ __forceinline__ int g_max(int a, int b) { return a > b ? a : b; }
__device__ __forceinline__ int g_min(int a, int b) { return a < b ? a : b; }
__device__ __forceinline__ unsigned int g_max(unsigned int a, unsigned int b) { return a > b ? a : b; }
__device__ __forceinline__ unsigned int g_min(unsigned int a, unsigned int b) { return a < b ? a : b; }
__device__ __forceinline__ float g_sign(float a) { return a > 0.0f ? 1.0f : (a < 0.0f ? -1.0f : a); }
__device__ __forceinline__ int g_sign(int a) { return (a > 0) - (a < 0); }
__device__ __forceinline__ float g_mod(float a, float b) { float r = fmodf(a, b); return (r != 0.0f && ((r < 0.0f) != (b < 0.0f))) ? r + b : r; }
__device__ __forceinline__ int g_mod(int a, int b) { int r = a % b; return (r != 0 && ((r < 0) != (b < 0))) ? r + b : r; }
__device__ __forceinline__ unsigned int g_shl(unsigned int x, unsigned int s) { return s < 32u ? x << s : 0u; }
__device__ __forceinline__ unsigned int g_shr(unsigned int x, unsigned int s) { return s < 32u ? x >> s : 0u; }
__device__ __forceinline__ unsigned int g_rotl(unsigned int x, int r) { return (x << r) | (x >> (32 - r)); }
// Threefry-2x32-20 of the counter [c0 c1] under the key [k0 k1] (gen.random.threefry)
__device__ __forceinline__ void g_threefry(unsigned int k0, unsigned int k1, unsigned int c0, unsigned int c1,
                                            unsigned int* w0, unsigned int* w1) {
  const int r[8] = {13, 15, 26, 6, 17, 29, 16, 24};
  unsigned int ks[3] = {k0, k1, k0 ^ k1 ^ 0x1BD11BDAu};
  unsigned int x0 = c0 + k0, x1 = c1 + k1;
  #pragma unroll
  for (int i = 0; i < 5; i++) {
    #pragma unroll
    for (int j = 0; j < 4; j++) { x0 += x1; x1 = g_rotl(x1, r[(i % 2) * 4 + j]) ^ x0; }
    x0 += ks[(i + 1) % 3]; x1 += ks[(i + 2) % 3] + (unsigned int)(i + 1);
  }
  *w0 = x0; *w1 = x1;
}
")

;; ---------------------------------------------------------------------------
;; Generation
;; ---------------------------------------------------------------------------

(defn- emitter [] {:out (StringBuilder.) :n (volatile! 0) :indent (volatile! 1)})

(defn- line! [{:keys [^StringBuilder out indent]} s]
  (.append out (apply str (repeat @indent "  "))) (.append out ^String s) (.append out "\n"))

(defn- simple? [^String e] (boolean (re-matches #"[A-Za-z_][A-Za-z0-9_]*|[-0-9.eE+fux]+|true|false|__int_as_float\(0x[0-9a-f]+\)" e)))

(defn- bind!
  "A C name for the expression e of dtype (e itself when it is a name or a
   literal)."
  [em dtype ^String e]
  (if (simple? e)
    e
    (let [v (str "v" (vswap! (:n em) inc))]
      (line! em (str (ctype dtype) " " v " = " e ";"))
      v)))

(defn- value [shape dtype el] {:shape (vec shape) :dtype dtype :el (vec el)})

(def ^:private float-unary
  {:exp "expf" :log "logf" :expm1 "expm1f" :log1p "log1pf" :sqrt "sqrtf" :sin "sinf" :cos "cosf" :tan "tanf"
   :tanh "tanhf" :sinh "sinhf" :cosh "coshf" :arcsin "asinf" :arccos "acosf" :arctan "atanf" :erf "erff"
   :erfinv "erfinvf" :floor "floorf" :ceil "ceilf" :round "rintf" :abs "fabsf"})

(defn- unary-expr [op dtype a]
  (cond (= op :negative) (str "(-" a ")")
        (= op :logical-not) (str "(!" a ")")
        (= op :sign) (str "g_sign(" a ")")
        (= op :isnan) (if (= dtype :float32) (str "(" a " != " a ")") "false")
        (= op :isinf) (if (= dtype :float32) (str "(fabsf(" a ") == __int_as_float(0x7f800000))") "false")
        (and (= op :abs) (= dtype :int32)) (str "abs(" a ")")
        (and (float-unary op) (= dtype :float32)) (str (float-unary op) "(" a ")")
        (#{:floor :ceil :round} op) a
        :else (throw (ex-info (str "CUDA C: no " op " on " dtype " yet") {:op op :dtype dtype}))))

(def ^:private infix
  {:add "+" :subtract "-" :multiply "*" :divide "/" :logical-and "&&" :logical-or "||" :bit-xor "^" :bit-or "|"
   :less "<" :less-equal "<=" :greater ">" :greater-equal ">=" :equal "==" :not-equal "!="})

(defn- binary-expr [op dtype a b]
  (cond (infix op) (str "(" a " " (infix op) " " b ")")
        (= op :maximum) (str "g_max(" a ", " b ")")
        (= op :minimum) (str "g_min(" a ", " b ")")
        (= op :remainder) (str "g_mod(" a ", " b ")")
        (= op :shift-left) (str "g_shl(" a ", " b ")")
        (= op :shift-right-logical) (str "g_shr(" a ", " b ")")
        (and (= op :power) (= dtype :float32)) (str "powf(" a ", " b ")")
        (and (= op :arctan2) (= dtype :float32)) (str "atan2f(" a ", " b ")")
        :else (throw (ex-info (str "CUDA C: no " op " on " dtype " yet") {:op op :dtype dtype}))))

(def ^:private bit-ops #{:bit-xor :bit-or :shift-left :shift-right-logical})
(def ^:private comparisons #{:less :less-equal :greater :greater-equal :equal :not-equal})
(def ^:private binary-ops (into (set (keys infix)) [:maximum :minimum :remainder :shift-left :shift-right-logical :power :arctan2]))
(def ^:private unary-ops (into (set (keys float-unary)) [:negative :logical-not :sign :isnan :isinf]))

(def ^:private reducers
  {:sum ["(" " + " ")" 0] :prod ["(" " * " ")" 1] :all ["(" " && " ")" true] :any ["(" " || " ")" false]})

(defn- reduce-expr [op a b]
  (case op
    :max (str "g_max(" a ", " b ")")
    :min (str "g_min(" a ", " b ")")
    (let [[l m r] (reducers op)] (str l a m b r))))

(defn- max-elements ^long [] (Long/getLong "gen.cuda.max-elements" 256))

(defn- lanes-default ^long [] (Long/getLong "gen.cuda.lanes" 1))

(declare emit-graph! emit-scan!)

(defn- operand
  "An operand of a node: its value, or a constant's (a {:const x} of dtype)."
  [env a dtype]
  (if (map? a)
    (let [d (if (boolean? (:const a)) :bool dtype)] (value [] d [(lit (:const a) d)]))
    (let [v (or (env a) (throw (ex-info "CUDA C: an operand with no value" {:id a})))]
      (when (:ys v) (throw (ex-info "CUDA C: a scan's ys read in the thread (only a program's outputs may be ys)" {:id a})))
      v)))

(defn- elementwise [em shape dtype operands f]
  (let [maps (mapv #(broadcast-map shape (:shape %)) operands)]
    (value shape dtype (mapv (fn [i] (bind! em dtype (apply f (map (fn [o m] (nth (:el o) (nth m i))) operands maps))))
                             (range (size shape))))))

(defn- reduce-value [em {:keys [shape dtype el]} op axes out-shape out-dtype]
  (when-not (or (reducers op) (#{:max :min} op))
    (throw (ex-info (str "CUDA C: no reduction " op " yet") {:op op})))
  (let [axes (set axes)
        groups (group-by (fn [i] (let [idx (unravel i shape)] (vec (keep-indexed (fn [k x] (when-not (axes k) x)) idx))))
                         (range (size shape)))
        kept (vec (keep-indexed (fn [k d] (when-not (axes k) d)) shape))]
    (value out-shape out-dtype
           (mapv (fn [o] (let [is (get groups (unravel o kept))]
                           (if (empty? is)
                             (lit (case op :max Double/NEGATIVE_INFINITY :min Double/POSITIVE_INFINITY (nth (reducers op) 3)) out-dtype)
                             (reduce (fn [acc i] (bind! em out-dtype (reduce-expr op acc (nth el i)))) (nth el (first is)) (rest is)))))
                 (range (size kept))))))

(defn- emit-small-node!
  "env with node's value: {:shape :dtype :el} (C expressions), or a scan's
   or its ys result's own entry."
  [em env {:keys [id op args shape dtype attrs] :as node} ctx]
  (let [in (:in attrs)
        put #(assoc env id %)
        arg (fn [i d] (operand env (nth args i) d))]
    (cond
      (= op :param) (if (env id) env (throw (ex-info "CUDA C: an unbound param" {:node node})))
      (= op :constant) (let [vs (:values attrs) n (size shape)]
                         (put (value shape dtype (mapv #(lit (nth vs (if (= 1 (count vs)) 0 %)) dtype) (range n)))))
      (= op :iota) (put (value shape dtype (mapv #(lit % dtype) (range (size shape)))))
      (= op :broadcast) (let [o (arg 0 dtype) m (broadcast-map shape (:shape o))]
                          (put (value shape dtype (mapv #(nth (:el o) %) m))))
      (= op :convert) (let [o (arg 0 nil)]
                        (put (if (= (:dtype o) dtype)
                               (assoc o :shape shape)
                               (value shape dtype (mapv #(bind! em dtype (if (= dtype :bool) (str "(" % " != 0)") (str "(" (ctype dtype) ")(" % ")")))
                                                        (:el o))))))
      (= op :bitcast) (let [o (arg 0 nil)
                            f (case [(:dtype o) dtype] [:uint32 :float32] "__uint_as_float" [:float32 :uint32] "__float_as_uint"
                                    (throw (ex-info "CUDA C: a bitcast other than uint32 ↔ float32" {:node node})))]
                        (put (value shape dtype (mapv #(bind! em dtype (str f "(" % ")")) (:el o)))))
      (= op :stop-gradient) (put (assoc (arg 0 in) :shape shape))
      (unary-ops op) (put (elementwise em shape dtype [(arg 0 in)] #(unary-expr op in %)))
      (bit-ops op) (put (elementwise em shape :uint32 [(arg 0 :uint32) (arg 1 :uint32)] #(binary-expr op :uint32 %1 %2)))
      (comparisons op) (put (elementwise em shape :bool [(arg 0 in) (arg 1 in)] #(binary-expr op in %1 %2)))
      (binary-ops op) (put (elementwise em shape dtype [(arg 0 in) (arg 1 in)] #(binary-expr op in %1 %2)))
      (= op :where) (put (elementwise em shape dtype [(arg 0 :bool) (arg 1 in) (arg 2 in)] #(str "(" %1 " ? " %2 " : " %3 ")")))
      (= op :reduce) (put (reduce-value em (arg 0 nil) (:op attrs) (:axes attrs) shape dtype))
      (= op :reshape) (put (assoc (arg 0 nil) :shape shape))
      (= op :transpose) (let [o (arg 0 nil) perm (:perm attrs)]
                          (put (value shape dtype (mapv (fn [i] (let [oi (unravel i shape)
                                                                      ii (reduce (fn [v k] (assoc v (nth perm k) (nth oi k)))
                                                                                 (vec (repeat (count shape) 0)) (range (count shape)))]
                                                                  (nth (:el o) (ravel ii (:shape o)))))
                                                        (range (size shape))))))
      (= op :concatenate) (let [ax (:axis attrs)
                                os (mapv #(operand env % dtype) args)
                                offsets (vec (reductions + 0 (map #(nth (:shape %) ax) os)))]
                            (put (value shape dtype
                                        (mapv (fn [i] (let [oi (unravel i shape)
                                                            p (nth oi ax)
                                                            k (count (take-while #(<= % p) (rest offsets)))
                                                            o (nth os k)]
                                                        (nth (:el o) (ravel (assoc oi ax (- p (nth offsets k))) (:shape o)))))
                                              (range (size shape))))))
      (= op :slice) (let [o (arg 0 nil) {:keys [starts steps]} attrs]
                      (put (value shape dtype (mapv (fn [i] (let [oi (unravel i shape)]
                                                              (nth (:el o) (ravel (map (fn [s k x] (+ s (* k x))) starts steps oi) (:shape o)))))
                                                    (range (size shape))))))
      (= op :pad) (let [o (arg 0 nil) {:keys [axis low interior]} attrs
                        m (nth (:shape o) axis) z (lit 0 dtype)]
                    (put (value shape dtype (mapv (fn [i] (let [oi (unravel i shape)
                                                                p (- (nth oi axis) low)
                                                                j (quot p (inc interior))]
                                                            (if (and (>= p 0) (zero? (mod p (inc interior))) (< j m))
                                                              (nth (:el o) (ravel (assoc oi axis j) (:shape o)))
                                                              z)))
                                                  (range (size shape))))))
      (= op :matmul) (let [x (arg 0 dtype) y (arg 1 dtype)
                           [sx sy] [(:shape x) (:shape y)]
                           _ (when (or (> (count sx) 2) (> (count sy) 2)) (throw (ex-info "CUDA C: batched matmul" {:node node})))
                           mx (if (= 1 (count sx)) [1 (first sx)] sx)
                           my (if (= 1 (count sy)) [(first sy) 1] sy)
                           [r k] mx c (second my)]
                       (put (value shape dtype
                                   (for [a (range r) b (range c)]
                                     (reduce (fn [acc j] (bind! em dtype (str "(" acc " + " (nth (:el x) (+ (* a k) j)) " * " (nth (:el y) (+ (* j c) b)) ")")))
                                             (bind! em dtype (str "(" (nth (:el x) (* a k)) " * " (nth (:el y) b) ")"))
                                             (range 1 k))))))
      (= op :scan) (put (emit-scan! em env node ctx))
      (= op :result) (let [s (env (first args)) i (:index attrs)]
                       (put (if (< i (:n-carry s))
                              (nth (:carries s) i)
                              {:shape shape :dtype dtype :ys (- i (:n-carry s)) :scan (first args)})))
      :else (throw (ex-info (str "CUDA C: no lowering of " op " yet") {:node (dissoc node :attrs)})))))


;; ---------------------------------------------------------------------------
;; Arrays in memory: data inputs and arrays past max-elements, filled by
;; loops split across a particle's lanes (genclj-4zmv)

(defn- buf? [v] (contains? v :buf))

(defn- lanes ^long [ctx] (long (:lanes ctx 1)))

(defn- sync! [em ctx] (when (> (lanes ctx) 1) (line! em "__syncwarp();")))

(defn- fresh [em prefix] (str prefix (vswap! (:n em) inc)))

(defn- read-at
  "The C expression of buffer s's element i (a C expression) of dtype."
  [s i dtype]
  (if (= dtype :bool) (str "(" s "[" i "] != 0)") (str s "[" i "]")))

(defn- buf-value [shape dtype s] {:shape (vec shape) :dtype dtype :buf s})

(defn- local-array!
  "v, an :el value, as a local C array declared here: its name."
  [em {:keys [dtype el]}]
  (let [a (fresh em "l")]
    (line! em (str (ctype dtype) " " a "[" (count el) "] = {" (str/join ", " el) "};"))
    a))

(defn- accessor
  "A function of a flat index (a C expression) giving v's element there,
   declared here (an :el value of several elements becomes a local array)."
  [em v]
  (cond (buf? v) (fn [i] (read-at (:buf v) i (:dtype v)))
        (= 1 (count (:el v))) (let [e (first (:el v))] (fn [_] e))
        :else (let [a (local-array! em v)] (fn [i] (str a "[" i "]")))))

(defn- bidx
  "The flat index into an operand of in-shape that NumPy's broadcasting
   pairs with the element at flat index i (a C expression) of out-shape."
  [i out-shape in-shape]
  (if (= (vec out-shape) (vec in-shape))
    i
    (let [k (- (count out-shape) (count in-shape))
          os (strides out-shape)
          is (strides in-shape)
          terms (keep (fn [j] (let [d (nth in-shape j) oj (+ k j)]
                                (when (not= 1 d)
                                  (str "((" i " / " (nth os oj) ") % " (nth out-shape oj) ") * " (nth is j)))))
                      (range (count in-shape)))]
      (if (empty? terms) "0" (str "(" (str/join " + " terms) ")")))))

(defn- coords
  "C expressions of the coordinates of flat index i (a C expression) in shape."
  [i shape]
  (mapv (fn [st d] (str "((" i " / " st ") % " d ")")) (strides shape) shape))

(defn- flat
  "The flat index of coordinates cs (C expressions) in shape."
  [cs shape]
  (if (empty? cs) "0" (str "(" (str/join " + " (map (fn [c st] (str c " * " st)) cs (strides shape))) ")")))

(def ^:private stypes {:float32 "float" :uint32 "unsigned int" :int32 "int" :bool "unsigned char"})

(defn- scratch!
  "A slice of the particle's scratch for n elements of dtype: its C name."
  [em ctx n dtype]
  (let [off @(:scratch ctx)
        bytes (* 4 (quot (+ (* (long n) (long (buffer-bytes dtype))) 3) 4))
        s (fresh em "s")]
    (vswap! (:scratch ctx) + bytes)
    (line! em (str (stypes dtype) "* " s " = (" (stypes dtype) "*)(scratch + " off ");"))
    s))

(defn- as-small
  "A buffer of at most max-elements elements as an :el value (its elements
   read from memory), a larger one as it is."
  [{:keys [shape dtype buf] :as v}]
  (if (and buf (<= (size shape) (max-elements)))
    (value shape dtype (mapv #(read-at buf % dtype) (range (size shape))))
    v))

(defn- fill!
  "A new array of shape and dtype whose element i is (f i), filled by its
   lanes' loop."
  [em ctx shape dtype f]
  (let [n (size shape) s (scratch! em ctx n dtype) i (fresh em "i")]
    (line! em (str "for (int " i " = ln; " i " < " n "; " i " += " (lanes ctx) ") " s "[" i "] = " (f i) ";"))
    (sync! em ctx)
    (as-small (buf-value shape dtype s))))

(def ^:private reduce-init {:sum 0 :prod 1 :max Double/NEGATIVE_INFINITY :min Double/POSITIVE_INFINITY :all true :any false})

(defn- combine-lanes!
  "acc (a C variable of dtype) combined across the particle's lanes by op,
   every lane left with the total."
  [em ctx op acc dtype]
  (when (> (lanes ctx) 1)
    (let [o (fresh em "o")
          t (if (= dtype :bool) "int" (ctype dtype))]
      (line! em (str "for (int " o " = " (quot (lanes ctx) 2) "; " o " > 0; " o " >>= 1) " acc " = "
                     (reduce-expr op acc (str "(" (ctype dtype) ")__shfl_xor_sync(0xffffffffu, (" t ")" acc ", " o ")")) ";")))))

(defn- reduce-big!
  "op of operand v over axes, shaped out-shape: across the lanes when it
   makes one value, over the outputs' lanes otherwise."
  [em ctx v op axes out-shape dtype]
  (when-not (contains? reduce-init op) (throw (ex-info (str "CUDA C: no reduction " op " yet") {:op op})))
  (let [in-shape (:shape v)
        axes (set axes)
        kept (vec (keep-indexed (fn [k d] (when-not (axes k) d)) in-shape))
        red (vec (keep-indexed (fn [k d] (when (axes k) d)) in-shape))
        x (accessor em v)
        idx (fn [o r] (let [ko (coords o kept) ro (coords r red)]
                        (flat (loop [k 0 ko ko ro ro out []]
                                (if (= k (count in-shape)) out
                                    (if (axes k) (recur (inc k) ko (rest ro) (conj out (first ro)))
                                        (recur (inc k) (rest ko) ro (conj out (first ko))))))
                              in-shape)))
        init (lit (reduce-init op) dtype)]
    (if (= 1 (size kept))
      (let [acc (fresh em "a") r (fresh em "r")]
        (line! em (str (ctype dtype) " " acc " = " init ";"))
        (line! em (str "for (int " r " = ln; " r " < " (size red) "; " r " += " (lanes ctx) ") " acc " = "
                       (reduce-expr op acc (x (idx "0" r))) ";"))
        (combine-lanes! em ctx op acc dtype)
        (value out-shape dtype [acc]))
      (let [s (scratch! em ctx (size kept) dtype) o (fresh em "o") r (fresh em "r") acc (fresh em "a")]
        (line! em (str "for (int " o " = ln; " o " < " (size kept) "; " o " += " (lanes ctx) ") { " (ctype dtype) " " acc " = " init
                       "; for (int " r " = 0; " r " < " (size red) "; ++" r ") " acc " = " (reduce-expr op acc (x (idx o r)))
                       "; " s "[" o "] = " acc "; }"))
        (sync! em ctx)
        (as-small (buf-value out-shape dtype s))))))

(defn- matmul-big!
  "x · y for vectors and matrices, an output element per lane-loop step."
  [em ctx x y out-shape dtype]
  (let [[sx sy] [(:shape x) (:shape y)]
        _ (when (or (> (count sx) 2) (> (count sy) 2)) (throw (ex-info "CUDA C: batched matmul" {:shapes [sx sy]})))
        [_ k] (if (= 1 (count sx)) [1 (first sx)] sx)
        c (if (= 1 (count sy)) 1 (second sy))
        xa (accessor em x) ya (accessor em y)
        o (fresh em "o") j (fresh em "j") acc (fresh em "a")
        s (scratch! em ctx (size out-shape) dtype)]
    (line! em (str "for (int " o " = ln; " o " < " (size out-shape) "; " o " += " (lanes ctx) ") { float " acc " = 0.0f; "
                   "for (int " j " = 0; " j " < " k "; ++" j ") " acc " += " (xa (str "(" o " / " c ") * " k " + " j))
                   " * " (ya (str j " * " c " + " o " % " c)) "; " s "[" o "] = " acc "; }"))
    (sync! em ctx)
    (as-small (buf-value out-shape dtype s))))

(defn- emit-big-node!
  "node over arrays in memory: data inputs, or arrays past max-elements."
  [em env {:keys [id op args shape dtype attrs] :as node} ctx]
  (let [in (:in attrs)
        put #(assoc env id %)
        arg (fn [i d] (operand env (nth args i) d))
        ew (fn [operands f] (let [as (mapv #(accessor em %) operands)]
                              (put (fill! em ctx shape dtype
                                          (fn [i] (apply f (map (fn [o a] (a (bidx i shape (:shape o)))) operands as)))))))]
    (cond
      (= op :constant) (let [vs (:values attrs)]
                         (if (apply = vs)
                           (put (fill! em ctx shape dtype (constantly (lit (first vs) dtype))))
                           (let [g (fresh em "g")]
                             (swap! (:globals ctx) conj (str "__device__ const " (stypes dtype) " " g "[" (count vs) "] = {"
                                                             (str/join ", " (map #(lit % dtype) vs)) "};"))
                             (put (as-small (buf-value shape dtype g))))))
      (= op :iota) (put (fill! em ctx shape dtype (fn [i] (str "(" (ctype dtype) ")" i))))
      (= op :broadcast) (ew [(arg 0 dtype)] identity)
      (= op :convert) (let [o (arg 0 nil)]
                        (ew [o] (fn [e] (if (= dtype :bool) (str "(" e " != 0)") (str "(" (ctype dtype) ")(" e ")")))))
      (= op :bitcast) (let [o (arg 0 nil)
                            f (case [(:dtype o) dtype] [:uint32 :float32] "__uint_as_float" [:float32 :uint32] "__float_as_uint"
                                    (throw (ex-info "CUDA C: a bitcast other than uint32 ↔ float32" {:node node})))]
                        (ew [o] (fn [e] (str f "(" e ")"))))
      (= op :stop-gradient) (put (assoc (arg 0 in) :shape shape))
      (unary-ops op) (ew [(arg 0 in)] #(unary-expr op in %))
      (bit-ops op) (ew [(arg 0 :uint32) (arg 1 :uint32)] #(binary-expr op :uint32 %1 %2))
      (comparisons op) (ew [(arg 0 in) (arg 1 in)] #(binary-expr op in %1 %2))
      (binary-ops op) (ew [(arg 0 in) (arg 1 in)] #(binary-expr op in %1 %2))
      (= op :where) (ew [(arg 0 :bool) (arg 1 in) (arg 2 in)] #(str "(" %1 " ? " %2 " : " %3 ")"))
      (= op :reduce) (put (reduce-big! em ctx (arg 0 nil) (:op attrs) (:axes attrs) shape dtype))
      (= op :reshape) (let [o (arg 0 nil)] (put (if (buf? o) (as-small (assoc o :shape shape)) (assoc o :shape shape))))
      (= op :transpose) (let [o (arg 0 nil) a (accessor em o) perm (:perm attrs)]
                          (put (fill! em ctx shape dtype
                                      (fn [i] (let [oc (coords i shape)
                                                    ic (reduce (fn [v k] (assoc v (nth perm k) (nth oc k))) (vec (repeat (count shape) "0")) (range (count shape)))]
                                                (a (flat ic (:shape o))))))))
      (= op :slice) (let [o (arg 0 nil) a (accessor em o) {:keys [starts steps]} attrs]
                      (put (fill! em ctx shape dtype
                                  (fn [i] (a (flat (map (fn [st k c] (str "(" st " + " k " * " c ")")) starts steps (coords i shape)) (:shape o)))))))
      (= op :concatenate) (let [ax (:axis attrs)
                                os (mapv #(operand env % dtype) args)
                                as (mapv #(accessor em %) os)
                                offsets (vec (reductions + 0 (map #(nth (:shape %) ax) os)))]
                            (put (fill! em ctx shape dtype
                                        (fn [i] (let [cs (coords i shape) p (nth cs ax)]
                                                  (reduce (fn [e k] (let [o (nth os k)
                                                                          v ((nth as k) (flat (assoc cs ax (str "(" p " - " (nth offsets k) ")")) (:shape o)))]
                                                                      (if e (str "(" p " < " (nth offsets (inc k)) " ? " v " : " e ")") v)))
                                                          nil (reverse (range (count os)))))))))
      (= op :pad) (let [o (arg 0 nil) a (accessor em o) {:keys [axis low interior]} attrs
                        m (nth (:shape o) axis) z (lit 0 dtype)]
                    (put (fill! em ctx shape dtype
                                (fn [i] (let [cs (coords i shape)
                                              p (str "(" (nth cs axis) " - " low ")")
                                              j (str "(" p " / " (inc interior) ")")]
                                          (str "((" p " >= 0 && " p " % " (inc interior) " == 0 && " j " < " m ") ? "
                                               (a (flat (assoc cs axis j) (:shape o))) " : " z ")"))))))
      (= op :matmul) (put (matmul-big! em ctx (arg 0 dtype) (arg 1 dtype) shape dtype))
      :else (throw (ex-info (str "CUDA C: no lowering of " op " over arrays in memory yet") {:node (dissoc node :attrs)})))))

(defn- emit-node!
  "env with node's value: scalarized ({:el}), an array in memory ({:buf}),
   or a scan's or its ys result's own entry."
  [em env {:keys [op args shape] :as node} ctx]
  (if (and (not (#{:param :scan :result} op))
           (or (> (size (or shape [])) (max-elements))
               (some #(and (not (map? %)) (buf? (get env % {}))) args)))
    (emit-big-node! em env node ctx)
    (emit-small-node! em env node ctx)))

(defn- emit-scan!
  "A counted scan as a for loop: {:carries [values after the loop]
   :n-carry}. ys the program returns are written to their outputs in the
   loop (ctx's :ys-out maps [scan id, ys index] to an output)."
  [em env {:keys [id args attrs]} {:keys [ys-out nested?] :as ctx}]
  (when nested? (throw (ex-info "CUDA C: a scan inside a scan, not yet" {})))
  (let [{:keys [body length n-carry n-x]} attrs
        {:keys [graph carries captures carry-out y-out]} body]
    (when (pos? n-x) (throw (ex-info "CUDA C: a scan over xs (use a counted scan, xs nil)" {:n-x n-x})))
    (let [inits (mapv #(env %) (subvec args 0 n-carry))
          _ (when (some buf? inits) (throw (ex-info "CUDA C: a scan carrying an array in memory, not yet" {})))
          caps (mapv #(env %) (subvec args (+ n-carry n-x)))
          vars (mapv (fn [{:keys [shape dtype el]}]
                       (value shape dtype (mapv (fn [e] (let [v (str "c" (vswap! (:n em) inc))]
                                                          (line! em (str (ctype dtype) " " v " = " e ";"))
                                                          v))
                                                el)))
                     inits)
          it (str "it" (vswap! (:n em) inc))
          _ (line! em (str "for (int " it " = 0; " it " < " length "; ++" it ") {"))
          _ (vswap! (:indent em) inc)
          benv (merge (zipmap carries vars) (zipmap captures caps))
          benv (emit-graph! em graph benv (assoc ctx :nested? true))]
      (doseq [[j y] (map-indexed vector y-out)
              :let [k (get ys-out [id j])]
              :when k
              :let [{:keys [shape el] :as v} (benv y) n (size shape)]]
        (when (buf? v) (throw (ex-info "CUDA C: a scan's ys as an array in memory, not yet" {})))
        (doseq [[e x] (map-indexed vector el)]
          (line! em (str "if (ln == 0 && act) out" k "[(long long)tid * " (* length n) " + (long long)" it " * " n " + " e "] = " x ";"))))
      (let [outs (mapv #(:el (benv %)) carry-out)
            tmps (mapv (fn [v o] (mapv (fn [e] (bind! em (:dtype v) (str "(" e ")"))) o)) vars outs)]
        (doseq [[v t] (map vector vars tmps) [c x] (map vector (:el v) t)]
          (when-not (= c x) (line! em (str c " = " x ";")))))
      (vswap! (:indent em) dec)
      (line! em "}")
      {:carries vars :n-carry n-carry})))

(defn- emit-graph! [em nodes env ctx]
  (reduce (fn [env node] (emit-node! em env node ctx)) env nodes))

(def ^:private log-mean-exp-kernel
  "A second kernel of every module: log mean exp of n float32 weights, the
   log-ML estimate of importance sampling, on the device. One block of 1024
   threads: a strided max and a tree; then a strided sum of exp(w - max)
   in double and a tree; one double out. The order is fixed, so the result
   is the same on every run for the same weights. NaN propagates; all
   weights -Inf give -Inf."
  "
extern \"C\" __global__ void log_mean_exp(const float* w, int n, double* out) {
  __shared__ float smax[1024];
  __shared__ double ssum[1024];
  int t = threadIdx.x, T = blockDim.x;
  float m = __int_as_float(0xff800000);
  for (int i = t; i < n; i += T) m = g_max(m, w[i]);
  smax[t] = m;
  __syncthreads();
  for (int s = T / 2; s > 0; s >>= 1) { if (t < s) smax[t] = g_max(smax[t], smax[t + s]); __syncthreads(); }
  float M = smax[0];
  bool finite = (M == M) && (M != __int_as_float(0xff800000)) && (M != __int_as_float(0x7f800000));
  double acc = 0.0;
  if (finite) for (int i = t; i < n; i += T) acc += (double) expf(w[i] - M);
  ssum[t] = acc;
  __syncthreads();
  for (int s = T / 2; s > 0; s >>= 1) { if (t < s) ssum[t] += ssum[t + s]; __syncthreads(); }
  if (t == 0) out[0] = finite ? (double) M + log(ssum[0]) - log((double) n) : (double) M;
}
")

(defn source
  "The CUDA C++ text of the kernel `chains` for program (gen.device.stage's,
   of a key and data inputs): (chains r0 r1 n scratch sbytes in1 … out0 …),
   particle p running program on (rng/split [r0 r1] n)[p] with lanes
   threads (opts :lanes, gen.cuda.lanes). Returns {:source :outputs
   [{:shape :dtype} per output, without the particle axis] :inputs (the
   data inputs' [shape dtype]) :scratch-bytes (per particle) :lanes}."
  ([program] (source program nil))
  ([{:keys [graph inputs outputs]} {:keys [lanes] :or {lanes (lanes-default)}}]
   (when-not (= (first inputs) [[2] :uint32])
     (throw (ex-info "CUDA C: a program of a key, then its data" {:inputs inputs})))
   (when-not (and (pos? lanes) (<= lanes 32) (zero? (bit-and lanes (dec lanes))))
     (throw (ex-info (str "CUDA C: lanes a power of two up to 32, not " lanes) {:lanes lanes})))
   (let [em (emitter)
         by-id (into {} (map (juxt :id identity) graph))
         ;; ys a program returns: [scan id, ys index] → output
         ys-out (into {} (keep-indexed (fn [k o] (let [{:keys [op args attrs]} (by-id o)]
                                                   (when (= op :result)
                                                     (let [s (by-id (first args)) nc (get-in s [:attrs :n-carry])]
                                                       (when (>= (:index attrs) nc) [[(:id s) (- (:index attrs) nc)] k])))))
                                       outputs))
         params (filter #(= :param (:op %)) graph)
         data (vec (rest inputs))
         env0 (into {} (map (fn [{:keys [id attrs]}]
                              (let [i (:index attrs)]
                                (if (zero? i)
                                  [id (value [2] :uint32 ["key0" "key1"])]
                                  (let [[shape dtype] (nth inputs i)] [id (as-small (buf-value shape dtype (str "in" i)))]))))
                            params))
         ctx {:ys-out ys-out :lanes lanes :scratch (volatile! 0) :globals (atom [])}
         env (emit-graph! em graph env0 ctx)
         outs (mapv env outputs)
         body (.toString ^StringBuilder (:out em))
         tail (str/join (for [[k {:keys [shape el ys buf dtype]}] (map-indexed vector outs)
                              :when (nil? ys)]
                          (if buf
                            (str "  for (int w = ln; w < " (size shape) "; w += " lanes ") if (act) out" k
                                 "[(long long)tid * " (size shape) " + w] = " (read-at buf "w" dtype) ";\n")
                            (str/join (for [[e x] (map-indexed vector el)]
                                        (str "  if (ln == 0 && act) out" k "[(long long)tid * " (size shape) " + " e "] = " x ";\n"))))))
         sig-in (str/join (map-indexed (fn [k [_ dtype]] (str ", const " (stypes dtype) "* __restrict__ in" (inc k))) data))
         sig-out (str/join (map-indexed (fn [k {:keys [dtype]}] (str ", " (buffer-types dtype) "* out" k)) outs))
         sbytes @(:scratch ctx)]
     {:source (str prelude
                   (str/join "\n" @(:globals ctx)) "\n"
                   "\n__device__ __forceinline__ void chain(int tid, int ln, bool act, unsigned int key0, unsigned int key1,"
                   " char* scratch" sig-in sig-out ") {\n"
                   body tail "}\n\n"
                   "#ifdef __CUDACC_RTC__\n"
                   "extern \"C\" __global__ void chains(unsigned int r0, unsigned int r1, int n, char* scratch_all,"
                   " long long sbytes" sig-in sig-out ") {\n"
                   "  long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;\n"
                   "  int p = (int)(g / " lanes "), ln = (int)(g % " lanes ");\n"
                   "  bool act = p < n;\n"
                   ;; an inactive particle runs on the last one's key, its own scratch, writing nothing:
                   ;; a warp's lanes stay converged for __syncwarp
                   "  int q = act ? p : n - 1;\n"
                   "  unsigned int key0, key1;\n"
                   "  g_threefry(r0, r1, 0u, (unsigned int)q, &key0, &key1);\n"
                   "  chain(q, ln, act, key0, key1, scratch_all + (long long)p * sbytes"
                   (str/join (map #(str ", in" (inc %)) (range (count data))))
                   (str/join (map #(str ", out" %) (range (count outs)))) ");\n"
                   "}\n"
                   log-mean-exp-kernel
                   "#endif\n")
      :outputs (mapv (fn [{:keys [shape dtype ys]} o]
                       {:shape (if ys (:shape (by-id o)) shape) :dtype dtype})
                     outs outputs)
      :inputs data
      :scratch-bytes sbytes
      :lanes lanes})))

;; ---------------------------------------------------------------------------
;; NVRTC and the driver API, by FFM
;; ---------------------------------------------------------------------------

(def ^:private ^AddressLayout A ValueLayout/ADDRESS)
(def ^:private ^ValueLayout$OfInt I ValueLayout/JAVA_INT)
(def ^:private ^ValueLayout$OfLong J ValueLayout/JAVA_LONG)

(defn- lookup [^String path]
  (SymbolLookup/libraryLookup (Path/of path (make-array String 0)) (Arena/global)))

(defn- handle ^MethodHandle [^SymbolLookup lib ^String name ret & args]
  (.downcallHandle (Linker/nativeLinker) (.orElseThrow (.find lib name))
                   (if ret
                     (FunctionDescriptor/of ret (into-array MemoryLayout args))
                     (FunctionDescriptor/ofVoid (into-array MemoryLayout args)))
                   (make-array Linker$Option 0)))

(defn- call [^MethodHandle h & args] (.invokeWithArguments h ^objects (object-array args)))

(defn- cstr ^String [^MemorySegment p]
  (if (zero? (.address p)) "" (.getString (.reinterpret p Long/MAX_VALUE) 0)))

(def ^:private libs
  (delay
    (let [nvrtc (lookup (or (System/getProperty "gen.cuda.nvrtc")
                            (throw (ex-info "No NVRTC: -Dgen.cuda.nvrtc=<libnvrtc.so>" {}))))
          cu (lookup (System/getProperty "gen.cuda.driver" "/usr/lib/x86_64-linux-gnu/libcuda.so.1"))
          f {:nvrtcCreateProgram (handle nvrtc "nvrtcCreateProgram" I A A A I A A)
             :nvrtcCompileProgram (handle nvrtc "nvrtcCompileProgram" I A I A)
             :nvrtcGetProgramLogSize (handle nvrtc "nvrtcGetProgramLogSize" I A A)
             :nvrtcGetProgramLog (handle nvrtc "nvrtcGetProgramLog" I A A)
             :nvrtcGetCUBINSize (handle nvrtc "nvrtcGetCUBINSize" I A A)
             :nvrtcGetCUBIN (handle nvrtc "nvrtcGetCUBIN" I A A)
             :nvrtcDestroyProgram (handle nvrtc "nvrtcDestroyProgram" I A)
             :nvrtcGetErrorString (handle nvrtc "nvrtcGetErrorString" A I)
             :cuInit (handle cu "cuInit" I I)
             :cuDeviceGet (handle cu "cuDeviceGet" I A I)
             :cuDevicePrimaryCtxRetain (handle cu "cuDevicePrimaryCtxRetain" I A I)
             :cuCtxSetCurrent (handle cu "cuCtxSetCurrent" I A)
             :cuModuleLoadData (handle cu "cuModuleLoadData" I A A)
             :cuModuleUnload (handle cu "cuModuleUnload" I A)
             :cuModuleGetFunction (handle cu "cuModuleGetFunction" I A A A)
             :cuFuncGetAttribute (handle cu "cuFuncGetAttribute" I A I A)
             :cuMemAlloc (handle cu "cuMemAlloc_v2" I A J)
             :cuMemFree (handle cu "cuMemFree_v2" I J)
             :cuMemAllocHost (handle cu "cuMemAllocHost_v2" I A J)
             :cuMemFreeHost (handle cu "cuMemFreeHost" I A)
             :cuMemcpyDtoH (handle cu "cuMemcpyDtoH_v2" I A J J)
             :cuMemcpyHtoD (handle cu "cuMemcpyHtoD_v2" I J A J)
             :cuLaunchKernel (handle cu "cuLaunchKernel" I A I I I I I I I A A A)
             :cuCtxSynchronize (handle cu "cuCtxSynchronize" I)
             :cuGetErrorString (handle cu "cuGetErrorString" I I A)}]
      f)))

(defn- fn-of ^MethodHandle [k] (get @libs k))

(defn- cu! [k & args]
  (let [r (int (apply call (fn-of k) args))]
    (when-not (zero? r)
      (let [msg (with-open [arena (Arena/ofConfined)]
                  (let [p (.allocate arena A)]
                    (call (fn-of :cuGetErrorString) (int r) p)
                    (cstr (.get p A 0))))]
        (throw (ex-info (str "CUDA " (name k) ": " msg " (" r ")") {:fn k :code r}))))))

(defn- nvrtc! [k & args]
  (let [r (int (apply call (fn-of k) args))]
    (when-not (zero? r)
      (throw (ex-info (str "NVRTC " (name k) ": " (cstr (call (fn-of :nvrtcGetErrorString) (int r))) " (" r ")")
                      {:fn k :code r})))))

(def ^:private context
  (delay
    (with-open [arena (Arena/ofConfined)]
      (cu! :cuInit (int 0))
      (let [dev (.allocate arena I) ctx (.allocate arena A)]
        (cu! :cuDeviceGet dev (int 0))
        (cu! :cuDevicePrimaryCtxRetain ctx (.get dev I 0))
        (.get ctx A 0)))))

(defn- current! [] (cu! :cuCtxSetCurrent @context))

(defn- strings ^MemorySegment [^Arena arena xs]
  (let [a (.allocate arena A (max 1 (count xs)))]
    (doseq [[i ^String x] (map-indexed vector xs)]
      (.setAtIndex a A (long i) (.allocateFrom arena x)))
    a))

(defn compile
  "Compiles the kernel of (source program) with NVRTC for the device and
   loads it: {:function :module :outputs :log (NVRTC's, with ptxas's
   report) :registers :local-bytes :compile-ms}."
  [{:keys [source outputs inputs scratch-bytes lanes]}]
  (current!)
  (let [arch (System/getProperty "gen.cuda.arch" "sm_120")
        t0 (System/nanoTime)]
    (with-open [arena (Arena/ofConfined)]
      (let [prog (.allocate arena A)
            _ (nvrtc! :nvrtcCreateProgram prog (.allocateFrom arena ^String source) (.allocateFrom arena "chains.cu")
                      (int 0) MemorySegment/NULL MemorySegment/NULL)
            p (.get prog A 0)
            opts [(str "--gpu-architecture=" arch) "--fmad=false" "--std=c++17" "--ptxas-options=-v"]
            r (int (call (fn-of :nvrtcCompileProgram) p (int (count opts)) (strings arena opts)))
            log (let [n (.allocate arena J)]
                  (nvrtc! :nvrtcGetProgramLogSize p n)
                  (let [b (.allocate arena (max 1 (.get n J 0)))]
                    (nvrtc! :nvrtcGetProgramLog p b)
                    (.getString b 0)))]
        (when-not (zero? r)
          (nvrtc! :nvrtcDestroyProgram prog)
          (throw (ex-info (str "NVRTC could not compile the kernel:\n" (subs log 0 (min 4000 (count log)))) {:log log})))
        (let [n (.allocate arena J)
              _ (nvrtc! :nvrtcGetCUBINSize p n)
              cubin (.allocate arena (.get n J 0) 16)
              _ (nvrtc! :nvrtcGetCUBIN p cubin)
              _ (nvrtc! :nvrtcDestroyProgram prog)
              m (.allocate arena A)
              _ (cu! :cuModuleLoadData m cubin)
              module (.get m A 0)
              f (.allocate arena A)
              _ (cu! :cuModuleGetFunction f module (.allocateFrom arena "chains"))
              function (.get f A 0)
              rf (.allocate arena A)
              _ (cu! :cuModuleGetFunction rf module (.allocateFrom arena "log_mean_exp"))
              reduce-out (.allocate arena J)
              _ (cu! :cuMemAlloc reduce-out (long 8))
              attr (fn [k] (let [v (.allocate arena I)] (cu! :cuFuncGetAttribute v (int k) function) (.get v I 0)))]
          {:function function :module module :outputs outputs :inputs inputs
           :reduce-function (.get rf A 0) :reduce-out (.get reduce-out J 0)
           :scratch-bytes (or scratch-bytes 0) :lanes (or lanes 1) :log log
           :registers (attr 4) :local-bytes (attr 3)
           :compile-ms (/ (- (System/nanoTime) t0) 1e6)})))))

(defn buffers
  "Device and pinned host buffers for n threads' outputs of k (reused by
   run): close with free-buffers."
  [{:keys [outputs]} n]
  (current!)
  (with-open [arena (Arena/ofConfined)]
    (mapv (fn [{:keys [shape dtype]}]
            (let [bytes (* (long n) (size shape) (long (buffer-bytes dtype)))
                  d (.allocate arena J)
                  h (.allocate arena A)]
              (cu! :cuMemAlloc d (long (max 1 bytes)))
              (cu! :cuMemAllocHost h (long (max 1 bytes)))
              {:device (.get d J 0) :host (.reinterpret ^MemorySegment (.get h A 0) (max 1 bytes))
               :bytes bytes :shape (into [n] shape) :dtype dtype}))
          outputs)))

(defn free-buffers [bufs]
  (current!)
  (doseq [{:keys [device host]} bufs]
    (cu! :cuMemFree (long device))
    (cu! :cuMemFreeHost host)))

(def ^:private block-size 128)

(defn- grid ^long [n lanes] (quot (+ (* (long n) (long lanes)) (dec block-size)) block-size))

(defn scratch
  "The device scratch for n particles of k (every launched thread's
   particle, so the last block's spare ones too): {:device :bytes}, or nil
   when k needs none. Free with free-device."
  [{:keys [scratch-bytes lanes]} n]
  (when (pos? (long scratch-bytes))
    (current!)
    (with-open [arena (Arena/ofConfined)]
      (let [particles (quot (* (grid n lanes) block-size) (long lanes))
            bytes (* particles (long scratch-bytes))
            d (.allocate arena J)]
        (cu! :cuMemAlloc d (long bytes))
        {:device (.get d J 0) :bytes bytes}))))

(defn upload
  "A float array copied to the device: {:device :bytes}. Free with
   free-device."
  [^floats xs]
  (current!)
  (with-open [arena (Arena/ofConfined)]
    (let [bytes (* 4 (alength xs))
          d (.allocate arena J)
          h (.allocateFrom arena ValueLayout/JAVA_FLOAT xs)]
      (cu! :cuMemAlloc d (long (max 4 bytes)))
      (cu! :cuMemcpyHtoD (.get d J 0) h (long bytes))
      {:device (.get d J 0) :bytes bytes})))

(defn free-device [{:keys [device]}]
  (when device (current!) (cu! :cuMemFree (long device))))

(defn launch!
  "Launches n particles of k (k's lanes threads each) on the root key words
   [r0 r1], waits, and copies every output to its pinned host buffer (bufs,
   from buffers). opts: :inputs (the data inputs' device buffers, from
   upload), :scratch (from scratch), :copy? (true; false times the kernel
   alone). Returns bufs."
  ([k bufs n key-words] (launch! k bufs n key-words {}))
  ([k bufs n key-words opts]
   (let [{:keys [function scratch-bytes lanes]} k
         {:keys [inputs scratch copy?] :or {copy? true}} (if (map? opts) opts {:copy? opts})
         [r0 r1] key-words]
     (current!)
     (with-open [arena (Arena/ofConfined)]
       (let [ptr (fn [v] (doto (.allocate arena J) (.set J 0 (long v))))
             vals (-> [(doto (.allocate arena I) (.set I 0 (unchecked-int (long r0))))
                       (doto (.allocate arena I) (.set I 0 (unchecked-int (long r1))))
                       (doto (.allocate arena I) (.set I 0 (int n)))
                       (ptr (or (:device scratch) 0))
                       (ptr (or scratch-bytes 0))]
                      (into (map (comp ptr :device) inputs))
                      (into (map (comp ptr :device) bufs)))
             params (.allocate arena A (count vals))]
         (doseq [[i v] (map-indexed vector vals)] (.setAtIndex params A (long i) ^MemorySegment v))
         (cu! :cuLaunchKernel function (int (grid n (or lanes 1))) (int 1) (int 1) (int block-size) (int 1) (int 1) (int 0)
              MemorySegment/NULL params MemorySegment/NULL)
         (cu! :cuCtxSynchronize)
         (when copy?
           (doseq [{:keys [device host bytes]} bufs]
             (cu! :cuMemcpyDtoH host (long device) (long bytes))))
         bufs)))))

(defn log-mean-exp!
  "log mean exp of the n float32 values of output buffer k-th of bufs (on
   the device, from launch!): importance sampling's log-ML estimate,
   reduced on the device by the module's log_mean_exp kernel and returned
   as a double (8 bytes copied back). Launch with :copy? false to keep the
   weights on the device."
  ^double [{:keys [reduce-function reduce-out]} bufs n output]
  (current!)
  (with-open [arena (Arena/ofConfined)]
    (let [{:keys [device]} (nth bufs output)
          vals [(doto (.allocate arena J) (.set J 0 (long device)))
                (doto (.allocate arena I) (.set I 0 (int n)))
                (doto (.allocate arena J) (.set J 0 (long reduce-out)))]
          params (.allocate arena A (count vals))
          host (.allocate arena ValueLayout/JAVA_DOUBLE)]
      (doseq [[i v] (map-indexed vector vals)] (.setAtIndex params A (long i) ^MemorySegment v))
      (cu! :cuLaunchKernel reduce-function (int 1) (int 1) (int 1) (int 1024) (int 1) (int 1) (int 0)
           MemorySegment/NULL params MemorySegment/NULL)
      (cu! :cuCtxSynchronize)
      (cu! :cuMemcpyDtoH host (long reduce-out) (long 8))
      (.get host ValueLayout/JAVA_DOUBLE 0))))

(defn read-floats
  "The floats of a float32 buffer's host copy (row-major, [n …])."
  ^floats [{:keys [^MemorySegment host bytes]}]
  (.toArray (.asSlice host 0 (long bytes)) ValueLayout/JAVA_FLOAT))

(defn read-bytes
  "The bytes of a bool buffer's host copy."
  ^bytes [{:keys [^MemorySegment host bytes]}]
  (.toArray (.asSlice host 0 (long bytes)) ValueLayout/JAVA_BYTE))

(defn unload!
  "Unloads k's module (and frees its reduction's output)."
  [{:keys [module reduce-out]}]
  (current!)
  (when reduce-out (cu! :cuMemFree (long reduce-out)))
  (cu! :cuModuleUnload module))
