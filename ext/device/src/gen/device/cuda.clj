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

   Refused, loudly, by node: scatter-add (pads replace the slices'
   gradients), take, a scan over xs (counted scans only), a nested scan, a
   scan's ys read in the thread, float64, batched matmul, and arrays past
   a per-node size limit (gen.cuda.max-elements, 4096 elements)."
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

(defn- max-elements ^long [] (Long/getLong "gen.cuda.max-elements" 4096))

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

(defn- emit-node!
  "env with node's value: {:shape :dtype :el} (C expressions), or a scan's
   or its ys result's own entry."
  [em env {:keys [id op args shape dtype attrs] :as node} ctx]
  (when (and shape (> (size shape) (max-elements)))
    (throw (ex-info (str "CUDA C: " op " makes " (size shape) " elements in a thread, past gen.cuda.max-elements")
                    {:node (dissoc node :attrs) :limit (max-elements)})))
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
              :let [{:keys [shape el]} (benv y) n (size shape)]]
        (doseq [[e x] (map-indexed vector el)]
          (line! em (str "out" k "[(long long)tid * " (* length n) " + (long long)" it " * " n " + " e "] = " x ";"))))
      (let [outs (mapv #(:el (benv %)) carry-out)
            tmps (mapv (fn [v o] (mapv (fn [e] (bind! em (:dtype v) (str "(" e ")"))) o)) vars outs)]
        (doseq [[v t] (map vector vars tmps) [c x] (map vector (:el v) t)]
          (when-not (= c x) (line! em (str c " = " x ";")))))
      (vswap! (:indent em) dec)
      (line! em "}")
      {:carries vars :n-carry n-carry})))

(defn- emit-graph! [em nodes env ctx]
  (reduce (fn [env node] (emit-node! em env node ctx)) env nodes))

(defn source
  "The CUDA C++ text of the kernel `chains` for program (gen.device.stage's,
   of one key): (chains r0 r1 n out0 out1 …), thread i running program on
   (rng/split [r0 r1] n)[i]. Returns {:source :outputs [{:shape :dtype}
   per output, without the thread axis]}."
  [{:keys [graph inputs outputs]}]
  (when-not (= inputs [[[2] :uint32]])
    (throw (ex-info "CUDA C: a program of one key only" {:inputs inputs})))
  (let [em (emitter)
        by-id (into {} (map (juxt :id identity) graph))
        ;; ys a program returns: [scan id, ys index] → output
        ys-out (into {} (keep-indexed (fn [k o] (let [{:keys [op args attrs]} (by-id o)]
                                                  (when (= op :result)
                                                    (let [s (by-id (first args)) nc (get-in s [:attrs :n-carry])]
                                                      (when (>= (:index attrs) nc) [[(:id s) (- (:index attrs) nc)] k])))))
                                      outputs))
        param (first (filter #(= :param (:op %)) graph))
        env (emit-graph! em graph {(:id param) (value [2] :uint32 ["key0" "key1"])} {:ys-out ys-out})
        outs (mapv env outputs)
        body (.toString ^StringBuilder (:out em))
        tail (str/join (for [[k {:keys [shape el ys]}] (map-indexed vector outs)
                             :when (nil? ys)
                             [e x] (map-indexed vector el)]
                         (str "  out" k "[(long long)tid * " (size shape) " + " e "] = " x ";\n")))
        sig (str/join ", " (map-indexed (fn [k {:keys [dtype]}] (str (buffer-types dtype) "* out" k)) outs))]
    {:source (str prelude
                  "\n__device__ __forceinline__ void chain(int tid, unsigned int key0, unsigned int key1, " sig ") {\n"
                  body tail "}\n\n"
                  "#ifdef __CUDACC_RTC__\n"
                  "extern \"C\" __global__ void chains(unsigned int r0, unsigned int r1, int n, " sig ") {\n"
                  "  int tid = blockIdx.x * blockDim.x + threadIdx.x;\n"
                  "  if (tid >= n) return;\n"
                  "  unsigned int key0, key1;\n"
                  "  g_threefry(r0, r1, 0u, (unsigned int)tid, &key0, &key1);\n"
                  "  chain(tid, key0, key1, " (str/join ", " (map #(str "out" %) (range (count outs)))) ");\n"
                  "}\n#endif\n")
     :outputs (mapv (fn [{:keys [shape dtype ys]} o]
                      {:shape (if ys (:shape (by-id o)) shape) :dtype dtype})
                    outs outputs)}))

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
  [{:keys [source outputs]}]
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
              attr (fn [k] (let [v (.allocate arena I)] (cu! :cuFuncGetAttribute v (int k) function) (.get v I 0)))]
          {:function function :module module :outputs outputs :log log
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

(defn launch!
  "Launches n threads of k on the root key words [r0 r1], waits, and
   copies every output to its pinned host buffer (bufs, from buffers).
   Returns bufs. :copy? false skips the copies (the kernel alone)."
  ([k bufs n key-words] (launch! k bufs n key-words true))
  ([{:keys [function]} bufs n [r0 r1] copy?]
   (current!)
   (with-open [arena (Arena/ofConfined)]
     (let [block (min 128 (long n))
           grid (quot (+ (long n) (dec block)) block)
           vals (into [(doto (.allocate arena I) (.set I 0 (unchecked-int (long r0))))
                       (doto (.allocate arena I) (.set I 0 (unchecked-int (long r1))))
                       (doto (.allocate arena I) (.set I 0 (int n)))]
                      (map (fn [{:keys [device]}] (doto (.allocate arena J) (.set J 0 (long device)))) bufs))
           params (.allocate arena A (count vals))]
       (doseq [[i v] (map-indexed vector vals)] (.setAtIndex params A (long i) ^MemorySegment v))
       (cu! :cuLaunchKernel function (int grid) (int 1) (int 1) (int block) (int 1) (int 1) (int 0)
            MemorySegment/NULL params MemorySegment/NULL)
       (cu! :cuCtxSynchronize)
       (when copy?
         (doseq [{:keys [device host bytes]} bufs]
           (cu! :cuMemcpyDtoH host (long device) (long bytes))))
       bufs))))

(defn read-floats
  "The floats of a float32 buffer's host copy (row-major, [n …])."
  ^floats [{:keys [^MemorySegment host bytes]}]
  (.toArray (.asSlice host 0 (long bytes)) ValueLayout/JAVA_FLOAT))

(defn read-bytes
  "The bytes of a bool buffer's host copy."
  ^bytes [{:keys [^MemorySegment host bytes]}]
  (.toArray (.asSlice host 0 (long bytes)) ValueLayout/JAVA_BYTE))

(defn unload!
  "Unloads k's module."
  [{:keys [module]}]
  (current!)
  (cu! :cuModuleUnload module))
