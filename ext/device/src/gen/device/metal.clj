(ns gen.device.metal
  "A staged program (gen.device.stage) as one Metal kernel run through MLX
   (mlx_fast_metal_kernel): thread i runs the per-particle program on
   (rng/split root n)[i], its key derived in the thread, the whole run
   (a chain's steps, a particle's generate) one dispatch. The Metal target
   of bean genclj-h8rn (research prototype, ext/device), beside the JVM
   lowering and CUDA C; the program is scalarized by gen.device.scalar.

   (compile program) makes the kernel (MLX compiles the Metal source on its
   first run and caches it); (run k n root-words) dispatches n threads on
   the MLX backend's stream and returns the outputs as MLX arrays, run i's
   at [i …], unevaluated (array/eval! or a host read runs the kernel).

   float32 only (Metal has no float64). The arithmetic is Metal's; exp,
   log and the other transcendental functions are metal::precise's, erfinv
   Giles' single-precision approximation, so draws equal the handler's to
   float32 rounding.

   Refused, loudly: data inputs, arrays past gen.device.max-elements, a scan
   over xs or inside a scan, a scan's ys read in the program, batched
   matmul, float64, erf."
  {:api :internal}
  (:refer-clojure :exclude [compile])
  (:require [gen.array :as array]
            [gen.backend.mlx]
            [gen.device.scalar :as sc])
  (:import [gen.mlx Mlx MlxRaw MlxRuntime]
           [java.lang.foreign Arena ValueLayout]))

(set! *warn-on-reflection* true)

;; ---------------------------------------------------------------------------
;; Metal's expressions
;; ---------------------------------------------------------------------------

(def ^:private mtypes {:float32 "float" :uint32 "uint" :int32 "int" :bool "bool"})

(defn- mtype [dtype]
  (or (mtypes dtype) (throw (ex-info (str "Metal lowering: no " dtype " (float32, uint32, int32 and bool only)") {:dtype dtype}))))

(defn- lit [v dtype]
  (case dtype
    :float32 (let [f (unchecked-float (if (boolean? v) (if v 1.0 0.0) (double v)))]
               (if (or (Float/isNaN f) (Float/isInfinite f))
                 (format "as_type<float>(0x%08xu)" (Float/floatToRawIntBits f))
                 (str (Float/toString f) "f")))
    :uint32 (format "0x%08xu" (bit-and (long (if (boolean? v) (if v 1 0) v)) 0xFFFFFFFF))
    :int32 (str (long (if (boolean? v) (if v 1 0) v)))
    :bool (str (if (boolean? v) v (not (zero? (double v)))))
    (mtype dtype)))

(defn- simple? [^String e]
  (boolean (re-matches #"[A-Za-z_][A-Za-z0-9_]*|-?[0-9][0-9.eE+xa-f-]*[fu]?|as_type<float>\(0x[0-9a-f]+u\)" e)))

(def ^:private precise
  {:exp "exp" :log "log" :sqrt "sqrt" :sin "sin" :cos "cos" :tan "tan" :tanh "tanh" :sinh "sinh" :cosh "cosh"
   :arcsin "asin" :arccos "acos" :arctan "atan"})

(defn- unary-expr [op dtype a]
  (cond (= op :negative) (str "(- " a ")")
        (= op :logical-not) (str "(!" a ")")
        (= op :sign) (str "g_sign(" a ")")
        (= op :isnan) (if (= dtype :float32) (str "metal::isnan(" a ")") "false")
        (= op :isinf) (if (= dtype :float32) (str "metal::isinf(" a ")") "false")
        (= op :abs) (if (= dtype :uint32) a (str "metal::abs(" a ")"))
        (not= dtype :float32) (if (#{:floor :ceil :round} op) a
                                  (throw (ex-info (str "Metal lowering: no " op " on " dtype " yet") {:op op :dtype dtype})))
        (= op :erfinv) (str "g_erfinv(" a ")")
        (= op :expm1) (str "g_expm1(" a ")")
        (= op :log1p) (str "g_log1p(" a ")")
        (precise op) (str "metal::precise::" (precise op) "(" a ")")
        (= op :floor) (str "metal::floor(" a ")")
        (= op :ceil) (str "metal::ceil(" a ")")
        (= op :round) (str "metal::rint(" a ")")
        :else (throw (ex-info (str "Metal lowering: no " op " on " dtype " yet") {:op op :dtype dtype}))))

(def ^:private infix
  {:add "+" :subtract "-" :multiply "*" :divide "/" :logical-and "&&" :logical-or "||" :bit-xor "^" :bit-or "|"
   :bit-and "&" :less "<" :less-equal "<=" :greater ">" :greater-equal ">=" :equal "==" :not-equal "!="})

(defn- binary-expr [op dtype a b]
  (cond (infix op) (str "(" a " " (infix op) " " b ")")
        (= op :maximum) (str "g_max(" a ", " b ")")
        (= op :minimum) (str "g_min(" a ", " b ")")
        (= op :remainder) (if (= dtype :uint32) (str "(" a " % " b ")") (str "g_mod(" a ", " b ")"))
        (#{:shift-left :shift-right-logical} op)
        (let [s (sc/int-literal b)
              [cop helper] (if (= op :shift-left) ["<<" "g_shl"] [">>" "g_shr"])]
          (cond (nil? s) (str helper "(" a ", " b ")")
                (< -1 s 32) (str "(" a " " cop " " s "u)")
                :else "0u"))
        (and (= op :power) (= dtype :float32)) (str "metal::precise::pow(" a ", " b ")")
        (and (= op :arctan2) (= dtype :float32)) (str "metal::precise::atan2(" a ", " b ")")
        :else (throw (ex-info (str "Metal lowering: no " op " on " dtype " yet") {:op op :dtype dtype}))))

(defn- convert-expr [from to a]
  (cond (= from to) a
        (= to :bool) (str "(" a (if (= from :float32) " != 0.0f)" " != 0)"))
        (= from :bool) (str "(" a (case to :float32 " ? 1.0f : 0.0f)" :uint32 " ? 1u : 0u)" " ? 1 : 0)"))
        :else (str "((" (mtype to) ")(" a "))")))

(defn- bitcast-expr [from to a]
  (case [from to]
    [:uint32 :float32] (str "as_type<float>(" a ")")
    [:float32 :uint32] (str "as_type<uint>(" a ")")
    (throw (ex-info "Metal lowering: a bitcast other than uint32 ↔ float32" {:from from :to to}))))

(def dialect
  "Metal's expressions (gen.device.scalar's dialect)."
  {:name "Metal lowering" :lit lit :simple? simple? :unary unary-expr :binary binary-expr
   :convert convert-expr :bitcast bitcast-expr})

(def ^:private header
  "Helpers every kernel uses: Threefry for the threads' keys, and the ops
   whose Metal operators do not have gen.array's semantics."
  (let [rot [13 15 26 6 17 29 16 24]
        ks ["k0" "k1" "k2"]]
    (str "
inline float g_max(float a, float b) { return (metal::isnan(a) || metal::isnan(b)) ? as_type<float>(0x7fc00000u) : metal::fmax(a, b); }
inline float g_min(float a, float b) { return (metal::isnan(a) || metal::isnan(b)) ? as_type<float>(0x7fc00000u) : metal::fmin(a, b); }
inline int g_max(int a, int b) { return a > b ? a : b; }
inline int g_min(int a, int b) { return a < b ? a : b; }
inline uint g_max(uint a, uint b) { return a > b ? a : b; }
inline uint g_min(uint a, uint b) { return a < b ? a : b; }
inline float g_sign(float a) { return a > 0.0f ? 1.0f : (a < 0.0f ? -1.0f : a); }
inline int g_sign(int a) { return (a > 0) - (a < 0); }
inline float g_mod(float a, float b) { float r = metal::fmod(a, b); return (r != 0.0f && ((r < 0.0f) != (b < 0.0f))) ? r + b : r; }
inline int g_mod(int a, int b) { int r = a % b; return (r != 0 && ((r < 0) != (b < 0))) ? r + b : r; }
inline uint g_shl(uint x, uint s) { return s < 32u ? x << s : 0u; }
inline uint g_shr(uint x, uint s) { return s < 32u ? x >> s : 0u; }
inline float g_log1p(float x) { float u = 1.0f + x; return u == 1.0f ? x : metal::precise::log(u) * (x / (u - 1.0f)); }
inline float g_expm1(float x) { float u = metal::precise::exp(x); return u == 1.0f ? x : (u - 1.0f == -1.0f ? -1.0f : (u - 1.0f) * x / metal::precise::log(u)); }
// Giles' single-precision erfinv
inline float g_erfinv(float x) {
  float w = -metal::precise::log((1.0f - x) * (1.0f + x));
  float p;
  if (w < 5.0f) {
    w = w - 2.5f;
    p = 2.81022636e-08f;
    p = 3.43273939e-07f + p * w; p = -3.5233877e-06f + p * w; p = -4.39150654e-06f + p * w;
    p = 0.00021858087f + p * w; p = -0.00125372503f + p * w; p = -0.00417768164f + p * w;
    p = 0.246640727f + p * w; p = 1.50140941f + p * w;
  } else {
    w = metal::precise::sqrt(w) - 3.0f;
    p = -0.000200214257f;
    p = 0.000100950558f + p * w; p = 0.00134934322f + p * w; p = -0.00367342844f + p * w;
    p = 0.00573950773f + p * w; p = -0.0076224613f + p * w; p = 0.00943887047f + p * w;
    p = 1.00167406f + p * w; p = 2.83297682f + p * w;
  }
  return p * x;
}
// Threefry-2x32-20 of the counter [c0 c1] under the key [k0 k1] (gen.random.threefry)
inline void g_threefry(uint k0, uint k1, uint c0, uint c1, thread uint& w0, thread uint& w1) {
  uint k2 = k0 ^ k1 ^ 0x1BD11BDAu;
  uint x0 = c0 + k0, x1 = c1 + k1;
"
         (apply str (for [i (range 5)]
                      (str (apply str (for [j (range 4)]
                                        (let [r (nth rot (+ (* (mod i 2) 4) j))]
                                          (str "  x0 += x1; x1 = ((x1 << " r "u) | (x1 >> " (- 32 r) "u)) ^ x0;\n"))))
                           "  x0 += " (nth ks (mod (inc i) 3)) "; x1 += " (nth ks (mod (+ i 2) 3)) " + " (inc i) "u;\n")))
         "  w0 = x0; w1 = x1;\n}\n")))

;; ---------------------------------------------------------------------------
;; The kernel's text
;; ---------------------------------------------------------------------------

(defn- body-text
  "The statements as Metal, run p's outputs at p·size."
  [stmts out-specs]
  (let [sb (StringBuilder.)
        line (fn [indent s] (.append sb (apply str (repeat indent "  "))) (.append sb ^String s) (.append sb "\n"))
        emit (fn emit [indent stmts]
               (doseq [[kind a b c :as s] stmts]
                 (case kind
                   :decl (line indent (str (mtype a) " " b " = " c ";"))
                   :set (line indent (str a " = " b ";"))
                   :store (line indent (str "out" a "[p * " (sc/size (:shape (nth out-specs a))) "u + " b "] = " c ";"))
                   :loop (let [[_ it length body] s]
                           (line indent (str "for (uint " it " = 0; " it " < " length "u; ++" it ") {"))
                           (emit (inc indent) body)
                           (line indent "}")))))]
    (line 0 "uint p = thread_position_in_grid.x;")
    (line 0 "if (p >= nn[0]) return;")
    (line 0 "uint key0, key1;")
    (line 0 "g_threefry(root[0], root[1], 0u, p, key0, key1);")
    (emit 0 stmts)
    (.toString sb)))

(defn source
  "The Metal kernel of program (gen.device.stage's, of one key):
   {:source (its body) :header :outputs [{:shape :dtype} per run]
   :statements}. Thread p runs program on (rng/split [root0 root1] n)[p]
   (inputs root, uint32 [2], and nn, uint32 [1] holding n)."
  [program]
  (let [{:keys [stmts outputs statements]} (sc/scalarize program dialect)]
    {:source (body-text stmts outputs) :header header :outputs outputs :statements statements}))

;; ---------------------------------------------------------------------------
;; Running through MLX
;; ---------------------------------------------------------------------------

(def ^:private dtype-code @#'gen.backend.mlx/code)
(def ^:private wrap @#'gen.backend.mlx/wrap)
(def ^:private stream-of @#'gen.backend.mlx/stream)

(def ^:private counter (atom 0))

(defn- strings ^long [xs]
  (let [v (Mlx/mlx_vector_string_new)]
    (doseq [x xs] (Mlx/mlx_vector_string_append_value v ^String x))
    v))

(defn compile
  "program as an MLX Metal kernel: {:kernel (its handle) :outputs :source
   :header :statements :name}. MLX compiles the source on the kernel's
   first run."
  [program]
  (let [{:keys [source header outputs] :as src} (source program)
        name (str "gen_program_" (swap! counter inc))
        ins (strings ["root" "nn"])
        outs (strings (map #(str "out" %) (range (count outputs))))]
    (try
      (with-open [arena (Arena/ofConfined)]
        (assoc src
               :name name
               :kernel (Mlx/mlx_fast_metal_kernel_new (.allocateFrom arena ^String name) ins outs
                                                      (.allocateFrom arena ^String source) (.allocateFrom arena ^String header)
                                                      true false)))
      (finally (Mlx/mlx_vector_string_free ins) (Mlx/mlx_vector_string_free outs)))))

(defn run
  "n threads of k on root's words: the outputs as MLX arrays of the MLX
   backend b ([n …] each), unevaluated. opts: :thread-group (threads per
   group, 64 by default)."
  ([k n root-words] (run k n root-words {}))
  ([{:keys [kernel outputs]} n [r0 r1] {:keys [thread-group backend] :or {thread-group 64}}]
   (let [b (or backend array/*backend*)
         n (long n)
         root (array/array [(bit-and (long r0) 0xFFFFFFFF) (bit-and (long r1) 0xFFFFFFFF)] {:dtype :uint32})
         nn (array/array [n] {:dtype :uint32})
         cfg (Mlx/mlx_fast_metal_kernel_config_new)]
     (try
       (with-open [arena (Arena/ofConfined)]
         (doseq [{:keys [shape dtype]} outputs]
           (let [shp (into [n] shape)]
             (Mlx/mlx_fast_metal_kernel_config_add_output_arg cfg (.allocateFrom arena ValueLayout/JAVA_INT (int-array (map int shp)))
                                                              (long (count shp)) (int (dtype-code dtype))))))
       (Mlx/mlx_fast_metal_kernel_config_set_grid cfg (int n) 1 1)
       (Mlx/mlx_fast_metal_kernel_config_set_thread_group cfg (int (min n thread-group)) 1 1)
       (let [in (MlxRuntime/vectorOf (long-array [(.-h ^gen.backend.mlx.MlxArray root) (.-h ^gen.backend.mlx.MlxArray nn)]))]
         (try
           (let [out (Mlx/mlx_fast_metal_kernel_apply kernel in cfg (stream-of b))]
             (try (mapv #(wrap b %) (MlxRuntime/vectorArrays out))
                  (finally (MlxRaw/mlx_vector_array_free out))))
           (finally (MlxRaw/mlx_vector_array_free in)
                    (java.lang.ref.Reference/reachabilityFence root)
                    (java.lang.ref.Reference/reachabilityFence nn))))
       (finally (Mlx/mlx_fast_metal_kernel_config_free cfg))))))
