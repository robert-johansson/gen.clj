(ns gen.backend.mlx
  "The MLX backend (M2): arrays are MLX arrays, computed lazily by MLX on the
   CPU or the GPU (Metal; CUDA on Linux), through mlx-c over FFM. The Java
   binding (src-java/gen/mlx, generated from mlx-c's headers) is compiled with
   `clojure -T:build java`; the native libraries come from native/build-mlx.sh.

   - Arrays are MlxArray values owning an MLX handle (a long), freed once the
     value is unreachable (gen.mlx.MlxRuntime/adopt: phantom references
     drained by the threads making arrays). MLX's reference counting keeps
     any graph that still needs an array alive. The GC does not see MLX's
     memory, so the backend collects under pressure: when 100,000 arrays are
     live (gen.mlx.gc-arrays; Metal allows about 500,000 buffers) or MLX's
     active memory passes an eighth of RAM (gen.mlx.gc-bytes).
   - MLX's own limits are bounded too: its buffer cache (freed buffers kept
     for reuse) to a sixteenth of RAM, and its memory limit, past which it
     frees the cache and waits for running work before allocating, to half
     of RAM. MLX's defaults are all of RAM for both, which let one JVM take
     the whole machine (2026-10-19: a kernel panic from memory exhaustion).
     MLX's limit is no cap, though: past it MLX goes on allocating, without
     an error (genclj-zztc, 2026-10-20).
   - So the backend keeps a budget of its own (budget, set-budget!): MLX's
     active memory may reach the budget it shares with native engines on
     the same machine (an LLM engine's weights and caches, which they
     reserve in gen.array.memory), less those reservations and a headroom.
     After every evaluation the backend checks it; past it, it collects
     unreachable arrays and checks again, and then it is a loud error
     (:reason :mlx-budget-exceeded) with the numbers.
   - 0-d arrays are MLX arrays too (the JVM backend's are host values), so
     scalar code stays lazy. Reading one on the host (item, truthy?) evaluates.
   - Ops run on the backend's stream, a stream MLX lets any thread use. One
     lock serializes evaluation, so the backend can be used from several
     threads.
   - Decompositions and solves run on the CPU stream (MLX implements them only
     there).
   - Keys are gen.random's (JAX's partitionable Threefry), computed with MLX's
     uint32 ops, so a key draws the same uniforms as on the JVM backend.
   - Gradients use the backend-generic tape (gen.array.tape), or MLX's own
     with gen.mlx.grad=native (docs/array-api.md §8).

   This namespace holds the backend's ops and its record; its parts are
   gen.backend.mlx.array (the MlxArray type, dtype codes, the backend's
   configuration), .memory (the budget, Metal's buffer limit, evaluation),
   .operands (host scalars' constants, temporaries, an op's operands, host
   data), .linalg, .random (Threefry), .transforms (MLX's gradients, custom
   VJPs and compile), .exact (exact routes around MLX's kernels), .nn (its
   fast kernels and quantized ops) and .io (GGUF files).

   Configuration (system properties, read when the namespace loads):
     gen.mlx.device  cpu or gpu (default gpu when MLX has one)
     gen.mlx.dtypes  32 (default) or 64: the default dtypes, float32/int32 or
                     float64/int64 (64 needs the CPU device; MLX's GPUs have no
                     float64)
     gen.mlx.grad    tape (default) or native
     gen.mlx.cache-bytes   MLX's buffer cache (default a sixteenth of RAM)
     gen.mlx.memory-bytes  MLX's memory limit (default half of RAM)
     gen.mlx.budget-bytes  the budget MLX shares with native engines (default
                           half of RAM)
     gen.mlx.budget-headroom-bytes  kept free of both (default a sixteenth
                           of RAM)
     gen.mlx.gc-bytes, gen.mlx.gc-arrays  when the backend collects (above)
     gen.mlx.gc-new-bytes  collect after this many bytes of new arrays
                           (default a sixteenth of RAM)
     gen.mlx.constants     host-scalar constants kept per dtype (default
                           4096; each holds one buffer)
     gen.mlx.cpu-compile   true or false: skip the probe of compile on the
                           CPU device (cpu-compiles?)
     gen.mlx.tf32          true: MLX's float32 matrix products in TF32 on
                           CUDA (and through Metal's neural accelerators
                           where a Mac has them), faster and about 1e-3
                           relative; default false, full float32 (MlxRuntime
                           sets MLX_ENABLE_TF32=0 unless the environment
                           sets it; the property decides over it)
     gen.mlx.lib           the directory of libmlxc (default
                           native/build/<platform>/lib; MlxRuntime)"
  {:api :internal}
  (:require [gen.array.dtype :as dt]
            [gen.array.protocols :as p]
            [gen.array.shape :as sh]
            [gen.backend.mlx.array :as mlx-array
             :refer [->Config check-dtype code defaults fence mlx-array? stream with-device wrap]]
            [gen.backend.mlx.io :as io]
            [gen.backend.mlx.linalg :refer [linalg matmul]]
            [gen.backend.mlx.memory :as memory]
            [gen.backend.mlx.nn :as nn]
            [gen.backend.mlx.operands :as operands
             :refer [as-array dtype-on eval-arrays! from-host host? mlx-error operand scalar-handle shape-of
                     with-operands with-temps]]
            [gen.backend.mlx.random :refer [random-uniform]]
            [gen.backend.mlx.transforms :as transforms
             :refer [*differentiating* *tracing-compile* checkpoint compile-fn custom-vjp custom-vjps? grad-mode
                     native-value-and-grad tracing-compile?]])
  (:import [gen.backend.mlx.array Config Configured]
           [gen.mlx Mlx MlxRuntime MlxException]))

(set! *warn-on-reflection* true)


;; ---------------------------------------------------------------------------
;; The parts' vars that dependents call here
;; ---------------------------------------------------------------------------

(defmacro ^:private export
  "Vars of this namespace that are its parts' (syms, qualified): each the
   part's value, with its docstring and arglists."
  [& syms]
  `(do ~@(for [s syms :let [m (meta (resolve s))]]
           `(def ~(vary-meta (symbol (name s)) assoc :doc (:doc m) :arglists (list 'quote (:arglists m))) ~s))))

(export mlx-array/device operands/to-host
        memory/memory memory/evaluated? memory/collect! memory/budget memory/set-budget!
        memory/set-memory-limit! memory/set-cache-limit! memory/clear-cache!)

;; ---------------------------------------------------------------------------
;; Elementwise ops
;; ---------------------------------------------------------------------------

(defmacro ^:private mlx-fn
  "A fn of handles (and the stream) calling the mlx-c function f."
  [f n]
  (let [args (map #(with-meta % {:tag 'long}) (repeatedly n #(gensym "h")))
        s (with-meta (gensym "s") {:tag 'long})]
    `(fn ~(vec (concat args [s])) (~(symbol "Mlx" (str "mlx_" f)) ~@args ~s))))

(def ^:private unary-fns
  {:negative (mlx-fn negative 1) :abs (mlx-fn abs 1) :sign (mlx-fn sign 1) :sqrt (mlx-fn sqrt 1)
   :exp (mlx-fn exp 1) :log (mlx-fn log 1) :expm1 (mlx-fn expm1 1) :log1p (mlx-fn log1p 1)
   :sin (mlx-fn sin 1) :cos (mlx-fn cos 1) :tan (mlx-fn tan 1) :tanh (mlx-fn tanh 1)
   :sinh (mlx-fn sinh 1) :cosh (mlx-fn cosh 1)
   :arcsin (mlx-fn arcsin 1) :arccos (mlx-fn arccos 1) :arctan (mlx-fn arctan 1)
   :erf (mlx-fn erf 1) :erfinv (mlx-fn erfinv 1)
   :floor (mlx-fn floor 1) :ceil (mlx-fn ceil 1) :round (fn [^long h ^long s] (Mlx/mlx_round h (int 0) s))
   :isnan (mlx-fn isnan 1) :isinf (mlx-fn isinf 1) :logical-not (mlx-fn logical_not 1)
   :stop-gradient (mlx-fn stop_gradient 1)})

(def ^:private binary-fns
  {:add (mlx-fn add 2) :subtract (mlx-fn subtract 2) :multiply (mlx-fn multiply 2)
   :divide (mlx-fn divide 2) :power (mlx-fn power 2) :maximum (mlx-fn maximum 2)
   :minimum (mlx-fn minimum 2) :remainder (mlx-fn remainder 2) :floor-divide (mlx-fn floor_divide 2)
   :less (mlx-fn less 2) :less-equal (mlx-fn less_equal 2) :greater (mlx-fn greater 2)
   :greater-equal (mlx-fn greater_equal 2) :equal (mlx-fn equal 2) :not-equal (mlx-fn not_equal 2)
   :logical-and (mlx-fn logical_and 2) :logical-or (mlx-fn logical_or 2) :logaddexp (mlx-fn logaddexp 2)
   :arctan2 (mlx-fn arctan2 2)})

(declare exact-route broadcast-to)

(defn- unary [b op x dtype]
  (cond
    ;; MLX's kernels that are not exact take exact routes (gen.backend.mlx.exact,
    ;; loaded on first use: it is written in gen.array, which loads this namespace)
    (and (= dtype :float64) (#{:exp :sin :cos :erf :erfinv} op))
    (exact-route b op [(as-array b x dtype)])

    (and (= op :expm1) (= :gpu (device b)) (#{:float32 :float16 :bfloat16} dtype))
    (exact-route b :expm1 [(as-array b x dtype)])

    (and (#{:floor :ceil :round} op) (not= :float (dt/kind dtype)))
    ;; rounding an int is the identity
    (as-array b x dtype)

    :else
    (let [f (unary-fns op)
          s (stream b)]
      (with-operands b op [h x dtype]
        (wrap b (cond
                  (and (= op :sign) (= :float (dt/kind dtype)))
                  ;; MLX's sign of NaN is 0; NumPy's (and the JVM backend's) NaN
                  (with-temps [t]
                    (Mlx/mlx_where (t (Mlx/mlx_isnan h s)) h (t (f h s)) s))

                  (and (= op :arctan) (not= dtype :float64))
                  ;; Metal's float32 atan(-0.0) is +0.0: x itself at zero (where
                  ;; not x), as NumPy and the JVM backend; float64's keeps the sign
                  (with-temps [t]
                    (Mlx/mlx_where (t (Mlx/mlx_logical_not h s)) h (t (f h s)) s))

                  :else (f h s)))))))

(defn- int-division
  "remainder or floor-divide of int handles, as the JVM backend (and NumPy):
   0 for a zero divisor, since integer division by zero is undefined in
   MLX's kernels; and floor division rounding down, which MLX's (truncating)
   floor_divide does not: (x - x mod y) / y, an exact division. A -1
   divisor of a signed dtype never reaches the kernels either: MIN / -1
   overflows, which x86's idiv traps (SIGFPE, a JVM crash on MLX's CPU,
   genclj-kw88). x mod -1 is 0 and x floor-div -1 is -x, wrapping at MIN as
   Java's floorDiv."
  [b op hx hy dtype]
  (let [hx (long hx) hy (long hy)
        s (stream b)
        signed? (not= dtype :uint32)]
    (with-temps [t]
      (let [zero (t (scalar-handle 0 dtype))
            one (t (scalar-handle 1 dtype))
            z (t (Mlx/mlx_equal hy zero s))
            m1 (when signed? (t (Mlx/mlx_equal hy (t (scalar-handle -1 dtype)) s)))
            safe (t (Mlx/mlx_where (if m1 (t (Mlx/mlx_logical_or z m1 s)) z) one hy s))
            r (t (Mlx/mlx_remainder hx safe s))
            q (if (= op :remainder)
                r
                (let [q (t (Mlx/mlx_floor_divide (t (Mlx/mlx_subtract hx r s)) safe s))]
                  (if m1 (t (Mlx/mlx_where m1 (t (Mlx/mlx_negative q s)) q s)) q)))]
        (Mlx/mlx_where z zero q s)))))

(defn- float-floor-divide
  "floor-divide of float handles as NumPy's (gen.array.dtype/floor-div,
   DESIGN §9, 1.2.1's owner calls), where MLX's is floor(x/y): x/y for a
   zero y; otherwise (x - fmod(x, y)) / y, one less where fmod's sign is not
   y's, snapped to the nearest integer, and a zero quotient signed as x/y.
   MLX has no fmod: it is MLX's remainder of |x| and |y|, which is fmod's
   magnitude there, signed as x."
  [b hx hy dtype]
  (let [hx (long hx) hy (long hy)
        s (stream b)]
    (with-temps [t]
      (let [k (fn ^long [v] (t (scalar-handle v dtype)))
            zero (k 0.0)
            less0 (fn ^long [^long h] (t (Mlx/mlx_less h zero s)))
            r (t (Mlx/mlx_remainder (t (Mlx/mlx_abs hx s)) (t (Mlx/mlx_abs hy s)) s))
            m (t (Mlx/mlx_where (less0 hx) (t (Mlx/mlx_negative r s)) r s))
            x-over-y (t (Mlx/mlx_divide hx hy s))
            q (t (Mlx/mlx_divide (t (Mlx/mlx_subtract hx m s)) hy s))
            ;; NaN is no zero, and NaN < 0 is false, as NumPy's tests
            off (t (Mlx/mlx_logical_and (t (Mlx/mlx_not_equal m zero s))
                                        (t (Mlx/mlx_not_equal (less0 hy) (less0 m) s)) s))
            q (t (Mlx/mlx_where off (t (Mlx/mlx_subtract q (k 1.0) s)) q s))
            f (t (Mlx/mlx_floor q s))
            snapped (t (Mlx/mlx_where (t (Mlx/mlx_greater (t (Mlx/mlx_subtract q f s)) (k 0.5) s))
                                      (t (Mlx/mlx_add f (k 1.0) s)) f s))
            ;; a zero quotient has |x| < |y|, so x/y is finite there: x/y·0 is
            ;; the zero of its sign
            q (t (Mlx/mlx_where (t (Mlx/mlx_equal q zero s)) (t (Mlx/mlx_multiply x-over-y zero s)) snapped s))]
        (Mlx/mlx_where (t (Mlx/mlx_equal hy zero s)) x-over-y q s)))))

(defn- java-int-cast
  "Float handle h (of dtype from) cast to the int dtype as Java casts
   (DESIGN.md §9): truncated toward zero, saturating at the dtype's range
   (the infinities too), NaN 0. MLX's own cast is C++'s, undefined out of
   range: MIN for all of these on x86's CPU, MIN for an int64 NaN on the GPU.
   The bounds are powers of two, exact in every float dtype (or infinite,
   in float16, whose finite values all fit)."
  [b ^long h from dtype]
  (let [s (stream b)
        [lo hi lo-int hi-int] (case dtype
                                :int32 [-2147483648.0 2147483648.0 Integer/MIN_VALUE Integer/MAX_VALUE]
                                :int64 [-9.223372036854775808E18 9.223372036854775808E18 Long/MIN_VALUE Long/MAX_VALUE]
                                :uint32 [0.0 4294967296.0 0 4294967295])]
    (with-temps [t]
      (let [k (fn ^long [v d] (t (scalar-handle v d)))
            over (t (Mlx/mlx_greater_equal h (k hi from) s))
            under (t (Mlx/mlx_less h (k lo from) s))
            bad (t (Mlx/mlx_logical_or (t (Mlx/mlx_logical_or over under s)) (t (Mlx/mlx_isnan h s)) s))
            cast (t (Mlx/mlx_astype (t (Mlx/mlx_where bad (k 0.0 from) h s)) (int (code dtype)) s))]
        (Mlx/mlx_where over (k hi-int dtype) (t (Mlx/mlx_where under (k lo-int dtype) cast s)) s)))))

(defn- int-power
  "x^y of int handles, as the JVM backend (and host math): exact for y >= 0;
   for y < 0 through floats, cast as Java casts (truncated, saturating at
   the ±infinities, 0 for NaN): 1 and -1 give ±1, larger |x| give 0, and 0
   gives the int64 maximum. (MLX's integer power gives 0 for every negative
   exponent, and for every element of an array with one.)"
  [b hx hy dtype]
  (let [hx (long hx) hy (long hy)
        s (stream b)
        fdt (int (code (:float (defaults b))))
        i64 (int (code :int64))]
    (with-temps [t]
      ;; MLX's integer power gives 0 in every element once any exponent is
      ;; negative: the exact path sees only non-negative stand-ins
      (let [k (fn ^long [v d] (t (scalar-handle v d)))
            neg (t (Mlx/mlx_less hy (k 0 dtype) s))
            exact (t (Mlx/mlx_power hx (t (Mlx/mlx_where neg (k 0 dtype) hy s)) s))
            f (t (Mlx/mlx_power (t (Mlx/mlx_astype hx fdt s)) (t (Mlx/mlx_astype hy fdt s)) s))
            fin (t (Mlx/mlx_isinf f s))
            finite (t (Mlx/mlx_where (t (Mlx/mlx_logical_or fin (t (Mlx/mlx_isnan f s)) s))
                                     (k 0.0 (:float (defaults b))) f s))
            cast (t (Mlx/mlx_astype finite i64 s))
            pos-inf (t (Mlx/mlx_logical_and fin (t (Mlx/mlx_greater f (k 0.0 (:float (defaults b))) s)) s))
            neg-inf (t (Mlx/mlx_logical_and fin (t (Mlx/mlx_less f (k 0.0 (:float (defaults b))) s)) s))
            java (t (Mlx/mlx_where pos-inf (k Long/MAX_VALUE :int64)
                                   (t (Mlx/mlx_where neg-inf (k Long/MIN_VALUE :int64) cast s)) s))
            java (if (= dtype :int64) java (t (Mlx/mlx_astype java (int (code dtype)) s)))]
        (Mlx/mlx_where neg java exact s)))))

(defn- binary [b op x y dtype]
  (let [sx (shape-of x) sy (shape-of y)]
    (sh/broadcast sx sy)
    (if (and (= op :logaddexp) (= dtype :float64))
      ;; MLX's float64 exp is not exact (see unary)
      ;; broadcast first: a custom VJP's cotangents have its inputs' shapes
      (let [shape (sh/broadcast sx sy)]
        (exact-route b :logaddexp [(broadcast-to b (as-array b x dtype) shape) (broadcast-to b (as-array b y dtype) shape)]))
    (let [s (stream b)]
      (with-operands b op [hx x dtype hy y dtype]
        (wrap b (cond (and (#{:remainder :floor-divide} op) (= :int (dt/kind dtype)))
                      (int-division b op hx hy dtype)
                      (= op :floor-divide)
                      (float-floor-divide b hx hy dtype)
                      (and (= op :power) (= :int (dt/kind dtype)))
                      (int-power b hx hy dtype)
                      :else ((binary-fns op) hx hy s))))))))

(defn- where-op [b c x y dtype]
  (sh/broadcast-all [(shape-of c) (shape-of x) (shape-of y)])
  (let [s (stream b)]
    (with-operands b :where [hc c :bool hx x dtype hy y dtype]
      (wrap b (Mlx/mlx_where hc hx hy s)))))

(defn- elementwise [b op args dtype]
  (case (count args)
    1 (unary b op (nth args 0) dtype)
    2 (binary b op (nth args 0) (nth args 1) dtype)
    3 (where-op b (nth args 0) (nth args 1) (nth args 2) dtype)))


;; ---------------------------------------------------------------------------
;; Reductions and scans
;; ---------------------------------------------------------------------------

(defn- arg-reduce
  "argmax or argmin of handle h along axis, as the default int dtype. As in
   NumPy, the first NaN wins (MLX skips NaNs)."
  [b op h axis kd dtype]
  (let [h (long h)
        s (stream b)
        ax (int axis)
        kd (boolean kd)
        int-code (int (code (:int (defaults b))))]
    (with-temps [t]
      (let [i (t (if (= op :argmax) (Mlx/mlx_argmax_axis h ax kd s) (Mlx/mlx_argmin_axis h ax kd s)))
            i (if (= :float (dt/kind dtype))
                (let [nan (t (Mlx/mlx_isnan h s))
                      first-nan (t (Mlx/mlx_argmax_axis nan ax kd s))
                      any-nan (t (Mlx/mlx_any_axis nan ax kd s))]
                  (t (Mlx/mlx_where any-nan first-nan i s)))
                i)]
        ;; MLX's indices are uint32
        (Mlx/mlx_astype i int-code s)))))

(def ^:private exact-fns
  ;; gen.backend.mlx.exact's routes, resolved on first use, once: they are
  ;; written in gen.array, which loads this namespace
  (delay {:blocked-above (long @(requiring-resolve 'gen.backend.mlx.exact/blocked-above))
          :sum (requiring-resolve 'gen.backend.mlx.exact/sum)
          :half-sum (requiring-resolve 'gen.backend.mlx.exact/half-sum)
          :route (requiring-resolve 'gen.backend.mlx.exact/route)
          :vjp (requiring-resolve 'gen.backend.mlx.exact/vjp)}))

(def ^:private tape-value-and-grad
  ;; gen.array.tape's, resolved once, as exact-fns
  (delay (requiring-resolve 'gen.array.tape/value-and-grad)))

(defn- blocked-sum?
  "Does a sum of x over axes take gen.backend.mlx.exact's blocks: float32
   on the CPU, of more terms than exact/blocked-above (genclj-y0mp)?"
  [b x dtype axes]
  (and (= dtype :float32) (= :cpu (device b))
       (> (sh/size (mapv #(nth (shape-of x) %) axes)) (long (:blocked-above @exact-fns)))))


(def ^:private mac?
  "Does this JVM run on macOS (MLX's GPU Metal there, CUDA elsewhere)?"
  (delay (.contains (.toLowerCase (System/getProperty "os.name")) "mac")))

(defn- cuda?
  "Is b's device a CUDA GPU (MLX's GPU off macOS)?"
  [b]
  (and (= :gpu (device b)) (not @mac?)))

(defn- reduce-op [b op x axes keep-dims?]
  (let [s (stream b)
        dtype (dtype-on b x)]
    (cond
      (empty? axes)
      ;; over no axis: the identity (argmax/argmin always name one axis)
      (as-array b x dtype)

      (and (#{:argmax :argmin} op) (zero? (sh/size (shape-of x))))
      ;; an empty result (gen.array refuses an empty axis): MLX's
      ;; argmax/argmin refuse any empty array, NumPy and the JVM backend
      ;; give the empty indices
      (let [sx (shape-of x) a (long (first axes))]
        (p/-full b (if keep-dims? (assoc sx a 1) (into (subvec sx 0 a) (subvec sx (inc a)))) 0 (:int (defaults b))))

      (and (= op :sum) (blocked-sum? b x dtype axes))
      ;; MLX's CPU accumulates a float32 sum sequentially (DESIGN §9)
      ((:sum @exact-fns) (as-array b x dtype) axes keep-dims?)

      (and (= op :sum) (#{:float16 :bfloat16} dtype) (or (= :cpu (device b)) (cuda? b)))
      ;; and a half-float sum in its own precision, on MLX's CPU and on CUDA
      ;; (genclj-tblx): in float32 instead; Metal accumulates them in float32
      ((:half-sum @exact-fns) (as-array b x dtype) axes keep-dims?)

      :else
      (let [ax (int-array axes)
            kd (boolean keep-dims?)]
        (with-operands b op [h x dtype]
          (wrap b (case op
                    :sum (Mlx/mlx_sum_axes h ax kd s)
                    :prod (Mlx/mlx_prod_axes h ax kd s)
                    :max (Mlx/mlx_max_axes h ax kd s)
                    :min (Mlx/mlx_min_axes h ax kd s)
                    :all (Mlx/mlx_all_axes h ax kd s)
                    :any (Mlx/mlx_any_axes h ax kd s)
                    (:argmax :argmin) (arg-reduce b op h (first axes) kd dtype))))))))

(defn- scan-op [b op x axis reverse?]
  (let [s (stream b)
        ax (int axis)
        rev (boolean reverse?)]
    (with-operands b op [h x nil]
      (wrap b (case op
                :cumsum (Mlx/mlx_cumsum_axis h ax rev true nil s)
                :cumprod (Mlx/mlx_cumprod_axis h ax rev true nil s)
                :cummax (Mlx/mlx_cummax_axis h ax rev true s)
                :cummin (Mlx/mlx_cummin_axis h ax rev true s))))))

;; ---------------------------------------------------------------------------
;; Shape and indexing
;; ---------------------------------------------------------------------------

(defn- reshape [b x shape]
  (with-operands b :reshape [h x nil]
    (wrap b (Mlx/mlx_reshape h (int-array (map int shape)) (stream b)))))

(defn- transpose [b x perm]
  (with-operands b :transpose [h x nil]
    (wrap b (Mlx/mlx_transpose_axes h (int-array (map int perm)) (stream b)))))

(defn- broadcast-to [b x shape]
  (with-operands b :broadcast-to [h x nil]
    (wrap b (Mlx/mlx_broadcast_to h (int-array (map int shape)) (stream b)))))

(defn- concatenate [b xs axis]
  (let [d (dtype-on b (first xs))]
    (with-temps [t]
      (try
        (let [hs (mapv #(let [raw (operand b % d)] (if (neg? raw) (t (- raw)) raw)) xs)
              v (t :vector (MlxRuntime/vectorOf (long-array hs)))]
          (wrap b (Mlx/mlx_concatenate_axis v (int axis) (stream b))))
        (catch MlxException e (throw (mlx-error :concatenate xs e)))
        (finally (fence xs))))))

(defn- slice [b x starts stops steps]
  (let [shape (shape-of x)
        ;; MLX reads negative starts and stops from the end. With a negative
        ;; step, NumPy's normalized bounds use -1 for "before index 0": a
        ;; start there is an empty slice, and a stop there is -(n+1) to MLX
        neg-empty? (mapv (fn [st step] (and (neg? (long step)) (== -1 (long st)))) starts steps)
        starts (mapv (fn [st e?] (if e? 0 st)) starts neg-empty?)
        stops (mapv (fn [e st n e?]
                      (cond e? 0
                            (and (neg? (long st)) (== -1 (long e))) (- (inc (long n)))
                            :else e))
                    stops steps shape neg-empty?)]
    (with-operands b :slice [h x nil]
      (wrap b (Mlx/mlx_slice h (int-array (map int starts)) (int-array (map int stops))
                             (int-array (map int steps)) (stream b))))))

(defn- host-index
  "The host integer index v checked against a dimension of size d."
  ^long [v ^long d]
  (let [i (long v)
        i (if (neg? i) (+ i d) i)]
    (if (and (>= i 0) (< i d))
      i
      (throw (ex-info (str "Index " v " is out of bounds for a dimension of size " d)
                      {:index v :size d})))))

(defn- safe-indices
  "A handle of the int indices h normalized against a dimension of size d:
   negative ones count from the end, and out-of-range ones are clamped (MLX's
   gathers do not check bounds; a lazy backend cannot check them without
   evaluating)."
  ^long [b ^long h ^long d]
  (let [s (stream b)]
    (with-temps [t]
      (let [;; int32 indices, whatever the int dtype
            i32 (t (Mlx/mlx_astype h (code :int32) s))
            ;; through made, as every array made from host data: at Metal's
            ;; buffer limit a raw constructor gives an empty handle, which
            ;; the next op rejects (genclj-j9nx)
            zero (t (scalar-handle 0 :int32))
            dd (t (scalar-handle d :int32))
            top (t (scalar-handle (dec d) :int32))
            neg (t (Mlx/mlx_less i32 zero s))
            wrapped (t (Mlx/mlx_add i32 dd s))
            n (t (Mlx/mlx_where neg wrapped i32 s))]
        (Mlx/mlx_clip n zero top s)))))

(def ^:no-doc ^:dynamic *check-gathers*
  "Whether a gather's array indices are read on the host and checked, loud
   when one is out of range (as the JVM backend's are), in place of the
   documented clamp: one evaluation per gather, for debugging. Set by
   -Dgen.mlx.check-gathers=true (genclj-08un, DESIGN §9); off by default.
   A gather inside mlx_compile's trace cannot be read, and is not checked."
  (Boolean/parseBoolean (System/getProperty "gen.mlx.check-gathers" "false")))

(defn- check-indices!
  "With *check-gathers*, the array indices read and each checked against a
   dimension of size d (negative from the end), as host-index checks a
   host index."
  [indices ^long d]
  (when (and *check-gathers* (mlx-array? indices) (not (tracing-compile?)))
    (doseq [v (flatten [(to-host indices)])]
      (host-index v d))))

(defn- take-op [b x indices axis]
  (let [d (long (nth (shape-of x) axis))]
    (if (host? indices)
      (let [i (host-index indices d)]
        (with-operands b :take [h x nil hi i :int32]
          (wrap b (Mlx/mlx_take_axis h hi (int axis) (stream b)))))
      (do (check-indices! indices d)
          (with-operands b :take [h x nil hi indices nil]
            (with-temps [t]
              (wrap b (Mlx/mlx_take_axis h (t (safe-indices b hi d)) (int axis) (stream b)))))))))

(defn- take-along-axis [b x indices axis]
  (check-indices! indices (nth (shape-of x) axis))
  (with-operands b :take-along-axis [h x nil hi indices nil]
    (with-temps [t]
      (wrap b (Mlx/mlx_take_along_axis h (t (safe-indices b hi (nth (shape-of x) axis))) (int axis) (stream b))))))

(defn- scatter-add [b x indices updates axis]
  (let [d (dtype-on b x)]
    (with-operands b :scatter-add [h x d hi indices nil hu updates d]
      (with-temps [t]
        (wrap b (Mlx/mlx_scatter_add_axis h (t (safe-indices b hi (nth (shape-of x) axis))) hu (int axis) (stream b)))))))

(defn- put-along-axis
  "mlx_put_along_axis, its indices clamped as scatter-add's; repeated ones
   are not checked (which value lands is MLX's, unspecified)."
  [b x indices values axis]
  (let [d (dtype-on b x)]
    (with-operands b :put-along-axis [h x d hi indices nil hv values d]
      (with-temps [t]
        (wrap b (Mlx/mlx_put_along_axis h (t (safe-indices b hi (nth (shape-of x) axis))) hv (int axis) (stream b)))))))

(defn- exact-route
  "op of the MlxArrays xs by its exact composite (gen.backend.mlx.exact).
   In native mode it is one custom VJP with the function's derivative, not
   the composite's graph (branches and Newton steps) differentiated."
  [b op xs]
  (let [{:keys [route vjp]} @exact-fns]
    (if-not (custom-vjps?)
      (route op xs)
      (first (custom-vjp b
                         (fn [ys] [(route op ys)])
                         (fn [ps cs os] (vjp op ps (first cs) (first os)))
                         xs)))))

(def ^:private cpu-compile
  "Whether mlx_compile works on the CPU device: nil until the first compile
   there probes it (cpu-compiles?), or -Dgen.mlx.cpu-compile=true|false."
  (atom (some-> (System/getProperty "gen.mlx.cpu-compile") Boolean/parseBoolean)))

(defn- cpu-compiles?
  "Does mlx_compile work on the CPU device here? MLX's CPU compile builds each
   fused kernel at run time with the system's g++, from a header the wheel
   ships, and GCC 13 on Linux rejects that header (it redeclares the built-in
   _Float128: the Jetson Thor, 2026-10-19). The first compile on the CPU
   device compiles and runs a two-op function; where MLX cannot build it,
   compile is the identity on the CPU device (the same values, unfused), and
   a warning says so once."
  [b]
  (or @cpu-compile
      (locking cpu-compile
        (if (some? @cpu-compile)
          @cpu-compile
          (let [probe (compile-fn b (fn [[x]] [(elementwise b :exp [(elementwise b :negative [x] :float32)] :float32)]) nil)
                works? (try (to-host (first (probe [(from-host b [0.5 1.5] [2] :float32)])))
                            true
                            (catch Exception e
                              (if (re-find #"Failed to compile function" (str (ex-message e) " " (some-> (ex-cause e) ex-message)))
                                false
                                (throw e))))]
            (when-not works?
              (binding [*out* *err*]
                (println "gen.backend.mlx: MLX's CPU compile cannot build kernels with this system's g++;"
                         "array/compile runs functions unfused on the CPU device (docs/array-api.md §13)")))
            (reset! cpu-compile works?))))))

(defn compiles?
  "Does array/compile compile on MLX backend b: always on the GPU, and on the
   CPU device where MLX can build its kernels (see cpu-compiles?)?"
  [b]
  (or (not= :cpu (device b)) (cpu-compiles? b)))

(defmacro ^:private guarded
  "body, with an MLX error raised outside an op's own handling as ex-info
   naming op (gen.array's contract: errors are ex-info with data)."
  [op & body]
  `(try ~@body
        (catch MlxException e# (throw (mlx-error ~op [] e#)))))

(deftype MlxBackend [^Config config]
  Configured
  (config [_] config)

  p/ICustomVjp
  (-custom-vjp [b f vjp args]
    (when (and *tracing-compile* *differentiating*)
      (throw (ex-info (str "custom-vjp inside array/compile under native gradients: MLX's compiled graphs"
                           " leave custom VJPs out, so the gradient would be wrong; differentiate outside"
                           " compile, or use the tape")
                      {:op :custom-vjp})))
    (if-not (custom-vjps?)
      (apply f args)
      (first (custom-vjp b
                         (fn [xs] [(apply f xs)])
                         (fn [ps cs os]
                           (mapv (fn [ct p] (as-array b (or ct (p/-full b (p/-shape b p) 0 (p/-dtype b p))) nil))
                                 (vjp ps (first os) (first cs)) ps))
                         (mapv #(as-array b % nil) args)))))

  p/ICheckpoint
  (-checkpoint [b f xs]
    ;; only while MLX differentiates: elsewhere, and inside array/compile
    ;; (which leaves custom functions out), f itself, the same gradient
    (if (and (custom-vjps?) *differentiating*)
      (first (guarded :checkpoint (checkpoint b (fn [ys] [(f ys)]) xs)))
      (f xs)))

  p/ICompiling
  (-compile [b f opts] (if (compiles? b) (guarded :compile (compile-fn b f opts)) f))

  p/IFused
  (-fused [b op args] (guarded op (nn/fused b op args)))

  p/IQuantized
  (-quantized [b op args] (guarded op (nn/quantized b op args)))

  p/ILoading
  (-load-gguf [b header] (guarded :load-gguf (io/load-gguf b header)))
  (-load-safetensors [b header] (guarded :load-safetensors (io/load-safetensors b header)))

  p/IOverwriting
  (-put-along-axis [b x indices values axis] (guarded :put-along-axis (put-along-axis b x indices values axis)))

  p/IHostFloats
  (-to-floats [_ x] (guarded :->floats (operands/to-floats x)))

  p/IReleasing
  (-release! [_ xs]
    ;; as a compiled call's donated inputs: each repointed at the husk, its
    ;; bytes off the count of new ones
    (doseq [x xs] (transforms/donate! x))
    (MlxRuntime/settleFresh))

  p/INativeGrad
  (-native-grad? [_] (= grad-mode :native))

  p/IBackend
  (-name [_] :mlx)
  (-defaults [_] (.-defaults config))
  (-lazy? [_] true)
  (-native-ops [_] #{:logaddexp})

  (-from-host [b values shape dtype] (from-host b values shape dtype))
  (-to-host [_ x] (to-host x))
  (-eval! [_ xs] (eval-arrays! (filter mlx-array? xs)))

  (-shape [_ x] (shape-of x))
  (-dtype [b x] (dtype-on b x))

  (-full [b shape value dtype]
    (check-dtype dtype)
    (if (mlx-array? value)
      ;; an array value cast and broadcast, as the batch tracer fills, not
      ;; mlx_full: MLX's Full::vjp multiplies the cotangent by the value, so
      ;; MLX's own gradients scaled a fill's by it (genclj-hf5z;
      ;; docs/research/mlx-upstream/issue-1-full-vjp.md)
      (with-operands b :full [hv value nil]
        (with-temps [t]
          (let [hd (if (= dtype (dtype-on b value)) hv (t (Mlx/mlx_astype hv (int (code dtype)) (stream b))))]
            (wrap b (Mlx/mlx_broadcast_to hd (int-array (map int shape)) (stream b))))))
      (with-operands b :full [hv value dtype]
        (wrap b (Mlx/mlx_full (int-array (map int shape)) hv (int (code dtype)) (stream b))))))
  (-arange [b start stop step dtype]
    (guarded :arange
      (wrap b (Mlx/mlx_arange (double start) (double stop) (double step) (int (code dtype)) (stream b)))))
  (-astype [b x dtype]
    (with-operands b :astype [h x nil]
      (let [from (dtype-on b x)]
        (wrap b (if (and (= :float (dt/kind from)) (= :int (dt/kind dtype)))
                  (java-int-cast b h from dtype)
                  (Mlx/mlx_astype h (int (code dtype)) (stream b)))))))

  (-elementwise [b op args dtype] (guarded op (elementwise b op args dtype)))
  (-reduce [b op x axes keep-dims?] (guarded op (reduce-op b op x axes keep-dims?)))
  (-scan [b op x axis reverse?] (guarded op (scan-op b op x axis reverse?)))

  (-reshape [b x shape] (reshape b x shape))
  (-transpose [b x perm] (transpose b x perm))
  (-broadcast-to [b x shape] (broadcast-to b x shape))
  (-concatenate [b xs axis] (concatenate b xs axis))
  (-slice [b x starts stops steps] (slice b x starts stops steps))

  (-take [b x indices axis] (guarded :take (take-op b x indices axis)))
  (-take-along-axis [b x indices axis] (guarded :take-along-axis (take-along-axis b x indices axis)))
  (-scatter-add [b x indices updates axis] (guarded :scatter-add (scatter-add b x indices updates axis)))

  (-matmul [b x y] (guarded :matmul (matmul b x y)))
  (-linalg [b op args] (guarded op (linalg b op args)))
  (-random-uniform [b key shape dtype] (guarded :random-uniform (random-uniform b key shape dtype)))
  (-value-and-grad [b f args argnums]
    (guarded :value-and-grad
      (if (= grad-mode :native)
        (binding [*differentiating* true] (native-value-and-grad b f args argnums))
        (@tape-value-and-grad b f args argnums)))))

(defn gpu?
  "Does MLX have a GPU here (Metal, or CUDA on Linux)?"
  []
  (with-device :gpu #(Mlx/mlx_device_is_available (long %))))

(defn make-backend
  "An MLX backend. Options: :device (:cpu or :gpu, default :gpu when MLX has
   one), :dtypes (32 or 64: the default float and int dtypes; 64 needs the
   CPU)."
  [{:keys [device dtypes] :or {dtypes 32}}]
  (let [device (or device (if (gpu?) :gpu :cpu))]
    (when (and (= device :gpu) (not (gpu?)))
      (throw (ex-info "MLX has no GPU here" {:device device})))
    (when (and (= device :gpu) (= 64 dtypes))
      (throw (ex-info "MLX's GPUs have no float64: use :dtypes 64 with :device :cpu" {:device device :dtypes dtypes})))
    (vreset! p/lazy-backends? true)
    (MlxBackend.
     (->Config device
               (with-device device #(Mlx/mlx_stream_new_thread_unsafe (long %)))
               (with-device :cpu #(Mlx/mlx_stream_new_thread_unsafe (long %)))
               (case (long dtypes)
                 32 {:float :float32 :int :int32}
                 64 {:float :float64 :int :int64})))))

(def backend
  "The MLX backend configured by the system properties gen.mlx.device and
   gen.mlx.dtypes (see the namespace doc)."
  (make-backend {:device (some-> (System/getProperty "gen.mlx.device") keyword)
                 :dtypes (Long/parseLong (System/getProperty "gen.mlx.dtypes" "32"))}))
