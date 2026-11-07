(ns gen.array.protocols
  "The backend contract behind gen.array. Only backend implementers use this
   namespace; model and inference code calls gen.array.

   A backend is a value implementing IBackend. Its arrays implement IArray so an
   operation can find the backend from its operands. Clojure numbers and
   booleans are 0-d arrays that belong to no backend: gen.array computes on them
   directly on the host and coerces them when they meet a backend array.

   Ops are named by keyword, not by protocol method. The op set is data (see
   docs/array-api.md §6), so one method covers every elementwise op, one every
   reduction, and so on. That keeps backends small, makes autodiff rules a table
   keyed by the same keywords, and lets the conformance suite iterate over ops.

   gen.array guarantees, before calling into a backend:
     - operands are arrays of this backend or host scalars (Double, Long,
       Boolean) already converted to the operand dtype; a backend must accept
       host scalars wherever it accepts an array (they are 0-d arrays, and the
       JVM backend represents every 0-d array that way),
     - operands share one dtype (promotion already applied), except for the
       condition of :where and the index arguments of gathers,
     - axes are non-negative and in range; shapes are vectors of longs,
     - operands' shapes fit the op (gen.array.shape's rules: they agree off
       the concatenation axis, are square matrices with right-hand sides of
       their rows, and so on), checked on the model's shapes; elementwise
       operands are the exception, which the backend checks as it
       broadcasts them (the batch tracer per particle, before it lifts),
       since an elementwise op is the hot path,
     - host data is parsed: a shape, its values, and a dtype.
   A backend that wraps another (the batch tracer, the tape) keeps these
   when it calls the backend it wraps: its shapes are the checked ones,
   lifted."
  {:api :stable
   :implementers "the backend contract, which a new array backend implements"})

(defonce ^{:doc "true once a backend with lazy values to evaluate has been made:
   the MLX backend sets it when one is made, so that gen.array/eval! walks
   nothing where every backend evaluates eagerly. The staging backend,
   lazy but with nothing to evaluate (its -eval! does nothing), leaves it."}
  lazy-backends?
  (volatile! false))

(defprotocol IArray
  (-backend [x] "The IBackend this array belongs to."))

(defn ^:no-doc not-a-collection
  "The error for clojure.core's collection functions on an array (get, nth,
   count), which would otherwise answer nil or a message that says
   nothing: an array is no Clojure collection (rule 6). Its host values
   are gen.array/->clj's, one element gen.array/item's, and an index on
   the backend gen.array/take's."
  [op x]
  (ex-info (str op " on an array (" (.getSimpleName (class x)) "): arrays are not Clojure collections."
                " Read host values with gen.array/->clj (one element: gen.array/item),"
                " or index on the backend with gen.array/take")
           {:reason :not-a-collection :op op}))

(defn ^:no-doc linalg-operands
  "[arrays options] of a -linalg op's args, as -linalg's doc lists them:
   its array arguments (one for :cholesky and :slogdet, two for :solve and
   :solve-triangular), and what follows (:solve-triangular's options).
   For backends that treat the arrays apart: the batch tracer lifts them,
   MLX's native gradients make them a custom VJP's inputs."
  [op args]
  (let [args (vec args)
        n (case op (:cholesky :slogdet) 1 (:solve :solve-triangular) 2)]
    [(subvec args 0 n) (subvec args n)]))

(defprotocol IBatchedValue
  "A value that holds one value per particle of a batch (gen.array.batch)."
  (-all-particles [x] "The array of every particle's value, batch axes first."))

(defprotocol ITrackedValue
  "A value being differentiated on a tape (gen.array.tape), which gen.array
   recognizes without depending on the tape."
  (-primal [x] "The value it tracks: an array of the backend the tape wraps, or a host scalar."))

(defprotocol IWrappingBackend
  "A backend whose arrays build on another backend's, such as the tape of
   reverse-mode autodiff: an op mixing its arrays with the inner backend's
   runs on the wrapping backend."
  (-inner [b] "The backend this one wraps."))

(defprotocol ICustomVjp
  "A backend that differentiates through a function by a VJP it is given
   (gen.array/custom-vjp): the tape, and MLX in native mode."
  (-custom-vjp [b f vjp args]
    "(apply f args), whose gradient comes from (vjp args out cotangent), a
     vector of one cotangent per argument (nil for none)."))

(defprotocol ^{:api :experimental} ICheckpoint
  "A backend whose gradients can recompute a function's intermediates in
   the backward pass instead of keeping them from the forward
   (gen.array/checkpoint): the tape, and MLX in native mode. Every other
   backend calls the function, which has the same values and gradients
   (rule 1: a checkpoint changes memory, never values). Experimental:
   outside the 1.x promise (DESIGN §9)."
  (^{:api :experimental} -checkpoint [b f xs]
    "(f xs), xs a vector of float arrays (b's, or values b wraps), f a
     function of such a vector returning one float array, whose gradient
     comes from f run again in the backward pass on xs, its intermediates
     made there and dropped after, not kept from this call."))

(defprotocol INativeGrad
  "A backend whose -value-and-grad differentiates by its own transformation
   (MLX's mlx_value_and_grad), calling f with arrays of its own rather than
   with gen.array.tape's tracked values. A backend wrapping it (the batch
   tracer) therefore runs inside the transformation, where under the tape
   the tape wraps it."
  (-native-grad? [b] "Does -value-and-grad differentiate natively?"))

(defprotocol ^:no-doc IStagedKey
  "A key whose words are an array of a staging backend, not host values: a
   device program's key (gen.array.stage). gen.random splits and folds it
   by that backend's ops, the same Threefry as a host key's, and a draw on
   it is the backend's (-random-uniform receives the key). A staged key is
   a value too (rule 2; DESIGN §9). Experimental."
  (^:no-doc -split-key [k n] "n keys from k, a vector.")
  (^:no-doc -fold-in-key [k data] "k folded with the non-negative integer data.")
  (^:no-doc -key-words [k] "The key's words: a uint32 array of shape [2]."))

(defprotocol ^:no-doc ILooping
  "A backend that records a loop as one node (the staging backend,
   gen.array.stage): gen.array/scan's (f carry xs n), f run once on the
   backend's values. Every other backend runs scan as a host loop.
   Experimental."
  (^:no-doc -scan-loop [b f carry xs n] "[the last carry, the ys stacked on a leading axis of n]."))

(defprotocol ^{:api :experimental} IFused
  "A backend with fused kernels for some of gen.array's neural-network
   composites (rms-norm, rope, sdpa, gated-delta, paged-attention), which
   it may compute in one op instead of the composite (MLX: its fast ops,
   and Metal kernels for gated-delta and paged-attention). gen.array asks
   only a
   backend whose arrays are the operands themselves: a wrapping backend
   (the tape, the batch tracer) and the staging backend implement no
   kernel, so the composite runs there, whose ops they know. A kernel
   must be observationally equivalent to the composite, gradients
   included where its backend differentiates it natively (rule 1).
   Experimental: outside the 1.x promise (DESIGN §9)."
  (^{:api :experimental} -fused [b op args]
    "op's value on args by a fused kernel, or nil to decline (gen.array
     then computes the composite). gen.array has checked the shapes and
     options, and given the float operands one dtype:
       :rms-norm [x weight eps]   weight an array of x's last axis, or nil
       :rope [x dims opts]        opts {:traditional? :base :scale :offset
                                  :freqs}: base a host number or nil (with
                                  freqs), offset a host integer or an int
                                  array of one per row of x's axis 0
       :sdpa [q k v opts]         opts {:scale :mask}: mask nil, :causal, a
                                  bool array (true attends) or a float array
                                  added to the scores, broadcasting to
                                  [B Hq Tq Tk]
       :gated-delta [q k v g beta state opts]
                                  opts {:chunk-size}; the value [y state']
       :paged-write [k-pool v-pool k v slots]
                                  k and v of the pools' dtype; the value
                                  [k-pool' v-pool']
       :paged-attention [q k-pool v-pool tables lens opts]
                                  opts {:scale}, a host number; q and the
                                  pools of one dtype; tables and lens int
                                  arrays, their values unchecked on a lazy
                                  backend"))

(defprotocol ^{:api :experimental} IQuantized
  "A backend with quantized weights: gen.array's quantize, dequantize,
   quantized-matmul and gather-qmm, in MLX's layouts (a uint32 array of
   packed values along the last axis, and per-group scales and biases).
   A backend without it makes those ops a loud error. Experimental:
   outside the 1.x promise (DESIGN §9)."
  (^{:api :experimental} -quantized [b op args]
    "op on args, gen.array having checked shapes, dtypes and options
     (opts {:mode :group-size :bits}, and the op's own below; x, scales
     and biases of one float dtype):
       :quantize [w opts]                        [wq scales biases]
       :dequantize [wq scales biases opts]       w, of :dtype (opts)
       :quantized-matmul [x wq scales biases opts]
                                                 x·wᵀ, or x·w with
                                                 :transpose? false
       :gather-qmm [x wq scales biases lhs rhs opts]
                                                 x's matrices at lhs times
                                                 w's at rhs (int arrays of
                                                 one shape), :transpose? and
                                                 :sorted? as for MLX's"))

(defprotocol ^{:api :experimental} ILoading
  "A backend that loads arrays from a model's files: gen.array/load-gguf.
   A backend without it makes load-gguf a loud error. Experimental:
   outside the 1.x promise (DESIGN §9)."
  (^{:api :experimental} -load-gguf [b header]
    "The arrays of one GGUF file (a split model's file), whose header
     gen.array has read and checked (gen.array.gguf/header and
     check-tensors: {:path :metadata :value-types :tensors}, a tensor
     {:name :type :shape :offset :bytes}): a map from each tensor's name to
     its value, as MLX's GGUF loader gives them:
       :f32, :i32    an array of float32, int32
       :f16          an array of float16, or float32 (the same values)
                     on a backend without float16
       :q8_0, :q4_0, :q4_1, :q5_0, :q5_1
                     [wq scales biases]: MLX's affine quantization in
                     groups of 32 (8 bits for Q8_0, 5 for Q5_0 and Q5_1,
                     4 for the others), wq uint32 [.. N K·bits/32], scales
                     and biases float32 [.. N K/32]: the file's scale d,
                     and its bias (Q4_1's, Q5_1's), -128·d (Q8_0's), -8·d
                     (Q4_0's) or -16·d (Q5_0's) computed in float32, so
                     that the values are the file's exactly
       :q4_k, :q5_k  [wq scales biases] likewise at 4 and 5 bits, a
                     sub-block of 32 a group: scale d·sc and bias -(dmin·m)
                     in float32
       :q6_k         an array of float32, d·sc·(q - 32)
     gen.array.gguf/host-tensor decodes every type so on the host."))

(defprotocol ^{:api :experimental} IOverwriting
  "A backend that writes values over an array's entries at indices:
   gen.array/put-along-axis. A backend without it makes put-along-axis a
   loud error. Experimental: outside the 1.x promise (DESIGN §9)."
  (^{:api :experimental} -put-along-axis [b x indices values axis]
    "x with values written at indices along axis, overwriting what was
     there: indices (an int array) and values have one shape, equal to
     x's on every axis but axis (gen.array has checked the shapes and
     given x and values one dtype). Repeated indices along a line of axis
     are the backend's to refuse; which value lands is unspecified."))

(defprotocol ^{:api :experimental} IHostFloats
  "A backend that reads an array's values into a float array at once:
   gen.array/->floats. A backend without it is read through -to-host.
   Experimental: outside the 1.x promise (DESIGN §9)."
  (^{:api :experimental} -to-floats [b x]
    "x's values, row-major, as a new float array (each rounded to float32;
     booleans 0 or 1). Forces evaluation."))

(defprotocol ICompiling
  "A backend that compiles functions of its arrays (MLX: mlx_compile)."
  (-compile [b f opts]
    "A function of a vector of leaf arrays (b's) returning a vector of leaf
     arrays: f, which takes and returns such vectors, traced once per input
     signature and replayed. opts: :shapeless?, and :donate (experimental),
     the set of input positions each call consumes, whose arrays the
     backend may release once the call has returned (gen.array/compile's
     :donate)."))

(defprotocol IBackend
  (-name [b]
    "Keyword naming the backend, e.g. :jvm or :mlx.")
  (-defaults [b]
    "Map of default dtypes: {:float :float64 :int :int64}.")
  (-lazy? [b]
    "Does this backend defer computation, so that reading a value forces
     evaluation? (The JVM backend is eager; MLX is lazy.)")
  (-native-ops [b]
    "The optional ops this backend implements as primitive elementwise op
     keywords, a set. gen.array looks up one, :logaddexp: listed, it is an
     -elementwise op; otherwise gen.array composes it from primitives.")

  ;; --- host boundary ---------------------------------------------------------
  (-from-host [b values shape dtype]
    "An array of shape and dtype from host values in row-major order, as
     gen.array.host/parse gives them: a double[] or a seq of numbers and
     booleans, as many as shape holds (one for shape [], a 0-d array).")
  (-to-host [b x]
    "Host value of x: a number or boolean for 0-d arrays, nested vectors
     otherwise. Forces evaluation.")
  (-eval! [b xs]
    "Force evaluation of the arrays in xs (a seq). A no-op on eager backends.
     Returns nil.")

  ;; --- metadata --------------------------------------------------------------
  (-shape [b x] "Shape of x as a vector of longs; [] for 0-d.")
  (-dtype [b x] "Dtype keyword of x.")

  ;; --- creation and casting --------------------------------------------------
  (-full [b shape value dtype] "Array of the given shape filled with value.")
  (-arange [b start stop step dtype] "1-d array [start, stop) by step.")
  (-astype [b x dtype] "x converted to dtype.")

  ;; --- computation (op keywords: docs/array-api.md §6) ------------------------
  (-elementwise [b op args dtype]
    "Apply elementwise op to args (a vector of 1-3 arrays), broadcasting with
     NumPy rules. dtype is the operands' common dtype (for :where, that of x and
     y; the condition is :bool); the result dtype is
     (gen.array.dtype/result-dtype op dtype). Covers unary math, binary
     arithmetic, comparisons, logical ops and :where (args [condition x y]).")
  (-reduce [b op x axes keep-dims?]
    "Reduce x over axes (a vector of distinct axes; empty reduces over no axis,
     the identity, as NumPy's axis=()). op is one of
     :sum :prod :max :min :argmax :argmin :all :any. :argmax and :argmin take a
     single axis.")
  (-scan [b op x axis reverse?]
    "Inclusive scan along axis. op is one of :cumsum :cumprod :cummax :cummin.")

  ;; --- shape -----------------------------------------------------------------
  (-reshape [b x shape])
  (-transpose [b x perm] "Permute axes; perm is a vector of all axes.")
  (-broadcast-to [b x shape])
  (-concatenate [b xs axis] "Join a vector of arrays along an existing axis.")
  (-slice [b x starts stops steps]
    "Strided slice; starts, stops and steps are vectors with one entry per axis.")

  ;; --- indexing --------------------------------------------------------------
  (-take [b x indices axis]
    "Gather: the result's axis is replaced by the shape of indices (an int array).")
  (-take-along-axis [b x indices axis]
    "Gather with indices broadcast against x on every other axis.")
  (-scatter-add [b x indices updates axis]
    "x with updates added at indices along axis, accumulating repeats: the
     transpose of take-along-axis. indices (an int array) and updates have one
     shape, equal to x's on every axis but axis.")

  ;; --- linear algebra --------------------------------------------------------
  (-matmul [b x y] "Matrix product with NumPy batching rules.")
  (-linalg [b op args]
    "Decompositions and solves over the last two axes, batched over the rest:
       :cholesky [a]                 lower factor (NaN where a is not positive definite)
       :solve-triangular [a rhs {:lower? :transpose?}]   rhs is [.. n k]
       :solve [a rhs]                rhs is [.. n k]
       :slogdet [a]                  returns [sign log|det|] (sign 0, -Inf if singular)")

  ;; --- randomness and differentiation ---------------------------------------
  (-random-uniform [b key shape dtype]
    "Uniform [0, 1) array of shape and dtype, a pure function of key (a
     gen.random key value, not an array). Every other sampler in gen.random is
     built from this one primitive.")
  (-value-and-grad [b f args argnums]
    "Call f on args (a vector) and return [value grads], where grads holds the
     gradient of the scalar value with respect to each position in argnums.
     Positions may hold trees (nested maps and vectors of arrays and numbers);
     each gradient has the same tree structure."))

(defn ^:no-doc quantized
  "-quantized's op on b, or the loud error naming b when b has no
   quantized ops (gen.array, and backends that pass the op on)."
  [b op args]
  (if (instance? gen.array.protocols.IQuantized b)
    (-quantized b op args)
    (throw (ex-info (str (name op) ": the " (name (-name b)) " backend has no quantized ops"
                         (when (= :stage (-name b)) " (not stageable)"))
                    {:reason :unsupported :op op :backend (-name b)}))))

(defn ^:no-doc load-gguf
  "-load-gguf on b, or the loud error naming b when b loads no files."
  [b header]
  (if (instance? gen.array.protocols.ILoading b)
    (-load-gguf b header)
    (throw (ex-info (str "load-gguf: the " (name (-name b)) " backend loads no GGUF files")
                    {:reason :unsupported :op :load-gguf :backend (-name b)}))))

(defn ^:no-doc put-along-axis
  "-put-along-axis on b, or the loud error naming b when b overwrites
   nothing (gen.array, and backends that pass the op on)."
  [b x indices values axis]
  (if (instance? gen.array.protocols.IOverwriting b)
    (-put-along-axis b x indices values axis)
    (throw (ex-info (str "put-along-axis: the " (name (-name b)) " backend has no overwriting ops"
                         (when (= :stage (-name b)) " (not stageable)"))
                    {:reason :unsupported :op :put-along-axis :backend (-name b)}))))
