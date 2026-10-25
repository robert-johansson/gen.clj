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

(defonce ^{:doc "true once a lazy backend (-lazy? true) has been made: a backend
   sets it, so that gen.array/eval! walks nothing where all are eager."}
  lazy-backends?
  (volatile! false))

(defprotocol IArray
  (-backend [x] "The IBackend this array belongs to."))

(defprotocol IBatchedValue
  "A value that holds one value per particle of a batch (gen.array.batch)."
  (-all-particles [x] "The array of every particle's value, batch axes first."))

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

(defprotocol INativeGrad
  "A backend whose -value-and-grad differentiates by its own transformation
   (MLX's mlx_value_and_grad), calling f with arrays of its own rather than
   with gen.array.tape's tracked values. A backend wrapping it (the batch
   tracer) therefore runs inside the transformation, where under the tape
   the tape wraps it."
  (-native-grad? [b] "Does -value-and-grad differentiate natively?"))

(defprotocol ICompiling
  "A backend that compiles functions of its arrays (MLX: mlx_compile)."
  (-compile [b f opts]
    "A function of a vector of leaf arrays (b's) returning a vector of leaf
     arrays: f, which takes and returns such vectors, traced once per input
     signature and replayed. opts: :shapeless?."))

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
