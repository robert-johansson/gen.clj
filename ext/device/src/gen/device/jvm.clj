(ns gen.device.jvm
  "A staged program (gen.device.stage) as JVM code: the third lowering of
   bean genclj-a1hu, beside CUDA C and StableHLO (bean genclj-wbm7,
   research prototype, ext/device).

   (source program) scalarizes the program as gen.device.cuda does (an
   array is a vector of scalar expressions, so reshapes, broadcasts and
   slices cost nothing; a counted scan is a for loop whose carries are
   variables) and writes it as one Java class. HotSpot compiles no method
   over 8000 bytes of bytecode (DontCompileHugeMethods), and a chain's
   step is thousands of statements, so the statements are cut into
   methods of at most gen.jvm.chunk statements (300 by default): a value
   used only in its own method is a local, any other a field of the run's
   state object, and a loop calls its body's methods in order.

   (compile src) compiles the class in memory with the JDK's compiler
   (javax.tools, no dependency) and returns {:run (fn [key-words]) ...}:
   one run of the program on a key's two words, its outputs as Java arrays
   (float32 a float[], uint32 and int32 an int[], bool a boolean[]), each
   laid out flat in row-major order, a scan's ys [length …].

   float32 arithmetic is Java's (no fused multiply-add); exp, log and the
   other transcendental functions are Math's double functions rounded to
   float, and erfinv is Giles' single-precision approximation, so float32
   draws equal the handler's to float32 rounding, not bit for bit. A
   float64 program (staged under gen.device.stage/*float* :float64) is
   computed as the JVM backend computes: Math's functions, and erfinv and
   erf the backend's own (gen.array.special).

   Refused, loudly: data inputs (a program of one key only), arrays past
   gen.jvm.max-elements (4096), a scan over xs or inside a scan, a scan's
   ys read in the program, batched matmul, float32 erf."
  {:api :internal}
  (:refer-clojure :exclude [compile])
  (:require [clojure.string :as str])
  (:import [java.io ByteArrayOutputStream]
           [java.lang.invoke MethodHandle MethodHandles MethodType]
           [java.net URI]
           [javax.tools DiagnosticCollector ForwardingJavaFileManager JavaCompiler JavaCompiler$CompilationTask
            JavaFileObject$Kind SimpleJavaFileObject ToolProvider]))

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
;; Java text
;; ---------------------------------------------------------------------------

(def ^:private jtypes {:float32 "float" :float64 "double" :uint32 "int" :int32 "int" :bool "boolean"})

(defn- jtype [dtype]
  (or (jtypes dtype) (throw (ex-info (str "JVM lowering: no " dtype " (float32, float64, uint32, int32 and bool only)") {:dtype dtype}))))

(def ^:private array-types {:float32 "float[]" :float64 "double[]" :uint32 "int[]" :int32 "int[]" :bool "boolean[]"})

(def ^:private float-type? #{:float32 :float64})

(defn- lit
  "A host value as a Java literal of dtype, exactly."
  [v dtype]
  (case dtype
    :float32 (let [f (unchecked-float (if (boolean? v) (if v 1.0 0.0) (double v)))]
               (if (or (Float/isNaN f) (Float/isInfinite f))
                 (format "Float.intBitsToFloat(0x%08x)" (Float/floatToRawIntBits f))
                 (str (Float/toString f) "f")))
    :float64 (let [d (if (boolean? v) (if v 1.0 0.0) (double v))]
               (if (or (Double/isNaN d) (Double/isInfinite d))
                 (format "Double.longBitsToDouble(0x%016xL)" (Double/doubleToRawLongBits d))
                 (Double/toString d)))
    :uint32 (format "0x%08x" (bit-and (long (if (boolean? v) (if v 1 0) v)) 0xFFFFFFFF))
    :int32 (str (long (if (boolean? v) (if v 1 0) v)))
    :bool (str (if (boolean? v) v (not (zero? (double v)))))
    (jtype dtype)))

(defn- int-literal
  "The value of a uint32 or int32 literal e, or nil when e is not one."
  [^String e]
  (cond (re-matches #"0x[0-9a-f]{8}" e) (Long/parseLong (subs e 2) 16)
        (re-matches #"-?[0-9]+" e) (Long/parseLong e)))

(def ^:private helpers
  "Static helpers of every program class: the ops whose Java operators do
   not have gen.array's semantics, and erfinv."
  "
  static float g_sign(float a) { return a > 0.0f ? 1.0f : (a < 0.0f ? -1.0f : a); }
  static double g_sign(double a) { return a > 0.0 ? 1.0 : (a < 0.0 ? -1.0 : a); }
  static double g_mod(double a, double b) { double r = a % b; return (r != 0.0 && ((r < 0.0) != (b < 0.0))) ? r + b : r; }
  // float64's erfinv and erf: the JVM backend's own (gen.array.special), so draws can be its bit for bit
  static final clojure.lang.IFn.DD ERFINV = (clojure.lang.IFn.DD) ((clojure.lang.IDeref) clojure.java.api.Clojure.var(\"gen.array.special\", \"erfinv\")).deref();
  static final clojure.lang.IFn.DD ERF = (clojure.lang.IFn.DD) ((clojure.lang.IDeref) clojure.java.api.Clojure.var(\"gen.array.special\", \"erf\")).deref();
  static int g_sign(int a) { return (a > 0 ? 1 : 0) - (a < 0 ? 1 : 0); }
  static float g_mod(float a, float b) { float r = a % b; return (r != 0.0f && ((r < 0.0f) != (b < 0.0f))) ? r + b : r; }
  static int g_mod(int a, int b) { int r = a % b; return (r != 0 && ((r < 0) != (b < 0))) ? r + b : r; }
  static int g_umax(int a, int b) { return Integer.compareUnsigned(a, b) >= 0 ? a : b; }
  static int g_umin(int a, int b) { return Integer.compareUnsigned(a, b) <= 0 ? a : b; }
  static int g_shl(int x, int s) { return Integer.compareUnsigned(s, 32) < 0 ? x << s : 0; }
  static int g_shr(int x, int s) { return Integer.compareUnsigned(s, 32) < 0 ? x >>> s : 0; }
  // Giles' single-precision erfinv (gen.array.special's starting point)
  static float g_erfinv(float x) {
    float w = -(float) Math.log((1.0f - x) * (1.0f + x));
    float p;
    if (w < 5.0f) {
      w = w - 2.5f;
      p = 2.81022636e-08f;
      p = 3.43273939e-07f + p * w; p = -3.5233877e-06f + p * w; p = -4.39150654e-06f + p * w;
      p = 0.00021858087f + p * w; p = -0.00125372503f + p * w; p = -0.00417768164f + p * w;
      p = 0.246640727f + p * w; p = 1.50140941f + p * w;
    } else {
      w = (float) Math.sqrt(w) - 3.0f;
      p = -0.000200214257f;
      p = 0.000100950558f + p * w; p = 0.00134934322f + p * w; p = -0.00367342844f + p * w;
      p = 0.00573950773f + p * w; p = -0.0076224613f + p * w; p = 0.00943887047f + p * w;
      p = 1.00167406f + p * w; p = 2.83297682f + p * w;
    }
    return p * x;
  }
")

(def ^:private threefry
  "Threefry-2x32-20 of the counter [c0 c1] under the key [k0 k1], the two
   words packed in a long (gen.random.threefry's scheme, unrolled)."
  (let [rot [13 15 26 6 17 29 16 24]
        ks ["k0" "k1" "k2"]]
    (str "  static long g_threefry(int k0, int k1, int c0, int c1) {\n"
         "    int k2 = k0 ^ k1 ^ 0x1BD11BDA;\n"
         "    int x0 = c0 + k0, x1 = c1 + k1;\n"
         (apply str (for [i (range 5)]
                      (str (apply str (for [j (range 4)]
                                        (str "    x0 += x1; x1 = Integer.rotateLeft(x1, " (nth rot (+ (* (mod i 2) 4) j)) ") ^ x0;\n")))
                           "    x0 += " (nth ks (mod (inc i) 3)) "; x1 += " (nth ks (mod (+ i 2) 3)) " + " (inc i) ";\n")))
         "    return ((long) x0 << 32) | (x1 & 0xFFFFFFFFL);\n"
         "  }\n")))

(def ^:private float-unary
  {:exp "Math.exp" :log "Math.log" :expm1 "Math.expm1" :log1p "Math.log1p" :sqrt "Math.sqrt" :sin "Math.sin"
   :cos "Math.cos" :tan "Math.tan" :tanh "Math.tanh" :sinh "Math.sinh" :cosh "Math.cosh" :arcsin "Math.asin"
   :arccos "Math.acos" :arctan "Math.atan" :floor "Math.floor" :ceil "Math.ceil" :round "Math.rint"})

(defn- unary-expr [op dtype a]
  (cond (= op :negative) (str "(- " a ")")
        (= op :logical-not) (str "(!" a ")")
        (= op :sign) (str "g_sign(" a ")")
        (= op :isnan) (if (float-type? dtype) (str "(" a " != " a ")") "false")
        (= op :isinf) (case dtype
                        :float32 (str "(Math.abs(" a ") == Float.POSITIVE_INFINITY)")
                        :float64 (str "(Math.abs(" a ") == Double.POSITIVE_INFINITY)")
                        "false")
        (= op :abs) (if (= dtype :uint32) a (str "Math.abs(" a ")"))
        (and (= op :erfinv) (= dtype :float32)) (str "g_erfinv(" a ")")
        (and (= op :erfinv) (= dtype :float64)) (str "ERFINV.invokePrim(" a ")")
        (and (= op :erf) (= dtype :float64)) (str "ERF.invokePrim(" a ")")
        (and (float-unary op) (= dtype :float32)) (str "((float) " (float-unary op) "(" a "))")
        (and (float-unary op) (= dtype :float64)) (str (float-unary op) "(" a ")")
        (#{:floor :ceil :round} op) a
        :else (throw (ex-info (str "JVM lowering: no " op " on " dtype " yet") {:op op :dtype dtype}))))

(def ^:private infix
  {:add "+" :subtract "-" :multiply "*" :divide "/" :logical-and "&" :logical-or "|" :bit-xor "^" :bit-or "|"
   :bit-and "&" :less "<" :less-equal "<=" :greater ">" :greater-equal ">=" :equal "==" :not-equal "!="})

(def ^:private ordered #{:less :less-equal :greater :greater-equal})

(defn- binary-expr [op dtype a b]
  (cond (and (= dtype :uint32) (ordered op)) (str "(Integer.compareUnsigned(" a ", " b ") " (infix op) " 0)")
        (and (= dtype :uint32) (= op :divide)) (str "Integer.divideUnsigned(" a ", " b ")")
        (infix op) (str "(" a " " (infix op) " " b ")")
        (= op :maximum) (if (= dtype :uint32) (str "g_umax(" a ", " b ")") (str "Math.max(" a ", " b ")"))
        (= op :minimum) (if (= dtype :uint32) (str "g_umin(" a ", " b ")") (str "Math.min(" a ", " b ")"))
        (= op :remainder) (if (= dtype :uint32) (str "Integer.remainderUnsigned(" a ", " b ")") (str "g_mod(" a ", " b ")"))
        (#{:shift-left :shift-right-logical} op)
        (let [s (int-literal b)
              [jop helper] (if (= op :shift-left) ["<<" "g_shl"] [">>>" "g_shr"])]
          (cond (nil? s) (str helper "(" a ", " b ")")
                (< -1 s 32) (str "(" a " " jop " " s ")")
                :else "0"))
        (and (= op :power) (= dtype :float32)) (str "((float) Math.pow(" a ", " b "))")
        (and (= op :arctan2) (= dtype :float32)) (str "((float) Math.atan2(" a ", " b "))")
        (and (= op :power) (= dtype :float64)) (str "Math.pow(" a ", " b ")")
        (and (= op :arctan2) (= dtype :float64)) (str "Math.atan2(" a ", " b ")")
        :else (throw (ex-info (str "JVM lowering: no " op " on " dtype " yet") {:op op :dtype dtype}))))

(defn- convert-expr [from to a]
  (cond (= from to) a
        (= to :bool) (str "(" a (case from :float32 " != 0.0f)" :float64 " != 0.0)" " != 0)"))
        (= from :bool) (str "(" a (case to :float32 " ? 1.0f : 0.0f)" :float64 " ? 1.0 : 0.0)" " ? 1 : 0)"))
        (and (float-type? from) (= to :int32)) (str "((int) " a ")")
        (and (float-type? from) (= to :uint32)) (str "((int) (long) " a ")")
        (= from :int32) (str "((" (jtype to) ") " a ")")
        (= from :uint32) (str "((" (jtype to) ") (" a " & 0xFFFFFFFFL))")
        (and (float-type? from) (float-type? to)) (str "((" (jtype to) ") " a ")")
        (#{[:int32 :uint32] [:uint32 :int32]} [from to]) a
        :else (throw (ex-info (str "JVM lowering: no conversion from " from " to " to) {:from from :to to}))))

(def ^:private bit-ops #{:bit-xor :bit-or :bit-and :shift-left :shift-right-logical})
(def ^:private comparisons #{:less :less-equal :greater :greater-equal :equal :not-equal})
(def ^:private binary-ops (into (set (keys infix)) [:maximum :minimum :remainder :power :arctan2]))
(def ^:private unary-ops (into (set (keys float-unary)) [:negative :logical-not :sign :isnan :isinf :abs :erfinv]))

(def ^:private reduce-init {:sum 0 :prod 1 :max Double/NEGATIVE_INFINITY :min Double/POSITIVE_INFINITY :all true :any false})

(defn- reduce-expr [op dtype a b]
  (case op
    :max (binary-expr :maximum dtype a b)
    :min (binary-expr :minimum dtype a b)
    :sum (str "(" a " + " b ")")
    :prod (str "(" a " * " b ")")
    :all (str "(" a " & " b ")")
    :any (str "(" a " | " b ")")
    (throw (ex-info (str "JVM lowering: no reduction " op " yet") {:op op}))))

;; ---------------------------------------------------------------------------
;; Scalarization: statements
;;
;; The emitter collects statements, not text, so that they can be cut into
;; methods: [:decl dtype name expr], [:set name expr] (a carry's update),
;; [:store output index-expr expr] and [:loop it length body].
;; ---------------------------------------------------------------------------

(defn- emitter
  ([] {:stmts (volatile! []) :n (volatile! 0)})
  ([em] {:stmts (volatile! []) :n (:n em)}))

(defn- stmt! [em s] (vswap! (:stmts em) conj s))

(defn- fresh [em prefix] (str prefix (vswap! (:n em) inc)))

(defn- simple? [^String e]
  (boolean (re-matches #"[A-Za-z_][A-Za-z0-9_]*|-?[0-9][0-9.eE+xa-f-]*f?|Float\.intBitsToFloat\(0x[0-9a-f]+\)|Double\.longBitsToDouble\(0x[0-9a-f]+L\)" e)))

(defn- bind!
  "A Java name for the expression e of dtype (e itself when it is a name or
   a literal)."
  [em dtype ^String e]
  (if (simple? e)
    e
    (let [v (fresh em "v")]
      (stmt! em [:decl dtype v e])
      v)))

(defn- value [shape dtype el] {:shape (vec shape) :dtype dtype :el (vec el)})

(defn- max-elements ^long [] (Long/getLong "gen.jvm.max-elements" 4096))

(declare emit-graph! emit-scan!)

(defn- operand
  "An operand of a node: its value, or a constant's (a {:const x} of dtype)."
  [env a dtype]
  (if (map? a)
    (let [d (if (boolean? (:const a)) :bool dtype)] (value [] d [(lit (:const a) d)]))
    (let [v (or (env a) (throw (ex-info "JVM lowering: an operand with no value" {:id a})))]
      (when (:ys v) (throw (ex-info "JVM lowering: a scan's ys read in the program (only its outputs may be ys)" {:id a})))
      v)))

(defn- elementwise [em shape dtype operands f]
  (let [maps (mapv #(broadcast-map shape (:shape %)) operands)]
    (value shape dtype (mapv (fn [i] (bind! em dtype (apply f (map (fn [o m] (nth (:el o) (nth m i))) operands maps))))
                             (range (size shape))))))

(defn- reduce-value [em {:keys [shape el]} op axes out-shape out-dtype]
  (when-not (contains? reduce-init op) (throw (ex-info (str "JVM lowering: no reduction " op " yet") {:op op})))
  (let [axes (set axes)
        groups (group-by (fn [i] (let [idx (unravel i shape)] (vec (keep-indexed (fn [k x] (when-not (axes k) x)) idx))))
                         (range (size shape)))
        kept (vec (keep-indexed (fn [k d] (when-not (axes k) d)) shape))]
    (value out-shape out-dtype
           (mapv (fn [o] (let [is (get groups (unravel o kept))]
                           (if (empty? is)
                             (lit (reduce-init op) out-dtype)
                             (reduce (fn [acc i] (bind! em out-dtype (reduce-expr op out-dtype acc (nth el i))))
                                     (nth el (first is)) (rest is)))))
                 (range (size kept))))))

(defn- permuted [shape perm in-shape el]
  (mapv (fn [i] (let [oi (unravel i shape)
                      ii (reduce (fn [v k] (assoc v (nth perm k) (nth oi k))) (vec (repeat (count shape) 0)) (range (count shape)))]
                  (nth el (ravel ii in-shape))))
        (range (size shape))))

(defn- emit-node!
  "env with node's value: {:shape :dtype :el} (Java expressions), or a
   scan's or its ys result's own entry."
  [em env {:keys [id op args shape dtype attrs] :as node} ctx]
  (when (> (size (or shape [])) (max-elements))
    (throw (ex-info (str "JVM lowering: an array of " (size shape) " elements, past gen.jvm.max-elements") {:node (dissoc node :attrs)})))
  (let [in (:in attrs)
        put #(assoc env id %)
        arg (fn [i d] (operand env (nth args i) d))]
    (cond
      (= op :param) (if (env id) env (throw (ex-info "JVM lowering: an unbound param" {:node node})))
      (= op :constant) (let [vs (:values attrs)]
                         (put (value shape dtype (mapv #(lit (nth vs (if (= 1 (count vs)) 0 %)) dtype) (range (size shape))))))
      (= op :iota) (put (value shape dtype (mapv #(lit % dtype) (range (size shape)))))
      (= op :broadcast) (let [o (arg 0 dtype) m (broadcast-map shape (:shape o))]
                          (put (value shape dtype (mapv #(nth (:el o) %) m))))
      (= op :convert) (let [o (arg 0 nil)]
                        (put (if (= (:dtype o) dtype)
                               (assoc o :shape shape)
                               (value shape dtype (mapv #(bind! em dtype (convert-expr (:dtype o) dtype %)) (:el o))))))
      (= op :bitcast) (let [o (arg 0 nil)
                            f (case [(:dtype o) dtype] [:uint32 :float32] "Float.intBitsToFloat" [:float32 :uint32] "Float.floatToRawIntBits"
                                    (throw (ex-info "JVM lowering: a bitcast other than uint32 ↔ float32" {:node node})))]
                        (put (value shape dtype (mapv #(bind! em dtype (str f "(" % ")")) (:el o)))))
      (= op :stop-gradient) (put (assoc (arg 0 in) :shape shape))
      (unary-ops op) (put (elementwise em shape dtype [(arg 0 in)] #(unary-expr op in %)))
      (bit-ops op) (put (elementwise em shape :uint32 [(arg 0 :uint32) (arg 1 :uint32)] #(binary-expr op :uint32 %1 %2)))
      (comparisons op) (put (elementwise em shape :bool [(arg 0 in) (arg 1 in)] #(binary-expr op in %1 %2)))
      (binary-ops op) (put (elementwise em shape dtype [(arg 0 in) (arg 1 in)] #(binary-expr op in %1 %2)))
      (= op :where) (put (elementwise em shape dtype [(arg 0 :bool) (arg 1 in) (arg 2 in)] #(str "(" %1 " ? " %2 " : " %3 ")")))
      (= op :reduce) (put (reduce-value em (arg 0 nil) (:op attrs) (:axes attrs) shape dtype))
      (= op :reshape) (put (assoc (arg 0 nil) :shape shape))
      (= op :transpose) (let [o (arg 0 nil)] (put (value shape dtype (permuted shape (:perm attrs) (:shape o) (:el o)))))
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
                           _ (when (or (> (count sx) 2) (> (count sy) 2)) (throw (ex-info "JVM lowering: batched matmul" {:node node})))
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
      :else (throw (ex-info (str "JVM lowering: no lowering of " op " yet") {:node (dissoc node :attrs)})))))

(defn- emit-scan!
  "A counted scan as a loop statement: {:carries [values after the loop]
   :n-carry}. ys the program returns are stored to their outputs in the
   loop (ctx's :ys-out maps [scan id, ys index] to an output)."
  [em env {:keys [id args attrs]} {:keys [ys-out nested?] :as ctx}]
  (when nested? (throw (ex-info "JVM lowering: a scan inside a scan, not yet" {})))
  (let [{:keys [body length n-carry n-x]} attrs
        {:keys [graph carries captures carry-out y-out]} body]
    (when (pos? n-x) (throw (ex-info "JVM lowering: a scan over xs (use a counted scan, xs nil)" {:n-x n-x})))
    (let [inits (mapv #(env %) (subvec args 0 n-carry))
          caps (mapv #(env %) (subvec args (+ n-carry n-x)))
          vars (mapv (fn [{:keys [shape dtype el]}]
                       (value shape dtype (mapv (fn [e] (let [v (fresh em "c")] (stmt! em [:decl dtype v e]) v)) el)))
                     inits)
          it (fresh em "it")
          bem (emitter em)
          benv (emit-graph! bem graph (merge (zipmap carries vars) (zipmap captures caps)) (assoc ctx :nested? true))]
      (doseq [[j y] (map-indexed vector y-out)
              :let [k (get ys-out [id j])]
              :when k
              :let [{:keys [shape el]} (benv y) n (size shape)]
              [e x] (map-indexed vector el)]
        (stmt! bem [:store k (str it " * " n " + " e) x]))
      (let [outs (mapv #(:el (benv %)) carry-out)
            ;; every new value bound before any carry changes, as a carry's
            ;; update may read another carry
            tmps (mapv (fn [v o] (mapv (fn [e] (if (re-matches #"c[0-9]+" e) (bind! bem (:dtype v) (str "(" e ")")) e)) o)) vars outs)]
        (doseq [[v t] (map vector vars tmps) [c x] (map vector (:el v) t)]
          (when-not (= c x) (stmt! bem [:set c x]))))
      (stmt! em [:loop it length @(:stmts bem)])
      {:carries vars :n-carry n-carry})))

(defn- emit-graph! [em nodes env ctx]
  (reduce (fn [env node] (emit-node! em env node ctx)) env nodes))

;; ---------------------------------------------------------------------------
;; Methods: statements cut into chunks, values crossing chunks as fields
;; ---------------------------------------------------------------------------

(defn- chunk-size ^long [] (Long/getLong "gen.jvm.chunk" 600))

(defn- names-in [^String e] (re-seq #"\b(?:v|c)[0-9]+\b" e))

(defn- stmt-uses [[kind & xs]]
  (case kind
    :decl (names-in (nth xs 2))
    :set (cons (first xs) (names-in (second xs)))
    :store (concat (names-in (nth xs 1)) (names-in (nth xs 2)))))

(defn- chunks
  "The program's statements as methods: [{:name :it (the loop counter it
   takes, or nil) :stmts}], and the run's order of calls: a seq of method
   names and [:loop it length [method names]]."
  [stmts chunk]
  (let [n (volatile! 0)
        cut (fn [it ss] (mapv (fn [part] {:name (str (if it "b" "s") (vswap! n inc)) :it it :stmts (vec part)})
                              (partition-all chunk ss)))
        segments (partition-by #(= :loop (first %)) stmts)]
    (reduce (fn [{:keys [methods order]} seg]
              (if (= :loop (ffirst seg))
                (reduce (fn [{:keys [methods order]} [_ it length body]]
                          (when (some #(= :loop (first %)) body) (throw (ex-info "JVM lowering: a scan inside a scan, not yet" {})))
                          (let [ms (cut it body)]
                            {:methods (into methods ms) :order (conj order [:loop it length (mapv :name ms)])}))
                        {:methods methods :order order} seg)
                (let [ms (cut nil seg)]
                  {:methods (into methods ms) :order (into order (map :name ms))})))
            {:methods [] :order []} segments)))

(defn- fields
  "{name dtype} of the values that cross methods: declared in one method
   and used (or set) in another."
  [methods]
  (let [declared (into {} (for [{:keys [name stmts]} methods [kind dtype v] stmts :when (= kind :decl)] [v [name dtype]]))]
    (into {} (for [{:keys [name stmts]} methods s stmts u (stmt-uses s)
                   :let [[home dtype] (declared u)]
                   :when (and home (not= home name))]
               [u dtype]))))

(defn- java-method [{:keys [name it stmts]} fields]
  (let [sb (StringBuilder.)]
    (.append sb (str "  void " name "(" (when it (str "int " it)) ") {\n"))
    (doseq [[kind a b c] stmts]
      (.append sb "    ")
      (.append sb ^String (case kind
                            :decl (if (contains? fields b) (str b " = " c ";") (str (jtype a) " " b " = " c ";"))
                            :set (str a " = " b ";")
                            :store (str "out" a "[i * SIZE" a " + " b "] = " c ";")))
      (.append sb "\n"))
    (.append sb "  }\n")
    (.toString sb)))

(defn source
  "The Java text of a class (named class-name) running program (gen.device.
   stage's, of one key):
   - (run key0 key1 out0 …): one run on the key's words;
   - (runRange r0 r1 from to out0 …): runs from ≤ i < to, run i on
     (rng/split [r0 r1] n)[i] (Threefry of the counter [0 i] under the
     root, derived in the loop), its outputs at offset i·SIZEk of each
     outk;
   - an instance, (new class n), is a LongConsumer whose accept(root) is
     runRange over all n on its own arrays, and a Supplier of them.
   Returns {:source :class-name :outputs [{:shape :dtype} per output, a
   scan's ys [length …]] :methods (their count) :statements}. chunk: the
   statements per method (gen.jvm.chunk, 600 by default)."
  ([program class-name] (source program class-name (chunk-size)))
  ([{:keys [graph inputs outputs]} class-name chunk]
   (when-not (= inputs [[[2] :uint32]])
     (throw (ex-info "JVM lowering: a program of one key (data inputs not yet)" {:inputs inputs})))
   (let [em (emitter)
         by-id (into {} (map (juxt :id identity) graph))
         ys-out (into {} (keep-indexed (fn [k o] (let [{:keys [op args attrs]} (by-id o)]
                                                   (when (= op :result)
                                                     (let [s (by-id (first args)) nc (get-in s [:attrs :n-carry])]
                                                       (when (>= (:index attrs) nc) [[(:id s) (- (:index attrs) nc)] k])))))
                                       outputs))
         env0 (into {} (map (fn [{:keys [id]}] [id (value [2] :uint32 ["key0" "key1"])]) (filter #(= :param (:op %)) graph)))
         env (emit-graph! em graph env0 {:ys-out ys-out})
         outs (mapv env outputs)
         _ (doseq [[k {:keys [ys el]}] (map-indexed vector outs) :when (nil? ys) [e x] (map-indexed vector el)]
             (stmt! em [:store k (str e) x]))
         stmts @(:stmts em)
         {:keys [methods order]} (chunks stmts chunk)
         fs (fields methods)
         out-specs (mapv (fn [{:keys [shape dtype ys]} o] {:shape (if ys (:shape (by-id o)) shape) :dtype dtype}) outs outputs)
         ks (range (count out-specs))
         sig (str/join (map (fn [k {:keys [dtype]}] (str ", " (array-types dtype) " out" k)) ks out-specs))
         outs-args (str/join (map #(str ", out" %) ks))
         set-outs (str/join (map #(str "    p.out" % " = out" % ";\n") ks))
         calls (fn [indent]
                 (str/join (for [o order]
                             (if (string? o)
                               (str indent "p." o "();\n")
                               (let [[_ it length ms] o]
                                 (str indent "for (int " it " = 0; " it " < " length "; ++" it ") {\n"
                                      (str/join (map #(str indent "  p." % "(" it ");\n") ms))
                                      indent "}\n"))))))
         new-array (fn [{:keys [dtype]} k] (str "new " (str/replace (array-types dtype) "[]" "") "[n * SIZE" k "]"))]
     {:source (str "public final class " class-name
                   " implements java.util.function.LongConsumer, java.util.function.Supplier<Object[]> {\n"
                   helpers threefry
                   (str/join (map (fn [k {:keys [shape]}] (str "  static final int SIZE" k " = " (size shape) ";\n")) ks out-specs))
                   "  int key0, key1, i, n;\n"
                   (str/join (map (fn [k {:keys [dtype]}] (str "  " (array-types dtype) " out" k ";\n")) ks out-specs))
                   (str/join (map (fn [[v dtype]] (str "  " (jtype dtype) " " v ";\n")) (sort-by key fs)))
                   (str/join (map #(java-method % fs) methods))
                   "  public " class-name "() {}\n"
                   "  public " class-name "(int n) {\n    this.n = n;\n"
                   (str/join (map (fn [k o] (str "    out" k " = " (new-array o k) ";\n")) ks out-specs))
                   "  }\n"
                   "  public void accept(long root) { runRange((int) (root >>> 32), (int) root, 0, n" outs-args "); }\n"
                   "  public Object[] get() { return new Object[] {" (str/join ", " (map #(str "out" %) ks)) "}; }\n"
                   "  public static void run(int key0, int key1" sig ") {\n"
                   "    " class-name " p = new " class-name "();\n"
                   "    p.key0 = key0; p.key1 = key1;\n"
                   set-outs
                   (calls "    ")
                   "  }\n"
                   "  public static void runRange(int r0, int r1, int from, int to" sig ") {\n"
                   "    " class-name " p = new " class-name "();\n"
                   set-outs
                   "    for (int i = from; i < to; ++i) {\n"
                   "      long k = g_threefry(r0, r1, 0, i);\n"
                   "      p.i = i; p.key0 = (int) (k >>> 32); p.key1 = (int) k;\n"
                   (calls "      ")
                   "    }\n"
                   "  }\n}\n")
      :class-name class-name
      :outputs out-specs
      :methods (count methods)
      :fields (count fs)
      :chunk chunk
      :statements (reduce + (map #(if (= :loop (first %)) (count (nth % 3)) 1) stmts))})))

;; ---------------------------------------------------------------------------
;; Compiling in memory
;; ---------------------------------------------------------------------------

(defn- java-source ^SimpleJavaFileObject [^String class-name ^String text]
  (proxy [SimpleJavaFileObject] [(URI/create (str "string:///" class-name ".java")) JavaFileObject$Kind/SOURCE]
    (getCharContent [_] text)))

(defn- javac
  "class-name's bytes, compiled from text in memory."
  ^bytes [^String class-name ^String text]
  (let [^JavaCompiler compiler (or (ToolProvider/getSystemJavaCompiler)
                     (throw (ex-info "JVM lowering: no Java compiler in this runtime (a JDK, not a JRE)" {})))
        outs (atom {})
        output (fn [^String name kind]
                 (let [baos (ByteArrayOutputStream.)]
                   (swap! outs assoc name baos)
                   (proxy [SimpleJavaFileObject] [(URI/create (str "mem:///" name ".class")) kind]
                     (openOutputStream [] baos))))
        fm (proxy [ForwardingJavaFileManager] [(.getStandardFileManager compiler nil nil nil)]
             (getJavaFileForOutput [_ name kind _] (output name kind))
             (getJavaFileForOutputForOriginatingFiles [_ name kind _] (output name kind)))
        diag (DiagnosticCollector.)
        ok (.call ^JavaCompiler$CompilationTask (.getTask compiler nil fm diag ["-g:none" "-proc:none" "-Xlint:none"] nil [(java-source class-name text)]))]
    (when-not ok
      (throw (ex-info (str "JVM lowering: javac failed:\n" (str/join "\n" (take 5 (.getDiagnostics diag)))) {})))
    (.toByteArray ^ByteArrayOutputStream (get @outs class-name))))

(defn method-sizes
  "{method name, its bytecode length} of a class's bytes (to check that no
   method passes HotSpot's 8000-byte limit for compilation)."
  [^bytes bytes]
  (into {} (for [^java.lang.classfile.MethodModel m (.methods (.parse (java.lang.classfile.ClassFile/of) bytes))
                 :let [code (.code m)]
                 :when (.isPresent code)]
             [(.stringValue (.methodName m)) (.codeLength ^java.lang.classfile.attribute.CodeAttribute (.get code))])))

(def ^:private counter (atom 0))

(defn- new-array [{:keys [shape dtype]}]
  (let [n (size shape)]
    (case dtype
      :float32 (float-array n)
      :float64 (double-array n)
      (:uint32 :int32) (int-array n)
      :bool (boolean-array n))))

(def ^:private huge-method
  "HotSpot compiles no method of more bytecode than this (DontCompileHugeMethods)."
  8000)

(defn- array-class [dtype]
  (case dtype :float32 (class (float-array 0)) :float64 (class (double-array 0)) (:uint32 :int32) (class (int-array 0))
        :bool (class (boolean-array 0))))

(defn compile
  "program (gen.device.stage's, of one key) as a loaded JVM class:
   {:run (fn [[key0 key1]]) → the outputs as Java arrays, :handle (run's
   static method), :source, :outputs, :method-sizes, :chunk, :ms {:generate
   :javac}}. When a method passes HotSpot's limit for compilation, the
   statements are cut again into methods half the size."
  [program]
  (let [class-name (str "GenProgram" (swap! counter inc))]
    (loop [chunk (chunk-size)]
      (let [t0 (System/nanoTime)
            src (source program class-name chunk)
            t1 (System/nanoTime)
            bytes (javac class-name (:source src))
            t2 (System/nanoTime)
            sizes (method-sizes bytes)]
        (if (and (> (long (apply max (vals sizes))) (long huge-method)) (> chunk 50))
          (recur (quot chunk 2))
          (let [^Class cls (.defineClass (clojure.lang.DynamicClassLoader.) class-name bytes nil)
                outs (:outputs src)
                arrays (map (comp array-class :dtype) outs)
                lookup (MethodHandles/publicLookup)
                ^MethodHandle h (.findStatic lookup cls "run" (MethodType/methodType Void/TYPE ^java.util.List (into [Integer/TYPE Integer/TYPE] arrays)))
                ^MethodHandle range-h (.findStatic lookup cls "runRange"
                                                    (MethodType/methodType Void/TYPE ^java.util.List (into [Integer/TYPE Integer/TYPE Integer/TYPE Integer/TYPE] arrays)))
                ctor (.getConstructor cls (into-array Class [Integer/TYPE]))]
            (assoc src
                   :bytes bytes
                   :method-sizes sizes
                   :ms {:generate (/ (- t1 t0) 1e6) :javac (/ (- t2 t1) 1e6)}
                   :handle h
                   :range-handle range-h
                   :instance (fn [n] (.newInstance ^java.lang.reflect.Constructor ctor (object-array [(int n)])))
                   :run (fn [[k0 k1]]
                          (let [arrays (mapv new-array outs)]
                            (.invokeWithArguments h ^java.util.List (into [(unchecked-int (long k0)) (unchecked-int (long k1))] arrays))
                            arrays)))))))))

(defn key-long
  "A key's two words as one long, an instance's accept takes."
  ^long [[k0 k1]]
  (bit-or (bit-shift-left (long k0) 32) (bit-and (long k1) 0xFFFFFFFF)))

(defonce ^:private pool
  (delay (java.util.concurrent.ForkJoinPool. (.availableProcessors (Runtime/getRuntime)))))

(defn- block-lse
  "[max, sum of exp(w - max)] of w[from, to), w a float[] or a double[]."
  [w ^long from ^long to]
  (let [at (if (instance? (Class/forName "[F") w)
             (let [^floats w w] (fn ^double [^long j] (double (aget w j))))
             (let [^doubles w w] (fn ^double [^long j] (aget w j))))
        m (loop [j from m Double/NEGATIVE_INFINITY] (if (< j to) (recur (inc j) (Math/max m (.invokePrim ^clojure.lang.IFn$LD at j))) m))
        s (if (== m Double/NEGATIVE_INFINITY)
            0.0
            (loop [j from s 0.0] (if (< j to) (recur (inc j) (+ s (Math/exp (- (.invokePrim ^clojure.lang.IFn$LD at j) m)))) s)))]
    [m s]))

(defn- combine-lse
  "log(sum exp) of blocks' [max sum] partials, in block order."
  ^double [parts]
  (let [m (reduce max Double/NEGATIVE_INFINITY (map first parts))]
    (if (== m Double/NEGATIVE_INFINITY)
      m
      (+ m (Math/log (reduce + (map (fn [[bm bs]] (* (double bs) (Math/exp (- (double bm) m)))) parts)))))))

(defn run-many
  "n runs of k on the keys (rng/split root n)[i] (root's words), in
   contiguous blocks of opts :block runs (by default n/64, at least 1, so
   the blocks depend on n only) spread over a pool of every core (:threads
   1 runs them in this thread). Returns {:outputs [arrays, run i's at
   offset i·size], :log-sum-exp (with :lse k: log Σ exp of output k, a
   float32 [1] per run, combined from the blocks in order, so the same
   for any number of threads)}."
  [{:keys [outputs ^MethodHandle range-handle]} n [r0 r1] {:keys [block threads lse]}]
  (let [n (long n)
        block (long (or block (max 1 (quot n 64))))
        arrays (mapv (fn [{:keys [shape dtype]}] (new-array {:shape (into [n] shape) :dtype dtype})) outputs)
        r0 (unchecked-int (long r0)) r1 (unchecked-int (long r1))
        tasks (vec (for [from (range 0 n block)]
                     (let [to (min n (+ from block))]
                       (fn []
                         (.invokeWithArguments range-handle ^java.util.List (into [r0 r1 (int from) (int to)] arrays))
                         (when lse (block-lse (nth arrays lse) from to))))))
        parts (if (= 1 threads)
                (mapv #(%) tasks)
                (mapv #(.get ^java.util.concurrent.Future %) (.invokeAll ^java.util.concurrent.ForkJoinPool @pool ^java.util.Collection tasks)))]
    (cond-> {:outputs arrays}
      lse (assoc :log-sum-exp (combine-lse parts)))))
