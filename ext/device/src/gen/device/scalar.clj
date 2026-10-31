(ns gen.device.scalar
  "A staged program (gen.device.stage) as statements, the shared core of the
   per-particle lowerings (gen.device.jvm, gen.device.metal; research
   prototype, ext/device, beans genclj-wbm7, genclj-h8rn, genclj-2vj1).

   Arrays are of two kinds, so that code does not grow with data:
   - small (at most gen.device.small elements, 64 by default): a vector of
     expressions, one per element, so reshapes, broadcasts, slices and
     transposes cost nothing and the compiler downstream sees straight-line
     scalar code;
   - large: an accessor, a function of a loop's emitter and an index
     expression giving the element there. An elementwise chain stays an
     accessor (fused: one loop evaluates it), and is written to a buffer
     (materialized) when a node uses it twice, a matrix product reads it, a
     broadcast would recompute it, or the program stores it. Reductions and
     matrix products over large arrays are loops; so are a large result's
     stores. A program's data inputs (after its key) and its large constants
     are buffers the code receives, never code.
   A counted scan is a loop whose carries are variables (small) or buffers
   (large, updated through a temporary).

   (scalarize program dialect) returns {:stmts :outputs :buffers :inputs
   :constants :statements :error-output}: a program that gathers has one
   more output than the program, last, a bool per run set when an index
   fell outside its axis (:error-output its position; nil otherwise). Statements:
     [:decl dtype name expr]     a value (inside a :for, local to its body)
     [:set name expr]            a carry's or an accumulator's update
     [:store k index-expr expr]  output k's element index-expr in one run
     [:bset buffer index-expr expr]
     [:loop it length body]      a scan's loop
     [:for i n body]             a loop over an array's elements
   :buffers are the run's scratch arrays [{:name :dtype :n}], :inputs the
   data inputs [{:name :shape :dtype}] (in1, in2, …, the program's inputs
   after the key), :constants the large constants [{:name :dtype :values}].
   A dialect writes expressions in its language: a map of
     :name     the lowering's name, for errors
     :lit      (fn [v dtype]) a host value as a literal
     :simple?  (fn [expr]) a name or literal (not bound to a variable)
     :unary    (fn [op dtype a])
     :binary   (fn [op dtype a b])
     :convert  (fn [from to a])
     :bitcast  (fn [from to a])
     :read     (fn [buffer dtype index]) a buffer's element (optional;
               buffer[index] by default)
   each throwing for what it cannot write. Names: v<n> values, c<n> carries,
   it<n> scan counters, i<n> array loops, b<n> buffers, k<n> constants,
   in<n> inputs; the key's words are key0 and key1. Indices are int."
  {:api :internal}
  (:require [gen.random.threefry]))

(set! *warn-on-reflection* true)

;; ---------------------------------------------------------------------------
;; Shapes
;; ---------------------------------------------------------------------------

(defn size ^long [shape] (reduce * 1 shape))

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

(defn- small-limit ^long [] (Long/getLong "gen.device.small" 64))

(defn- large-shape? [shape] (> (size (or shape [])) (small-limit)))

;; Index expressions (int): a flat index's coordinates and back

(defn- coords
  "Expressions of the coordinates of flat index i (an expression) in shape."
  [i shape]
  (let [n (size shape)]
    (mapv (fn [st d] (cond (= 1 d) "0"
                           (= n (* st d)) (if (= 1 st) i (str "(" i " / " st ")"))
                           (= 1 st) (str "(" i " % " d ")")
                           :else (str "((" i " / " st ") % " d ")")))
          (strides shape) shape)))

(defn- flat
  "The flat index of coordinates cs (expressions) in shape."
  [cs shape]
  (let [terms (keep (fn [[c st d]] (when (and (not= "0" c) (not= 1 d)) (if (= 1 st) c (str c " * " st))))
                    (map vector cs (strides shape) shape))]
    (if (empty? terms) "0" (str "(" (apply str (interpose " + " terms)) ")"))))

(defn- bidx
  "The flat index into an operand of in-shape that NumPy's broadcasting
   pairs with the element at flat index i (an expression) of out-shape."
  [i out-shape in-shape]
  (cond (= (vec out-shape) (vec in-shape)) i
        (= 1 (size in-shape)) "0"
        :else (let [k (- (count out-shape) (count in-shape))
                    oc (coords i out-shape)]
                (flat (map-indexed (fn [j d] (if (= 1 d) "0" (nth oc (+ k j)))) in-shape) in-shape))))

;; ---------------------------------------------------------------------------
;; Statements
;; ---------------------------------------------------------------------------

(defn- emitter [dialect]
  {:stmts (volatile! []) :n (volatile! 0) :dialect dialect :buffers (volatile! []) :constants (volatile! [])
   :memo (java.util.IdentityHashMap.) :uses {}})

(defn- sub
  "An emitter for a loop's body: its own statements, everything else shared."
  [em]
  (assoc em :stmts (volatile! [])))

(defn- stmt! [em s] (vswap! (:stmts em) conj s))

(defn- fresh [em prefix] (str prefix (vswap! (:n em) inc)))

(defn- fail! [em msg data] (throw (ex-info (str (:name (:dialect em)) ": " msg) data)))

(defn- bind!
  "A name for the expression e of dtype (e itself when it is a name or a
   literal)."
  [em dtype ^String e]
  (if ((:simple? (:dialect em)) e)
    e
    (let [v (fresh em "v")]
      (stmt! em [:decl dtype v e])
      v)))

(defn- value [shape dtype el] {:shape (vec shape) :dtype dtype :el (vec el)})

(defn- large? [v] (contains? v :at))

(defn- read-expr [em b dtype i]
  (if-let [r (:read (:dialect em))] (r b dtype i) (str b "[" i "]")))

(defn- buffer-value [em shape dtype b]
  {:shape (vec shape) :dtype dtype :buf b :at (fn [bem i] (bind! bem dtype (read-expr em b dtype i)))})

(defn- buffer! [em dtype n]
  (let [b (fresh em "b")] (vswap! (:buffers em) conj {:name b :dtype dtype :n n}) b))

(defn- for!
  "A loop over i < n in em, its body written by (body bem i)."
  [em n body]
  (let [i (fresh em "i") bem (sub em)]
    (body bem i)
    (stmt! em [:for i n @(:stmts bem)])))

(defn- materialize!
  "v (small or large) as a buffer of em's run."
  [em {:keys [shape dtype] :as v}]
  (cond (:buf v) v
        (and (:el v) (.containsKey ^java.util.IdentityHashMap (:memo em) (:el v))) (.get ^java.util.IdentityHashMap (:memo em) (:el v))
        :else (let [b (buffer! em dtype (size shape))
                    bv (buffer-value em shape dtype b)]
                (if (:el v)
                  (do (doseq [[j e] (map-indexed vector (:el v))] (stmt! em [:bset b (str j) e]))
                      (.put ^java.util.IdentityHashMap (:memo em) (:el v) bv))
                  (for! em (size shape) (fn [bem i] (stmt! bem [:bset b i ((:at v) bem i)]))))
                bv)))

(defn- loopable
  "v readable inside a loop by an index: a large value as it is, a small one
   of one element as a constant accessor, any other small one materialized."
  [em v]
  (cond (large? v) v
        (= 1 (count (:el v))) (let [e (first (:el v))] (assoc v :at (fn [_ _] e)))
        :else (materialize! em v)))

(defn- elt
  "v's element k (an integer) as an expression in em: a small value's, or a
   large one's read here."
  [em v k]
  (if (:el v) (nth (:el v) k) ((:at v) em (str k))))

(defn- small-from
  "A small value of shape whose element o is (f o), f reading operands
   through elt."
  [shape dtype f]
  (value shape dtype (mapv f (range (size shape)))))

;; ---------------------------------------------------------------------------
;; Gathers: an index read at run time, normalized as the JVM backend does
;; (truncated, negative from the end), then clamped into [0, d) so that a
;; read stays in bounds (the handler throws for an index out of range; the
;; code cannot, genclj-ryg0)
;; ---------------------------------------------------------------------------

(defn- index-expr
  "The int index at expression e (of dtype) against a dimension of size d,
   bound in em. An index outside [0, d) after normalization sets the run's
   error flag (em's :flag), and is clamped meanwhile so that the read stays
   in bounds (DESIGN §9, genclj-i2c4)."
  [em dtype e d]
  (let [{:keys [convert binary unary]} (:dialect em)
        i (bind! em :int32 (if (#{:int32} dtype) e (convert dtype :int32 e)))
        i (bind! em :int32 (str "(" i " < 0 ? " i " + " d " : " i ")"))]
    (when-let [f (:flag em)]
      (let [ok (bind! em :bool (str "(" i " >= 0 && " i " < " d ")"))]
        (stmt! em [:set f (binary :logical-or :bool f (unary :logical-not :bool ok))])))
    (bind! em :int32 (str "(" i " < 0 ? 0 : (" i " > " (dec d) " ? " (dec d) " : " i "))"))))

(defn- gathers?
  "Does graph (or a scan body in it) gather?"
  [graph]
  (some (fn [{:keys [op attrs]}]
          (or (#{:take :take-along-axis} op)
              (and (= op :scan) (gathers? (get-in attrs [:body :graph])))))
        graph))

(defn- select-chain
  "The expression choosing (candidates k) where i (an int expression) is k."
  [i candidates]
  (reduce (fn [e k] (str "(" i " == " k " ? " (nth candidates k) " : " e ")"))
          (peek candidates) (range (- (count candidates) 2) -1 -1)))

;; ---------------------------------------------------------------------------
;; argmax and argmin: the first extreme, the first NaN winning (the JVM
;; backend's, NumPy's)
;; ---------------------------------------------------------------------------

(defn- arg-step!
  "[best-index best-value] after considering element v at index k (an int
   expression), op :argmax or :argmin, values of dtype, bound in em (or set
   when vars, [index-var value-var], are given)."
  [em op dtype [bk bv] k v]
  (let [{:keys [unary binary]} (:dialect em)
        better (bind! em :bool (binary :logical-and :bool (unary :logical-not :bool (unary :isnan dtype bv))
                                       (binary :logical-or :bool (unary :isnan dtype v)
                                               (binary (if (= op :argmax) :greater :less) dtype v bv))))]
    [(str "(" better " ? " k " : " bk ")") (str "(" better " ? " v " : " bv ")")]))

(defn- arg-reduce-small
  "argmax/argmin of the expressions es (in the reduced axes' order): an
   int32 expression."
  [em op dtype es]
  (first (reduce (fn [[bk bv] k]
                   (let [[nk nv] (arg-step! em op dtype [bk bv] (str k) (nth es k))]
                     [(bind! em :int32 nk) (bind! em dtype nv)]))
                 ["0" (first es)] (range 1 (count es)))))

;; ---------------------------------------------------------------------------
;; Nodes
;; ---------------------------------------------------------------------------

(def ^:private bit-ops #{:bit-xor :bit-or :bit-and :shift-left :shift-right-logical})
(def ^:private comparisons #{:less :less-equal :greater :greater-equal :equal :not-equal})
(def ^:private binary-ops #{:add :subtract :multiply :divide :logical-and :logical-or :maximum :minimum :remainder :power :arctan2})
(def ^:private unary-ops #{:negative :logical-not :sign :isnan :isinf :abs :erfinv :erf :exp :log :expm1 :log1p :sqrt :sin :cos :tan
                           :tanh :sinh :cosh :arcsin :arccos :arctan :floor :ceil :round})

(def ^:private reduce-init {:sum 0 :prod 1 :max Double/NEGATIVE_INFINITY :min Double/POSITIVE_INFINITY :all true :any false})
(def ^:private reduce-op {:sum :add :prod :multiply :max :maximum :min :minimum :all :logical-and :any :logical-or})

(defn- reduce-start
  "A reduction's starting value in dtype (the JVM backend's: 0 for a sum,
   -Inf for a float max, an integer max from the type's least value)."
  [op dtype]
  (case [op dtype]
    [:max :int32] Integer/MIN_VALUE
    [:min :int32] Integer/MAX_VALUE
    [:max :uint32] 0
    [:min :uint32] 0xFFFFFFFF
    (reduce-init op)))

(declare emit-graph! emit-scan!)

(defn- operand
  "An operand of a node: its value, or a constant's (a {:const x} of dtype)."
  [em env a dtype]
  (if (map? a)
    (let [d (if (boolean? (:const a)) :bool dtype)] (value [] d [((:lit (:dialect em)) (:const a) d)]))
    (let [v (or (env a) (fail! em "an operand with no value" {:id a}))]
      (when (:ys v) (fail! em "a scan's ys read in the program (only its outputs may be ys)" {:id a}))
      v)))

;; small arrays: every element an expression

(defn- elementwise [em shape dtype operands f]
  (let [maps (mapv #(broadcast-map shape (:shape %)) operands)]
    (value shape dtype (mapv (fn [i] (bind! em dtype (apply f (map (fn [o m] (nth (:el o) (nth m i))) operands maps))))
                             (range (size shape))))))

(defn- reduce-small [em {:keys [shape el dtype]} op axes out-shape out-dtype]
  (when-not (or (contains? reduce-init op) (#{:argmax :argmin} op)) (fail! em (str "no reduction " op " yet") {:op op}))
  (if (#{:argmax :argmin} op)
    (let [axes (set axes)
          groups (group-by (fn [i] (let [idx (unravel i shape)] (vec (keep-indexed (fn [k x] (when-not (axes k) x)) idx))))
                           (range (size shape)))
          kept (vec (keep-indexed (fn [k d] (when-not (axes k) d)) shape))]
      (value out-shape out-dtype (mapv (fn [o] (arg-reduce-small em op dtype (mapv #(nth el %) (get groups (unravel o kept)))))
                                       (range (size kept)))))
  (let [{:keys [lit binary]} (:dialect em)
        axes (set axes)
        groups (group-by (fn [i] (let [idx (unravel i shape)] (vec (keep-indexed (fn [k x] (when-not (axes k) x)) idx))))
                         (range (size shape)))
        kept (vec (keep-indexed (fn [k d] (when-not (axes k) d)) shape))]
    (value out-shape out-dtype
           (mapv (fn [o] (let [is (get groups (unravel o kept))]
                           (if (empty? is)
                             (lit (reduce-init op) out-dtype)
                             (reduce (fn [acc i] (bind! em out-dtype (binary (reduce-op op) out-dtype acc (nth el i))))
                                     (nth el (first is)) (rest is)))))
                 (range (size kept)))))))

(defn- permuted-index
  "The flat index into the operand (in-shape) of a transpose by perm at flat
   index i of shape: a function of integers or of expressions."
  [shape perm in-shape]
  (fn [cs] (reduce (fn [v k] (assoc v (nth perm k) (nth cs k))) (vec (repeat (count shape) 0)) (range (count shape)))))

(defn- emit-small!
  "A node whose operands and result are small."
  [em env {:keys [id op args shape dtype attrs] :as node} ctx]
  (let [{:keys [lit unary binary convert bitcast]} (:dialect em)
        in (:in attrs)
        put #(assoc env id %)
        arg (fn [i d] (operand em env (nth args i) (or d in dtype)))]
    (cond
      (= op :constant) (let [vs (:values attrs)]
                         (put (value shape dtype (mapv #(lit (nth vs (if (= 1 (count vs)) 0 %)) dtype) (range (size shape))))))
      (= op :iota) (put (value shape dtype (mapv #(lit % dtype) (range (size shape)))))
      (= op :broadcast) (let [o (arg 0 dtype) m (broadcast-map shape (:shape o))]
                          (put (value shape dtype (mapv #(nth (:el o) %) m))))
      (= op :convert) (let [o (arg 0 nil)]
                        (put (if (= (:dtype o) dtype)
                               (assoc o :shape shape)
                               (value shape dtype (mapv #(bind! em dtype (convert (:dtype o) dtype %)) (:el o))))))
      (= op :bitcast) (let [o (arg 0 nil)]
                        (put (value shape dtype (mapv #(bind! em dtype (bitcast (:dtype o) dtype %)) (:el o)))))
      (= op :stop-gradient) (put (assoc (arg 0 in) :shape shape))
      (unary-ops op) (put (elementwise em shape dtype [(arg 0 in)] #(unary op in %)))
      (bit-ops op) (put (elementwise em shape :uint32 [(arg 0 :uint32) (arg 1 :uint32)] #(binary op :uint32 %1 %2)))
      (comparisons op) (put (elementwise em shape :bool [(arg 0 in) (arg 1 in)] #(binary op in %1 %2)))
      (binary-ops op) (put (elementwise em shape dtype [(arg 0 in) (arg 1 in)] #(binary op in %1 %2)))
      (= op :where) (put (elementwise em shape dtype [(arg 0 :bool) (arg 1 in) (arg 2 in)] #(str "(" %1 " ? " %2 " : " %3 ")")))
      (= op :reduce) (put (reduce-small em (arg 0 nil) (:op attrs) (:axes attrs) shape dtype))
      (= op :reshape) (put (assoc (arg 0 nil) :shape shape))
      (= op :transpose) (let [o (arg 0 nil) pi (permuted-index shape (:perm attrs) (:shape o))]
                          (put (small-from shape dtype (fn [i] (nth (:el o) (ravel (pi (unravel i shape)) (:shape o)))))))
      (= op :concatenate) (let [ax (:axis attrs)
                                os (mapv #(operand em env % dtype) args)
                                offsets (vec (reductions + 0 (map #(nth (:shape %) ax) os)))]
                            (put (small-from shape dtype
                                             (fn [i] (let [oi (unravel i shape)
                                                           p (nth oi ax)
                                                           k (count (take-while #(<= % p) (rest offsets)))
                                                           o (nth os k)]
                                                       (nth (:el o) (ravel (assoc oi ax (- p (nth offsets k))) (:shape o))))))))
      (= op :slice) (let [o (arg 0 nil) {:keys [starts steps]} attrs]
                      (put (small-from shape dtype (fn [i] (nth (:el o) (ravel (map (fn [s k x] (+ s (* k x))) starts steps (unravel i shape))
                                                                               (:shape o)))))))
      (= op :pad) (let [o (arg 0 nil) {:keys [axis low interior]} attrs
                        m (nth (:shape o) axis) z (lit 0 dtype)]
                    (put (small-from shape dtype (fn [i] (let [oi (unravel i shape)
                                                               p (- (nth oi axis) low)
                                                               j (quot p (inc interior))]
                                                           (if (and (>= p 0) (zero? (mod p (inc interior))) (< j m))
                                                             (nth (:el o) (ravel (assoc oi axis j) (:shape o)))
                                                             z))))))
      (= op :matmul) (let [x (arg 0 dtype) y (arg 1 dtype)
                           [sx sy] [(:shape x) (:shape y)]
                           _ (when (or (> (count sx) 2) (> (count sy) 2)) (fail! em "batched matmul" {:node node}))
                           mx (if (= 1 (count sx)) [1 (first sx)] sx)
                           my (if (= 1 (count sy)) [(first sy) 1] sy)
                           [r k] mx c (second my)]
                       (put (value shape dtype
                                   (for [a (range r) b (range c)]
                                     (reduce (fn [acc j] (bind! em dtype (str "(" acc " + " (nth (:el x) (+ (* a k) j)) " * " (nth (:el y) (+ (* j c) b)) ")")))
                                             (bind! em dtype (str "(" (nth (:el x) (* a k)) " * " (nth (:el y) b) ")"))
                                             (range 1 k))))))
      (= op :take) (let [x (arg 0 dtype) ind (arg 1 :int32) ax (:axis attrs)
                         sx (:shape x) si (:shape ind) d (nth sx ax)
                         idx (mapv #(index-expr em (:dtype ind) % d) (:el ind))]
                     (put (small-from shape dtype
                                      (fn [i] (let [c (unravel i shape)
                                                    pre (subvec c 0 ax) j (ravel (subvec c ax (+ ax (count si))) si) post (subvec c (+ ax (count si)))]
                                                (bind! em dtype (select-chain (nth idx j)
                                                                              (mapv #(nth (:el x) (ravel (concat pre [%] post) sx)) (range d)))))))))
      (= op :take-along-axis) (let [x (arg 0 dtype) ind (arg 1 :int32) ax (:axis attrs)
                                    sx (:shape x) si (:shape ind) d (nth sx ax)
                                    imap (broadcast-map (assoc shape ax (nth si ax)) si)]
                                (put (small-from shape dtype
                                                 (fn [i] (let [c (unravel i shape)
                                                               k (index-expr em (:dtype ind) (nth (:el ind) (nth imap i)) d)
                                                               xc (fn [a] (ravel (map-indexed (fn [kk dim] (cond (= kk ax) a (= 1 dim) 0 :else (nth c kk))) sx) sx))]
                                                           (bind! em dtype (select-chain k (mapv #(nth (:el x) (xc %)) (range d)))))))))
      :else (fail! em (str "no lowering of " op " yet") {:node (dissoc node :attrs)}))))

;; large arrays: accessors and loops

(defn- lazy
  "A large value of shape whose element at index i is (f bem i)."
  [shape dtype f]
  {:shape (vec shape) :dtype dtype :at f})

(defn- elementwise-large
  "A large elementwise result: an accessor over the operands' (fused)."
  [em shape dtype operands f]
  (let [os (mapv #(loopable em %) operands)]
    (lazy shape dtype (fn [bem i] (bind! bem dtype (apply f (map (fn [o] ((:at o) bem (bidx i shape (:shape o)))) os)))))))

(defn- reduce-large
  "op over axes of v (any size), shaped out-shape: loops, sequential in the
   reduced axes' row-major order."
  [em v op axes out-shape out-dtype]
  (when-not (or (contains? reduce-init op) (#{:argmax :argmin} op)) (fail! em (str "no reduction " op " yet") {:op op}))
  (let [{:keys [lit binary]} (:dialect em)
        arg? (#{:argmax :argmin} op)
        v (loopable em v)
        in-shape (:shape v)
        axes (set axes)
        kept (vec (keep-indexed (fn [k d] (when-not (axes k) d)) in-shape))
        red (vec (keep-indexed (fn [k d] (when (axes k) d)) in-shape))
        index (fn [o r] (let [ko (coords o kept) ro (coords r red)]
                          (flat (loop [k 0 ko ko ro ro out []]
                                  (if (= k (count in-shape)) out
                                      (if (axes k) (recur (inc k) ko (rest ro) (conj out (first ro)))
                                          (recur (inc k) (rest ko) ro (conj out (first ko))))))
                                in-shape)))
        init (when-not arg? (lit (reduce-start op out-dtype) out-dtype))
        accumulate (fn [em o]
                     (if arg?
                       (let [bk (fresh em "v") bv (fresh em "v")]
                         (stmt! em [:decl :int32 bk "0"])
                         (stmt! em [:decl (:dtype v) bv ((:at v) em (index o "0"))])
                         (for! em (dec (size red))
                               (fn [bem r]
                                 (let [k (bind! bem :int32 (str "(" r " + 1)"))
                                       [nk nv] (arg-step! bem op (:dtype v) [bk bv] k ((:at v) bem (index o k)))]
                                   (stmt! bem [:set bk nk])
                                   (stmt! bem [:set bv nv]))))
                         bk)
                       (let [acc (fresh em "v")]
                         (stmt! em [:decl out-dtype acc init])
                         (for! em (size red) (fn [bem r] (stmt! bem [:set acc (binary (reduce-op op) out-dtype acc ((:at v) bem (index o r)))])))
                         acc)))]
    (if (large-shape? out-shape)
      (let [b (buffer! em out-dtype (size kept))]
        (for! em (size kept) (fn [bem o] (stmt! bem [:bset b o (accumulate bem o)])))
        (buffer-value em out-shape out-dtype b))
      (value out-shape out-dtype (mapv #(accumulate em (str %)) (range (size kept)))))))

(defn- matmul-large
  "x · y for vectors and matrices: each output element a loop over the
   shared axis, from 0 in the JVM backend's order."
  [em node x y out-shape dtype]
  (let [{:keys [lit]} (:dialect em)
        [sx sy] [(:shape x) (:shape y)]
        _ (when (or (> (count sx) 2) (> (count sy) 2)) (fail! em "batched matmul" {:node (dissoc node :attrs)}))
        x (if (large? x) (materialize! em x) (loopable em x))
        y (if (large? y) (materialize! em y) (loopable em y))
        k (peek sx)
        c (if (= 1 (count sy)) 1 (second sy))
        element (fn [em o]
                  (let [acc (fresh em "v")]
                    (stmt! em [:decl dtype acc (lit 0 dtype)])
                    (for! em k (fn [bem j]
                                 (let [a ((:at x) bem (if (= 1 c) (str "(" o ") * " k " + " j) (str "((" o ") / " c ") * " k " + " j)))
                                       b ((:at y) bem (if (= 1 c) j (str j " * " c " + (" o ") % " c)))]
                                   (stmt! bem [:set acc (str "(" acc " + " a " * " b ")")]))))
                    acc))]
    (if (large-shape? out-shape)
      (let [b (buffer! em dtype (size out-shape))]
        (for! em (size out-shape) (fn [bem o] (stmt! bem [:bset b o (element bem o)])))
        (buffer-value em out-shape dtype b))
      (value out-shape dtype (mapv #(element em (str %)) (range (size out-shape)))))))

(defn- in-range
  "An index expression clamped to 0 where cond (an expression) is false, so
   that a read hoisted out of a conditional stays in bounds."
  [cond idx]
  (str "(" cond " ? " idx " : 0)"))

(defn- emit-large!
  "A node with a large operand or result."
  [em env {:keys [id op args shape dtype attrs] :as node} ctx]
  (let [{:keys [lit unary binary convert bitcast]} (:dialect em)
        in (:in attrs)
        arg (fn [i d] (operand em env (nth args i) (or d in dtype)))
        result (fn [v] (if (and (large? v) (not (:buf v)) (> (long (get (:uses em) id 0)) 1)) (materialize! em v) v))
        put #(assoc env id (result %))
        small-out? (not (large-shape? shape))]
    (cond
      (= op :constant) (let [vs (:values attrs)]
                         (put (if (apply = vs)
                                (let [e (lit (first vs) dtype)] (lazy shape dtype (fn [_ _] e)))
                                (let [k (fresh em "k")]
                                  (vswap! (:constants em) conj {:name k :dtype dtype :values vs})
                                  (buffer-value em shape dtype k)))))
      (= op :iota) (put (lazy shape dtype (fn [bem i] (bind! bem dtype (convert :int32 dtype i)))))
      (= op :broadcast) (let [o (arg 0 dtype)
                              o (if (and (large? o) (not (:buf o)) (> (size shape) (size (:shape o)))) (materialize! em o) (loopable em o))]
                          (put (lazy shape dtype (fn [bem i] ((:at o) bem (bidx i shape (:shape o)))))))
      (= op :reshape) (let [o (arg 0 nil)]
                        (put (if (and small-out? (:el o)) (assoc o :shape shape) (assoc (loopable em o) :shape (vec shape)))))
      (= op :stop-gradient) (put (assoc (arg 0 in) :shape shape))
      (= op :convert) (let [o (arg 0 nil)]
                        (put (if (= (:dtype o) dtype)
                               (assoc o :shape shape)
                               (elementwise-large em shape dtype [o] #(convert (:dtype o) dtype %)))))
      (= op :bitcast) (let [o (arg 0 nil)] (put (elementwise-large em shape dtype [o] #(bitcast (:dtype o) dtype %))))
      (unary-ops op) (put (elementwise-large em shape dtype [(arg 0 in)] #(unary op in %)))
      (bit-ops op) (put (elementwise-large em shape :uint32 [(arg 0 :uint32) (arg 1 :uint32)] #(binary op :uint32 %1 %2)))
      (comparisons op) (put (elementwise-large em shape :bool [(arg 0 in) (arg 1 in)] #(binary op in %1 %2)))
      (binary-ops op) (put (elementwise-large em shape dtype [(arg 0 in) (arg 1 in)] #(binary op in %1 %2)))
      (= op :where) (put (elementwise-large em shape dtype [(arg 0 :bool) (arg 1 in) (arg 2 in)] #(str "(" %1 " ? " %2 " : " %3 ")")))
      (= op :reduce) (put (reduce-large em (arg 0 nil) (:op attrs) (:axes attrs) shape dtype))
      (= op :matmul) (put (matmul-large em node (arg 0 dtype) (arg 1 dtype) shape dtype))
      (= op :transpose) (let [o (loopable em (arg 0 nil))
                              pi (permuted-index shape (:perm attrs) (:shape o))
                              index (fn [i] (flat (pi (coords i shape)) (:shape o)))]
                          (put (if small-out?
                                 (small-from shape dtype (fn [i] (elt em o (ravel (pi (unravel i shape)) (:shape o)))))
                                 (lazy shape dtype (fn [bem i] ((:at o) bem (index i)))))))
      (= op :slice) (let [o (arg 0 nil) {:keys [starts steps]} attrs
                          index (fn [i] (flat (map (fn [s k c] (if (= 1 k) (str "(" s " + " c ")") (str "(" s " + " k " * " c ")")))
                                                   starts steps (coords i shape))
                                              (:shape o)))]
                      (put (if small-out?
                             (small-from shape dtype (fn [i] (elt em o (ravel (map (fn [s k x] (+ s (* k x))) starts steps (unravel i shape)) (:shape o)))))
                             (let [o (loopable em o)] (lazy shape dtype (fn [bem i] ((:at o) bem (index i))))))))
      (= op :pad) (let [o (loopable em (arg 0 nil)) {:keys [axis low interior]} attrs
                        m (nth (:shape o) axis) z (lit 0 dtype)]
                    (put (lazy shape dtype
                               (fn [bem i]
                                 (let [cs (coords i shape)
                                       p (str "(" (nth cs axis) " - " low ")")
                                       j (str "(" p " / " (inc interior) ")")
                                       ok (bind! bem :bool (str "(" p " >= 0 && " p " % " (inc interior) " == 0 && " j " < " m ")"))
                                       x ((:at o) bem (in-range ok (flat (assoc cs axis j) (:shape o))))]
                                   (bind! bem dtype (str "(" ok " ? " x " : " z ")")))))))
      (= op :concatenate) (let [ax (:axis attrs)
                                os (mapv #(loopable em (operand em env % dtype)) args)
                                offsets (vec (reductions + 0 (map #(nth (:shape %) ax) os)))]
                            (put (lazy shape dtype
                                       (fn [bem i]
                                         (let [cs (coords i shape) p (nth cs ax)]
                                           (reduce (fn [e k]
                                                     (let [o (nth os k)
                                                           lo (nth offsets k) hi (nth offsets (inc k))
                                                           in? (bind! bem :bool (str "(" p " >= " lo " && " p " < " hi ")"))
                                                           x ((:at o) bem (in-range in? (flat (assoc cs ax (str "(" p " - " lo ")")) (:shape o))))]
                                                       (if e (bind! bem dtype (str "(" in? " ? " x " : " e ")")) x)))
                                                   nil (range (count os))))))))
      (= op :take) (let [x (loopable em (arg 0 dtype)) ind (loopable em (arg 1 :int32)) ax (:axis attrs)
                         sx (:shape x) si (:shape ind) d (nth sx ax)
                         read (fn [bem i]
                                (let [c (coords i shape)
                                      j (flat (subvec c ax (+ ax (count si))) si)
                                      k (index-expr bem (:dtype ind) ((:at ind) bem j) d)]
                                  ((:at x) bem (flat (concat (subvec c 0 ax) [k] (subvec c (+ ax (count si)))) sx))))]
                     (put (if small-out?
                            (small-from shape dtype #(read em (str %)))
                            (lazy shape dtype read))))
      (= op :take-along-axis) (let [x (loopable em (arg 0 dtype)) ind (loopable em (arg 1 :int32)) ax (:axis attrs)
                                    sx (:shape x) si (:shape ind) d (nth sx ax)
                                    read (fn [bem i]
                                           (let [c (coords i shape)
                                                 k (index-expr bem (:dtype ind) ((:at ind) bem (bidx i (assoc (vec shape) ax (nth si ax)) si)) d)]
                                             ((:at x) bem (flat (map-indexed (fn [kk dim] (cond (= kk ax) k (= 1 dim) "0" :else (nth c kk))) sx) sx))))]
                                (put (if small-out?
                                       (small-from shape dtype #(read em (str %)))
                                       (lazy shape dtype read))))
      :else (fail! em (str "no lowering of " op " over large arrays yet") {:node (dissoc node :attrs)}))))

(defn- emit-node!
  "env with node's value, small ({:el}) or large ({:at}), or a scan's or its
   ys result's own entry."
  [em env {:keys [id op args shape] :as node} ctx]
  (cond
    (= op :param) (if (env id) env (fail! em "an unbound param" {:node node}))
    (= op :scan) (assoc env id (emit-scan! em env node ctx))
    (= op :result) (let [s (env (first args)) i (:index (:attrs node))]
                     (assoc env id (if (< i (:n-carry s))
                                     (nth (:carries s) i)
                                     {:shape shape :dtype (:dtype node) :ys (- i (:n-carry s)) :scan (first args)})))
    (or (large-shape? shape) (some #(and (not (map? %)) (large? (get env % {}))) args)) (emit-large! em env node ctx)
    :else (emit-small! em env node ctx)))

(defn- uses
  "How many times each node of graph is read: by its nodes and by extra
   (the graph's own outputs)."
  [graph extra]
  (frequencies (concat (mapcat (fn [{:keys [args]}] (remove map? args)) graph) extra)))

(defn- emit-scan!
  "A counted scan as a loop statement: {:carries [values after the loop]
   :n-carry}. ys the program returns are stored to their outputs in the
   loop (ctx's :ys-out maps [scan id, ys index] to an output)."
  [em env {:keys [id args attrs]} {:keys [ys-out nested?] :as ctx}]
  (when nested? (fail! em "a scan inside a scan, not yet" {}))
  (let [{:keys [body length n-carry n-x]} attrs
        {:keys [graph carries captures carry-out y-out]} body]
    (when (pos? n-x) (fail! em "a scan over xs (use a counted scan, xs nil)" {:n-x n-x}))
    (let [inits (mapv #(env %) (subvec args 0 n-carry))
          ;; a large capture is read every iteration: evaluated once, before
          caps (mapv #(let [v (env %)] (if (and (large? v) (not (:buf v))) (materialize! em v) v)) (subvec args (+ n-carry n-x)))
          vars (mapv (fn [{:keys [shape dtype] :as v}]
                       (if (large? v)
                         (let [b (buffer! em dtype (size shape))]
                           (for! em (size shape) (fn [bem i] (stmt! bem [:bset b i ((:at v) bem i)])))
                           (buffer-value em shape dtype b))
                         (value shape dtype (mapv (fn [e] (let [c (fresh em "c")] (stmt! em [:decl dtype c e]) c)) (:el v)))))
                     inits)
          it (fresh em "it")
          bem (assoc (sub em) :uses (uses graph (concat carry-out y-out)))
          benv (emit-graph! bem graph (merge (zipmap carries vars) (zipmap captures caps)) (assoc ctx :nested? true))]
      (doseq [[j y] (map-indexed vector y-out)
              :let [k (get ys-out [id j])]
              :when k
              :let [{:keys [shape] :as v} (benv y) n (size shape)]]
        (if (large? v)
          (for! bem n (fn [lem w] (stmt! lem [:store k (str it " * " n " + " w) ((:at v) lem w)])))
          (doseq [[e x] (map-indexed vector (:el v))]
            (stmt! bem [:store k (str it " * " n " + " e) x]))))
      ;; every new value computed before any carry changes, as a carry's
      ;; update may read another carry
      (let [news (mapv (fn [var o] (let [v (benv o)]
                                     (if (large? var)
                                       (if (= (:buf v) (:buf var)) nil (materialize! bem (assoc v :buf nil)))
                                       (mapv (fn [e] (if (re-matches #"c[0-9]+" e) (bind! bem (:dtype var) (str "(" e ")")) e)) (:el v)))))
                       vars carry-out)]
        (doseq [[var nv] (map vector vars news) :when nv]
          (if (large? var)
            (for! bem (size (:shape var)) (fn [lem i] (stmt! lem [:bset (:buf var) i ((:at nv) lem i)])))
            (doseq [[c x] (map vector (:el var) nv)]
              (when-not (= c x) (stmt! bem [:set c x]))))))
      (stmt! em [:loop it length @(:stmts bem)])
      {:carries vars :n-carry n-carry})))

(defn- emit-graph! [em nodes env ctx]
  (reduce (fn [env node] (emit-node! em env node ctx)) env nodes))

(defn- count-statements [stmts]
  (reduce + (map (fn [[kind & xs :as s]] (if (#{:loop :for} kind) (inc (count-statements (last s))) 1)) stmts)))

(defn scalarize
  "program (gen.device.stage's: a key, then data inputs) as statements in
   dialect: {:stmts :outputs [{:shape :dtype} per output, a scan's ys
   [length …]] :buffers :inputs :constants :statements (their count, loop
   bodies included)}. Each non-ys output's elements are stored at the end."
  [{:keys [graph inputs outputs]} dialect]
  (when-not (= (first inputs) [[2] :uint32])
    (throw (ex-info (str (:name dialect) ": a program of a key, then its data") {:inputs inputs})))
  (let [em (assoc (emitter dialect) :uses (uses graph outputs))
        ;; a program that gathers carries an error flag, false at the start
        ;; of every run, stored as a last, hidden output
        flag (when (gathers? graph)
               (let [c (fresh em "c")] (stmt! em [:decl :bool c ((:lit dialect) false :bool)]) c))
        em (assoc em :flag flag)
        by-id (into {} (map (juxt :id identity) graph))
        ys-out (into {} (keep-indexed (fn [k o] (let [{:keys [op args attrs]} (by-id o)]
                                                  (when (= op :result)
                                                    (let [s (by-id (first args)) nc (get-in s [:attrs :n-carry])]
                                                      (when (>= (:index attrs) nc) [[(:id s) (- (:index attrs) nc)] k])))))
                                      outputs))
        data (vec (map-indexed (fn [i [shape dtype]] {:name (str "in" (inc i)) :shape (vec shape) :dtype dtype}) (rest inputs)))
        env0 (into {} (map (fn [{:keys [id attrs]}]
                             (let [i (:index attrs)]
                               [id (if (zero? i)
                                     (value [2] :uint32 ["key0" "key1"])
                                     (let [{:keys [name shape dtype]} (nth data (dec i))]
                                       (if (large-shape? shape)
                                         (buffer-value em shape dtype name)
                                         ;; a small input read once into values
                                         (value shape dtype (mapv #(bind! em dtype (read-expr em name dtype (str %))) (range (size shape)))))))]))
                           (filter #(and (= :param (:op %)) (not (:capture (:attrs %)))) graph)))
        env (emit-graph! em graph env0 {:ys-out ys-out})
        outs (mapv env outputs)
        _ (doseq [[k {:keys [ys shape] :as v}] (map-indexed vector outs) :when (nil? ys)]
            (if (large? v)
              (for! em (size shape) (fn [bem w] (stmt! bem [:store k w ((:at v) bem w)])))
              (doseq [[e x] (map-indexed vector (:el v))] (stmt! em [:store k (str e) x]))))
        _ (when flag (stmt! em [:store (count outputs) "0" flag]))
        stmts @(:stmts em)]
    {:stmts stmts
     :outputs (cond-> (mapv (fn [{:keys [shape dtype ys]} o] {:shape (if ys (:shape (by-id o)) shape) :dtype dtype}) outs outputs)
                flag (conj {:shape [] :dtype :bool}))
     :error-output (when flag (count outputs))
     :buffers @(:buffers em)
     :inputs data
     :constants @(:constants em)
     :statements (count-statements stmts)}))

(defn int-literal
  "The value of a uint32 or int32 literal e (0x… with an optional u, or
   decimal), or nil when e is not one."
  [^String e]
  (cond (re-matches #"0x[0-9a-f]{8}u?" e) (Long/parseLong (subs e 2 10) 16)
        (re-matches #"-?[0-9]+" e) (Long/parseLong e)))

(defn words->key "A key of the two uint32 words [k0 k1]." [[k0 k1]] (gen.random.threefry.Key. (long k0) (long k1)))

(defn raise-run-error!
  "Run `run` of program read an array at an index out of range (its error
   flag set): its staged function run on the handler from key raises the
   handler's own error; when it does not (their rounding differs), a
   generic one (DESIGN §9, genclj-i2c4). An entry point calls it after the
   run, before returning anything."
  [program key run]
  (when-let [replay (or (:replay program) (when (= 1 (count (:inputs program))) (:f program)))]
    (replay key))
  (throw (ex-info (str "Device program: run " run " read an array at an index out of range (its run on the handler from"
                       " the same key did not throw: their rounding differs)")
                  {:reason :index-out-of-range :run run})))

