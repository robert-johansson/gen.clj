(ns gen.device.scalar
  "A staged program (gen.device.stage) scalarized into statements, the
   shared core of the per-particle lowerings (gen.device.jvm, gen.device.
   metal; research prototype, ext/device, beans genclj-wbm7 and genclj-h8rn).

   Every node is scalarized at generation time, as gen.device.cuda does: an
   array is a vector of expressions, one per element, so reshapes,
   broadcasts, slices, transposes, pads and concatenations cost nothing and
   the compiler downstream sees straight-line scalar code. A counted scan
   is a loop whose carries are variables.

   (scalarize program dialect) returns the statements:
   [:decl dtype name expr], [:set name expr] (a carry's update), [:store
   output index-expr expr] (output k's element index-expr within one run)
   and [:loop it length body], with the outputs' specs. A dialect writes
   expressions in its language: a map of
     :name     the lowering's name, for errors
     :lit      (fn [v dtype]) a host value as a literal
     :simple?  (fn [expr]) a name or literal (not bound to a variable)
     :unary    (fn [op dtype a])
     :binary   (fn [op dtype a b])
     :convert  (fn [from to a])
     :bitcast  (fn [from to a])
   each throwing for what it cannot write. Variables are v<n> (values),
   c<n> (carries) and it<n> (loop counters); the key's words are key0 and
   key1."
  {:api :internal})

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

;; ---------------------------------------------------------------------------
;; Statements
;; ---------------------------------------------------------------------------

(defn- emitter
  ([dialect] {:stmts (volatile! []) :n (volatile! 0) :dialect dialect})
  ([em _] {:stmts (volatile! []) :n (:n em) :dialect (:dialect em)}))

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

(defn- max-elements ^long [] (Long/getLong "gen.device.max-elements" 4096))

(def ^:private bit-ops #{:bit-xor :bit-or :bit-and :shift-left :shift-right-logical})
(def ^:private comparisons #{:less :less-equal :greater :greater-equal :equal :not-equal})
(def ^:private binary-ops #{:add :subtract :multiply :divide :logical-and :logical-or :maximum :minimum :remainder :power :arctan2})
(def ^:private unary-ops #{:negative :logical-not :sign :isnan :isinf :abs :erfinv :erf :exp :log :expm1 :log1p :sqrt :sin :cos :tan
                           :tanh :sinh :cosh :arcsin :arccos :arctan :floor :ceil :round})

(def ^:private reduce-init {:sum 0 :prod 1 :max Double/NEGATIVE_INFINITY :min Double/POSITIVE_INFINITY :all true :any false})
(def ^:private reduce-op {:sum :add :prod :multiply :max :maximum :min :minimum :all :logical-and :any :logical-or})

(declare emit-graph! emit-scan!)

(defn- operand
  "An operand of a node: its value, or a constant's (a {:const x} of dtype)."
  [em env a dtype]
  (if (map? a)
    (let [d (if (boolean? (:const a)) :bool dtype)] (value [] d [((:lit (:dialect em)) (:const a) d)]))
    (let [v (or (env a) (fail! em "an operand with no value" {:id a}))]
      (when (:ys v) (fail! em "a scan's ys read in the program (only its outputs may be ys)" {:id a}))
      v)))

(defn- elementwise [em shape dtype operands f]
  (let [maps (mapv #(broadcast-map shape (:shape %)) operands)]
    (value shape dtype (mapv (fn [i] (bind! em dtype (apply f (map (fn [o m] (nth (:el o) (nth m i))) operands maps))))
                             (range (size shape))))))

(defn- reduce-value [em {:keys [shape el]} op axes out-shape out-dtype]
  (when-not (contains? reduce-init op) (fail! em (str "no reduction " op " yet") {:op op}))
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
                 (range (size kept))))))

(defn- permuted [shape perm in-shape el]
  (mapv (fn [i] (let [oi (unravel i shape)
                      ii (reduce (fn [v k] (assoc v (nth perm k) (nth oi k))) (vec (repeat (count shape) 0)) (range (count shape)))]
                  (nth el (ravel ii in-shape))))
        (range (size shape))))

(defn- emit-node!
  "env with node's value: {:shape :dtype :el} (expressions), or a scan's or
   its ys result's own entry."
  [em env {:keys [id op args shape dtype attrs] :as node} ctx]
  (when (> (size (or shape [])) (max-elements))
    (fail! em (str "an array of " (size shape) " elements, past gen.device.max-elements") {:node (dissoc node :attrs)}))
  (let [{:keys [lit unary binary convert bitcast]} (:dialect em)
        in (:in attrs)
        put #(assoc env id %)
        arg (fn [i d] (operand em env (nth args i) d))]
    (cond
      (= op :param) (if (env id) env (fail! em "an unbound param" {:node node}))
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
      (= op :reduce) (put (reduce-value em (arg 0 nil) (:op attrs) (:axes attrs) shape dtype))
      (= op :reshape) (put (assoc (arg 0 nil) :shape shape))
      (= op :transpose) (let [o (arg 0 nil)] (put (value shape dtype (permuted shape (:perm attrs) (:shape o) (:el o)))))
      (= op :concatenate) (let [ax (:axis attrs)
                                os (mapv #(operand em env % dtype) args)
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
                           _ (when (or (> (count sx) 2) (> (count sy) 2)) (fail! em "batched matmul" {:node node}))
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
      :else (fail! em (str "no lowering of " op " yet") {:node (dissoc node :attrs)}))))

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
          caps (mapv #(env %) (subvec args (+ n-carry n-x)))
          vars (mapv (fn [{:keys [shape dtype el]}]
                       (value shape dtype (mapv (fn [e] (let [v (fresh em "c")] (stmt! em [:decl dtype v e]) v)) el)))
                     inits)
          it (fresh em "it")
          bem (emitter em :nested)
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

(defn scalarize
  "program (gen.device.stage's, of one key) as statements in dialect:
   {:stmts, :outputs [{:shape :dtype} per output, a scan's ys [length …]],
   :statements (their count, loop bodies included)}. Each non-ys output's
   elements are stored at the end."
  [{:keys [graph inputs outputs]} dialect]
  (when-not (= inputs [[[2] :uint32]])
    (throw (ex-info (str (:name dialect) ": a program of one key (data inputs not yet)") {:inputs inputs})))
  (let [em (emitter dialect)
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
        stmts @(:stmts em)]
    {:stmts stmts
     :outputs (mapv (fn [{:keys [shape dtype ys]} o] {:shape (if ys (:shape (by-id o)) shape) :dtype dtype}) outs outputs)
     :statements (reduce + (map #(if (= :loop (first %)) (count (nth % 3)) 1) stmts))}))

(defn int-literal
  "The value of a uint32 or int32 literal e (0x… with an optional u, or
   decimal), or nil when e is not one."
  [^String e]
  (cond (re-matches #"0x[0-9a-f]{8}u?" e) (Long/parseLong (subs e 2 10) 16)
        (re-matches #"-?[0-9]+" e) (Long/parseLong e)))
