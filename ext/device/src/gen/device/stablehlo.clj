(ns gen.device.stablehlo
  "A staged program (gen.device.stage) as StableHLO, MLIR's text form that
   XLA compiles (through PJRT, gen.device.pjrt). Research prototype (bean
   genclj-a1hu, ext/device).

   (module program) is the text of a module whose @main takes the
   program's inputs and returns its outputs. Ops are written in MLIR's
   generic form; elementwise operands are broadcast to the result's shape
   (broadcast_in_dim, NumPy's alignment), host numbers become constants,
   and a scan is a stablehlo.while whose operands are its counter, its
   carries, its outputs' buffers, its inputs and what its body captures."
  {:api :internal}
  (:require [clojure.string :as str]))

(set! *warn-on-reflection* true)

;; ---------------------------------------------------------------------------
;; Types and constants
;; ---------------------------------------------------------------------------

(def ^:private element-types
  {:float32 "f32" :float64 "f64" :int32 "i32" :int64 "i64" :uint32 "ui32" :bool "i1"})

(defn- element-type [dtype]
  (or (element-types dtype) (throw (ex-info (str "StableHLO: no element type for " dtype) {:dtype dtype}))))

(defn- ty [shape dtype]
  (str "tensor<" (str/join (map #(str % "x") shape)) (element-type dtype) ">"))

(defn- literal
  "A host value as an MLIR literal of dtype: floats exactly (a bit pattern in
   hex where they are not finite)."
  [v dtype]
  (case dtype
    :bool (if (boolean? v) (str v) (str (not (zero? (double v)))))
    :float32 (let [f (unchecked-float (if (boolean? v) (if v 1.0 0.0) (double v)))]
               (if (or (Float/isNaN f) (Float/isInfinite f))
                 (format "0x%08X" (Float/floatToRawIntBits f))
                 (str f)))
    :float64 (let [d (if (boolean? v) (if v 1.0 0.0) (double v))]
               (if (or (Double/isNaN d) (Double/isInfinite d))
                 (format "0x%016X" (Double/doubleToRawLongBits d))
                 (str d)))
    (:int32 :int64 :uint32) (str (if (boolean? v) (if v 1 0) (long v)))))

(defn- nested
  "The values vs (row-major) as nested brackets of shape."
  [vs shape dtype]
  (if (empty? shape)
    (literal (first vs) dtype)
    (let [n (long (first shape))
          inner (rest shape)
          step (reduce * 1 inner)]
      (str "[" (str/join ", " (map #(nested (take step (drop (* % step) vs)) inner dtype) (range n))) "]"))))

(defn- dense [vs shape dtype]
  (if (or (= 1 (count vs)) (apply = vs))
    (str "dense<" (literal (first vs) dtype) ">")
    (str "dense<" (nested vs shape dtype) ">")))

(defn- i64s [xs] (str "array<i64" (when (seq xs) (str ": " (str/join ", " xs))) ">"))

;; ---------------------------------------------------------------------------
;; Emitting
;; ---------------------------------------------------------------------------

(defn- emitter
  "A fresh name source and line sink."
  []
  {:n (volatile! 0) :out (StringBuilder.)})

(defn- fresh [{:keys [n]}] (str "%t" (vswap! n inc)))

(defn- line! [{:keys [^StringBuilder out]} indent s]
  (.append out (apply str (repeat indent "  "))) (.append out ^String s) (.append out "\n"))

(defn- op-line!
  "One generic op: its result's name."
  [em indent op operands in-types out-type & [attrs]]
  (let [r (fresh em)]
    (line! em indent (str r " = \"" op "\"(" (str/join ", " operands) ")"
                          (when attrs (str " {" attrs "}"))
                          " : (" (str/join ", " in-types) ") -> " out-type))
    r))

(defn- constant! [em indent vs shape dtype]
  (op-line! em indent "stablehlo.constant" [] [] (ty shape dtype) (str "value = " (dense vs shape dtype) " : " (ty shape dtype))))

(defn- broadcast!
  "v (of shape from) broadcast to shape, NumPy's trailing alignment."
  [em indent v from shape dtype]
  (if (= (vec from) (vec shape))
    v
    (let [k (- (count shape) (count from))]
      (op-line! em indent "stablehlo.broadcast_in_dim" [v] [(ty from dtype)] (ty shape dtype)
                (str "broadcast_dimensions = " (i64s (range k (count shape))))))))

(def ^:private unary
  {:negative "stablehlo.negate" :abs "stablehlo.abs" :sign "stablehlo.sign" :exp "stablehlo.exponential"
   :log "stablehlo.log" :expm1 "stablehlo.exponential_minus_one" :log1p "stablehlo.log_plus_one"
   :sqrt "stablehlo.sqrt" :sin "stablehlo.sine" :cos "stablehlo.cosine" :tan "stablehlo.tan"
   :tanh "stablehlo.tanh" :sinh "chlo.sinh" :cosh "chlo.cosh" :arcsin "chlo.asin" :arccos "chlo.acos"
   :arctan "chlo.atan" :erf "chlo.erf" :erfinv "chlo.erf_inv" :floor "stablehlo.floor" :ceil "stablehlo.ceil"
   :round "stablehlo.round_nearest_even" :logical-not "stablehlo.not"})

(def ^:private binary
  {:add "stablehlo.add" :subtract "stablehlo.subtract" :multiply "stablehlo.multiply" :divide "stablehlo.divide"
   :power "stablehlo.power" :maximum "stablehlo.maximum" :minimum "stablehlo.minimum" :arctan2 "stablehlo.atan2"
   :logical-and "stablehlo.and" :logical-or "stablehlo.or"
   :bit-xor "stablehlo.xor" :bit-or "stablehlo.or" :shift-left "stablehlo.shift_left"
   :shift-right-logical "stablehlo.shift_right_logical"})

(def ^:private comparisons
  {:less "LT" :less-equal "LE" :greater "GT" :greater-equal "GE" :equal "EQ" :not-equal "NE"})

(defn- compare-type [dtype]
  (case dtype (:float32 :float64) "FLOAT" (:int32 :int64) "SIGNED" (:uint32 :bool) "UNSIGNED"))

(defn- compare! [em indent dir a b shape dtype]
  (op-line! em indent "stablehlo.compare" [a b] [(ty shape dtype) (ty shape dtype)] (ty shape :bool)
            (str "comparison_direction = #stablehlo<comparison_direction " dir ">, compare_type = #stablehlo<comparison_type "
                 (compare-type dtype) ">")))

(def ^:private bit-ops #{:bit-xor :bit-or :shift-left :shift-right-logical})

(declare emit-nodes!)

(defn- reducer
  "A stablehlo.reduce of v over axes with op: [name, the reduced shape]."
  [em indent op v shape dtype axes]
  (let [[body init] (case op
                      :sum ["stablehlo.add" 0]
                      :prod ["stablehlo.multiply" 1]
                      :max ["stablehlo.maximum" Double/NEGATIVE_INFINITY]
                      :min ["stablehlo.minimum" Double/POSITIVE_INFINITY]
                      :all ["stablehlo.and" true]
                      :any ["stablehlo.or" false]
                      (throw (ex-info (str "StableHLO: no reduction " op " yet") {:op op})))
        init (if (and (#{:max :min} op) (#{:int32 :int64 :uint32} dtype))
               (if (= op :max) Integer/MIN_VALUE Integer/MAX_VALUE)
               init)
        out-shape (vec (keep-indexed (fn [i d] (when-not ((set axes) i) d)) shape))
        c (constant! em indent [init] [] dtype)
        r (fresh em)
        a (fresh em) b (fresh em) s (fresh em)
        st (ty [] dtype)]
    (line! em indent (str r " = \"stablehlo.reduce\"(" v ", " c ") ({"))
    (line! em (inc indent) (str "^bb0(" a ": " st ", " b ": " st "):"))
    (line! em (inc indent) (str s " = \"" body "\"(" a ", " b ") : (" st ", " st ") -> " st))
    (line! em (inc indent) (str "\"stablehlo.return\"(" s ") : (" st ") -> ()"))
    (line! em indent (str "}) {dimensions = " (i64s axes) "} : (" (ty shape dtype) ", " st ") -> " (ty out-shape dtype)))
    [r out-shape]))

(def ^:private small-reducers
  {:sum "stablehlo.add" :prod "stablehlo.multiply" :max "stablehlo.maximum" :min "stablehlo.minimum"
   :all "stablehlo.and" :any "stablehlo.or"})

(defn- small-reduce!
  "The whole of v (n ≤ 16 elements) reduced by op as a chain of its
   elements, shaped out-shape (all ones, or [])."
  [em indent op v shape dtype out-shape]
  (let [n (reduce * 1 shape)
        flat (if (= [n] (vec shape)) v (op-line! em indent "stablehlo.reshape" [v] [(ty shape dtype)] (ty [n] dtype)))
        elems (mapv (fn [i] (let [sl (op-line! em indent "stablehlo.slice" [flat] [(ty [n] dtype)] (ty [1] dtype)
                                              (str "start_indices = " (i64s [i]) ", limit_indices = " (i64s [(inc i)])
                                                   ", strides = " (i64s [1])))]
                              (op-line! em indent "stablehlo.reshape" [sl] [(ty [1] dtype)] (ty [] dtype))))
                    (range n))
        r (reduce (fn [a b] (op-line! em indent (small-reducers op) [a b] [(ty [] dtype) (ty [] dtype)] (ty [] dtype))) elems)]
    (if (empty? out-shape) r (op-line! em indent "stablehlo.reshape" [r] [(ty [] dtype)] (ty out-shape dtype)))))

(def ^:private highest
  "Float matrix products at full precision: never TF32 (DESIGN §9, TF32 off
   for densities)."
  "precision_config = [#stablehlo<precision HIGHEST>, #stablehlo<precision HIGHEST>]")

(defn- dot! [em indent x y sx sy dtype out-shape {:keys [lb rb lc rc]}]
  (op-line! em indent "stablehlo.dot_general" [x y] [(ty sx dtype) (ty sy dtype)] (ty out-shape dtype)
            (str "dot_dimension_numbers = #stablehlo.dot<"
                 (when (seq lb) (str "lhs_batching_dimensions = [" (str/join ", " lb) "], "
                                     "rhs_batching_dimensions = [" (str/join ", " rb) "], "))
                 "lhs_contracting_dimensions = [" lc "], rhs_contracting_dimensions = [" rc "]>, " highest)))

(defn- matmul!
  "NumPy's matmul: vectors and matrices as they are; batched matrices with
   equal batch axes by dot_general's batching; one side's batch axes all 1
   (an unbatched operand meeting a batched one, as the batch tracer lifts
   it) contracted without them, never broadcast to the other's batch (at
   ndreg's largest, 10^10 elements)."
  [em indent [x y] [sx sy] dtype out-shape]
  (let [rx (count sx) ry (count sy)
        reshape (fn [v from to] (if (= from to) v (op-line! em indent "stablehlo.reshape" [v] [(ty from dtype)] (ty to dtype))))]
    (cond
      (and (<= rx 2) (<= ry 2))
      (dot! em indent x y sx sy dtype out-shape {:lc (dec rx) :rc (if (= ry 1) 0 (- ry 2))})

      (and (= rx ry) (= (subvec sx 0 (- rx 2)) (subvec sy 0 (- ry 2))))
      (dot! em indent x y sx sy dtype out-shape {:lb (range (- rx 2)) :rb (range (- rx 2)) :lc (dec rx) :rc (- ry 2)})

      ;; x unbatched: [1… m k] · [B… k n] → [m B… n] → [B… m n]
      (and (= rx ry) (every? #(= 1 %) (subvec sx 0 (- rx 2))))
      (let [mk (subvec sx (- rx 2))
            x' (reshape x sx mk)
            bs (subvec sy 0 (- ry 2))
            n (peek sy)
            raw (into [(first mk)] (conj bs n))
            d (dot! em indent x' y mk sy dtype raw {:lc 1 :rc (- ry 2)})
            perm (vec (concat (range 1 (inc (count bs))) [0] [(inc (count bs))]))]
        (op-line! em indent "stablehlo.transpose" [d] [(ty raw dtype)] (ty out-shape dtype)
                  (str "permutation = " (i64s perm))))

      ;; y unbatched: [B… m k] · [1… k n] → [B… m n]
      (and (= rx ry) (every? #(= 1 %) (subvec sy 0 (- ry 2))))
      (let [kn (subvec sy (- ry 2))]
        (dot! em indent x (reshape y sy kn) sx kn dtype out-shape {:lc (dec rx) :rc 0}))

      :else (throw (ex-info "StableHLO: this batched matmul's broadcasting, not yet" {:shapes [sx sy]})))))

(defn- scatter-add!
  "x with updates added at indices along axis (indices and updates of one
   shape): a stablehlo.scatter of full index tuples."
  [em indent [x idx upd] [sx si su] dtype idx-dtype axis]
  (let [r (count sx)
        it (fn [s d] (ty s d))
        ishape (conj (vec su) 1)
        cols (mapv (fn [k]
                     (let [col (if (= k axis)
                                 (op-line! em indent "stablehlo.convert" [idx] [(it si idx-dtype)] (it si :int32))
                                 (op-line! em indent "stablehlo.iota" [] [] (it su :int32) (str "iota_dimension = " k " : i64")))]
                       (op-line! em indent "stablehlo.reshape" [col] [(it su :int32)] (it ishape :int32))))
                   (range r))
        full (if (= 1 r)
               (first cols)
               (op-line! em indent "stablehlo.concatenate" cols (repeat r (it ishape :int32)) (it (conj (vec su) r) :int32)
                         (str "dimension = " r " : i64")))
        res (fresh em) a (fresh em) b (fresh em) s (fresh em)
        st (ty [] dtype)]
    (line! em indent (str res " = \"stablehlo.scatter\"(" x ", " full ", " upd ") ({"))
    (line! em (inc indent) (str "^bb0(" a ": " st ", " b ": " st "):"))
    (line! em (inc indent) (str s " = \"stablehlo.add\"(" a ", " b ") : (" st ", " st ") -> " st))
    (line! em (inc indent) (str "\"stablehlo.return\"(" s ") : (" st ") -> ()"))
    (line! em indent (str "}) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<"
                          "inserted_window_dims = [" (str/join ", " (range r)) "], "
                          "scatter_dims_to_operand_dims = [" (str/join ", " (range r)) "], index_vector_dim = " r ">, "
                          "unique_indices = false} : (" (it sx dtype) ", " (it (conj (vec su) r) :int32) ", " (it su dtype)
                          ") -> " (it sx dtype)))
    res))

(defn- emit-scan!
  "A scan node as a stablehlo.while: returns the name of its results (name#i)."
  [em indent {:keys [args attrs]} env]
  (let [{:keys [body length n-carry n-x]} attrs
        {:keys [graph carries xs captures carry-out y-out]} body
        arg-info (mapv env args)
        [carry-in x-in cap-in] [(subvec arg-info 0 n-carry) (subvec arg-info n-carry (+ n-carry n-x))
                                (subvec arg-info (+ n-carry n-x))]
        y-shapes (mapv (fn [id] (let [{:keys [shape dtype]} (nth graph id)] [shape dtype])) y-out)
        accs (mapv (fn [[shape dtype]] (constant! em indent [(if (= dtype :bool) false 0)] (into [length] shape) dtype)) y-shapes)
        i0 (constant! em indent [0] [] :int32)
        operands (vec (concat [[i0 [] :int32]] carry-in (map (fn [a [shape dtype]] [a (into [length] shape) dtype]) accs y-shapes)
                              x-in cap-in))
        types (mapv (fn [[_ shape dtype]] (ty shape dtype)) operands)
        w (fresh em)
        block-args (fn [] (mapv (fn [_] (fresh em)) operands))
        sig (fn [names] (str/join ", " (map (fn [nm t] (str nm ": " t)) names types)))
        ret (fn [indent names] (line! em indent (str "\"stablehlo.return\"(" (str/join ", " names) ") : ("
                                                     (str/join ", " types) ") -> ()")))]
    (line! em indent (str w ":" (count operands) " = \"stablehlo.while\"(" (str/join ", " (map first operands)) ") ({"))
    ;; cond: the counter below length
    (let [names (block-args)
          _ (line! em (inc indent) (str "^bb0(" (sig names) "):"))
          lim (constant! em (inc indent) [length] [] :int32)
          c (compare! em (inc indent) "LT" (first names) lim [] :int32)]
      (line! em (inc indent) (str "\"stablehlo.return\"(" c ") : (tensor<i1>) -> ()")))
    (line! em indent "}, {")
    (let [names (block-args)
          _ (line! em (inc indent) (str "^bb0(" (sig names) "):"))
          i (first names)
          [carry-names acc-names x-names cap-names] [(subvec names 1 (inc n-carry))
                                                    (subvec names (inc n-carry) (+ 1 n-carry (count accs)))
                                                    (subvec names (+ 1 n-carry (count accs)) (+ 1 n-carry (count accs) n-x))
                                                    (subvec names (+ 1 n-carry (count accs) n-x))]
          z (constant! em (inc indent) [0] [] :int32)
          x-vals (mapv (fn [nm [_ shape dtype]]
                         (let [inner (vec (rest shape))
                               sl (op-line! em (inc indent) "stablehlo.dynamic_slice" (into [nm i] (repeat (count inner) z))
                                            (into [(ty shape dtype)] (repeat (inc (count inner)) "tensor<i32>"))
                                            (ty (into [1] inner) dtype)
                                            (str "slice_sizes = " (i64s (into [1] inner))))]
                           (op-line! em (inc indent) "stablehlo.reshape" [sl] [(ty (into [1] inner) dtype)] (ty inner dtype))))
                       x-names x-in)
          benv (merge (zipmap carries (map (fn [nm [_ shape dtype]] [nm shape dtype]) carry-names carry-in))
                      (zipmap xs (map (fn [v [_ shape dtype]] [v (vec (rest shape)) dtype]) x-vals x-in))
                      (zipmap captures (map (fn [nm [_ shape dtype]] [nm shape dtype]) cap-names cap-in)))
          benv (emit-nodes! em (inc indent) graph benv)
          accs' (mapv (fn [acc id [shape dtype]]
                        (let [[y] (benv id)
                              r (op-line! em (inc indent) "stablehlo.reshape" [y] [(ty shape dtype)] (ty (into [1] shape) dtype))]
                          (op-line! em (inc indent) "stablehlo.dynamic_update_slice" (into [acc r i] (repeat (count shape) z))
                                    (into [(ty (into [length] shape) dtype) (ty (into [1] shape) dtype)]
                                          (repeat (inc (count shape)) "tensor<i32>"))
                                    (ty (into [length] shape) dtype))))
                      acc-names y-out y-shapes)
          one (constant! em (inc indent) [1] [] :int32)
          i' (op-line! em (inc indent) "stablehlo.add" [i one] ["tensor<i32>" "tensor<i32>"] "tensor<i32>")]
      (ret (inc indent) (vec (concat [i'] (map #(first (benv %)) carry-out) accs' x-names cap-names))))
    (line! em indent (str "}) : (" (str/join ", " types) ") -> (" (str/join ", " types) ")"))
    {:name w :n-carry n-carry}))

(defn- emit-node!
  "Emits node; returns env with its [name shape dtype]."
  [em indent env {:keys [id op args shape dtype attrs] :as node}]
  (let [val-of (fn [a in-dtype]
                 (if (map? a)
                   (let [d (if (boolean? (:const a)) :bool in-dtype)] [(constant! em indent [(:const a)] [] d) [] d])
                   (env a)))
        bcast (fn [[v s d]] [(broadcast! em indent v s shape d) shape d])
        put (fn [name] (assoc env id [name shape dtype]))
        in (:in attrs)]
    (cond
      (= op :param) (if (env id) env (throw (ex-info "StableHLO: an unbound param" {:node node})))
      (= op :constant) (put (constant! em indent (:values attrs) shape dtype))
      (= op :iota) (put (op-line! em indent "stablehlo.iota" [] [] (ty shape dtype) "iota_dimension = 0 : i64"))
      (= op :broadcast) (let [[v s d] (val-of (first args) dtype)] (put (broadcast! em indent v s shape d)))
      (= op :convert) (let [[v s d] (env (first args))]
                        (put (if (= d dtype) v (op-line! em indent "stablehlo.convert" [v] [(ty s d)] (ty shape dtype)))))
      (= op :bitcast) (let [[v s d] (env (first args))]
                        (put (op-line! em indent "stablehlo.bitcast_convert" [v] [(ty s d)] (ty shape dtype))))
      (= op :stop-gradient) (assoc env id (env (first args)))
      (contains? unary op) (let [[v] (bcast (val-of (first args) in))]
                             (put (op-line! em indent (unary op) [v] [(ty shape in)] (ty shape dtype))))
      (contains? comparisons op) (let [[a] (bcast (val-of (first args) in)) [b] (bcast (val-of (second args) in))]
                                   (put (compare! em indent (comparisons op) a b shape in)))
      (= op :isnan) (let [[a] (val-of (first args) in)] (put (compare! em indent "NE" a a shape in)))
      (= op :isinf) (let [[a] (val-of (first args) in)
                          ab (op-line! em indent "stablehlo.abs" [a] [(ty shape in)] (ty shape in))
                          inf (broadcast! em indent (constant! em indent [Double/POSITIVE_INFINITY] [] in) [] shape in)]
                      (put (compare! em indent "EQ" ab inf shape in)))
      (contains? bit-ops op) (let [vs (mapv #(first (bcast (val-of % :uint32))) args)]
                               (put (op-line! em indent (binary op) vs [(ty shape :uint32) (ty shape :uint32)] (ty shape :uint32))))
      (contains? binary op) (let [vs (mapv #(first (bcast (val-of % in))) args)]
                              (put (op-line! em indent (binary op) vs [(ty shape in) (ty shape in)] (ty shape dtype))))
      (= op :remainder) (let [[a b] (mapv #(first (bcast (val-of % in))) args)
                              t (ty shape in)
                              r (op-line! em indent "stablehlo.remainder" [a b] [t t] t)
                              z (broadcast! em indent (constant! em indent [0] [] in) [] shape in)
                              fix (op-line! em indent "stablehlo.and"
                                            [(compare! em indent "NE" r z shape in)
                                             (compare! em indent "NE" (compare! em indent "LT" r z shape in)
                                                       (compare! em indent "LT" b z shape in) shape :bool)]
                                            [(ty shape :bool) (ty shape :bool)] (ty shape :bool))]
                          (put (op-line! em indent "stablehlo.select" [fix (op-line! em indent "stablehlo.add" [r b] [t t] t) r]
                                         [(ty shape :bool) t t] t)))
      (= op :where) (let [[c] (bcast (val-of (first args) :bool))
                          [x] (bcast (val-of (second args) in))
                          [y] (bcast (val-of (nth args 2) in))]
                      (put (op-line! em indent "stablehlo.select" [c x y] [(ty shape :bool) (ty shape in) (ty shape in)] (ty shape dtype))))
      (= op :reduce) (let [[v s d] (env (first args))
                           {rop :op axes :axes keep :keep-dims?} attrs
                           n (reduce * 1 s)]
                       (if (and (= (count axes) (count s)) (<= 1 n 16) (small-reducers rop))
                         ;; a whole small array: an elementwise chain, which
                         ;; fuses with its producers and consumers
                         (put (small-reduce! em indent rop v s d shape))
                         (let [[r rs] (reducer em indent rop v s d axes)]
                           (put (if keep (op-line! em indent "stablehlo.reshape" [r] [(ty rs dtype)] (ty shape dtype)) r)))))
      (= op :pad) (let [[v s d] (env (first args)) {:keys [axis low high interior]} attrs
                        r (count s)
                        at (fn [k] (i64s (map #(if (= % axis) k 0) (range r))))
                        z (constant! em indent [0] [] d)]
                    (put (op-line! em indent "stablehlo.pad" [v z] [(ty s d) (ty [] d)] (ty shape dtype)
                                   (str "edge_padding_low = " (at low) ", edge_padding_high = " (at high)
                                        ", interior_padding = " (at interior)))))
      (= op :reshape) (let [[v s d] (env (first args))]
                        (put (if (= s shape) v (op-line! em indent "stablehlo.reshape" [v] [(ty s d)] (ty shape dtype)))))
      (= op :transpose) (let [[v s d] (env (first args))]
                          (put (op-line! em indent "stablehlo.transpose" [v] [(ty s d)] (ty shape dtype)
                                         (str "permutation = " (i64s (:perm attrs))))))
      (= op :concatenate) (let [vs (mapv env args)]
                            (put (op-line! em indent "stablehlo.concatenate" (mapv first vs) (mapv (fn [[_ s d]] (ty s d)) vs)
                                           (ty shape dtype) (str "dimension = " (:axis attrs) " : i64"))))
      (= op :slice) (let [[v s d] (env (first args)) {:keys [starts stops steps]} attrs]
                      (when (some neg? steps) (throw (ex-info "StableHLO: a reversed slice, not yet" {:node node})))
                      (put (op-line! em indent "stablehlo.slice" [v] [(ty s d)] (ty shape dtype)
                                     (str "start_indices = " (i64s starts) ", limit_indices = " (i64s stops)
                                          ", strides = " (i64s steps)))))
      (= op :matmul) (let [vs (mapv env args)]
                       (put (matmul! em indent (mapv first vs) (mapv second vs) dtype shape)))
      (= op :scatter-add) (let [[x i u] (mapv env args)]
                            (put (scatter-add! em indent [(first x) (first i) (first u)] [(second x) (second i) (second u)]
                                               dtype (nth i 2) (:axis attrs))))
      (= op :scan) (assoc env id (emit-scan! em indent node env))
      (= op :result) (let [{:keys [name]} (env (first args))
                           i (:index attrs)]
                       ;; while's results: the counter, the carries, the ys' buffers
                       (assoc env id [(str name "#" (inc i)) shape dtype]))
      :else (throw (ex-info (str "StableHLO: no lowering of " op " yet") {:node node})))))

(defn- emit-nodes! [em indent nodes env]
  (reduce (fn [env node] (emit-node! em indent env node)) env nodes))

(defn module
  "The program (gen.device.stage/stage's) as a StableHLO module's text."
  [{:keys [graph inputs outputs]}]
  (let [em (emitter)
        params (filter #(= :param (:op %)) graph)
        arg-names (mapv #(str "%arg" %) (range (count inputs)))
        env (into {} (map (fn [{:keys [id shape dtype attrs]}] [id [(arg-names (:index attrs)) shape dtype]]) params))
        env (emit-nodes! em 2 graph env)
        outs (mapv env outputs)
        header (str "module @program {\n  func.func public @main("
                    (str/join ", " (map (fn [nm [shape dtype]] (str nm ": " (ty shape dtype))) arg-names inputs))
                    ") -> (" (str/join ", " (map (fn [[_ s d]] (ty s d)) outs)) ") {\n")]
    (str header (.toString ^StringBuilder (:out em))
         "    \"func.return\"(" (str/join ", " (map first outs)) ") : (" (str/join ", " (map (fn [[_ s d]] (ty s d)) outs)) ") -> ()\n"
         "  }\n}\n")))
