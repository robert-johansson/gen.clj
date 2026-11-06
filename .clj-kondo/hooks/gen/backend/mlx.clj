(ns hooks.gen.backend.mlx
  "The MLX backend's macros: export's defs, mlx-fn's fn of handles, and
   gen.backend.mlx.operands' binding macros, their handles bound around the
   body.")

(defmacro export [& syms]
  `(do ~@(for [s syms] `(def ~(symbol (name s)) ~s))))

(defmacro mlx-fn [f n]
  (let [args (repeatedly n #(gensym "h"))]
    `(fn [~@args s#] (~(symbol "Mlx" (str "mlx_" f)) ~@args s#))))

(defmacro with-temps [[t] & body]
  `(let [~t (fn ([h#] h#) ([kind# h#] [kind# h#]))] ~@body))

;; each handle bound to its operand form (with its dtype form evaluated),
;; as the operand's handle is
(defmacro with-operands [b op bindings & body]
  `(do ~b ~op
       (let [~@(mapcat (fn [[h x d]] [h `(do ~d ~x)]) (partition 3 bindings))]
         ~@body)))
