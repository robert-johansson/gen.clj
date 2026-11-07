(ns hooks.gen.array.gguf
  "gen.array.gguf's loop over a range of blocks, as it expands.")

(defmacro each-block [[i from to] & body]
  `(let [to# (long ~to)]
     (loop [~i (long ~from)]
       (when (< ~i to#)
         ~@body
         (recur (inc ~i))))))
