(ns hooks.gen.array
  "gen.array's def-macros, as they expand.")

(defmacro defunary [fname op doc]
  `(defn ~fname ~doc [~'x] (~'ew1 ~op ~'x)))

(defmacro defbinary [fname op doc]
  `(defn ~fname ~doc [~'x ~'y] (~'ew2 ~op ~'x ~'y)))
