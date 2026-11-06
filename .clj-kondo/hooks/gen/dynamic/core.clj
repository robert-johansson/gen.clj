(ns hooks.gen.dynamic.core
  "gen.dynamic.core's def-macros, as they expand (the guarded function's
   name, metadata only, left out). defguard rewrites the nodes, so that a
   syntax quote in its attr-map (get's :inline) keeps its unquotes, which a
   macroexpand hook would read as data."
  (:require [clj-kondo.hooks-api :as api]))

(defn defguard [{:keys [node]}]
  (let [[_ fname _guarded & fdecl] (:children node)]
    {:node (api/list-node (list* (api/token-node 'clojure.core/defn) fname fdecl))}))

(defmacro def-representation-tests [& fnames]
  `(do ~@(for [n fnames]
           `(defn ~n [~'x] (~(symbol "clojure.core" (name n)) (~'particle ~'x))))))
