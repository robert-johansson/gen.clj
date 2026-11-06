(ns hooks.gen.inference
  "gen.inference.kernel's kern and gen.inference.transform's transform and
   deftransform: their statements in the scope of the names they bind."
  (:require [clj-kondo.hooks-api :as api]
            [hooks.gen.util :refer [used]]))

(defn- trace-flag?
  "Does the node carry ^:trace? (Read from the node: a macroexpand hook's
   form has lost its metadata.)"
  [node]
  (some (fn [m] (let [v (api/sexpr m)] (or (= :trace v) (and (map? v) (:trace v))))) (:meta node)))

(defn kern
  "An analyze-call hook, to see the parameters' ^:trace: with it, a kernel
   (a fn of a trace and a key) whose statements read the trace, inside a fn
   of the other arguments; else a fn of the arguments, or with none the
   kernel itself."
  [{:keys [node]}]
  (let [[_ params & body] (:children node)
        fn-node (fn [ps forms] (api/list-node (list* (api/token-node 'clojure.core/fn) (api/vector-node ps) forms)))]
    {:node (cond
             (trace-flag? params) (let [[tr & args] (:children params)
                                        k (fn-node [tr (with-meta (api/token-node 'key) {:clj-kondo/mark-used true})] body)]
                                    (if (seq args) (fn-node args [k]) k))
             (seq (:children params)) (fn-node (:children params) body)
             :else (api/list-node (list* (api/token-node 'do) body)))}))

;; A transform's value is a map (the Transform record), its body a fn of the
;; four spaces. The spaces are positional, named in full by convention
;; (Gen.jl's [model-in aux-in model-out aux-out]) whether or not the body
;; reads each: never an unused binding.
(defn- transform-form [spaces body]
  `{:body (fn [~@(map used spaces)] ~@body nil)})

(defmacro transform [spaces & body]
  (transform-form spaces body))

(defmacro deftransform [tname & more]
  (let [[doc [spaces & body]] (if (string? (first more)) [(first more) (rest more)] [nil more])]
    `(def ~tname ~@(when doc [doc]) ~(transform-form spaces body))))
