(ns gen.dynamic.static
  "The static modeling language (Gen.jl's @gen (static)): a gen body of a
   fixed structure, (gen ^:static [args] (let [bindings…] ret)), whose
   graph lets update and regenerate skip what did not change.

   The body is one let. Each binding is a call at a constant address,
   (trace addr dist) or (splice addr gf & args), or a pure expression of the
   arguments and earlier bindings (any pure Clojure, including destructuring);
   then the return expression. Branching happens only inside called
   generative functions. This namespace checks a body at macroexpansion and
   builds its graph as data; gen.dynamic runs it.

     (ir gf)    the graph: the arguments, the nodes in order ({:kind :trace,
                :splice or :compute, :addr, :names, :inputs, :form}) and the
                return expression's inputs
     (dot gf)   the graph in Graphviz's DOT (Gen.jl's draw_graph)"
  (:require [clojure.string :as str]))

(def ^:private effects
  "The body's own calls, which a static body makes only at a binding's head."
  '#{trace splice untraced untraced-key})

(defn- symbols [form] (filter symbol? (tree-seq coll? seq form)))

(defn- not-static [why form]
  (throw (ex-info (str "gen ^:static: " why ": " (pr-str form)
                       " (a static body is one let of calls at constant addresses and pure bindings;"
                       " use gen without ^:static for any other body)")
                  {:form form})))

(defn- constant-address? [addr]
  (or (keyword? addr) (string? addr) (number? addr) (boolean? addr)
      (and (vector? addr) (seq addr) (every? constant-address? addr))))

(defn- pure!
  "form, checked to make none of the body's own calls."
  [form what]
  (when-let [e (first (filter effects (symbols form)))]
    (not-static (str what " calls " e ", which a static body may do only as a binding's whole expression") form))
  form)

(defn bound-names
  "The symbols a binding form binds (destructuring included), in order;
   _ binds nothing."
  [binding]
  (let [written (disj (set (symbols binding)) '_)]
    (vec (distinct (filter written (take-nth 2 (destructure [binding nil])))))))

(defn- inputs [form known] (vec (distinct (filter known (symbols form)))))

(defn- node
  "The node of one binding, given the names bound so far."
  [binding init known]
  (let [names (bound-names binding)
        head (when (seq? init) (first init))]
    (merge {:binding binding :names names :form init}
           (cond
             (= 'trace head)
             (let [[_ addr dist & more] init]
               (when (seq more) (not-static "trace takes an address and a distribution" init))
               (when-not (constant-address? addr) (not-static "trace's address must be a constant" init))
               {:kind :trace :addr addr :expr (pure! dist "a distribution") :inputs (inputs dist known)})

             (= 'splice head)
             (let [[_ addr gf & args] init]
               (when-not (constant-address? addr)
                 (not-static "splice needs a constant address (a splice without one is dynamic)" init))
               (let [expr `[~gf [~@args]]]
                 {:kind :splice :addr addr :expr (pure! expr "a splice's arguments") :inputs (inputs expr known)}))

             :else
             {:kind :compute :expr (pure! init "a binding") :inputs (inputs init known)}))))

(defn arg-names
  "[[names…] per argument], the rest parameter's names last, and whether
   there is one."
  [params]
  (let [i (.indexOf ^java.util.List params '&)
        fixed (if (neg? i) params (subvec params 0 i))]
    {:fixed (mapv bound-names fixed)
     :rest (when-not (neg? i) (bound-names (nth params (inc i))))}))

(defn analyze
  "The graph of a static body (a body of one let), or a loud error: {:args
   (arg-names), :nodes [{:kind :addr :binding :names :inputs :expr :form}],
   :ret {:inputs :expr}}."
  [params body]
  (when-not (= 1 (count body)) (not-static "the body is one let form" (vec body)))
  (let [form (first body)]
    (when-not (and (seq? form) (#{'let 'clojure.core/let} (first form)))
      (not-static "the body is one let form" form))
    (let [[_ bindings & ret] form
          _ (when-not (and (vector? bindings) (even? (count bindings))) (not-static "a let needs pairs of bindings" form))
          _ (when-not (= 1 (count ret)) (not-static "the let returns one expression" form))
          args (arg-names params)
          known (set (concat (apply concat (:fixed args)) (:rest args)))
          [nodes _] (reduce (fn [[nodes known] [binding init]]
                              (let [n (node binding init known)]
                                [(conj nodes n) (into known (:names n))]))
                            [[] known] (partition 2 bindings))
          names (into known (mapcat :names nodes))]
      {:args args
       :nodes nodes
       :ret {:expr (pure! (first ret) "the return expression") :inputs (inputs (first ret) names)}})))

(defn ir
  "gf's graph as data (see the namespace doc), or nil when gf is not static."
  [gf]
  (when-let [{:keys [args nodes ret]} (:static gf)]
    {:args args
     :nodes (mapv #(select-keys % [:kind :addr :names :inputs :form]) nodes)
     :return {:inputs (:inputs ret)}}))

(defn dot
  "gf's graph in Graphviz's DOT: argument, choice, call and computation
   nodes, an edge from each input."
  [gf]
  (let [{:keys [args nodes return]} (or (ir gf) (throw (ex-info "dot: not a static generative function" {})))
        id #(str "\"" % "\"")
        arg-lines (for [n (concat (apply concat (:fixed args)) (:rest args))] (str "  " (id n) " [shape=plaintext];"))
        node-id (fn [i n] (if (:addr n) (pr-str (:addr n)) (str "#" i)))
        owner (into {} (for [[i n] (map-indexed vector nodes) s (:names n)] [s (node-id i n)]))
        lines (for [[i n] (map-indexed vector nodes)
                    :let [me (node-id i n)]
                    line (cons (str "  " (id me) " [shape=" (case (:kind n) :trace "ellipse" :splice "box" "diamond")
                                    " label=" (id (str me (when (seq (:names n)) (str " → " (str/join " " (:names n))))))
                                    "];")
                               (for [s (:inputs n)] (str "  " (id (get owner s s)) " -> " (id me) ";")))]
                line)
        ret-lines (cons "  \"return\" [shape=doublecircle];" (for [s (:inputs return)] (str "  " (id (get owner s s)) " -> \"return\";")))]
    (str "digraph {\n" (str/join "\n" (concat arg-lines lines ret-lines)) "\n}\n")))
