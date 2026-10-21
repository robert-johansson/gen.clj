(ns gen.dynamic.static
  "The static modeling language (Gen.jl's @gen (static)): a gen body of a
   fixed structure, (gen ^:static [args] (let [bindings…] ret)), whose
   graph lets update and regenerate skip what did not change.

   The body is one let. Each binding is a call at a constant address,
   (trace addr dist) or (splice addr gf & args), or a pure expression of the
   arguments and earlier bindings (any pure Clojure, including destructuring);
   then the return expression, which may itself be such a call. A binding's
   head is macroexpanded first, as the dynamic gen expands its body, so a
   macro that expands to a call is that call; quoted data is data, and may
   mention trace. Branching happens only inside called generative functions.
   This namespace checks a body at macroexpansion and builds its graph as
   data; gen.dynamic runs it.

     (ir gf)    the graph: the arguments, the nodes in order ({:kind :trace,
                :splice or :compute, :addr, :names, :inputs, :form}) and the
                return expression's inputs
     (dot gf)   the graph in Graphviz's DOT (Gen.jl's draw_graph)

   A lint says which dynamic generative functions the flag would take, so
   that a model gets the static language's speed where it fits, without an
   implicit switch:

     (could-be-static gf)  {:static? true}, or {:static? false :reason …}
     (lint ns …)           per generative function var of the namespaces
                           (the current one by default): {:var :static?
                           :reason}, the flagged ones as {:already true}"
  (:require [clojure.string :as str]))

(def ^:private effects
  "The body's own calls, which a static body makes only at a binding's head."
  '#{trace splice untraced untraced-key})

(defn- quoted? [form] (and (seq? form) (= 'quote (first form))))

(defn- symbols
  "The symbols of form, but quoted data's (which is not code)."
  [form]
  (filter symbol? (tree-seq #(and (coll? %) (not (quoted? %))) seq form)))

(defn- not-static [why form]
  (throw (ex-info (str "gen ^:static: " why ": " (pr-str form)
                       " (a static body is one let of calls at constant addresses and pure bindings;"
                       " use gen without ^:static for any other body)")
                  {:form form})))

(defn- constant-address? [addr]
  (or (keyword? addr) (string? addr) (number? addr) (boolean? addr)
      (and (vector? addr) (seq addr) (every? constant-address? addr))))

(defn- outside-gens
  "The symbols of form outside the gen forms and the quoted data in it (a
   nested gen's trace is its own, and quoted data is not code)."
  [form]
  (filter symbol? (tree-seq #(and (coll? %) (not (quoted? %)) (not (and (seq? %) ('#{gen gen.dynamic/gen} (first %)))))
                            seq form)))

(defn- pure!
  "form, checked to make none of the body's own calls."
  [form what]
  (when-let [e (first (filter effects (outside-gens form)))]
    (not-static (str what " calls " e ", which a static body may do only as a binding's whole expression") form))
  form)

(defn bound-names
  "The symbols a binding form binds (destructuring included, :keys with
   namespaced or keyword entries too), in order; _ binds nothing."
  [binding]
  (vec (distinct (remove #(or (= '_ %) (re-matches #"(vec|map|seq|first|p)__\d+" (name %)))
                         (take-nth 2 (destructure [binding nil]))))))

(defn- inputs [form known] (vec (distinct (filter known (symbols form)))))

(defn- macro-call?
  "Is form a call of a macro (not of a local, nor one of the body's own
   calls, nor gen, whose body is its own)?"
  [{:keys [env locals]} form]
  (and (seq? form) (symbol? (first form)) (not (effects (first form)))
       (not (contains? env (first form))) (not (contains? locals (first form)))
       (let [v (resolve env (first form))]
         (boolean (and (var? v) (:macro (meta v)) (not= 'gen.dynamic/gen (symbol v)))))))

(defn- expanded
  "form, its head macroexpanded until it is no macro's: what a binding is,
   as the dynamic gen expands its body, so that a user macro expanding to
   (trace …) or (splice …) is that call."
  [env form]
  (if (macro-call? env form) (recur env (macroexpand-1 form)) form))

(defn- simple-arg? [form] (or (symbol? form) (not (coll? form))))

(defn- call-var
  "The var form calls, when form is one call of a function var (not a
   local, a macro or a special form) on symbols and constants: the
   binding whose diff gen.diff/propagate can give precisely. locals are
   the enclosing locals (a gen body's too, which &env does not hold)."
  [{:keys [env locals]} form]
  (when (and (seq? form) (symbol? (first form)) (not (contains? env (first form))) (not (contains? locals (first form)))
             (every? simple-arg? (rest form)))
    (let [v (resolve env (first form))]
      (when (and (var? v) (not (:macro (meta v))))
        (symbol (str (.-ns ^clojure.lang.Var v)) (str (.-sym ^clojure.lang.Var v)))))))

(defn- arg-spec
  "How an argument's diff is found: its symbol's, none for a constant, or
   from the inputs of an expression."
  [form known]
  (cond (symbol? form) {:symbol form}
        (not (coll? form)) {:constant true}
        :else {:inputs (inputs form known)}))

(defn- compute-node
  "A binding or return expression: its inputs, and for one call of a var,
   the var and its arguments (for gen.diff/propagate)."
  [env form known]
  (let [v (call-var env form)]
    (cond-> {:kind :compute :expr (pure! form "a binding") :inputs (inputs form known)}
      (and (symbol? form) (known form)) (assoc :alias form)
      v (assoc :call v :args (vec (rest form)) :arg-specs (mapv #(arg-spec % known) (rest form))))))

(defn- node
  "The node of one binding, given the names bound so far: a call when the
   binding, its head macroexpanded, is one, else a computation as written
   (its expansion checked too)."
  [env binding init known]
  (let [names (bound-names binding)
        call (expanded env init)
        head (when (seq? call) (first call))]
    (merge {:binding binding :names names :form init}
           (cond
             (= 'trace head)
             (let [[_ addr dist & more] call]
               (when (seq more) (not-static "trace takes an address and a distribution" call))
               (when-not (constant-address? addr) (not-static "trace's address must be a constant" call))
               {:kind :trace :addr addr :expr (pure! dist "a distribution") :inputs (inputs dist known)})

             (= 'splice head)
             (let [[_ addr gf & args] call]
               (when-not (constant-address? addr)
                 (not-static "splice needs a constant address (a splice without one is dynamic)" call))
               (let [expr `[~gf [~@args]]]
                 {:kind :splice :addr addr :expr (pure! expr "a splice's arguments") :inputs (inputs expr known)
                  :arg-specs (mapv #(arg-spec % known) args)}))

             :else
             (do (pure! call "a binding")
                 (compute-node env init known))))))

(defn arg-names
  "[[names…] per argument], the rest parameter's names last, and whether
   there is one."
  [params]
  (let [i (.indexOf ^java.util.List params '&)
        fixed (if (neg? i) params (subvec params 0 i))]
    {:fixed (mapv bound-names fixed)
     :rest (when-not (neg? i) (bound-names (nth params (inc i))))}))

(defn- check-addresses!
  "Every trace and splice address once, none under another, as the handler
   checks at run time (a repeated address would otherwise count twice in
   the score fn's assess)."
  [nodes]
  (let [paths (for [{:keys [kind addr form]} nodes :when (not= kind :compute)]
                [(if (vector? addr) addr [addr]) form])]
    (doseq [[i [p _]] (map-indexed vector paths)
            [q form] (drop (inc i) paths)
            :let [n (min (count p) (count q))]
            :when (= (subvec p 0 n) (subvec q 0 n))]
      (not-static (if (= p q) (str "the address " p " is used twice") (str "the addresses " p " and " q " overlap"))
                  form))))

(defn analyze
  "The graph of a static body (a body of one let), or a loud error: {:args
   (arg-names), :nodes [{:kind :addr :binding :names :inputs :expr :form,
   and :call :args :arg-specs for one call of a var, :arg-specs for a
   splice}], :ret (a compute node)}. env is the macro's &env, locals the
   enclosing locals."
  ([params body env] (analyze params body env #{}))
  ([params body env locals]
  (when-not (= 1 (count body)) (not-static "the body is one let form" (vec body)))
  (let [env {:env env :locals (set locals)}
        form (first body)]
    (when-not (and (seq? form) (#{'let 'clojure.core/let} (first form)))
      (not-static "the body is one let form" form))
    (let [[_ bindings & ret] form
          _ (when-not (and (vector? bindings) (even? (count bindings))) (not-static "a let needs pairs of bindings" form))
          _ (when-not (= 1 (count ret)) (not-static "the let returns one expression" form))
          args (arg-names params)
          known (set (concat (apply concat (:fixed args)) (:rest args)))
          ;; the arguments and the bindings so far are locals, not the vars
          ;; or macros they shadow
          in-scope (fn [names] (update env :locals into names))
          ;; a call in return position is a last node, its value returned
          [ret pairs] (let [r (first ret)
                            call (expanded (in-scope (into known (mapcat (comp bound-names first) (partition 2 bindings)))) r)]
                        (if (and (seq? call) ('#{trace splice} (first call)))
                          (let [v (gensym "ret")] [v (concat (partition 2 bindings) [[v r]])])
                          [r (partition 2 bindings)]))
          [nodes _] (reduce (fn [[nodes known] [binding init]]
                              (let [n (node (in-scope known) binding init known)]
                                [(conj nodes n) (into known (:names n))]))
                            [[] known] pairs)
          _ (check-addresses! nodes)
          names (into known (mapcat :names nodes))]
      {:args args
       :nodes nodes
       :ret (do (pure! (expanded (in-scope names) ret) "the return expression")
                (compute-node (in-scope names) ret names))}))))

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
        id #(str "\"" (str/escape (str %) {\" "\\\"" \\ "\\\\"}) "\"")
        arg-lines (for [n (concat (apply concat (:fixed args)) (:rest args))] (str "  " (id n) " [shape=plaintext];"))
        node-id (fn [i n] (if (some? (:addr n)) (pr-str (:addr n)) (str "#" i)))
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

;; ---------------------------------------------------------------------------
;; The lint
;; ---------------------------------------------------------------------------

(defn could-be-static
  "Would ^:static take the dynamic generative function gf? Its source is
   analyzed as the flag would analyze it (in ns, the namespace it was
   defined in, by default the current one): {:static? true}, or
   {:static? false :reason the compile error's message}."
  ([gf] (could-be-static gf *ns*))
  ([gf ns]
   (let [[params & body] (:source gf)]
     (try (binding [*ns* (the-ns ns)] (analyze (vec params) body {}))
          {:static? true}
          (catch clojure.lang.ExceptionInfo e {:static? false :reason (ex-message e)})))))

(defn lint
  "The generative function vars of the namespaces (the current one by
   default), each with could-be-static's verdict, or {:already true} when
   it is static: [{:var :static? :reason :already}], in name order."
  [& nss]
  (let [dynamic? (requiring-resolve 'gen.dynamic/dynamic-gf?)]
    (vec (for [ns (or (seq nss) [*ns*])
               [_ v] (sort-by key (ns-interns (the-ns ns)))
               :let [gf (when (bound? v) @v)]
               :when (and gf (dynamic? gf) (:source gf))]
           (if (:static gf)
             {:var v :already true}
             (assoc (could-be-static gf ns) :var v))))))
