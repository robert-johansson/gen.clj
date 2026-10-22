(ns gen.dynamic.static
  "The static modeling language (Gen.jl's @gen (static)): a gen body of a
   fixed structure, (gen ^:static [args] (let [bindings…] ret)), whose
   graph lets update and regenerate skip what did not change.

   The body is one let. Each binding is a call, (trace addr dist) or
   (splice addr gf & args), or a pure expression of the arguments and
   earlier bindings (any pure Clojure, including destructuring); then the
   return expression, which may itself be such a call. A binding's head is
   macroexpanded first, as the dynamic gen expands its body, so a macro that
   expands to a call is that call, and one that expands to a computation is
   that computation, its inputs the names the expansion reads. A macro
   nested deeper is the compiler's to expand, after the analysis found the
   inputs from the names the binding mentions, so it must be hygienic: it
   reads only the locals it is given. Quoted data is data, and may mention
   trace. Branching happens only inside called generative functions.

   An address starts with a constant key: [:x i] is a node at :x whose
   further keys are computed from its inputs, run as the dynamic body runs
   it. A call nested where it is evaluated exactly once whenever its
   expression is (a function call's arguments, a literal's elements, an
   if's test) is hoisted into a node of its own before the binding, in
   evaluation order, so the static graph draws what the body draws (DESIGN.md
   §9). This namespace checks a body at macroexpansion and builds its graph
   as data, with the code of its nodes and its score fn (static-form,
   guarded by gen.dynamic.guard); gen.dynamic's handler runs them.

     (ir gf)    the graph: the arguments, the nodes in order ({:kind :trace,
                :splice or :compute, :addr, :names, :inputs, :form, and
                :keys, a computed address's further keys}) and the return
                expression's inputs
     (dot gf)   the graph in Graphviz's DOT (Gen.jl's draw_graph)

   A lint says which dynamic generative functions the flag would take, so
   that a model gets the static language's speed where it fits, without an
   implicit switch:

     (could-be-static gf)  {:static? true}, or {:static? false :reason …}
     (lint ns …)           per generative function var of the namespaces
                           (the current one by default): {:var :static?
                           :reason}, the flagged ones as {:already true}"
  (:require [clojure.string :as str]
            [gen.array :as array]
            [gen.choicemap :as cm]
            [gen.dynamic.guard :as guard]
            [gen.random :as rng]))

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
                       " (a static body is one let of calls at addresses that start with a constant, and pure"
                       " bindings;"
                       " use gen without ^:static for any other body)")
                  {:form form})))

(defn- constant-address? [addr]
  (or (keyword? addr) (string? addr) (number? addr) (boolean? addr)
      (and (vector? addr) (seq addr) (every? constant-address? addr))))

(defn- split-address
  "[address keys] of a call: the longest leading run of constant keys of
   its address, the node's, and the keys computed after it (nil when the
   address is constant). The first key must be a constant, as Gen.jl's
   must be a Symbol."
  [addr call]
  (cond (constant-address? addr) [addr nil]
        (and (vector? addr) (constant-address? (first addr)))
        (let [[prefix ks] (split-with constant-address? addr)]
          [(if (next prefix) (vec prefix) (first prefix)) (vec ks)])
        :else (not-static (str (first call) "'s address must start with a constant key"
                               (when (= 'splice (first call)) " (a splice without an address is dynamic)"))
                          call)))

(defn- nested-gen?
  "Is form a gen, whose trace is its own: a call whose head is no local and
   resolves to gen.dynamic/gen, however it is spelled (gen, dyn/gen, …)?"
  [{:keys [env locals]} form]
  (and (seq? form) (symbol? (first form))
       (not (contains? env (first form))) (not (contains? locals (first form)))
       (let [v (resolve env (first form))]
         (and (var? v) (= 'gen.dynamic/gen (symbol v))))))

(defn- outside-gens
  "The symbols of form outside the gens and the quoted data in it (a nested
   gen's trace is its own, and quoted data is not code)."
  [env form]
  (filter symbol? (tree-seq #(and (coll? %) (not (quoted? %)) (not (nested-gen? env %))) seq form)))

(defn- pure!
  "form, checked to make none of the body's own calls."
  [env form what]
  (when-let [e (first (filter effects (outside-gens env form)))]
    (not-static (str what " calls " e ", which a static body may do only as a binding's whole expression") form))
  form)

(defn bound-names
  "The symbols a binding form binds (destructuring included, :keys with
   namespaced or keyword entries too), in order; _ binds nothing."
  [binding]
  (vec (distinct (remove #(or (= '_ %) (re-matches #"(vec|map|seq|first|p)__\d+" (name %)))
                         (take-nth 2 (destructure [binding nil]))))))

(defn- inputs [form known] (vec (distinct (filter known (symbols form)))))

(defn- macro-name
  "The macro form calls, as a qualified symbol: not a local, nor one of the
   body's own calls, nor gen, whose body is its own (nil otherwise)."
  [{:keys [env locals]} form]
  (when (and (seq? form) (symbol? (first form)) (not (effects (first form)))
             (not (contains? env (first form))) (not (contains? locals (first form))))
    (let [v (resolve env (first form))]
      (when (and (var? v) (:macro (meta v)) (not= 'gen.dynamic/gen (symbol v)))
        (symbol v)))))

(defn- expanded
  "form, its head macroexpanded until it is no macro's: what a binding is,
   as the dynamic gen expands its body, so that a user macro expanding to
   (trace …) or (splice …) is that call."
  [env form]
  (if (macro-name env form) (recur env (macroexpand-1 form)) form))

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
    (cond-> {:kind :compute :expr (pure! env form "a binding") :inputs (inputs form known)}
      (and (symbol? form) (known form)) (assoc :alias form)
      v (assoc :call v :args (vec (rest form)) :arg-specs (mapv #(arg-spec % known) (rest form))))))

(defn- node
  "The node of one binding, given the names bound so far, from the binding
   with its head macroexpanded: a call, or a computation (its form as
   written)."
  [env binding init known]
  (let [names (bound-names binding)
        call (expanded env init)
        head (when (seq? call) (first call))]
    (merge {:binding binding :names names :form init}
           (cond
             (= 'trace head)
             (let [[_ addr dist & more] call
                   _ (when (seq more) (not-static "trace takes an address and a distribution" call))
                   [addr ks] (split-address addr call)]
               (cond-> {:kind :trace :addr addr :expr (pure! env dist "a distribution") :inputs (inputs [dist ks] known)}
                 ks (assoc :keys (pure! env ks "an address"))))

             (= 'splice head)
             (let [[_ addr gf & args] call
                   [addr ks] (split-address addr call)
                   expr `[~gf [~@args]]]
               (cond-> {:kind :splice :addr addr :expr (pure! env expr "a splice's arguments") :inputs (inputs [expr ks] known)
                        :arg-specs (mapv #(arg-spec % known) args)}
                 ks (assoc :keys (pure! env ks "an address"))))

             :else
             (compute-node env call known)))))

;; ---------------------------------------------------------------------------
;; Hoisting: a call nested where it is evaluated exactly once
;; ---------------------------------------------------------------------------

(defn- call-form? [form] (and (seq? form) (contains? '#{trace splice} (first form))))

(defn- calls?
  "Does form call trace or splice (outside nested gens and quoted data)?"
  [env form]
  (boolean (some '#{trace splice} (outside-gens env form))))

(defn- not-once
  "The error for a call inside form, where (inside where) it is not
   evaluated exactly once whenever its expression is: it names the call."
  [where why form]
  (let [call (or (first (filter call-form? (tree-seq #(and (coll? %) (not (quoted? %))) seq form))) form)]
    (not-static (str where " calls " (if (call-form? call) (first call) "trace") (when why (str ", " why))) call)))

(declare hoist-in)

(defn- hoist-all
  "[forms pairs]: hoist-in over forms, in order."
  [env forms]
  (reduce (fn [[out pairs] f] (let [[f' ps] (hoist-in env f)] [(conj out f') (into pairs ps)]))
          [[] []] forms))

(defn- hoist-call
  "[name pairs]: call bound to a fresh name, after the calls its address
   and arguments evaluate (theirs first)."
  [env [head & args]]
  (let [[args' pairs] (hoist-all env args)
        sym (gensym (str head "__"))]
    [sym (conj pairs [sym (apply list head args')])]))

(defn- hoist-in
  "[form' pairs]: form with each trace or splice call it evaluates exactly
   once whenever it is evaluated (a function call's arguments, a vector,
   map or set's elements, an if's test) replaced by a fresh name, and the
   pairs [name call] in Clojure's evaluation order (arguments left to
   right, the innermost call first), so the graph draws what the body
   draws. A call anywhere else is not static. A form without calls is as
   written."
  [env form]
  (if-not (calls? env form)
    [form []]
    (let [m (macro-name env form)]
      (cond
        ('#{clojure.core/and clojure.core/or} m)
        (not-once (str "an " (name m)) (str "which runs only when the arguments before it are "
                                            (if (= 'clojure.core/and m) "true" "false"))
                  form)
        m (hoist-in env (macroexpand-1 form))
        (call-form? form) (hoist-call env form)
        (vector? form) (hoist-all env form)
        (map? form) (let [[xs pairs] (hoist-all env (apply concat form))] [(into {} (map vec) (partition 2 xs)) pairs])
        (set? form) (let [[xs pairs] (hoist-all env form)] [(set xs) pairs])
        (seq? form)
        (let [h (first form)]
          (cond
            (= 'if h) (let [[_ test & branches] form
                            [test' pairs] (hoist-in env test)]
                        (when-let [b (first (filter #(calls? env %) branches))]
                          (not-once "a branch of an if"
                                    "which would run only when the branch is taken, where a node runs every time" b))
                        [(apply list 'if test' branches) pairs])
            ('#{fn* letfn*} h) (not-once "a function inside an expression"
                                         "which may run any number of times, where a node runs once" form)
            ('#{let* loop*} h) (not-once "a let inside an expression" "bind that value in the body's let instead" form)
            (special-symbol? h) (not-once (str "a " h " inside an expression") nil form)
            :else (let [[xs pairs] (hoist-all env form)] [(apply list xs) pairs])))
        :else [form []]))))

(defn- hoisted
  "The pairs [binding init] one binding becomes: the calls its expression
   (or its call's address and arguments) evaluates exactly once, hoisted
   before it, then the binding reading them, written as it was when
   nothing was hoisted."
  [env [binding init]]
  (let [call (expanded env init)
        [init' pairs] (if (call-form? call)
                        (let [[head & args] call
                              [args' pairs] (hoist-all env args)]
                          [(apply list head args') pairs])
                        (hoist-in env init))]
    (conj pairs [binding (if (seq pairs) init' init)])))

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
                [(cm/path* addr) form])]
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
           [ret pairs] (let [r (first ret)]
                         (if (call-form? (expanded (in-scope (into known (mapcat (comp bound-names first) (partition 2 bindings))))
                                                   r))
                           (let [v (gensym "ret")] [v (concat (partition 2 bindings) [[v r]])])
                           [r (partition 2 bindings)]))
           add (fn [[nodes known] [binding init]]
                 (let [n (node (in-scope known) binding init known)]
                   [(conj nodes n) (into known (:names n))]))
           [nodes names] (reduce (fn [acc pair] (reduce add acc (hoisted (in-scope (second acc)) pair)))
                                 [[] known] pairs)
           [ret ret-pairs] (hoist-in (in-scope names) ret)
           [nodes names] (reduce add [nodes names] ret-pairs)
           _ (check-addresses! nodes)]
       {:args args
        :nodes nodes
        :ret (let [ret (expanded (in-scope names) ret)]
               (pure! (in-scope names) ret "the return expression")
               (compute-node (in-scope names) ret names))}))))

;; ---------------------------------------------------------------------------
;; The code: the graph's runtime form and its score fn, which gen.dynamic's
;; handler runs (static-site, static-call and check-computed! are its)
;; ---------------------------------------------------------------------------

(defn- score-form
  "The form of a static body's score fn, (fn [param args choices key] →
   [weight retval]): the body's bindings in order as locals, a trace node's
   value from choices with its log density, a splice's callee assessed on
   its sub-map, keys split as the handler's assess splits them (none when
   key is nil)."
  [st params coercions locals]
  (let [args (gensym "args") choices (gensym "choices") key (gensym "key")
        {:keys [bindings terms]}
        (reduce (fn [{:keys [bindings terms k]} {:keys [kind addr binding expr] computed :keys}]
                  (let [;; a computed address's path, checked to hold every
                        ;; choice under its prefix
                        p (gensym "p")
                        at (when (some? addr)
                             (if computed
                               [p `(into ~(cm/path* addr) [~@computed])
                                '_ `(gen.dynamic/check-computed! ~choices ~(cm/path* addr) ~p ~(= kind :trace))]
                               [p (cm/path* addr)]))]
                    (case kind
                      :trace (let [v (gensym "v") lp (gensym "lp") k' (gensym "k")]
                               {:bindings (-> bindings (into at)
                                              (into [[v lp] `(gen.dynamic/static-site ~expr ~choices ~p)
                                                     binding v
                                                     k' `(when ~k (first (rng/split ~k)))]))
                                :terms (conj terms lp) :k k'})
                      :splice (let [[g as] expr w (gensym "w") rv (gensym "rv") k' (gensym "k") ks (gensym "ks")]
                                {:bindings (-> bindings (into at)
                                               (into [[k' ks] `(if ~k (rng/split ~k) [nil nil])
                                                      [w rv] `(gen.dynamic/static-call ~p ~g ~as
                                                                                       (cm/get-submap ~choices ~p cm/EMPTY) ~ks)
                                                      binding rv]))
                                 :terms (conj terms w) :k k'})
                      {:bindings (into bindings [binding expr]) :terms terms :k k})))
                {:bindings [] :terms [] :k key}
                (:nodes st))]
    (guard/guard-branches
     `(fn [~'param ~args ~choices ~key]
        (let [~params ~args ~@coercions ~@bindings]
          [(reduce array/+ 0.0 [~@terms]) ~(:expr (:ret st))]))
     locals)))

(defn ^:no-doc static-form
  "The runtime graph of the static body st (gen.dynamic.static/analyze),
   its names in slots (the arguments' first): per node, its kind, address,
   names and inputs, the slots it reads and writes, a guarded fn of (param,
   env) for its distribution, call or value, a fn writing its names' slots
   from its value, and how its diff is found (a rule, an alias, argument
   specs)."
  [st params coercions locals]
  (let [;; each name once: a name bound twice ([w w z]) holds its last binding, as let's
        arg-names (vec (distinct (concat (apply concat (:fixed (:args st))) (:rest (:args st)))))
        names (vec (distinct (concat arg-names (mapcat :names (:nodes st)))))
        slots (zipmap names (range))
        env (gensym "env")
        reads (fn [inputs] (vec (mapcat (fn [x] [x `(nth ~env ~(slots x))]) inputs)))
        node-fn (fn [{:keys [inputs expr]}]
                  (guard/guard-branches `(fn [~'param ~env] (let ~(reads inputs) ~expr)) (into locals inputs)))
        keys-fn (fn [{:keys [inputs] ks :keys}]
                  (guard/guard-branches `(fn [~'param ~env] (let ~(reads inputs) [~@ks])) (into locals inputs)))
        param? (fn [expr] (boolean (some #{'param} (tree-seq coll? seq expr))))
        spec (fn [{:keys [symbol constant inputs]}]
               (cond (and symbol (contains? slots symbol)) {:slot (slots symbol)}
                     (or symbol constant) {:constant true}
                     :else {:in (mapv slots inputs)}))
        call (fn [{:keys [call args arg-specs inputs alias]}]
               (cond call {:rule `(var ~call) :arg-specs (mapv spec arg-specs)
                           :args-f `(fn [~'param ~env] (let ~(reads inputs) [~@args]))}
                     alias {:alias (slots alias)}))
        v (gensym "v")
        args (gensym "args")]
    `{:args '~(:args st)
      :params '~params
      :slots '~slots
      :size ~(count names)
      :args-bind (fn [~args] (let [~params ~args ~@coercions] [~@arg-names ~@(repeat (- (count names) (count arg-names)) nil)]))
      :nodes [~@(for [n (:nodes st)]
                  (merge `{:kind ~(:kind n) :addr ~(:addr n) :names '~(:names n) :inputs '~(:inputs n) :form '~(:form n)
                           :in ~(mapv slots (:inputs n)) :out ~(mapv slots (:names n))
                           :symbol? ~(symbol? (:binding n))
                           :param? ~(param? [(:expr n) (:keys n)])
                           :f ~(node-fn n)
                           :keys-f ~(when (:keys n) (keys-fn n))
                           :bind (fn [~env ~v] (let [~(:binding n) ~v] (-> ~env ~@(map (fn [x] `(assoc ~(slots x) ~x)) (:names n)))))}
                         (when (= :splice (:kind n)) {:arg-specs (mapv spec (:arg-specs n))})
                         (when (:keys n) {:keys (list 'quote (:keys n))})
                         (call n)))]
      :ret ~(merge {:inputs `'~(:inputs (:ret st)) :in (mapv slots (:inputs (:ret st)))
                    :param? (param? (:expr (:ret st))) :f (node-fn (:ret st))}
                   (call (:ret st)))
      :score ~(score-form st params coercions locals)
      :roots ~(set (keep #(when-some [a (:addr %)] (first (cm/path* a))) (:nodes st)))
      ;; a node at a computed address holds its prefix, as a call does
      ;; (its score fn checks the rest: check-computed!)
      :paths ~(when (some #(vector? (:addr %)) (:nodes st))
                (vec (keep #(when-some [a (:addr %)] [(if (:keys %) :splice (:kind %)) (cm/path* a)]) (:nodes st))))}))

;; ---------------------------------------------------------------------------
;; The graph as data
;; ---------------------------------------------------------------------------

(defn ir
  "gf's graph as data (see the namespace doc), or nil when gf is not static."
  [gf]
  (when-let [{:keys [args nodes ret]} (:static gf)]
    {:args args
     :nodes (mapv #(select-keys % [:kind :addr :keys :names :inputs :form]) nodes)
     :return {:inputs (:inputs ret)}}))

(defn dot
  "gf's graph in Graphviz's DOT: argument, choice, call and computation
   nodes, an edge from each input."
  [gf]
  (let [{:keys [args nodes return]} (or (ir gf) (throw (ex-info "dot: not a static generative function" {})))
        id #(str "\"" (str/escape (str %) {\" "\\\"" \\ "\\\\"}) "\"")
        arg-lines (for [n (concat (apply concat (:fixed args)) (:rest args))] (str "  " (id n) " [shape=plaintext];"))
        node-id (fn [i n] (cond (:keys n) (pr-str (into (cm/path* (:addr n)) (:keys n)))
                                (some? (:addr n)) (pr-str (:addr n))
                                :else (str "#" i)))
        owner (into {} (for [[i n] (map-indexed vector nodes) s (:names n)] [s (node-id i n)]))
        lines (for [[i n] (map-indexed vector nodes)
                    :let [me (node-id i n)]
                    line (cons (str "  " (id me) " [shape=" (case (:kind n) :trace "ellipse" :splice "box" "diamond")
                                    " label=" (id (str me (when (seq (:names n)) (str " → " (str/join " " (:names n))))))
                                    "];")
                               (for [s (:inputs n)] (str "  " (id (get owner s s)) " -> " (id me) ";")))]
                line)
        ret-lines (cons "  \"return\" [shape=doublecircle];"
                        (for [s (:inputs return)] (str "  " (id (get owner s s)) " -> \"return\";")))]
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
