(ns gen.dynamic.guard
  "The gen body rewriter, run at macroexpansion. A gen body (or a
   guarded-fn helper) is macroexpanded with every if test wrapped in
   gen.array/truthy?, the truthiness tests, lookups and representation
   tests of gen.dynamic.core in place of the functions they guard, catch
   clauses raising the fallback in a batched run, and case dispatching on a
   checked value (case-key, its one runtime helper); and a lazy call whose
   trace calls would never run (its value discarded, or bound and never
   used) is a compile error. It knows nothing of the handler: gen.dynamic's
   macros call it, and gen.dynamic.static guards the static graph's code
   with it."
  {:api :internal}
  (:require [gen.array :as array]
            [gen.array.batch :as batch]
            [gen.dynamic.core]))

(set! *warn-on-reflection* true)

(defn ^:no-doc case-key
  "case's dispatch value, checked: a value that differs across the particles
   of a batch cannot pick one clause (gen.array.batch), so it raises the
   batched-branch error rather than falling to the default clause; a 0-d
   backend array (an MLX scalar) dispatches on its host value, as on the JVM
   backend."
  [x]
  (if (batch/batched? x)
    (throw (array/batched-branch "case on a batched value: it differs across particles" {}))
    (gen.dynamic.core/host-value "case on" x)))

(declare guard-branches)

(def ^:private guarded-core
  "The functions that test truthiness, look up by key or test
   representation, to their guarded versions in gen.dynamic.core (each names
   the function it guards in its :guards metadata)."
  (into {} (keep (fn [[s v]] (when-let [g (:guards (meta v))] [g (symbol "gen.dynamic.core" (name s))])))
        (ns-publics 'gen.dynamic.core)))

(def ^:dynamic ^:private *outer-locals*
  "The locals of the gen body being guarded, while forms inside it are
   macroexpanded: a gen (or guarded-fn) nested in it sees them."
  #{})

(defn- expand
  "form macroexpanded, with the locals around it visible to nested gens."
  [form locals]
  (binding [*outer-locals* locals] (macroexpand form)))

(defn- guard-symbol
  "sym, or its guarded version when it names one of guarded-core's
   functions (and is not a local)."
  [sym locals]
  (if (contains? locals sym)
    sym
    (let [v (try (resolve sym) (catch Exception _ nil))]
      (or (when (var? v) (guarded-core (symbol v))) sym))))

(defn- guard-body [forms locals] (map #(guard-branches % locals) forms))

(def ^:private lazy-fns
  '#{clojure.core/map clojure.core/map-indexed clojure.core/mapcat clojure.core/keep
     clojure.core/keep-indexed clojure.core/filter clojure.core/remove clojure.core/for
     gen.dynamic.core/filter gen.dynamic.core/remove})

(defn- discarded-lazy-trace?
  "Is form a lazy call (map, for, keep, …) whose trace or splice calls would
   never run because its value is discarded?"
  [form locals]
  (and (seq? form)
       (let [head (first form)]
         (and (symbol? head) (not (contains? locals head))
              (when-let [v (resolve head)] (lazy-fns (symbol v)))
              (some #{'trace 'splice 'untraced 'untraced-key} (tree-seq coll? seq (rest form)))))))

(defn- discarded-lazy-call
  "The lazy call with trace calls that would be discarded with form's value:
   form itself (as written, (for …), or macroexpanded, (->> xs (map …))), or
   the value of an if, do, let or letfn it expands to (when, cond, …)."
  [form locals]
  (let [expanded (if (and (seq? form) (symbol? (first form)) (not (contains? locals (first form))))
                   (expand form locals)
                   form)]
    (cond (or (discarded-lazy-trace? form locals) (discarded-lazy-trace? expanded locals)) form
          (not (seq? expanded)) nil
          :else (case (first expanded)
                  if (some #(discarded-lazy-call % locals) (drop 2 expanded))
                  do (some-> (last (rest expanded)) (discarded-lazy-call locals))
                  (let* loop* letfn*) (let [[_ bindings & body] expanded]
                                        (some-> (last body)
                                                (discarded-lazy-call (into locals (take-nth 2 bindings)))))
                  nil))))

(defn- guard-statements
  "A body's forms, guarded; every form but the last is a statement, whose
   value is discarded, so a lazy call there with trace calls is an error."
  [forms locals]
  (doseq [f (butlast forms)]
    (when-let [lazy (discarded-lazy-call f locals)]
      (throw (ex-info (str "In gen: " (pr-str (first lazy)) " here is lazy and its value is discarded,"
                           " so its trace calls would never run. Use doseq, run! or mapv.")
                      {:form f}))))
  (guard-body forms locals))

(defn- check-lazy-bindings
  "A let binding of a lazy call with trace calls whose symbol is never used:
   its trace calls would never run, so it is an error, as for a statement."
  [bindings body locals]
  (let [pairs (vec (partition 2 bindings))]
    (doseq [[i [sym init]] (map-indexed vector pairs)
            :let [expanded (if (and (seq? init) (symbol? (first init)) (not (contains? locals (first init))))
                             (expand init locals)
                             init)]]
      (when (and (symbol? sym)
                 (or (discarded-lazy-trace? init locals) (discarded-lazy-trace? expanded locals))
                 (not (some #{sym} (tree-seq coll? seq (concat (map second (subvec pairs (inc i))) body)))))
        (throw (ex-info (str "In gen: " sym " is bound to a lazy " (pr-str (first init)) " and never used,"
                             " so its trace calls would never run. Use doseq, run! or mapv.")
                        {:form init}))))))

(defn- guard-bindings
  "[bindings' locals'] for a let*/loop* binding vector."
  [bindings locals]
  (reduce (fn [[acc ls] [sym init]] [(conj acc sym (guard-branches init ls)) (conj ls sym)])
          [[] locals]
          (partition 2 bindings)))

(defn- guard-fn [[_ & more] locals]
  (let [[fname more] (if (symbol? (first more)) [(first more) (rest more)] [nil more])
        locals (cond-> locals fname (conj fname))
        arities (if (vector? (first more)) [more] more)]
    (list* 'fn* (concat (when fname [fname])
                        (for [[params & body] arities]
                          (list* params (guard-statements body (into locals (remove #{'&} params)))))))))

(defn- guard-special [form locals]
  (case (first form)
    (quote var) form
    if (let [[_ test then else] form]
         (list* 'if (list `array/truthy? (guard-branches test locals))
                (guard-branches then locals)
                (when (= 4 (count form)) [(guard-branches else locals)])))
    (let* loop*) (let [[op bindings & body] form
                       _ (when (= 'let* op) (check-lazy-bindings bindings body locals))
                       [bindings locals] (guard-bindings bindings locals)]
                   (list* op bindings (guard-statements body locals)))
    do (list* 'do (guard-statements (rest form) locals))
    fn* (guard-fn form locals)
    ;; (reify* [interfaces] (method [this params…] body…) …): the parameters are locals
    reify* (let [[_ interfaces & methods] form]
             (list* 'reify* interfaces
                    (for [[mname params & body] methods]
                      (list* mname params (guard-statements body (into locals params))))))
    ;; (. target member args…) or (. target (member args…)): a member name is
    ;; not a var, even when it is spelled like one (.get)
    . (let [[_ target member & args] form]
        (list* '. (guard-branches target locals)
               (if (seq? member)
                 (list* (first member) (guard-body (rest member) locals))
                 member)
               (guard-body args locals)))
    letfn* (let [[_ bindings & body] form
                 locals (into locals (take-nth 2 bindings))]
             (list* 'letfn*
                    (vec (mapcat (fn [[n f]] [n (guard-branches f locals)]) (partition 2 bindings)))
                    (guard-body body locals)))
    ;; a caught exception may be thrown for some particles only: in a batched
    ;; run, a catch clause raises the fallback first (gen.dynamic.core/caught)
    try (list* 'try (for [f (rest form)]
                      (cond (and (seq? f) (= 'catch (first f)))
                            (let [[_ cls sym & body] f]
                              (list* 'catch cls sym (list `gen.dynamic.core/caught sym)
                                     (guard-body body (conj locals sym))))
                            (and (seq? f) (= 'finally (first f)))
                            (list* 'finally (guard-body (rest f) locals))
                            :else (guard-branches f locals))))
    ;; case*'s dispatch value is a local symbol (case binds it): check it first
    ;; dispatching on case-key's value (case* needs a local: rebind it)
    case* (let [[_ ge shift mask default imap & more] form]
            (list 'let* [ge (list `case-key ge)]
                  (list* 'case* ge shift mask (guard-branches default locals)
                   (into (empty imap) (map (fn [[k [test then]]] [k [test (guard-branches then locals)]])) imap)
                   more)))
    (apply list (guard-body form locals))))

(defn- keep-meta
  "guarded, with the metadata of the form it came from (type hints, lines)."
  [form guarded]
  (if-let [m (meta form)]
    (if (instance? clojure.lang.IObj guarded) (with-meta guarded (merge (meta guarded) m)) guarded)
    guarded))

(defn ^:no-doc guard-branches
  "form, macroexpanded, with every if test wrapped in gen.array/truthy?, and
   the truthiness tests, lookups and representation tests of
   gen.dynamic.core in place of the functions they guard. locals are the
   symbols bound around form (never macroexpanded as heads)."
  [form locals]
  (cond
    (seq? form) (let [head (first form)]
                  (if (and (symbol? head) (contains? locals head))
                    (keep-meta form (apply list (guard-body form locals)))
                    (let [expanded (expand form locals)]
                      (if (seq? expanded)
                        (keep-meta expanded (guard-special expanded locals))
                        (guard-branches expanded locals)))))
    (vector? form) (keep-meta form (mapv #(guard-branches % locals) form))
    (map? form) (into (empty form) (map (fn [[k v]] [(guard-branches k locals) (guard-branches v locals)])) form)
    (set? form) (into (empty form) (map #(guard-branches % locals)) form)
    (symbol? form) (guard-symbol form locals)
    :else form))

(defn ^:no-doc locals
  "The locals around a gen (or guarded-fn) form: those of its &env, and
   those of the body being guarded around it, while its forms are
   macroexpanded (a nested gen sees its enclosing body's)."
  [env]
  (into (set (keys env)) *outer-locals*))
