(ns gen.codegen.session
  "A REPL session for an LLM writing code: a namespace kept across its
   turns, where its forms are read, repaired, vetted and evaluated by hooks
   the caller chooses, and an error is explained as the caller explains
   it. With gen.llm.agent it makes an eval tool (eval-tool).

     (open opts)                 → a session
     (eval! session code)        → {:text :ok? :value :values :forms :repaired? :printed}
     (eval! session code hooks)
     (lookup session sym)        → the value of a var the session defined, or nil
     (program session)           → the code of every form that evaluated, in order
     (close! session)            → removes its namespace
     (eval-tool hooks)           → a gen.llm.agent tool, {:description :params :fn},
                                   its state the session
     (within ms f)               → (f), or an ex-info with ::timeout after ms

   opts of open: :prelude, the clauses of the namespace's ns form after its
   name ((:refer-clojure …) (:require …)); :vars, a map of symbol to value
   interned in it (the task's data, say); :prefix, the namespace's name
   before a unique suffix (\"gen.codegen.session-\").

   hooks (eval! and eval-tool):
     :read     (fn [code] forms), default read-forms (*read-eval* false)
     :repair   (fn [code] code'), tried when :read throws (parinfer's indent
               mode, say); the reply notes that it ran
     :vet      (fn [forms] → nil, or a text that refuses them): run before any
               form is evaluated
     :explain  (fn [throwable ns forms] → text): an evaluation error as the
               model reads it, given the session's namespace and the forms
               (default: the message and its causes')
     :show     (fn [value] → text): a value as the model reads it (default:
               printed, cut to 300 characters)
     :each?    every form's value, each after its form (cut to 60
               characters), not only the last's; on an error, the values
               of the forms before it, then the error (false: a model may
               take the last value for the only one)
     :timeout-ms  each eval's limit (30000; nil for none, the caller keeping
               its own); past it the eval is abandoned, its thread
               interrupted (it may go on: the JVM is then tainted, and
               :timeout? true says so)

   This evaluates code on the calling JVM, the caller's choice and risk:
   run sessions in a JVM of their own (gen.codegen.process), never one that
   holds anything else of value. The vetting is the caller's; SCI is the
   sandbox to come."
  {:api :experimental}
  (:require [clojure.string :as str]))

(defn within
  "(f) on another thread (dynamic bindings conveyed), or an ex-info with
   ::timeout when it takes more than ms (its thread is interrupted, and may
   go on). An exception f throws is rethrown."
  [ms f]
  (let [fut (future-call f)
        v (try (deref fut ms ::timeout)
               (catch java.util.concurrent.ExecutionException e (throw (or (.getCause e) e))))]
    (if (= v ::timeout)
      (do (future-cancel fut)
          (throw (ex-info (str "timed out after " ms " ms") {::timeout ms})))
      v)))

(defn read-forms
  "The forms of code, read with *read-eval* false (#= refused) and no
   default tagged-literal reader."
  [code]
  (binding [*read-eval* false *default-data-reader-fn* nil]
    (let [r (clojure.lang.LineNumberingPushbackReader. (java.io.StringReader. code))]
      (loop [forms []]
        (let [f (read {:eof ::eof} r)]
          (if (= f ::eof) forms (recur (conj forms f))))))))

(defn- causes
  "A throwable's message and its causes', joined."
  [^Throwable e]
  (->> (take-while some? (iterate ex-cause e))
       (map #(or (ex-message %) (str (class %))))
       distinct
       (str/join " Caused by: ")))

(defn open
  "A new session. See the namespace's docstring."
  ([] (open nil))
  ([{:keys [prelude vars prefix] :or {prefix "gen.codegen.session-"}}]
   (let [n (gensym prefix)]
     (binding [*ns* *ns*]
       (eval (list* 'ns n prelude)))
     (doseq [[sym v] vars] (intern n sym v))
     {:ns n :program (atom [])})))

(defn close!
  "Removes the session's namespace."
  [{n :ns}]
  (remove-ns n)
  nil)

(defn lookup
  "The value of sym's var in the session, nil when it has none."
  [{n :ns} sym]
  (some-> (ns-resolve (the-ns n) (symbol sym)) deref))

(defn program
  "The code of every form the session evaluated without error, in order."
  [{:keys [program]}]
  (str/join "\n\n" @program))

(defn- clip [s n] (if (> (count s) n) (str (subs s 0 n) "…") s))

(defn eval!
  "code's forms read, vetted and evaluated in the session, in order: the
   last one's value (:each?, every one's), or the first error (the forms
   before it stay). :values holds the value of every form evaluated. See
   the namespace's docstring."
  ([session code] (eval! session code nil))
  ([{n :ns :keys [program]} code {:keys [read repair vet explain show each? timeout-ms]
                                  :or {read read-forms timeout-ms 30000}}]
   (let [[forms repaired? read-error]
         (try [(read code) false nil]
              (catch Exception e
                (if repair
                  (try [(read (repair code)) true nil] (catch Exception _ [nil false e]))
                  [nil false e])))
         note (when repaired? "(The delimiters did not balance; they were rebuilt from the indentation.)\n")]
     (if read-error
       {:text (str "error: the code does not read: " (causes read-error)) :ok? false}
       (if-let [refusal (when vet (vet forms))]
         {:text (str note "error: " refusal) :ok? false :forms forms :repaired? repaired?}
         (let [out (java.io.StringWriter.)
               printed #(let [s (str out)] (when (seq s) (str "printed: " (clip s 400) "\n")))
               shown #(if show (show %) (clip (pr-str %) 300))
               ;; [form value] of each form evaluated, in order
               done (atom [])
               each-text #(str/join "\n" (map (fn [[f v]] (str (clip (pr-str f) 60) "\n=> " (shown v))) @done))]
           (try
             (let [run #(binding [*ns* (the-ns n) *out* out]
                          (doseq [f forms]
                            (let [v (eval f)]
                              (swap! program conj (pr-str f))
                              (swap! done conj [f v]))))
                   _ (if timeout-ms (within timeout-ms run) (run))
                   v (second (peek @done))]
               {:text (str note (printed) (if (and each? (seq @done)) (each-text) (str "=> " (shown v))))
                :ok? true :value v :values (mapv second @done) :forms forms
                :repaired? repaired? :printed (str out)})
             (catch Throwable e
               {:text (str note (printed) (when (and each? (seq @done)) (str (each-text) "\n"))
                           "error: " (if explain (explain e (the-ns n) forms) (causes e)))
                :ok? false :values (mapv second @done) :forms forms :repaired? repaired? :printed (str out)
                :timeout? (boolean (::timeout (ex-data e)))}))))))))

(defn eval-tool
  "gen.llm.agent's tool that evaluates code in the session (the episode's
   state, or (:session state) when the state is a map holding one), with
   hooks: {:description :params :fn}. The result's :ok?, :repaired? and
   :timeout? stay in the log beside :text."
  ([] (eval-tool nil))
  ([hooks]
   {:description (str "Evaluate Clojure forms in the session's namespace. Define one or two small things at a time. Returns "
                      (if (:each? hooks) "each form's value" "the last form's value") ", or the error.")
    :params [{:name "code" :description "One or a few Clojure forms."}]
    :fn (fn [{:strs [code]} state]
          (let [s (if (and (map? state) (:session state)) (:session state) state)]
            (select-keys (eval! s (or code "") hooks) [:text :ok? :repaired? :timeout?])))}))
