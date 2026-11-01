(ns gen.codegen.candidate
  "A generated model's code, from an LLM's reply to a model (the codegen
   bake-off, genclj-ihc4): its text read without evaluation, checked
   against a list of forbidden names, and evaluated in a namespace of its
   own under the prelude every candidate is written against.

     (code-of reply)        → the code in a reply (its last fenced block)
     (read-forms text)      → its forms, read with *read-eval* false
     (repair-delimiters code) → its delimiters rebuilt from its indentation
     (forbidden forms)      → the forbidden names it uses, sorted
     (evaluate forms)       → [namespace model]: the forms evaluated, model their var's value
     (discard! namespace)   → removes the namespace
     (within ms f)          → (f), or an ex-info with ::timeout after ms

   The check is a guard against accidents, not a sandbox: a local model's
   code runs on this JVM, with names that reach files, processes,
   threads, other namespaces or Java refused. A timed-out evaluation
   leaves its thread running (the JVM is then tainted: the driver starts
   another). SCI is the sandbox to come (GenMLX's codegen/eval,
   genclj-0izr)."
  (:require [clojure.string :as str]))

(def prelude
  "The ns form a candidate's code is evaluated under, the namespace's name
   left out: gen.array's operators in place of clojure.core's, and gen.clj's
   aliases. The prompt gives it to the model."
  '(:refer-clojure :exclude [+ - * / < > <= >= ==]))

(def requires
  "The prelude's requires."
  '(:require [gen.array :as array :refer [+ - * / < > <= >= ==]]
             [gen.choicemap :as cm]
             [gen.combinators :as comb]
             [gen.dist :as d]
             [gen.dynamic :refer [gen]]
             [gen.gfi :as gfi]))

(defn prelude-text
  "The prelude as the prompt shows it."
  []
  (str "(ns candidate\n  " (pr-str prelude) "\n  "
       (str/replace (pr-str requires) #"\] \[" "]\n             [") ")"))

(defn code-of
  "The code in a reply: its last fenced block (```clojure or ```; one cut
   off before its closing fence runs to the reply's end), else the whole
   reply."
  [reply]
  (let [blocks (loop [lines (str/split-lines reply) block nil blocks []]
                 (if-let [[l & more] (seq lines)]
                   (cond (and (nil? block) (re-matches #"\s*```[a-zA-Z]*\s*" l)) (recur more [] blocks)
                         (nil? block) (recur more nil blocks)
                         (re-matches #"\s*```\s*" l) (recur more nil (conj blocks block))
                         :else (recur more (conj block l) blocks))
                   (cond-> blocks block (conj block))))]
    (if (seq blocks) (str (str/join "\n" (peek blocks)) "\n") reply)))

(defn repair-delimiters
  "code's delimiters rebuilt from its indentation (parinfer's indent mode,
   parinferish): what clojure-mcp-light's hook does for an LLM's edits."
  [code]
  (let [parse (requiring-resolve 'parinferish.core/parse)
        flat (requiring-resolve 'parinferish.core/flatten)]
    (flat (parse code {:mode :indent}))))

(defn read-forms
  "The forms of text, read with *read-eval* false (#= refused) and no
   default tagged-literal reader."
  [text]
  (binding [*read-eval* false *default-data-reader-fn* nil]
    (let [r (clojure.lang.LineNumberingPushbackReader. (java.io.StringReader. text))]
      (loop [forms []]
        (let [f (read {:eof ::eof} r)]
          (if (= f ::eof) forms (recur (conj forms f))))))))

(def ^:private forbidden-names
  "Names a model has no use for that reach past it: evaluation and loading,
   namespaces and vars, files and processes, threads, Java classes."
  '#{eval load load-file load-string load-reader read read-string require use import refer
     refer-clojure ns in-ns create-ns remove-ns ns-unmap ns-unalias ns-resolve resolve intern
     alter-var-root with-redefs with-redefs-fn find-var var-set set! new monitor-enter
     monitor-exit future future-call pmap pcalls pvalues agent send send-off send-via
     spit slurp file-seq line-seq sh System Runtime Thread ProcessBuilder Class ClassLoader
     proxy reify gen-class gen-interface definterface deftype defrecord extend extend-type
     extend-protocol compile})

(def ^:private allowed-qualifiers
  "The namespaces and classes a qualified name may name: the prelude's
   aliases and java.lang.Math."
  #{"array" "cm" "comb" "d" "gfi" "Math" "clojure.core"})

(defn- forbidden? [s]
  (let [n (name s) q (namespace s)]
    (or (contains? forbidden-names (symbol n))
        (and q (not (allowed-qualifiers q)))
        (and (nil? q) (or (str/starts-with? n ".") (and (str/ends-with? n ".") (not= n "."))
                          (re-find #"\w\.\w" n))))))

(defn forbidden
  "The forbidden names forms use, sorted: forbidden-names, a name
   qualified by anything but the prelude's aliases and Math, a method call
   (.m), a constructor (C.) or a class name (java.io.File)."
  [forms]
  (->> (tree-seq coll? seq forms)
       (filter symbol?)
       (filter forbidden?)
       distinct
       (sort-by str)
       vec))

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

(defn evaluate
  "forms evaluated in a new namespace under the prelude: [the namespace's
   name, the value of its var model]. Throws when model is not defined."
  [forms]
  (let [n (gensym "gen.codegen.candidate-")]
    (binding [*ns* *ns*]
      (eval (list 'ns n prelude requires))
      (doseq [f forms] (eval f))
      (let [v (ns-resolve (the-ns n) 'model)]
        (when-not v
          (throw (ex-info "the code defines no model: (def model (gen …))" {})))
        [n @v]))))

(defn discard!
  "Removes a candidate's namespace."
  [n]
  (remove-ns n))
