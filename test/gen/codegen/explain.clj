(ns gen.codegen.explain
  "Errors an LLM can act on (genclj-t1xh, genclj-hu8z): what a candidate's
   code got wrong, said briefly and with what to use instead.

     (lint code)               → clj-kondo's findings on code, as lines:
                                 unresolved names and vars (d/point), wrong
                                 arities ((array/sin a b)), malformed
                                 bindings; gen.clj's namespaces known to it
                                 (a cache of src/, made once)
     (explain e ns)            → an evaluation error's message, with its
                                 causes, enriched: did you mean, for a name
                                 that is not there (the alias's real names);
                                 the arglists, for a wrong arity; a spec
                                 failure of a macro (let, defn) as one line
                                 a problem

   clj-kondo is on the :test alias; without it lint returns nil."
  (:require [clojure.java.io :as io]
            [clojure.string :as str]))

;; ---------------------------------------------------------------------------
;; Did you mean
;; ---------------------------------------------------------------------------

(defn- distance
  "The edit distance of strings a and b."
  [^String a ^String b]
  (let [n (count b)]
    (peek (reduce (fn [prev [i ca]]
                    (reduce (fn [row j]
                              (conj row (min (inc (peek row)) (inc (prev (inc j)))
                                             (+ (prev j) (if (= ca (.charAt b j)) 0 1)))))
                            [(inc i)] (range n)))
                  (vec (range (inc n))) (map-indexed vector a)))))

(defn- closest
  "Up to k of names nearest to s (by edit distance, then length), not s."
  [s names k]
  (->> names (remove #{s}) (sort-by (juxt #(distance s %) count)) (take k)))

(def ^:private aliases
  {"d" 'gen.dist "array" 'gen.array "comb" 'gen.combinators "cm" 'gen.choicemap "gfi" 'gen.gfi})

(defn- publics [ns-sym]
  (try (require ns-sym) (sort (map str (keys (ns-publics ns-sym)))) (catch Exception _ nil)))

(defn- no-such-var
  "For No such var: alias/name, what the alias has that is close."
  [alias nm]
  (when-let [ns-sym (aliases alias)]
    (let [names (publics ns-sym)]
      (str "There is no " alias "/" nm ". Did you mean " (str/join ", " (map #(str alias "/" %) (closest nm names 3)))
           "? " alias "/ is " ns-sym ", whose names are: " (str/join " " (take 60 names))
           (when (> (count names) 60) " …") "."))))

(defn- unresolved
  "For Unable to resolve symbol: s, what is close in the namespace (its
   own definitions and clojure.core's) and in the aliases."
  [s ns]
  (cond (#{"trace" "splice"} s)
        (str s " exists only inside a (gen [...] ...) body, where it records a random choice.")
        (= "gen" s) "gen comes from gen.dynamic and is referred by the prelude: (gen [args] body)."
        :else
        (let [here (concat ["trace" "splice" "gen"] (when ns (map str (keys (ns-map ns)))))
              qualified (for [[a n] aliases :when (#{"d" "array" "comb"} a) p (publics n)] (str a "/" p))
              near (closest s (concat here qualified) 3)]
          (when (seq near) (str "Did you mean " (str/join ", " near) "?")))))

(defn- arglists-of
  "For Wrong number of args (n) passed to: ns/name, that var's arglists."
  [fqn]
  (when-let [v (try (requiring-resolve (symbol fqn)) (catch Exception _ nil))]
    (when-let [al (:arglists (meta v))]
      (str (symbol fqn) " takes " (str/join " or " (map #(pr-str (list* (symbol (name (symbol fqn))) %)) al)) "."))))

;; ---------------------------------------------------------------------------
;; Spec failures of macros, briefly
;; ---------------------------------------------------------------------------

(defn- spec-problems
  "A macro's spec failure (clojure.spec's ex-data) as one line a problem."
  [macro data]
  (let [problems (:clojure.spec.alpha/problems data)]
    (when (seq problems)
      (str/join "\n" (for [{:keys [path pred val]} (take 3 problems)]
                       (str "In (" macro " …)" (when (seq path) (str ", at " (str/join " " (map name path))))
                            ": expected " (let [p (pr-str pred)] (subs p 0 (min 80 (count p))))
                            ", got " (let [v (pr-str val)] (subs v 0 (min 120 (count v))))
                            (when (and (= "let" (name (symbol macro)))
                                       (some #(and (seq? %) (#{'defn 'def 'defn-} (first %))) (tree-seq coll? seq val)))
                              " (a let binding is a name and a value: define functions before the let, or bind (fn …))")))))))

(defn explain
  "e's message and its causes', each enriched where it can be."
  [^Throwable e ns]
  (let [chain (take-while some? (iterate ex-cause e))
        msgs (distinct (map #(or (ex-message %) (str (class %))) chain))
        spec (some #(when (:clojure.spec.alpha/problems (ex-data %)) %) chain)
        hints (keep (fn [m]
                      (or (when-let [[_ a n] (re-find #"No such var: ([\w-]+)/([^\s]+)" m)] (no-such-var a n))
                          (when-let [[_ s] (re-find #"Unable to resolve symbol: ([^\s]+) in this context" m)] (unresolved s ns))
                          (when-let [[_ f] (re-find #"Wrong number of args \(\d+\) passed to: ([\w.-]+/[^\s]+)" m)]
                            (arglists-of (str/replace f "_" "-")))
                          (when (re-find #"No matching method (sin|cos|exp|log|sqrt|pow|tan)\b" m)
                            "java.lang.Math's methods take host numbers and fixed arities; use gen.array's: array/sin, array/exp, … (they take arrays too).")))
                    msgs)
        spec-text (when spec
                    (spec-problems (or (second (re-find #"Call to ([\w./-]+) did not conform" (str/join " " msgs))) "the macro")
                                   (ex-data spec)))]
    (str/join "\n" (concat [(str/join " Caused by: " msgs)] (distinct hints) (when spec-text [spec-text])))))

;; ---------------------------------------------------------------------------
;; clj-kondo
;; ---------------------------------------------------------------------------

(def ^:private kondo-config
  {:lint-as {'gen.dynamic/gen 'clojure.core/fn}
   :linters {:unresolved-symbol {:exclude ['trace 'splice 'args 'observations]}
             :unused-binding {:level :off} :unused-private-var {:level :off}
             :redefined-var {:level :off} :inline-def {:level :off}}
   :output {:canonical-paths true}})

(def ^:private kondo-dir
  "clj-kondo's config and cache directory, its cache holding gen.clj's
   sources (src/), so that d/…, array/… and comb/… names and arities are
   known: made once a JVM."
  (delay (let [d (io/file (System/getProperty "java.io.tmpdir") "gen-codegen-kondo")
               run! (requiring-resolve 'clj-kondo.core/run!)]
           (.mkdirs (io/file d ".cache"))
           (run! {:lint ["src"] :config-dir (str d) :cache true :cache-dir (str (io/file d ".cache")) :parallel true :repro true})
           d)))

(def ^:private prelude-lines 8)

(defn lint
  "clj-kondo's findings of level warning or error on code, under the
   prelude and after the forms before (a session's program, so that its
   definitions are known): one line each (its line in code, the message),
   or nil when there are none or clj-kondo is not on the classpath."
  ([code] (lint code ""))
  ([code before]
  (when-let [run! (try (requiring-resolve 'clj-kondo.core/run!) (catch Exception _ nil))]
    (let [prelude (str "(ns candidate\n  (:refer-clojure :exclude [+ - * / < > <= >= ==])\n"
                       "  (:require [gen.array :as array :refer [+ - * / < > <= >= ==]]\n"
                       "            [gen.choicemap :as cm]\n            [gen.combinators :as comb]\n"
                       "            [gen.dist :as d]\n            [gen.dynamic :refer [gen]]\n"
                       "            [gen.gfi :as gfi]))\n")
          offset (+ prelude-lines (count (re-seq #"\n" (str before "\n"))))
          findings (with-in-str (str prelude before "\n" code)
                     (:findings (run! {:lint ["-"] :lang :clj :config-dir (str @kondo-dir) :cache true
                                       :cache-dir (str (io/file @kondo-dir ".cache")) :config kondo-config :repro true})))
          ;; gen.array's and gen.dist's names are made by macros clj-kondo cannot see
          ;; through: an unresolved var is checked against the namespace itself
          real? (fn [message]
                  (when-let [[_ a nm] (re-find #"Unresolved var: ([\w-]+)/(\S+)" message)]
                    (when-let [ns-sym (aliases a)] (some #{nm} (publics ns-sym)))))
          lines (for [{:keys [row message level type]} findings
                      :when (and (#{:error :warning} level) (> row offset)
                                 (not (#{:unused-namespace :unused-referred-var :refer-all :missing-docstring} type))
                                 (not (real? message)))]
                  (str "line " (- row offset) ": "
                       (or (when-let [[_ a nm] (re-find #"Unresolved var: ([\w-]+)/(\S+)" message)] (no-such-var a nm))
                           message)))]
      (when (seq lines) (str/join "\n" (distinct lines)))))))
