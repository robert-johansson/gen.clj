(ns gen.api
  "The API's tiers and the stable surface's snapshot (DESIGN.md §9, \"The
   1.0's shape\").

   Tiers. Every namespace of src/ and of ext/'s modules (ext/*/src, gen.laws
   in ext/laws among them) carries its tier in its ns form's attribute map:

     (ns gen.trace \"…\" {:api :stable} (:require …))

   - :stable: through 1.x its names, arities, result-map keys and
     documented semantics stay put; minor releases add to them.
   - :experimental: shipped, outside the promise.
   - :internal: not for dependents.

   A stable namespace whose protocols other code implements says so with
   :implementers, a note naming them (gen.gfi, gen.array.protocols). It is
   the same promise, seen from the other side: through 1.x such a protocol
   keeps its methods and gains none. gen.api-test fails on a namespace
   without a tier, or with another value.

   The snapshot. test/fixtures/api-snapshot.edn holds every public var of
   every stable namespace, but those marked ^:no-doc: its arglists, or its
   kind (macro, protocol with its methods, multimethod, var), and
   :deprecated when it is. gen.api-test fails when the code's stable
   surface differs from it, naming what was added, removed or changed, and
   calling out apart a var removed without a deprecation first. Changing
   the surface is deliberate: write the snapshot,

     clojure -M:test -m gen.api write

   and commit it with the change and its CHANGELOG line. Until the 1.0
   tag a removal is written so too, its CHANGELOG line marked Breaking.
   `clojure -M:test -m gen.api check` prints the differences.

   Deprecation (DESIGN.md §9): a stable var to be removed in 2.0 gets
   :deprecated metadata, the version it was deprecated in, and a note in
   its docstring naming what replaces it. Nothing is printed when it is
   called. The snapshot records :deprecated, so deprecating is a snapshot
   change too, and its removal before 2.0 still fails.

     (defn ^{:deprecated \"1.2\"} old-name
       \"Deprecated (1.2): use new-name. …\"
       …)

   The book's appendix The Public API (book/063-api.md) is written from the
   tiers and the docstrings (gen.book)."
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [clojure.walk :as walk]))

(def tiers
  "The tiers a namespace can carry, as :api in its ns form."
  #{:stable :experimental :internal})

(def snapshot-file
  "The committed snapshot of the stable surface."
  "test/fixtures/api-snapshot.edn")

(def write-command
  "The command that writes the snapshot from the code."
  "clojure -M:test -m gen.api write")

;; ---------------------------------------------------------------------------
;; Tiers
;; ---------------------------------------------------------------------------

(defn source-files
  "The source files whose namespaces carry tiers: src/ and every
   extension's src/ (ext/*/src: ext/laws' gen.laws among them), in path
   order."
  []
  (let [clj? #(and (.isFile ^java.io.File %) (str/ends-with? (.getName ^java.io.File %) ".clj"))
        exts (->> (.listFiles (io/file "ext")) (map #(io/file % "src")) (filter #(.isDirectory ^java.io.File %)))]
    (->> (mapcat file-seq (cons (io/file "src") exts))
         (filter clj?)
         (map #(.getPath ^java.io.File %))
         sort
         vec)))

(defn ns-info
  "What an ns form says of its namespace, read without loading it: {:ns
   :doc :attrs}, attrs being its attribute map (nil when it has none), or
   nil when form is not an ns form."
  [form]
  (when (and (seq? form) (= 'ns (first form)) (symbol? (second form)))
    (let [[doc & more] (drop 2 form)
          [doc more] (if (string? doc) [doc more] [nil (cons doc more)])]
      {:ns (second form) :doc doc :attrs (when (map? (first more)) (first more))})))

(defn read-ns-form
  "The first form of a source file, read as data (no evaluation)."
  [file]
  (with-open [r (java.io.PushbackReader. (io/reader file))]
    (binding [*read-eval* false]
      (read {:eof nil} r))))

(defn namespaces
  "Every namespace that carries a tier, in path order: {:ns :doc :attrs
   :tier :file}, tier being (:api attrs)."
  []
  (mapv (fn [file]
          (let [info (ns-info (read-ns-form file))]
            (assoc info :file file :tier (get-in info [:attrs :api]))))
        (source-files)))

(defn tier-problems
  "What is wrong with namespaces' marks (as namespaces gives them), one
   sentence each: a file without an ns form, a namespace without a tier or
   with another value, an :implementers note on a namespace that is not
   stable or that is not a string."
  [infos]
  (let [allowed (str/join ", " (map #(str "{:api " % "}") (sort tiers)))]
    (for [{:keys [ns file tier attrs]} infos
          :let [note (:implementers attrs)
                problem (cond
                          (nil? ns) (str file " does not start with an ns form")
                          (nil? tier) (str ns " (" file ") has no tier: its ns form needs one of " allowed)
                          (not (tiers tier)) (str ns " (" file ") has the tier " (pr-str tier) ", not one of " allowed)
                          (and (some? note) (not= :stable tier)) (str ns " has :implementers but is " tier
                                                                      ": the note is for stable namespaces")
                          (and (some? note) (not (string? note))) (str ns "'s :implementers is "
                                                                       (pr-str note) ", not a string"))]
          :when problem]
      problem)))

(defn stable-namespaces
  "The names of the stable namespaces, sorted."
  []
  (->> (namespaces) (filter #(= :stable (:tier %))) (map :ns) sort vec))

;; ---------------------------------------------------------------------------
;; The stable surface
;; ---------------------------------------------------------------------------

(defn- as-edn
  "x as EDN reads it back: printed without metadata (type hints) and read."
  [x]
  (edn/read-string (binding [*print-meta* false *print-length* nil *print-level* nil *print-namespace-maps* false]
                     (pr-str x))))

(defn published?
  "Is v a var of the surface: public, not ^:no-doc, and not the var proxy
   interns to hold its class (gen.gfi.proxy$java.lang.ClassValue$…)?"
  [v]
  (let [m (meta v)]
    (not (or (:private m) (:no-doc m) (str/includes? (str (:name m)) "proxy$")))))

(defn- arglists
  "Arglists as EDN, a macro's generated names (defrecord's map->'s
   m__7997__auto__) without the counter, which changes with Clojure's
   build."
  [as]
  (walk/postwalk #(if-let [[_ base] (and (symbol? %) (re-matches #"(.+?)__\d+__auto__" (name %)))] (symbol base) %)
                 (as-edn as)))

(defn entry
  "A var's record in the snapshot: :arglists for a function, :kind
   (:macro, :protocol, :multimethod, :var, or :fn for a function without
   arglists) for the rest, with arglists where it has them and a
   protocol's :methods; :dynamic and :deprecated when the var is."
  [v]
  (let [m (meta v)
        x @v
        kind (cond (:macro m) :macro
                   (and (map? x) (:on-interface x)) :protocol
                   (instance? clojure.lang.MultiFn x) :multimethod
                   (or (fn? x) (:arglists m)) :fn
                   :else :var)]
    (cond-> {}
      (not (and (= :fn kind) (:arglists m))) (assoc :kind kind)
      (:arglists m) (assoc :arglists (arglists (:arglists m)))
      (= :protocol kind) (assoc :methods (vec (sort (map :name (vals (:sigs x))))))
      (:dynamic m) (assoc :dynamic true)
      (:deprecated m) (assoc :deprecated (as-edn (:deprecated m))))))

(defn surface
  "The stable surface of the loaded code: {namespace {var-name entry}}
   over the namespaces given (default: every stable one, loaded)."
  ([] (surface (stable-namespaces)))
  ([nses]
   (into (sorted-map)
         (for [n nses]
           (do (require n)
               (when-not (= :stable (:api (meta (the-ns n))))
                 (throw (ex-info (str n " loads without {:api :stable} in its metadata") {:ns n})))
               [n (into (sorted-map)
                        (for [[s v] (ns-publics n) :when (published? v)]
                          [s (entry v)]))])))))

(defn read-snapshot
  "The committed snapshot, {namespace {var-name entry}}."
  ([] (read-snapshot snapshot-file))
  ([file] (edn/read-string (slurp file))))

(defn snapshot-text
  "A surface as the snapshot file's text: a namespace per block, a var per
   line, both sorted, after a header saying how to update it."
  [surf]
  (binding [*print-meta* false *print-length* nil *print-level* nil *print-namespace-maps* false]
    (let [block (fn [[n vars]]
                  (str " " n "\n {"
                       (str/join "\n  " (map (fn [[s e]] (str s " " (pr-str e))) (sort-by (comp str key) vars)))
                       "}"))]
      (str ";; gen.clj's stable surface (DESIGN.md §9, \"The 1.0's shape\"): every public\n"
           ";; var, but ^:no-doc ones, of every namespace marked {:api :stable}, with its\n"
           ";; arglists, or its kind where they do not apply. gen.api-test fails when the\n"
           ";; code differs. A change of the surface is deliberate: write this file with\n"
           ";;   " write-command "\n"
           ";; and commit it with the change and its CHANGELOG line. A stable var is\n"
           ";; deprecated (:deprecated metadata, a docstring note) before it is removed.\n"
           "{" (subs (str/join "\n\n" (map block (sort-by (comp str key) surf))) 1) "}\n"))))

(defn- qualified [surf]
  (into {} (for [[n vars] surf [s e] vars] [(symbol (str n) (str s)) e])))

(defn diff
  "How the surface now differs from the surface was (a snapshot): {:added
   :removed :removed-deprecated :deprecated :changed :namespaces-added
   :namespaces-removed}, each sorted. removed holds the vars removed
   without a deprecation first, removed-deprecated those deprecated before;
   deprecated the vars deprecated since, otherwise unchanged; changed
   {:var :was :now} for every other difference."
  [was now]
  (let [w (qualified was)
        n (qualified now)
        both (filter #(contains? w %) (keys n))
        newly-deprecated? #(and (:deprecated (n %)) (not (:deprecated (w %)))
                                (= (dissoc (n %) :deprecated) (w %)))
        gone (remove #(contains? n %) (keys w))]
    {:added (vec (sort (remove #(contains? w %) (keys n))))
     :removed (vec (sort (remove #(:deprecated (w %)) gone)))
     :removed-deprecated (vec (sort (filter #(:deprecated (w %)) gone)))
     :deprecated (vec (sort (filter newly-deprecated? both)))
     :changed (->> both
                   (filter #(and (not= (w %) (n %)) (not (newly-deprecated? %))))
                   sort
                   (mapv (fn [v] {:var v :was (w v) :now (n v)})))
     :namespaces-added (vec (sort (remove (set (keys was)) (keys now))))
     :namespaces-removed (vec (sort (remove (set (keys now)) (keys was))))}))

(defn same?
  "Does a diff say the surfaces are the same?"
  [d]
  (every? empty? (vals d)))

(defn report
  "A diff as text, a line per difference, after the one line saying how to
   write the snapshot; empty when there is none."
  [d]
  (let [lines (fn [title xs] (when (seq xs) (cons (str title ":") (map #(str "  " %) xs))))]
    (if (same? d)
      ""
      (str/join "\n"
                (concat
                 [(str "The stable surface differs from " snapshot-file ". If that is deliberate, write it: "
                       write-command " (and add a CHANGELOG line).")]
                 (lines "Removed without a deprecation first (1.x keeps a stable var: deprecate it, remove it in 2.0)"
                        (:removed d))
                 (lines "Removed after a deprecation (a removal waits for 2.0)" (:removed-deprecated d))
                 (lines "Namespaces no longer stable" (:namespaces-removed d))
                 (lines "Namespaces newly stable" (:namespaces-added d))
                 (lines "Added" (:added d))
                 (lines "Deprecated" (:deprecated d))
                 (lines "Changed" (map (fn [{:keys [var was now]}] (str var ": " (pr-str was) " -> " (pr-str now)))
                                       (:changed d))))))))

(defn deprecated
  "The deprecated vars of a surface, sorted, with their :deprecated."
  [surf]
  (->> (qualified surf) (filter (comp :deprecated val)) (sort-by key) (mapv (fn [[v e]] [v (:deprecated e)]))))

(defn write!
  "Writes the snapshot from the loaded code's stable surface; returns the
   diff from the snapshot it replaces."
  []
  (let [now (surface)
        was (when (.exists (io/file snapshot-file)) (read-snapshot))]
    (spit snapshot-file (snapshot-text now))
    (diff (or was {}) now)))

(defn- counts [surf]
  (str (count surf) " namespaces, " (reduce + (map count (vals surf))) " vars"))

(defn -main
  "write: write the snapshot from the code, printing what changed.
   check: print how the code's stable surface differs from the snapshot,
   and the deprecated vars; exit 1 when it differs."
  [& [what]]
  (case what
    "write" (let [d (write!)
                  surf (read-snapshot)]
              (println "wrote" snapshot-file (str "(" (counts surf) ")"))
              (doseq [[k xs] d :when (seq xs)]
                (println (str (name k) ":") (str/join " " (map #(if (map? %) (:var %) %) xs)))))
    "check" (let [surf (surface)
                  d (diff (read-snapshot) surf)]
              (println (counts surf) "in the stable surface")
              (println "deprecated:" (if-let [ds (seq (deprecated surf))] (str/join ", " (map first ds)) "none"))
              (println (if (same? d) "the snapshot is current" (report d)))
              (shutdown-agents)
              (System/exit (if (same? d) 0 1)))
    (println "usage: clojure -M:test -m gen.api write|check"))
  (shutdown-agents))
