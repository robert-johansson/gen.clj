(ns gen.book
  "The book (book/) is Markdown whose code runs (DESIGN.md §9).

   A chapter book/NNN-name.md keeps its code in a namespace,
   book/src/book/name.clj, written and run at the REPL. A line
   `;; book: region` there starts a region, which runs to the next such
   line. The Markdown includes a region with a line `<!-- code: region -->`
   followed by a ```clojure block holding the region's text, which
   `clojure -M:test -m gen.book sync` writes. Every ```clojure block of a
   chapter is such an include, so no code in the book goes untested.

   A chapter's test checks that every include holds its region's text,
   then runs the namespace region by region, form by form, in one
   namespace (the first region's ns form names it), output discarded.
   Comments in the code make checks:

   - `;=> value` after a form: the form's result, read back as Clojure data
     (gen.array arrays by gen.array/->clj), equals value, which is read, not
     evaluated. A float checks to the digits it shows (half a unit in its
     last place, as Clojure prints it, plus the backend's tolerance), so one
     text passes on float64 and float32 backends; an integer exactly. So
     few digits make a weak check (`;=> 0.0` accepts anything within
     0.05): where values must agree up to rounding, the code says so with
     gen.array/allclose? and `;=> true`.
   - `;=> value ; float64`: the same, checked only where the default float
     is float64 (the JVM backend; MLX with -Dgen.mlx.dtypes=64). For a
     sampled value: a key draws other values in float32.
   - `;; error: message` as a region's first line: one of its forms throws
     an ex-info whose message begins with message. The forms after it do
     not run.
   - `;; figure: path` as a region's first line: its last form returns a
     figure's SVG text (gen.book.svg), which is the file book/path. On the
     JVM backend the test fails when the file differs (a stale figure);
     sync writes it.

   A statistical claim is a check written in the code (a band from the
   estimator's standard error, `;=> true`), so it holds on every backend.
   A region's lines are at most 80 characters, so they fit a printed page.

   (run-chapter \"book/011-getting-started.md\") runs one at the REPL;
   gen.book-test runs them all. `clojure -M:test -m gen.book pdf` builds
   the book as a PDF (target/book/gen-clj-book.pdf), with Pandoc, xelatex
   and librsvg's rsvg-convert (for the SVG figures)."
  (:require [clojure.java.io :as io]
            [clojure.string :as str]
            [clojure.test :refer [is testing]]
            [gen.array :as array]
            [gen.test-support :as ts]))

(def dir "book")

(defn chapters
  "The book's chapters, its numbered Markdown files (011-getting-started.md),
   in order."
  []
  (->> (.listFiles (io/file dir))
       (filter #(re-matches #"\d{3}-.*\.md" (.getName ^java.io.File %)))
       (sort-by #(.getName ^java.io.File %))
       (mapv #(.getPath ^java.io.File %))))

(defn source-file
  "The namespace file of a chapter: book/011-getting-started.md's is
   book/src/book/getting_started.clj."
  [chapter]
  (let [[_ name] (re-matches #"\d{3}-(.*)\.md" (.getName (io/file chapter)))]
    (str dir "/src/book/" (str/replace name "-" "_") ".clj")))

;; ---------------------------------------------------------------------------
;; Regions and includes
;; ---------------------------------------------------------------------------

(defn regions
  "The regions of a chapter's source, in order: {:name :line :code}, line
   being the `;; book:` line's. Text before the first is not a region."
  [text]
  (let [lines (str/split-lines text)
        starts (keep-indexed (fn [i l] (when-let [[_ n] (re-matches #";; book: (\S+)\s*" l)] [i n])) lines)
        rs (mapv (fn [[i n] end]
                   {:name n
                    :line (inc i)
                    :code (str (str/trimr (str/join "\n" (drop-while str/blank? (subvec lines (inc i) end)))) "\n")})
                 starts
                 (concat (map first (rest starts)) [(count lines)]))
        repeated (keep (fn [[n c]] (when (> c 1) n)) (frequencies (map :name rs)))]
    (when (seq repeated)
      (throw (ex-info (str "Regions named more than once: " (str/join ", " repeated)) {:names (vec repeated)})))
    rs))

(def ^:private include-re #"(?ms)^<!-- code: (\S+) -->\n(?:```clojure\n(.*?)^```\n)?")

(defn includes
  "The includes of a chapter's Markdown: {:name :code}, code being the
   block after the marker (nil when there is none)."
  [md]
  (let [m (re-matcher include-re md)]
    (loop [out []]
      (if (.find m) (recur (conj out {:name (.group m 1) :code (.group m 2)})) out))))

(defn synced
  "The Markdown with each include's block holding its region's text."
  [md regions]
  (let [by-name (into {} (map (juxt :name :code)) regions)]
    (str/replace md include-re
                 (fn [[_ name]]
                   (let [code (or (by-name name)
                                  (throw (ex-info (str "No region " name " to include") {:region name})))]
                     (str "<!-- code: " name " -->\n```clojure\n" code "```\n"))))))

(def max-line
  "The longest line a region may have: what fits a printed page."
  80)

(defn check-lines
  "Every line of every region fits in max-line characters."
  [regions]
  (doseq [{:keys [name line code]} regions
          [i l] (map-indexed vector (str/split-lines code))
          :when (> (count l) max-line)]
    (is false (str "the region " name ", line " (+ line i 1) ": " (count l) " characters, over " max-line))))

(defn check-includes
  "Every include names a region and holds its text, and every clojure
   block of the Markdown is an include's."
  [md regions]
  (let [by-name (into {} (map (juxt :name :code)) regions)
        incs (includes md)]
    (doseq [{:keys [name code]} incs]
      (testing (str "the include of " name)
        (is (contains? by-name name) (str "no region " name))
        (when (contains? by-name name)
          (is (= (by-name name) code) "stale: clojure -M:test -m gen.book sync"))))
    (is (= (count (re-seq #"(?m)^```clojure$" md)) (count (filter :code incs)))
        "every clojure block is an include (<!-- code: region -->)")))

;; ---------------------------------------------------------------------------
;; Running the code
;; ---------------------------------------------------------------------------

(defn- forms
  "A region's forms, each with the values its ;=> markers expect:
   {:form :expect [{:value :float64?}]}."
  [code]
  (let [marked (-> code
                   (str/replace #"(?m);=>(.*;\s*float64\s*)$" " :gen.book/=f64> $1")
                   (str/replace ";=>" " :gen.book/=> "))
        rdr (clojure.lang.LineNumberingPushbackReader. (java.io.StringReader. marked))
        next-form #(read {:eof ::eof} rdr)
        expect (fn [out x] (update-in out [(dec (count out)) :expect] (fnil conj []) x))]
    (loop [out []]
      (let [form (next-form)]
        (cond
          (= ::eof form) out
          (= ::=> form) (recur (expect out {:value (next-form)}))
          (= ::=f64> form) (recur (expect out {:value (next-form) :float64? true}))
          :else (recur (conj out {:form form})))))))

(defn- read-back [v]
  (cond (or (number? v) (boolean? v)) v
        (array/array? v) (read-back (array/->clj v))
        (map? v) (into {} (map (fn [[k x]] [k (read-back x)])) v)
        (sequential? v) (mapv read-back v)
        :else v))

(defn- shown-tolerance
  "Half a unit in the last place x shows, as Clojure prints it."
  ^double [x]
  (* 0.5 (Math/pow 10.0 (- (.scale (BigDecimal. (str x)))))))

(defn same?
  "Does the actual value match the expected one, a float to the digits it
   shows?"
  [expected actual]
  (cond
    (and (integer? expected) (integer? actual)) (== expected actual)
    (and (number? expected) (number? actual))
    (let [e (double expected) a (double actual)]
      (or (== e a)
          (<= (Math/abs (- e a)) (+ (if (integer? expected) 0.0 (shown-tolerance expected))
                                    (* 2e-5 (max 1.0 (Math/abs e)))))))
    (and (sequential? expected) (sequential? actual))
    (and (= (count expected) (count actual)) (every? true? (map same? expected actual)))
    (and (map? expected) (map? actual))
    (and (= (set (keys expected)) (set (keys actual)))
         (every? #(same? (get expected %) (get actual %)) (keys expected)))
    :else (= expected actual)))

(defn- marker [code re] (second (re-find re code)))

(defn- run-region
  "Evaluates a region's forms: {:checks n} and, for a figure, {:figure [path svg]}."
  [{:keys [name code]}]
  (testing (str "the region " name)
    (let [fs (forms code)
          error (marker code #"^;; error: (.*)")
          figure (marker code #"^;; figure: (\S+)")]
      (if error
        (let [e (try (doseq [{:keys [form]} fs] (eval form)) nil (catch Throwable e e))]
          (is (some? e) "no form threw")
          (is (some-> e ex-message (str/starts-with? error)) (str "the message is " (some-> e ex-message)))
          {:checks 1})
        (reduce (fn [acc {:keys [form expect]}]
                  (let [v (try (eval form) (catch Throwable e e))]
                    (if (instance? Throwable v)
                      (do (is false (str (pr-str form) " threw: " (ex-message v))) (reduced acc))
                      (let [checked (remove #(and (:float64? %) (ts/f32?)) expect)]
                        (doseq [{x :value} checked]
                          (is (same? (read-back x) (read-back v))
                              (str (pr-str form) "\nexpected " (pr-str x) "\nactual   " (pr-str (read-back v)))))
                        (cond-> (update acc :checks + (count checked))
                          figure (assoc :figure [figure v]))))))
                {:checks 0} fs)))))

(defn run-regions
  "Runs a chapter's regions in order, in the namespace the first one's ns
   form names: {:regions :checks :figures}, figures a map from path to SVG
   text."
  [regions]
  (let [first-form (some-> (first regions) :code forms first :form)
        ns-name (when (and (seq? first-form) (= 'ns (first first-form))) (second first-form))]
    (is (some? ns-name) "the first region is an ns form")
    (when ns-name (remove-ns ns-name))
    (binding [*ns* (the-ns 'user)
              *out* (java.io.StringWriter.)
              *err* (java.io.StringWriter.)]
      (reduce (fn [acc r]
                (let [{:keys [checks figure]} (run-region r)]
                  (cond-> (update acc :checks + checks)
                    figure (assoc-in [:figures (first figure)] (second figure)))))
              {:regions (count regions) :checks 0 :figures {}}
              regions))))

(defn check-figures
  "Each figure (a map from path to SVG text) is SVG, and on the JVM backend
   the file book/path holds exactly it."
  [figures]
  (doseq [[path svg] figures
          :let [f (io/file dir path)]]
    (testing (str "the figure " path)
      (is (and (string? svg) (str/starts-with? svg "<svg")) "a figure region returns SVG text")
      (when (ts/jvm?)
        (is (and (.exists f) (= svg (slurp f)))
            (str (.getPath f) " is stale: clojure -M:test -m gen.book sync"))))))

(defn run-chapter
  "Checks a chapter's includes, runs its namespace and checks its figures:
   {:regions :checks :figures}."
  [chapter]
  (testing chapter
    (let [rs (regions (slurp (source-file chapter)))]
      (check-lines rs)
      (check-includes (slurp chapter) rs)
      (let [r (run-regions rs)]
        (check-figures (:figures r))
        r))))

(def pdf-file "target/book/gen-clj-book.pdf")

(defn pdf!
  "Builds the book as a PDF with Pandoc: the chapters in order, with
   book/metadata.yaml. Loud when an include is stale (sync first) or Pandoc
   fails."
  []
  (let [stale (filter #(not= (slurp %) (synced (slurp %) (regions (slurp (source-file %))))) (chapters))]
    (when (seq stale)
      (throw (ex-info (str "Stale includes in " (str/join ", " stale) ": clojure -M:test -m gen.book sync")
                      {:chapters (vec stale)}))))
  (let [args (into ["pandoc" (str "--metadata-file=" dir "/metadata.yaml") "--top-level-division=chapter"
                    "--pdf-engine=xelatex" "--syntax-highlighting=tango" (str "--resource-path=" dir)
                    "-o" pdf-file]
                   (chapters))
        _ (io/make-parents pdf-file)
        exit (try (-> (ProcessBuilder. ^java.util.List args) (.inheritIO) (.start) (.waitFor))
                  (catch java.io.IOException e
                    (throw (ex-info (str "Pandoc did not start (" (ex-message e) "): the PDF needs pandoc, xelatex"
                                         " and rsvg-convert (librsvg)")
                                    {:args args}))))]
    (when-not (zero? exit)
      (throw (ex-info (str "Pandoc failed, exit " exit) {:exit exit :args args})))
    (println "wrote" pdf-file)))

(defn -main
  "sync: write every chapter's includes from its regions, and its figures.
   pdf: build the book as a PDF."
  [& [what]]
  (case what
    "sync" (doseq [chapter (chapters)
                   :let [rs (regions (slurp (source-file chapter)))
                         md (slurp chapter)
                         md' (synced md rs)]]
             (when (not= md md')
               (spit chapter md')
               (println "wrote" chapter))
             (doseq [[path svg] (:figures (run-regions rs))
                     :let [f (io/file dir path)]]
               (when-not (and (.exists f) (= svg (slurp f)))
                 (io/make-parents f)
                 (spit f svg)
                 (println "wrote" (.getPath f)))))
    "pdf" (pdf!)
    (println "usage: clojure -M:test -m gen.book sync|pdf"))
  (shutdown-agents))
