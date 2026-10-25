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
     not run. In the source, each of the region's forms is written behind
     `#_`, so that requiring the namespace does not throw; the test runs
     them, and the Markdown shows them, without it.
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
            [gen.api :as api]
            [gen.array :as array]
            [gen.test-support :as ts]))

(def dir "book")

(defn source-file
  "The namespace file of a chapter: book/011-getting-started.md's is
   book/src/book/getting_started.clj."
  [chapter]
  (let [[_ name] (re-matches #"\d{3}-(.*)\.md" (.getName (io/file chapter)))]
    (str dir "/src/book/" (str/replace name "-" "_") ".clj")))

(defn book-files
  "The book's numbered Markdown files, in order: its chapters, and the
   pages without code (a part's heading, front matter)."
  []
  (->> (.listFiles (io/file dir))
       (filter #(re-matches #"\d{3}-.*\.md" (.getName ^java.io.File %)))
       (sort-by #(.getName ^java.io.File %))
       (mapv #(.getPath ^java.io.File %))))

(defn chapters
  "The book's chapters: its numbered Markdown files with a namespace of
   code (source-file), in order."
  []
  (filterv #(.exists (io/file (source-file %))) (book-files)))

(defn- printed-files
  "The book's files the PDF holds: a part's heading (NN0-…, no code) only
   when its part has a chapter, so a planned part prints no empty page."
  []
  (let [decade #(subs (.getName (io/file %)) 0 2)
        written (set (map decade (chapters)))]
    (filterv #(or (not (re-matches #"\d\d0-.*\.md" (.getName (io/file %)))) (written (decade %)))
             (book-files))))

;; ---------------------------------------------------------------------------
;; Regions and includes
;; ---------------------------------------------------------------------------

(defn- error-region? [source] (str/starts-with? source ";; error:"))

(defn regions
  "The regions of a chapter's source, in order: {:name :line :source
   :code}, line being the `;; book:` line's, source the region's text and
   code what the Markdown shows and the test runs (an error region's
   without its forms' `#_`). Text before the first is not a region."
  [text]
  (let [lines (str/split-lines text)
        starts (keep-indexed (fn [i l] (when-let [[_ n] (re-matches #";; book: (\S+)\s*" l)] [i n])) lines)
        rs (mapv (fn [[i n] end]
                   (let [source (str (str/trimr (str/join "\n" (drop-while str/blank? (subvec lines (inc i) end))))
                                     "\n")]
                     {:name n
                      :line (inc i)
                      :source source
                      :code (if (error-region? source) (str/replace source #"(?m)^#_" "") source)}))
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

(defn check-error-regions
  "Every top-level form of an error region is written behind #_, so that
   requiring the chapter's namespace does not throw."
  [regions]
  (doseq [{:keys [name source]} regions
          :when (error-region? source)
          l (str/split-lines source)
          :when (re-find #"^[^\s;#]" l)]
    (is false (str "the error region " name " has a form not behind #_ (the namespace would throw on load): " l))))

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
  "Half a unit in the last place x shows, as Clojure prints it. In
   E-notation the printed mantissa's padding zero is not a digit shown:
   0.0007 prints 7.0E-4 and shows four decimals, as written."
  ^double [x]
  (let [s (str x)
        d (cond-> (BigDecimal. s) (str/includes? s "E") .stripTrailingZeros)]
    (* 0.5 (Math/pow 10.0 (- (.scale d))))))

(defn same?
  "Does the actual value match the expected one, a float to the digits it
   shows?"
  [expected actual]
  (cond
    (and (integer? expected) (integer? actual)) (== expected actual)
    (and (number? expected) (number? actual))
    (let [e (double expected) a (double actual)]
      (if (or (Double/isNaN e) (Double/isInfinite e))
        (or (== e a) (and (Double/isNaN e) (Double/isNaN a)))
        (or (== e a)
            (<= (Math/abs (- e a)) (+ (if (integer? expected) 0.0 (shown-tolerance expected))
                                      (* 2e-5 (max 1.0 (Math/abs e))))))))
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

(defn check-in-step
  "The checks of a chapter that run none of its code: its includes hold
   their regions, its lines fit, its error regions' forms sit behind #_."
  [chapter]
  (testing chapter
    (let [rs (regions (slurp (source-file chapter)))]
      (check-lines rs)
      (check-error-regions rs)
      (check-includes (slurp chapter) rs))))

(defn part-chapters
  "The chapters whose file names start with one of prefixes: \"02\" for
   Part II, \"01\" \"06\" for Part I and the appendices."
  [& prefixes]
  (filterv (fn [c] (some #(str/starts-with? (.getName (io/file c)) %) prefixes)) (chapters)))

(defn runs-here?
  "Does the book run on this backend? The JVM backend is its gate (DESIGN.md
   §9); elsewhere only when -Dgen.book.mlx=true asks, since its examples run
   one model at a time, minutes on MLX for seconds on the JVM."
  []
  (or (ts/jvm?) (= "true" (System/getProperty "gen.book.mlx"))))

(declare run-chapter)

(defn check-chapters
  "Runs chapters (run-chapter) where the book runs (runs-here?), each with
   code and checks."
  [chapters]
  (when (runs-here?)
    (is (seq chapters) "the part has chapters")
    (doseq [chapter chapters]
      (let [{:keys [regions checks]} (run-chapter chapter)]
        (is (pos? regions) (str chapter " has code"))
        (is (pos? checks) (str chapter " checks results"))))))

(defn run-chapter
  "Checks a chapter's includes, runs its namespace and checks its figures:
   {:regions :checks :figures}. Failures print to the current *out*, so a
   REPL connected after clojure.test loaded sees them."
  [chapter]
  (binding [clojure.test/*test-out* *out*]
    (testing chapter
      (let [rs (regions (slurp (source-file chapter)))]
        (check-lines rs)
        (check-error-regions rs)
        (check-includes (slurp chapter) rs)
        (let [r (run-regions rs)]
          (check-figures (:figures r))
          r)))))

;; ---------------------------------------------------------------------------
;; Pages written from the code
;; ---------------------------------------------------------------------------

(defn- first-paragraph [doc]
  (-> (str/trim (or doc "")) (str/split #"\n\s*\n") first (str/replace #"\s*\n\s*" " ") (str/replace "*" "\\*")))

(defn- flag?
  "Does the multimethod mm answer true for the distribution family named
   type-kw, whatever its parameters (a flag: ^:discrete, ^:pathwise)?"
  [mm type-kw]
  (boolean (when-let [m (get (methods mm) type-kw)]
             (try (m {:type type-kw}) (catch Exception _ false)))))

(defn distributions-page
  "The appendix of distributions: every constructor of gen.dist, in source
   order, with its parameters, whether its values are discrete, whether it
   samples pathwise, and its docstring's first paragraph."
  []
  (require 'gen.dist 'gen.dist.core)
  (let [constructor? #(get (meta %) (keyword "gen.dist.core" "constructor"))
        ctors (->> (ns-publics 'gen.dist) vals (filter constructor?) (sort-by #(:line (meta %))))
        discrete (resolve 'gen.dist.core/-discrete?)
        pathwise (resolve 'gen.dist.core/-pathwise?)
        entry (fn [v]
                (let [{:keys [name arglists doc]} (meta v)
                      type-kw (keyword "gen.dist" (str name))
                      call (if-let [args (first arglists)]
                             (str "(d/" name (apply str (map #(str " " %) args)) ")")
                             (str "(d/" name " …)"))
                      flags (cond-> [] (flag? @discrete type-kw) (conj "discrete") (flag? @pathwise type-kw) (conj "pathwise"))]
                  (str "## " name "\n\n`" call "`" (when (seq flags) (str " · " (str/join ", " flags))) "\n\n"
                       (first-paragraph doc) "\n")))]
    (str "# The Distributions\n\n"
         "Every distribution in `gen.dist`, in the order its source defines them, with its parameters and the first "
         "paragraph of its docstring. The names and parameterizations are Gen.jl's; families Gen.jl does not have take "
         "Distributions.jl's parameters. A *discrete* family has discrete values. A *pathwise* one samples by a "
         "differentiable transform of noise drawn apart from its parameters, which variational inference's pathwise "
         "estimator needs. `gen.dist.dsl/dist` defines more distributions as pushforwards, Gen.jl's `@dist`, and "
         "`gen.dist.host` wraps host libraries' families.\n\n"
         "This page is written from the code (`clojure -M:test -m gen.book sync`), and the book's test fails when it "
         "is out of date.\n\n"
         (str/join "\n" (map entry ctors)))))

(def ^:private layers
  "The API appendix's layers, in order: a title, and the names of the
   namespaces it holds."
  [["Arrays and randomness" #"gen\.(array|random|backend)(\..+)?"]
   ["Choice maps, selections, traces and diffs" #"gen\.(choicemap|selection|trace|diff)"]
   ["The generative function interface" #"gen\.(gfi|dynamic|serialize|batched|partition|laws)(\..+)?"]
   ["Distributions" #"gen\.dist(\..+)?"]
   ["Combinators" #"gen\.(combinators|custom|state-space)"]
   ["Inference" #"gen\.(inference\..+|gradients|learning)"]
   ["Language models" #"gen\.llm\..+"]])

(defn- layer-of [n]
  (or (some (fn [[title re]] (when (re-matches re (str n)) title)) layers)
      (throw (ex-info (str "The API appendix has no layer for " n ": add it to gen.book's layers") {:ns n}))))

(defn- paragraph
  "A docstring's first paragraph on one line: its lines up to a blank one,
   a list's item (- …) or a table's row (columns aligned by spaces), a
   closing colon (before such an example) made a period."
  [doc]
  (let [[line & more] (str/split-lines (str/trim (or doc "")))
        prose (take-while #(not (or (str/blank? %) (re-find #"^\s*[-*] |\S\s{3,}\S" %))) more)]
    (-> (str/join " " (map str/trim (cons line prose))) (str/replace #":$" "."))))

(defn- sentence
  "A docstring's first sentence: its first paragraph up to the first
   period before a capital letter, or question or exclamation mark before
   a capital letter or a parenthesis."
  [doc]
  (let [p (paragraph doc)]
    (or (second (re-find #"^(.*?(?:\.(?=\s+[A-Z])|[?!](?=\s+[A-Z(])))" p)) p)))

(defn- escape
  "Text that Markdown shows as written: the punctuation it would read
   (emphasis, a link's bracket, raw HTML, math, super- and subscripts, a
   citation) escaped, but in code spans (when the backquotes pair up)."
  [s]
  (let [plain #(-> %
                   (str/replace #"[\\*\[$^~@]" (fn [c] (str "\\" c)))
                   (str/replace #"(?<![A-Za-z0-9])_|_(?![A-Za-z0-9])" "\\\\_")
                   (str/replace #"<(?=[A-Za-z/!?])" "\\\\<"))]
    (if (even? (count (filter #{\`} s)))
      (apply str (map #(if (str/starts-with? % "`") % (plain %)) (re-seq #"`[^`]*`|[^`]+" s)))
      (str/replace (plain s) "`" "\\`"))))

(def ^:private width
  "The longest line of a page written from the code."
  79)

(defn- wrap
  "Markdown text as lines of at most width characters (a longer word on a
   line of its own), the first starting with lead and the rest indented as
   far. A word Markdown would read as a block's marker at a line's start
   (a list's, a heading's) stays on the line before."
  [lead text]
  (let [marker? #(or (re-matches #"[-+:#=|]+|#\.|\(?(\d+|[a-zA-Z]|[ivxlcdmIVXLCDM]+)[.)]|\(@\w*\)" %)
                     (str/starts-with? % ">"))
        words (reduce (fn [ws w] (if (and (seq ws) (marker? w)) (conj (pop ws) (str (peek ws) " " w)) (conj ws w)))
                      [] (str/split (str/trim text) #"\s+"))
        indent (apply str (repeat (count lead) " "))]
    (->> words
         (reduce (fn [lines w]
                   (let [line (peek lines)]
                     (cond (= line lead) (conj (pop lines) (str line w))
                           (> (+ (count line) 1 (count w)) width) (conj lines (str indent w))
                           :else (conj (pop lines) (str line " " w)))))
                 [lead])
         (str/join "\n"))))

(defn- shown-args
  "An arglist as the appendix shows it: a destructured map by its :as name,
   or else by its :keys, without defaults; no commas."
  [args]
  (let [show (fn show [x]
               (cond (and (map? x) (:as x)) (str (:as x))
                     (and (map? x) (:keys x)) (str "{:keys [" (str/join " " (:keys x)) "]}")
                     (map? x) (str "{" (str/join " " (map (fn [[k v]] (str (show k) " " (show v))) x)) "}")
                     (vector? x) (str "[" (str/join " " (map show x)) "]")
                     :else (pr-str x)))]
    (str/join " " (map show args))))

(defn- var-item
  "A stable var's line in the API appendix: its calls or its kind, and its
   docstring's first sentence."
  [n s {:keys [kind arglists methods dynamic deprecated]}]
  (let [doc (some-> (meta (ns-resolve n s)) :doc
                    ;; defrecord's factories: the class by its simple name, which fits a line
                    (str/replace #"^((?:Positional f|F)actory function for class )[\w.]+\.(\w+)" "$1$2"))
        calls (if (seq arglists)
                (str/join ", " (map #(str "`(" s (when (seq %) " ") (shown-args %) ")`") arglists))
                (str "`" s "`"))
        tags (cond-> []
               (#{:macro :multimethod} kind) (conj (name kind))
               (= :protocol kind) (conj (str "protocol: " (str/join ", " (map #(str "`" % "`") methods))))
               dynamic (conj "dynamic")
               deprecated (conj (if (string? deprecated) (str "deprecated in " deprecated) "deprecated")))]
    (wrap "- " (str calls (when (seq tags) (str " (" (str/join ", " tags) ")"))
                    (when doc (str ": " (escape (sentence doc))))))))

(defn- by-layer
  "Namespaces (gen.api/namespaces' maps) grouped by layer, in the layers'
   order and the namespaces' names' within each: [[title infos] …]."
  [infos]
  (let [groups (group-by (comp layer-of :ns) infos)]
    (for [[title] layers :when (groups title)]
      [title (sort-by (comp str :ns) (groups title))])))

(defn api-page
  "The appendix of the public API: the stable namespaces by layer, each
   with its public vars' calls and their docstrings' first sentences, as
   gen.api's snapshot records them; then the experimental namespaces by
   name and docstring. The internal ones are left out."
  []
  (let [infos (api/namespaces)
        surface (api/surface)
        tier (fn [t] (filter #(= t (:tier %)) infos))
        stable-ns (fn [{n :ns :keys [doc attrs]}]
                    (str "### " n "\n\n"
                         (wrap "" (escape (paragraph doc))) "\n\n"
                         (when-let [note (:implementers attrs)]
                           (str (wrap "" (str "*For implementers:* " (escape note) ".")) "\n\n"))
                         (str/join "\n" (map (fn [[s e]] (var-item n s e)) (sort-by (comp str key) (surface n))))
                         "\n"))
        experimental-ns (fn [{n :ns :keys [doc file]}]
                          (let [home (second (re-matches #"(.*?)/gen/.*" file))]
                            (wrap "- " (str "`" n "`" (when (not= "src" home) (str " (in " home ")"))
                                            ": " (escape (paragraph doc))))))]
    (str "# The Public API\n\n"
         (wrap "" (str "Each of gen.clj's namespaces carries a tier in its ns form: `{:api :stable}`, "
                       "`{:api :experimental}` or `{:api :internal}` (DESIGN.md §9). Through the 1.x releases a "
                       "stable namespace keeps its names, arities, result-map keys and documented semantics; minor "
                       "releases add to them. A removal waits for 2.0, after a deprecation: `:deprecated` metadata "
                       "and a note in the docstring, nothing printed. An experimental namespace ships outside that "
                       "promise, and an internal one is not for code built on gen.clj, which this appendix leaves "
                       "out."))
         "\n\n"
         (wrap "" (str "The stable namespaces come first, by layer: each with the first paragraph of its "
                       "docstring, and every public function, macro and var with the first sentence of its own. A "
                       "protocol names its methods, which are listed as functions too. Where a namespace's "
                       "protocols are for other code to implement, a note says so: through 1.x such a protocol "
                       "keeps its methods and gains none. The experimental namespaces follow, each by name and the "
                       "first paragraph of its docstring. A committed snapshot of the stable surface "
                       "(`test/fixtures/api-snapshot.edn`) fails a test when the surface changes, and `"
                       api/write-command "` writes it after a deliberate change."))
         "\n\n"
         (wrap "" (str "This page is written from the code (`clojure -M:test -m gen.book sync`), and the book's "
                       "test fails when it is out of date."))
         "\n\n"
         (str/join "\n" (for [[title nses] (by-layer (tier :stable))]
                          (str "## " title "\n\n" (str/join "\n" (map stable-ns nses)))))
         "\n## The experimental namespaces\n\n"
         (str/join "\n" (for [[title nses] (by-layer (tier :experimental))]
                          (str "### " title "\n\n" (str/join "\n" (map experimental-ns nses)) "\n"))))))

(def generated
  "The book's pages written from the code: file name to a function of no
   arguments giving its text."
  {"062-distributions.md" distributions-page
   "063-api.md" api-page})

(defn check-generated
  "Every page written from the code (generated, or pages in its shape)
   holds what the code gives now."
  ([] (check-generated generated))
  ([pages]
   (doseq [[file page] pages
           :let [f (io/file dir file)]]
     (testing file
       (is (and (.exists f) (= (page) (slurp f))) (str (.getPath f) " is stale: clojure -M:test -m gen.book sync"))))))

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
                   (printed-files))
        _ (io/make-parents pdf-file)
        exit (try (-> (ProcessBuilder. ^java.util.List args) (.inheritIO) (.start) (.waitFor))
                  (catch java.io.IOException e
                    (throw (ex-info (str "Pandoc did not start (" (ex-message e) "): the PDF needs pandoc, xelatex"
                                         " and rsvg-convert (librsvg)")
                                    {:args args}))))]
    (when-not (zero? exit)
      (throw (ex-info (str "Pandoc failed, exit " exit) {:exit exit :args args})))
    (println "wrote" pdf-file)))

(defn sync!
  "Writes chapters' includes from their regions, and their figures: those
   named (paths like \"book/011-getting-started.md\"), or every chapter and
   the pages written from the code when none is. A named page written from
   the code (book/063-api.md) is written again. For the REPL as for -main."
  [& named]
  (doseq [[file page] generated
          :let [f (io/file dir file)]
          :when (or (empty? named) (some #(= (.getName (io/file %)) file) named))
          :let [text (page)]]
    (when-not (and (.exists f) (= text (slurp f)))
      (spit f text)
      (println "wrote" (.getPath f))))
  (doseq [chapter (or (seq (remove #(contains? generated (.getName (io/file %))) named))
                      (when (empty? named) (chapters)))
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
        (println "wrote" (.getPath f))))))

(defn -main
  "sync [chapter …]: write the chapters' includes from their regions, and
   their figures (every chapter when none is named, and then the pages
   written from the code too); `sync!` does it at the REPL.
   check [chapter …]: run the chapters' checks, print a summary, and exit 1
   on a failure.
   pdf: build the book as a PDF."
  [& [what & named]]
  (case what
    "sync" (apply sync! named)
    "check" (let [counters (binding [clojure.test/*report-counters* (ref clojure.test/*initial-report-counters*)]
                             (doseq [chapter (or (seq named) (chapters))]
                               (let [{:keys [regions checks]} (run-chapter chapter)]
                                 (println chapter ":" regions "regions," checks "checks")))
                             @clojure.test/*report-counters*)]
              (println (:pass counters) "passed," (:fail counters) "failed," (:error counters) "errors")
              (shutdown-agents)
              (System/exit (if (zero? (+ (:fail counters) (:error counters))) 0 1)))
    "pdf" (pdf!)
    (println "usage: clojure -M:test -m gen.book sync|check [book/NNN-name.md …] | pdf"))
  (shutdown-agents))
