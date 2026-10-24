(ns gen.book
  "The book (book/) is Markdown whose code runs (DESIGN.md §9). Every
   ```clojure block of a chapter runs, in order, in one namespace (the first
   block's ns form names it), with output discarded. Markers make a block a
   check:

   - `;=> value` after a form: the form's result, read back as Clojure data
     (gen.array arrays by gen.array/->clj), equals value, which is read, not
     evaluated. A float checks to the digits it shows (half a unit in its
     last place, as Clojure prints it, plus the backend's tolerance), so one
     text passes on float64 and float32 backends; an integer exactly.
   - `;=> value ; float64`: the same, checked only where the default float
     is float64 (the JVM backend; MLX with -Dgen.mlx.dtypes=64). For a
     sampled value: a key draws other values in float32.
   - `;; error: message` as a block's first line: one of its forms throws an
     ex-info whose message begins with message. The forms after it do not
     run.
   - a line `prints` right after a block, then a ```text block: what the
     block printed, its lines trimmed.
   - `;; figure: path` as a block's first line: its last form returns a
     figure's SVG text (gen.book.svg), which is the file book/path. On the
     JVM backend the test fails when the file differs (a stale figure);
     `clojure -M:test -m gen.book figures` writes them.

   A statistical claim is a check written in the block (a band from the
   estimator's standard error, `;=> true`), so it holds on every backend.

   (run-chapter file) runs one at the REPL; gen.book-test runs them all."
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

(defn blocks
  "The clojure blocks of a Markdown text: {:line :code :printed}, line being
   the opening fence's, and printed the text block after a line `prints`."
  [text]
  (let [m (re-matcher #"(?ms)^```clojure\n(.*?)^```\n(?:\nprints\n\n```text\n(.*?)^```\n)?" text)]
    (loop [out []]
      (if (.find m)
        (recur (conj out {:line (inc (count (re-seq #"\n" (subs text 0 (.start m)))))
                          :code (.group m 1)
                          :printed (.group m 2)}))
        out))))

(defn- forms
  "A block's forms, each with the values its ;=> markers expect:
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

(defn- lines [s] (mapv str/trimr (str/split-lines (str/trim s))))

(defn- run-block
  "Evaluates a block's forms: {:checks n} and, for a figure, {:figure [path svg]}."
  [{:keys [line code printed]}]
  (testing (str "the block at line " line)
    (let [fs (forms code)
          error (marker code #"^;; error: (.*)")
          figure (marker code #"^;; figure: (\S+)")]
      (cond
        printed
        (let [out (with-out-str (doseq [{:keys [form]} fs] (eval form)))]
          (is (= (lines printed) (lines out)) (str "it printed\n" out))
          {:checks 1})

        error
        (let [e (try (doseq [{:keys [form]} fs] (eval form)) nil (catch Throwable e e))]
          (is (some? e) "no form threw")
          (is (some-> e ex-message (str/starts-with? error)) (str "the message is " (some-> e ex-message)))
          {:checks 1})

        :else
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

(defn run-text
  "Runs the clojure blocks of a Markdown text: {:blocks :checks :figures},
   figures a map from path to SVG text."
  [text]
  (let [bs (blocks text)
        first-form (some-> (first bs) :code forms first :form)
        ns-name (when (and (seq? first-form) (= 'ns (first first-form))) (second first-form))]
    (is (some? ns-name) "the first block is an ns form")
    (when ns-name (remove-ns ns-name))
    (binding [*ns* (the-ns 'user)
              *out* (java.io.StringWriter.)
              *err* (java.io.StringWriter.)]
      (reduce (fn [acc b]
                (let [{:keys [checks figure]} (run-block b)]
                  (cond-> (update acc :checks + checks)
                    figure (assoc-in [:figures (first figure)] (second figure)))))
              {:blocks (count bs) :checks 0 :figures {}}
              bs))))

(defn check-figures
  "Each figure (a map from path to SVG text) is SVG, and on the JVM backend
   the file book/path holds exactly it."
  [figures]
  (doseq [[path svg] figures
          :let [f (io/file dir path)]]
    (testing (str "the figure " path)
      (is (and (string? svg) (str/starts-with? svg "<svg")) "a figure block returns SVG text")
      (when (ts/jvm?)
        (is (and (.exists f) (= svg (slurp f)))
            (str (.getPath f) " is stale: clojure -M:test -m gen.book figures"))))))

(defn run-chapter
  "Runs a chapter and checks its figures: {:blocks :checks :figures}."
  [file]
  (testing file
    (let [r (run-text (slurp file))]
      (check-figures (:figures r))
      r)))

(defn -main
  "figures: run every chapter and write its figures under book/."
  [& [what]]
  (case what
    "figures" (doseq [file (chapters)
                      [path svg] (:figures (run-text (slurp file)))
                      :let [f (io/file dir path)]]
                (io/make-parents f)
                (spit f svg)
                (println "wrote" (.getPath f)))
    (println "usage: clojure -M:test -m gen.book figures"))
  (shutdown-agents))
