(ns gen.book-test
  "The book runs (gen.book): every chapter's includes hold their regions;
   Part I's chapters and the appendices run, their ;=> values and figures
   checked (Parts II-V run in gen.book.part2-test … part5-test, ^:slow).
   And the runner can fail: a wrong value, a float past the digits it
   shows, a throw, an error that does not come, a stale or missing include,
   a code block outside an include, a stale figure and a stale page
   written from the code each fail it. The API appendix follows the tiers
   (gen.api): the stable namespaces in full, the experimental ones by
   name, no internal one."
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [gen.api :as api]
            [gen.book :as book]
            [gen.book.svg :as svg]
            [gen.test-support :as ts]))

(deftest every-chapter-is-in-step
  (doseq [chapter (book/chapters)]
    (book/check-in-step chapter)))

(deftest foundations-and-appendices-run
  (book/check-chapters (book/part-chapters "01" "06")))

(deftest generated-pages-are-current
  (book/check-generated))

(deftest the-api-page-follows-the-tiers
  (let [page (book/api-page)
        tier (fn [t] (map :ns (filter #(= t (:tier %)) (api/namespaces))))]
    (doseq [n (tier :stable)]
      (is (str/includes? page (str "\n### " n "\n")) (str "the stable " n " has a section")))
    (is (str/includes? page "- `(simulate gf args)`, `(simulate gf args key)`: ") "a stable var with its calls")
    (doseq [n (tier :experimental)]
      (is (re-find (re-pattern (str "\n- `" (java.util.regex.Pattern/quote (str n)) "`")) page)
          (str "the experimental " n " is named")))
    (doseq [n (tier :internal)]
      (is (not (str/includes? page (str "\n- `" n "`"))) (str "the internal " n " is left out"))
      (is (not (str/includes? page (str "### " n "\n"))) (str "the internal " n " is left out")))
    (is (every? #(<= (count %) book/max-line) (str/split-lines page)) "its lines fit a printed page")))

(defn- failures
  "The number of failures and errors f reports."
  [f]
  (let [n (atom 0)]
    (binding [clojure.test/report (fn [m] (when (#{:fail :error} (:type m)) (swap! n inc)))]
      (f))
    @n))

(defn- source [& regions]
  (apply str ";; book: ns\n(ns gen.book-test.chapter)\n"
         (map-indexed (fn [i code] (str "\n;; book: r" i "\n" code "\n")) regions)))

(defn- runs [& regions] (failures #(book/run-regions (book/regions (apply source regions)))))

(deftest the-runner-can-fail
  (testing "a right value passes, a wrong one fails"
    (is (zero? (runs "(+ 1 2)\n;=> 3")))
    (is (= 1 (runs "(+ 1 2)\n;=> 4"))))
  (testing "a float checks to the digits it shows"
    (is (zero? (runs "(/ 2.0 3.0)\n;=> 0.667")))
    (is (= 1 (runs "(/ 2.0 3.0)\n;=> 0.668")))
    (is (= 1 (runs "(/ 2.0 3.0)\n;=> 0.6668")) "a digit more is a tighter check")
    (is (= 1 (runs "(+ 12 0.4)\n;=> 12")) "an integer is exact")
    (is (zero? (runs "(* 7.4 1e-4)\n;=> 0.0007")) "0.0007 prints as 7.0E-4 but shows four decimals")
    (is (= 1 (runs "(* 7.8 1e-4)\n;=> 0.0007"))))
  (testing "NaN and infinities check exactly, as one check each"
    (is (zero? (runs "(/ 0.0 0.0)\n;=> ##NaN")))
    (is (= 1 (runs "1.0\n;=> ##NaN")))
    (is (= 1 (runs "(/ 0.0 0.0)\n;=> 1.0")))
    (is (zero? (runs "(/ 1.0 0.0)\n;=> ##Inf")))
    (is (= 1 (runs "1e300\n;=> ##Inf"))))
  (testing "a float64 check runs where floats are float64, and only there"
    (is (zero? (runs "(/ 2.0 3.0)\n;=> 0.667 ; float64")))
    (is (= (if (ts/f32?) 0 1) (runs "(/ 2.0 3.0)\n;=> 0.668 ; float64"))))
  (testing "data compare element by element"
    (is (zero? (runs "{:a [1.0 (/ 1.0 3.0)]}\n;=> {:a [1.0 0.333]}")))
    (is (= 1 (runs "{:a [1.0 (/ 1.0 3.0)]}\n;=> {:a [1.0 0.334]}")))
    (is (= 1 (runs "[1 2]\n;=> [1 2 3]"))))
  (testing "definitions carry from region to region"
    (is (zero? (runs "(def x 2)" "(* x 3)\n;=> 6"))))
  (testing "a region that throws fails"
    (is (= 1 (runs "(throw (ex-info \"no\" {}))"))))
  (testing "an error marker needs the error"
    (is (zero? (runs ";; error: Bad input\n(throw (ex-info \"Bad input: 3\" {}))")))
    (is (= 1 (runs ";; error: Bad input\n(throw (ex-info \"Other\" {}))")))
    (is (= 2 (runs ";; error: Bad input\n(+ 1 2)"))))
  (testing "a source without an ns region fails"
    (is (= 1 (failures #(book/run-regions (book/regions ";; book: a\n(+ 1 2)\n"))))))
  (testing "a line over 80 characters fails"
    (is (zero? (failures #(book/check-lines (book/regions (source (apply str (repeat 80 "x"))))))))
    (is (= 1 (failures #(book/check-lines (book/regions (source (apply str (repeat 81 "x")))))))))
  (testing "an error region's forms run, and show, without their #_"
    (let [rs (book/regions (source ";; error: Bad input\n#_(throw (ex-info \"Bad input: 3\" {}))"))]
      (is (= ";; error: Bad input\n(throw (ex-info \"Bad input: 3\" {}))\n" (:code (peek rs))))
      (is (zero? (failures #(book/run-regions rs))))
      (is (zero? (failures #(book/check-error-regions rs))))))
  (testing "an error region with a form not behind #_ fails: requiring the namespace would throw"
    (is (= 1 (failures #(book/check-error-regions
                         (book/regions (source ";; error: Bad input\n(throw (ex-info \"Bad input: 3\" {}))")))))))
  (testing "a region named twice is an error"
    (is (thrown? clojure.lang.ExceptionInfo (book/regions ";; book: a\n1\n;; book: a\n2\n")))))

(deftest includes-hold-their-regions
  (let [rs (book/regions ";; book: ns\n(ns x)\n\n;; book: add\n(+ 1 2)\n;=> 3\n")
        fresh "Text.\n\n<!-- code: add -->\n```clojure\n(+ 1 2)\n;=> 3\n```\n\nMore.\n"]
    (testing "an include in step passes"
      (is (zero? (failures #(book/check-includes fresh rs)))))
    (testing "a stale include fails, and sync brings it back in step"
      (let [stale (clojure.string/replace fresh "(+ 1 2)" "(+ 1 5)")]
        (is (= 1 (failures #(book/check-includes stale rs))))
        (is (= fresh (book/synced stale rs)))))
    (testing "a marker without its block is written by sync"
      (let [bare "Text.\n\n<!-- code: add -->\n\nMore.\n"]
        (is (pos? (failures #(book/check-includes bare rs))) "nothing to compare is not in step")
        (is (= fresh (book/synced bare rs)))))
    (testing "an include of no region fails"
      (is (= 1 (failures #(book/check-includes "<!-- code: nope -->\n```clojure\n1\n```\n" rs))))
      (is (thrown? clojure.lang.ExceptionInfo (book/synced "<!-- code: nope -->\n" rs))))
    (testing "a clojure block outside an include fails"
      (is (= 1 (failures #(book/check-includes (str fresh "\n```clojure\n(+ 4 5)\n```\n") rs)))))))

(deftest figure-text-is-escaped
  (let [svg (svg/tree [["</s>" "a & \"b\""]])]
    (is (not (re-find #"<text[^>]*></s>" svg)) "a label's < is not markup")
    (is (re-find #"&lt;/s&gt;" svg))
    (is (re-find #"a &amp; " svg))))

(deftest stale-figures-fail
  (let [path "figures/011-prior-lines.svg"
        current (slurp (str book/dir "/" path))]
    (is (zero? (failures #(book/check-figures {path current}))))
    (when (ts/jvm?)
      (is (= 1 (failures #(book/check-figures {path (str current " ")}))))
      (is (= 1 (failures #(book/check-figures {"figures/no-such.svg" "<svg/>"})))))))

(deftest stale-pages-fail
  (let [[file page] (first book/generated)]
    (is (zero? (failures #(book/check-generated {file page}))))
    (is (= 1 (failures #(book/check-generated {file (fn [] (str (page) " "))}))))))
