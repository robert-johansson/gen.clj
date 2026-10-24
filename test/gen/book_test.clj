(ns gen.book-test
  "The book runs (gen.book): every chapter's blocks in order, their ;=>
   values and figures checked. And the runner can fail: a wrong value, a
   float past the digits it shows, a block that throws, an error that does
   not come, wrong printed output and a stale figure each fail it."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.book :as book]))

(deftest every-chapter-runs
  (let [chapters (book/chapters)]
    (is (seq chapters) "the book has chapters with code")
    (doseq [file chapters]
      (let [{:keys [blocks checks]} (book/run-chapter file)]
        (is (pos? blocks) (str file " has blocks"))
        (is (pos? checks) (str file " checks results"))))))

(defn- failures
  "The number of failures and errors f reports."
  [f]
  (let [n (atom 0)]
    (binding [clojure.test/report (fn [m] (when (#{:fail :error} (:type m)) (swap! n inc)))]
      (f))
    @n))

(defn- chapter [& blocks]
  (apply str "# A test\n\n```clojure\n(ns gen.book-test.chapter)\n```\n"
         (map #(str "\n" % "\n") blocks)))

(defn- runs [& blocks] (failures #(book/run-text (apply chapter blocks))))

(deftest the-runner-can-fail
  (testing "a right value passes, a wrong one fails"
    (is (zero? (runs "```clojure\n(+ 1 2)\n;=> 3\n```")))
    (is (= 1 (runs "```clojure\n(+ 1 2)\n;=> 4\n```"))))
  (testing "a float checks to the digits it shows"
    (is (zero? (runs "```clojure\n(/ 2.0 3.0)\n;=> 0.667\n```")))
    (is (= 1 (runs "```clojure\n(/ 2.0 3.0)\n;=> 0.668\n```")))
    (is (= 1 (runs "```clojure\n(/ 2.0 3.0)\n;=> 0.6668\n```")) "a digit more is a tighter check")
    (is (= 1 (runs "```clojure\n(+ 12 0.4)\n;=> 12\n```")) "an integer is exact"))
  (testing "data compare element by element"
    (is (zero? (runs "```clojure\n{:a [1.0 (/ 1.0 3.0)]}\n;=> {:a [1.0 0.333]}\n```")))
    (is (= 1 (runs "```clojure\n{:a [1.0 (/ 1.0 3.0)]}\n;=> {:a [1.0 0.334]}\n```")))
    (is (= 1 (runs "```clojure\n[1 2]\n;=> [1 2 3]\n```"))))
  (testing "a float64 check runs where floats are float64, and only there"
    (is (zero? (runs "```clojure\n(/ 2.0 3.0)\n;=> 0.667 ; float64\n```")))
    (is (= (if (gen.test-support/f32?) 0 1) (runs "```clojure\n(/ 2.0 3.0)\n;=> 0.668 ; float64\n```"))))
  (testing "definitions carry from block to block"
    (is (zero? (runs "```clojure\n(def x 2)\n```" "```clojure\n(* x 3)\n;=> 6\n```"))))
  (testing "a block that throws fails"
    (is (= 1 (runs "```clojure\n(throw (ex-info \"no\" {}))\n```"))))
  (testing "an error marker needs the error"
    (is (zero? (runs "```clojure\n;; error: Bad input\n(throw (ex-info \"Bad input: 3\" {}))\n```")))
    (is (= 1 (runs "```clojure\n;; error: Bad input\n(throw (ex-info \"Other\" {}))\n```")))
    (is (= 2 (runs "```clojure\n;; error: Bad input\n(+ 1 2)\n```"))))
  (testing "printed output is checked"
    (is (zero? (runs "```clojure\n(println \"hi\")\n```\n\nprints\n\n```text\nhi\n```")))
    (is (= 1 (runs "```clojure\n(println \"hi\")\n```\n\nprints\n\n```text\nho\n```"))))
  (testing "a stale or missing figure fails, on the JVM backend"
    (let [path "figures/011-prior-lines.svg"
          current (slurp (str book/dir "/" path))]
      (is (zero? (failures #(book/check-figures {path current}))))
      (when (gen.test-support/jvm?)
        (is (= 1 (failures #(book/check-figures {path (str current " ")}))))
        (is (= 1 (failures #(book/check-figures {"figures/no-such.svg" "<svg/>"}))))))))
