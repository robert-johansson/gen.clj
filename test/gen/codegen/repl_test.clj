(ns gen.codegen.repl-test
  "REPL sessions for the codegen agent (genclj-t1xh): a model built in
   steps (priors, simulated; the likelihood; checked against the data;
   compared; submitted) grades as the reference does; an error is text the
   LLM reads, a definition before it stays; delimiters are repaired and
   said to be; a model that misses the data's addresses is told which."
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [gen.codegen.repl :as repl]))

(deftest ^:slow a-model-built-in-steps
  (let [{id :session :as o} (repl/open! :line)]
    (is (str/includes? (:text o) "args = [[0.0 0.5"))
    (testing "the priors, simulated"
      (is (str/starts-with? (:text (repl/eval! id "(def prior (gen [] (let [slope (trace :slope (d/normal 0 5))] slope)))")) "=> #'"))
      (is (re-find #"score .*\n\[:slope\] " (:text (repl/simulate! id "prior" "[]")))))
    (testing "an error is text, and what came before it stays"
      (is (re-find #"error: .*Unable to resolve symbol: nope" (:text (repl/eval! id "(def a 1) (nope)"))))
      (is (= "=> 1" (:text (repl/eval! id "a")))))
    (testing "delimiters repaired, and said to be"
      (is (str/starts-with? (:text (repl/eval! id "(def b (+ 1 2)")) "(The delimiters did not balance")))
    (testing "the model, checked against the data and compared"
      (repl/eval! id "(def line
  (gen [xs]
    (let [slope (trace :slope (d/normal 0 5))
          intercept (trace :intercept (d/normal 0 5))]
      (doseq [[i x] (map-indexed vector xs)]
        (trace [:y i] (d/normal (+ (* slope x) intercept) 0.3))))))")
      (let [c (:text (repl/check! id "line"))]
        (is (str/includes? c "It takes the data"))
        (is (re-find #"\[:slope\] " c))
        (is (re-find #"\[:y \d\] " c)))
      (is (re-find #"log marginal likelihood -?\d" (:text (repl/evidence! id "line")))))
    (testing "a model at the wrong addresses is told which"
      (repl/eval! id "(def wrong (gen [xs] (doseq [[i x] (map-indexed vector xs)] (trace [:obs i] (d/normal x 1)))))")
      (is (thrown-with-msg? Exception #"not visited" (repl/check! id "wrong"))))
    (testing "submitted: graded as the reference is, with the session's program"
      (let [{:keys [grading program]} (repl/submit! id "line")]
        (is (= 7 (:passed grading)) (pr-str (:stages grading)))
        (is (str/includes? program "(def model line)"))
        (is (thrown? Exception (repl/eval! id "1")) "the session is closed")))))
