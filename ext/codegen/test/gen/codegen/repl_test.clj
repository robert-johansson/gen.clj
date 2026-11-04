(ns gen.codegen.repl-test
  "REPL sessions for the codegen agent (genclj-t1xh): a model built in
   steps (priors, simulated; the likelihood; checked against the data;
   compared; submitted) grades as the reference does; an error is text the
   LLM reads, a definition before it stays; delimiters are repaired and
   said to be; a model that misses the data's addresses is told which."
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [gen.codegen.grade]
            [gen.codegen.repl :as repl]
            [gen.codegen.tasks]))

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
    (let [slope (trace :slope (d/normal 0 10))
          intercept (trace :intercept (d/normal 0 10))]
      (doseq [[i x] (map-indexed vector xs)]
        (trace [:y i] (d/normal (+ (* slope x) intercept) 0.3))))))")
      (let [c (:text (repl/check! id "line"))]
        (is (str/includes? c "It takes the data"))
        (is (re-find #"\[:slope\] " c))
        (is (re-find #"\[:y \d\] " c))
        (is (str/includes? c "No observation stands out") "the stopping rule's signal, for a model that fits"))
      (is (re-find #"log marginal likelihood -?\d" (:text (repl/evidence! id "line")))))
    (testing "a model at the wrong addresses is told which"
      (repl/eval! id "(def wrong (gen [xs] (doseq [[i x] (map-indexed vector xs)] (trace [:obs i] (d/normal x 1)))))")
      (is (thrown-with-msg? Exception #"not visited" (repl/check! id "wrong"))))
    (testing "submitted: graded as the reference is, with the session's program"
      (let [{:keys [grading program]} (repl/submit! id "line")]
        (is (= 7 (:passed grading)) (pr-str (:stages grading)))
        (is (= (get-in (gen.codegen.grade/grade (gen.codegen.tasks/by-id :line) (:answer (gen.codegen.tasks/by-id :line))) [:log-ml :candidate :median])
               (get-in grading [:log-ml :reference :median]))
            "the reference is the task's answer, not the submitted model")
        (is (str/includes? program "(def model line)"))
        (is (thrown? Exception (repl/eval! id "1")) "the session is closed")))))

(deftest the-verdict-flags-an-observation-that-stands-out
  (let [{id :session} (repl/open! :spread)]
    (repl/eval! id "(def tight (gen [n] (let [mu (trace :mu (d/normal 0 5))] (dotimes [i n] (trace [:y i] (d/normal mu 0.3))))))")
    (repl/eval! id "(def robust (gen [n] (let [mu (trace :mu (d/normal 0 5)) s (trace :s (d/gamma 2 1))] (dotimes [i n] (trace [:y i] (d/cauchy mu s))))))")
    (is (str/includes? (:text (repl/check! id "tight")) "explained far worse than the rest")
        "the 8.5 among values near 3 stands out under a tight normal")
    (is (str/includes? (:text (repl/check! id "robust")) "No observation stands out"))
    (repl/close! id)))
