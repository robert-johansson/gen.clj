(ns gen.codegen.prompt-test
  "The codegen bake-off's prompts and sampler (genclj-ihc4): the worked
   examples the prompt teaches run (simulate, and generate with their own
   draws as data); the system message holds the prelude and no task's
   answer; a failed grading becomes feedback and a passing one none; the
   chat is Qwen's template with thinking off; and the sampler keeps to the
   top-k tokens and the top-p mass."
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [gen.choicemap :as cm]
            [gen.codegen.candidate :as c]
            [gen.codegen.driver :as driver]
            [gen.codegen.prompt :as prompt]
            [gen.codegen.tasks :as tasks]
            [gen.gfi :as gfi]
            [gen.random :as rng]
            [gen.test-support :as ts]))

(deftest the-worked-examples-run
  (doseq [[[_ code] args] (map vector prompt/examples [[[0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0]] [[3 2 4]]])]
    (let [[n model] (c/evaluate (c/read-forms code))
          tr (gfi/simulate model args (rng/key 1))]
      (is (Double/isFinite (double (ts/host (:score tr)))))
      (is (Double/isFinite (double (ts/host (:weight (gfi/generate model args (:choices tr) (rng/key 2)))))))
      (c/discard! n))))

(deftest the-system-message
  (let [s (prompt/system-message)]
    (is (str/includes? s "(:refer-clojure :exclude [+ - * / < > <= >= ==])"))
    (is (str/includes? s "[gen.combinators :as comb]"))
    (doseq [t tasks/tasks]
      (is (not (str/includes? s (:answer t))) (:id t)))))

(deftest feedback
  (is (nil? (prompt/feedback-message {:stages [{:stage :read :ok true}]})))
  (is (re-find #"^Running the model with the data as observations failed: generate: the constraint at \[:y 3\]"
               (prompt/feedback-message {:stages [{:stage :read :ok true}
                                                  {:stage :accepts-data :ok false
                                                   :error "generate: the constraint at [:y 3] was not visited"}]}))))

(deftest feedback-after-a-delimiter-repair-says-so
  (is (re-find #"^Your code's parentheses and brackets did not balance.*\n\nEvaluating the code failed: x"
               (prompt/feedback-message {:stages [{:stage :read :ok true :repaired-delimiters true}
                                                  {:stage :evaluates :ok false :error "x"}]}))))

(deftest the-chat-template
  (is (= "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
         (prompt/chat [[:system "S"] [:user "U\n"]]))))

(deftest the-sampler-keeps-to-top-k-and-top-p
  (let [lp (float-array (map #(Math/log %) [0.5 0.3 0.1 0.05 0.05]))]
    (testing "top-p 0.95 keeps the four most likely of five (0.95 of the mass)"
      (is (= #{0 1 2 3} (set (map #(driver/pick lp {:temp 1.0 :top-k 20 :top-p 0.95} %) (map #(/ % 100.0) (range 100)))))))
    (testing "top-k 2 keeps two"
      (is (= #{0 1} (set (map #(driver/pick lp {:temp 1.0 :top-k 2 :top-p 1.0} %) (map #(/ % 100.0) (range 100)))))))
    (testing "the draws follow the kept mass: token 0 for u below 0.5 / 0.95"
      (is (= 0 (driver/pick lp {:temp 1.0 :top-k 20 :top-p 0.95} 0.52)))
      (is (= 1 (driver/pick lp {:temp 1.0 :top-k 20 :top-p 0.95} 0.54))))))
