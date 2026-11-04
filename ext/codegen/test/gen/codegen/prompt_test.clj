(ns gen.codegen.prompt-test
  "The codegen bake-off's prompts (genclj-ihc4): the worked examples the
   prompt teaches run (simulate, and generate with their own draws as
   data); the system message holds the prelude and no task's answer; a
   failed grading becomes feedback and a passing one none; the REPL
   agent's prompt offers its tools in Qwen's template (gen.llm.chat),
   evidence only when asked, and its task asks for the tools."
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [gen.choicemap :as cm]
            [gen.codegen.candidate :as c]
            [gen.codegen.prompt :as prompt]
            [gen.codegen.session :as session]
            [gen.codegen.tasks :as tasks]
            [gen.gfi :as gfi]
            [gen.llm.chat :as chat]
            [gen.random :as rng]
            [gen.test-support :as ts]))

(deftest the-worked-examples-run
  (doseq [[[_ code] args] (map vector prompt/examples [[[0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0]] [[3 2 4]]])]
    (let [[n model] (c/evaluate (session/read-forms code))
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

(deftest the-repl-agents-prompt
  (let [task (tasks/by-id :line)
        render #(chat/render [{:role :system :content (prompt/repl-system-message %)}
                              {:role :user :content (prompt/repl-task-message task)}]
                             {:tools (prompt/tools-for %)})
        with (render true) without (render false)]
    (is (str/starts-with? with "<|im_start|>system\n# Tools\n"))
    (is (str/includes? with "\"name\": \"evidence\""))
    (is (not (str/includes? without "\"name\": \"evidence\"")) "evidence off: not offered")
    (is (not (str/includes? without "compare variants with evidence")) "nor mentioned")
    (is (str/includes? with "\"required\": [\"name\"]}") "simulate's args optional")
    (is (str/ends-with? with "Build the model with the tools, then submit it.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"))))
