(ns gen.llm.chat-test
  "Qwen's chat template (gen.llm.chat), the texts written out from the
   template the Qwen3.6 and 3.8 GGUFs carry: system and user turns; the
   generation prompt with thinking off (an empty think block) and on; an
   assistant turn's think block kept after the last user message and lost
   before it; consecutive tool responses as one user turn; the tools
   section; continuation of tool turns (and its refusal of a user turn);
   the tool-call parser and the think-block split."
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [gen.llm.chat :as chat]))

(deftest turns-and-the-generation-prompt
  (testing "system and user, thinking off"
    (is (= "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
           (chat/render [{:role :system :content "S"} {:role :user :content "U\n"}]))))
  (testing "thinking on"
    (is (= "<|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\n<think>\n"
           (chat/render [{:role :user :content "U"}] {:thinking? true}))))
  (testing "no generation prompt"
    (is (= "<|im_start|>user\nU<|im_end|>\n" (chat/render [{:role :user :content "U"}] {:generation-prompt? false})))))

(deftest think-blocks-before-and-after-the-last-user-message
  (let [msgs [{:role :user :content "Q1"}
              {:role :assistant :content "A1" :reasoning "r1"}
              {:role :user :content "Q2"}
              {:role :assistant :content "A2" :reasoning "r2"}
              {:role :tool :content "T1"}
              {:role :tool :content "T2"}]]
    (is (= (str "<|im_start|>user\nQ1<|im_end|>\n"
                "<|im_start|>assistant\nA1<|im_end|>\n"                              ; before the last user message: no think block
                "<|im_start|>user\nQ2<|im_end|>\n"
                "<|im_start|>assistant\n<think>\nr2\n</think>\n\nA2<|im_end|>\n"     ; after it: kept
                "<|im_start|>user\n<tool_response>\nT1\n</tool_response>\n<tool_response>\nT2\n</tool_response><|im_end|>\n"
                "<|im_start|>assistant\n<think>\n\n</think>\n\n")
           (chat/render msgs)))))

(deftest the-tools-section
  (let [t (chat/render [{:role :system :content "S"} {:role :user :content "U"}]
                       {:tools [{:name "eval" :description "Eval \"it\"." :params [{:name "code" :description "Forms."}]}]})]
    (is (str/starts-with? t "<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n"))
    (is (str/includes? t "{\"type\": \"function\", \"function\": {\"name\": \"eval\", \"description\": \"Eval \\\"it\\\".\", \"parameters\": {\"type\": \"object\", \"properties\": {\"code\": {\"type\": \"string\", \"description\": \"Forms.\"}}, \"required\": [\"code\"]}}}"))
    (is (str/includes? t "</IMPORTANT>\n\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n"))))

(deftest continuation-of-tool-turns
  (let [opts {:thinking? false}
        msgs [{:role :user :content "Q"} {:role :assistant :content "A"}]
        new [{:role :tool :content "T"}]]
    (is (= (chat/render (into msgs new) opts)
           (str (chat/render msgs (assoc opts :generation-prompt? false)) (chat/continuation msgs new opts))))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"only tool responses"
                          (chat/continuation msgs [{:role :user :content "more"}] opts)))))

(deftest parsing-replies
  (is (= [{:name "eval" :params {"code" "(def a 1)\n(def b 2)"}} {:name "submit" :params {"name" "m"}}]
         (chat/tool-calls (str "I'll define it.\n<tool_call>\n<function=eval>\n<parameter=code>\n(def a 1)\n(def b 2)\n"
                               "</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=submit>\n<parameter=name>\nm\n"
                               "</parameter>\n</function>\n</tool_call>"))))
  (is (= [] (chat/tool-calls "no calls here")))
  (is (= ["thinking it over" "the answer"] (chat/split-reasoning "thinking it over\n</think>\n\nthe answer")))
  (is (= [nil "cut short"] (chat/split-reasoning "cut short"))))

(deftest loud-errors
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #":role" (chat/render [{:role :bot :content "x"}])))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"comes first"
                        (chat/render [{:role :user :content "U"} {:role :system :content "S"}]))))
