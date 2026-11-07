(ns gen.llm.chat-test
  "Qwen's chat template (gen.llm.chat), the texts written out from the
   template the Qwen3.6 and 3.8 GGUFs carry: system and user turns; the
   generation prompt with thinking off (an empty think block) and on; an
   assistant turn's think block kept after the last user message and lost
   before it; consecutive tool responses as one user turn; the tools
   section; continuation of tool turns (and its refusal of a user turn);
   the tool-call parser and the think-block split. Qwen3.8's (:template
   :qwen3.8) against its template's own renderings by jinja2
   (test/fixtures/qwen38-chat.edn, from test/fixtures/qwen_chat.py): the
   reasoning-effort sentences, preserve_thinking on and off, tools, an
   empty system message."
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [gen.llm.branching-contract :as bc]
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

;; ---------------------------------------------------------------------------
;; Qwen3.8 (genclj-6m7t), against its template's renderings
;; ---------------------------------------------------------------------------

(def ^:private qwen38 (edn/read-string (slurp (io/file "test/fixtures/qwen38-chat.edn"))))

(def ^:private eval-tool {:name "eval" :description "Eval \"it\"." :params [{:name "code" :description "Forms."}]})

(def ^:private conversation
  [{:role :user :content "Q1"}
   {:role :assistant :content "A1" :reasoning "r1"}
   {:role :user :content "Q2"}
   {:role :assistant :content "A2" :reasoning "r2"}
   {:role :tool :content "T1"}
   {:role :tool :content "T2"}])

(deftest qwen38-renders-as-its-template
  (let [su [{:role :system :content "S"} {:role :user :content "U"}]
        u [{:role :user :content "U"}]
        r (fn [messages opts] (chat/render messages (assoc opts :template :qwen3.8)))]
    (doseq [[case text] {:user-thinking (r u {:thinking? true})
                         :user-default-effort (r u {:thinking? true :reasoning-effort :xhigh})
                         :system-low (r su {:thinking? true :reasoning-effort :low})
                         :system-medium (r su {:thinking? true :reasoning-effort :medium})
                         :system-xhigh (r su {:thinking? true})
                         :empty-system-thinking (r [{:role :system :content "  "} {:role :user :content "U"}] {:thinking? true})
                         :user-off (r u {})
                         :system-off (r [{:role :system :content "S"} {:role :user :content "U\n"}] {})
                         :tools-thinking (r su {:thinking? true :tools [eval-tool]})
                         :tools-off (r su {:tools [eval-tool]})
                         :conversation-thinking (r conversation {:thinking? true})
                         :conversation-off (r conversation {})
                         :conversation-not-preserved (r conversation {:thinking? true :preserve-thinking? false})
                         :no-generation-prompt (r u {:thinking? true :generation-prompt? false})}]
      (is (= (qwen38 case) text) (name case)))))

(deftest qwen38-and-qwen35-differ-only-where-the-templates-do
  ;; thinking off, one user turn: the same text; a conversation keeps its
  ;; earlier think blocks under Qwen3.8 only
  (is (= (chat/render [{:role :user :content "U"}]) (chat/render [{:role :user :content "U"}] {:template :qwen3.8})))
  (is (not= (chat/render conversation) (chat/render conversation {:template :qwen3.8})))
  (is (= (chat/render conversation {:thinking? true})
         (chat/render conversation {:template :qwen3.8 :preserve-thinking? false :reasoning-effort :medium :thinking? true}))))

(deftest qwen38-options-are-checked
  (doseq [[what opts] [["an unknown template" {:template :qwen4}]
                       ["an unknown effort" {:template :qwen3.8 :reasoning-effort :max}]
                       ["Qwen3.8's options on Qwen3.5" {:reasoning-effort :low}]
                       ["preserve-thinking on Qwen3.5" {:preserve-thinking? true}]]]
    (is (= :invalid-option (bc/reason #(chat/render [{:role :user :content "U"}] opts))) what)))

;; ---------------------------------------------------------------------------
;; Segments: the rendering by message and part (gen.llm.sft's masks)
;; ---------------------------------------------------------------------------

(deftest segments-are-the-rendering-in-parts
  (testing "their texts are render's, for every case above"
    (doseq [[messages opts] [[[{:role :system :content "S"} {:role :user :content "U\n"}] nil]
                             [[{:role :user :content "U"}] {:thinking? true}]
                             [conversation nil]
                             [conversation {:thinking? true :generation-prompt? false}]
                             [[{:role :system :content "S"} {:role :user :content "U"}] {:tools [eval-tool]}]
                             [[{:role :user :content "U"}] {:tools [eval-tool] :template :qwen3.8 :thinking? true}]
                             [[{:role :system :content "  "} {:role :user :content "U"}] {:template :qwen3.8 :thinking? true}]
                             [conversation {:template :qwen3.8}]]]
      (is (= (chat/render messages opts) (apply str (map :text (chat/segments messages opts)))) (pr-str opts))))
  (testing "each segment's message and part"
    (is (= [["<|im_start|>system\n" :template 0 :system] ["S" :content 0 :system] ["<|im_end|>" :end 0 :system]
            ["\n" :template 0 :system]
            ["<|im_start|>user\n" :template 1 :user] ["U" :content 1 :user] ["<|im_end|>" :end 1 :user] ["\n" :template 1 :user]
            ["<|im_start|>assistant\n" :template 2 :assistant] ["A1" :content 2 :assistant] ["<|im_end|>" :end 2 :assistant]
            ["\n" :template 2 :assistant]
            ["<|im_start|>user\n" :template 3 :user] ["V" :content 3 :user] ["<|im_end|>" :end 3 :user] ["\n" :template 3 :user]
            ["<|im_start|>assistant\n<think>\n" :template 4 :assistant] ["r" :reasoning 4 :assistant]
            ["\n</think>\n\n" :template 4 :assistant] ["A2" :content 4 :assistant] ["<|im_end|>" :end 4 :assistant]
            ["\n" :template 4 :assistant]]
           (mapv (juxt :text :part :message :role)
                 (chat/segments [{:role :system :content "S"} {:role :user :content "U"} {:role :assistant :content " A1 " :reasoning "q"}
                                 {:role :user :content "V"} {:role :assistant :content "A2" :reasoning "r"}]
                                {:generation-prompt? false}))))
    (is (= [nil nil] ((juxt :message :role) (peek (chat/segments [{:role :user :content "U"}]))))
        "the generation prompt is no message's")))
