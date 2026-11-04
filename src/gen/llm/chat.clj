(ns gen.llm.chat
  "Qwen's chat template (ChatML), as the Qwen3.5, 3.6 and 3.8 GGUFs carry
   it: messages rendered to the text a model continues, thinking on or
   off, tools offered in the template's format, and the model's tool calls
   parsed back.

     (render messages)                 → text ending in the assistant's turn
     (render messages opts)
     (continuation messages new opts)  → the text that extends (render
                                         messages opts) to (render (into
                                         messages new) opts): tool turns only
     (tool-calls text)                 → [{:name :params}], in order
     (split-reasoning text)            → [reasoning content] of a reply made
                                         with thinking on (nil reasoning when
                                         it has no closing </think>)

   A message is {:role :content}: :system (first only), :user, :assistant
   (with :reasoning, its think block's text, when it has one), or :tool (a
   tool's response; consecutive ones make one user turn of
   <tool_response> blocks). opts:
     :tools               [{:name :description :params [{:name :description
                          :required?}]}], offered in the system turn as the
                          template writes them (JSON, Python's spacing)
     :thinking?           false (default): the assistant's turn opens with an
                          empty think block, as enable_thinking=false writes
                          it; true: it opens with \"<think>\\n\"
     :generation-prompt?  true (default): the assistant's turn begun

   As the template: an assistant turn after the last user message (one
   that is not tool responses) keeps its think block, an earlier one loses
   it, so a new user message re-renders the turns before it; tool turns
   extend the text (continuation). Text holds the special tokens as text:
   encode it with the tokenizer's :parse-special? true."
  {:api :experimental}
  (:require [clojure.string :as str]))

(defn- json
  "x as JSON with Python's json.dumps spacing (the template's tojson):
   strings, keywords (as strings), booleans, numbers, maps (in order) and
   sequences."
  [x]
  (cond (string? x) (str "\"" (-> x (str/replace "\\" "\\\\") (str/replace "\"" "\\\"")
                                  (str/replace "\n" "\\n") (str/replace "\t" "\\t")) "\"")
        (keyword? x) (json (name x))
        (boolean? x) (str x)
        (number? x) (str x)
        (nil? x) "null"
        (map? x) (str "{" (str/join ", " (map (fn [[k v]] (str (json k) ": " (json v))) x)) "}")
        (sequential? x) (str "[" (str/join ", " (map json x)) "]")
        :else (throw (ex-info (str "chat: no JSON for " (pr-str x)) {:value x}))))

(defn- tool-json [{tool-name :name :keys [description params]}]
  (json (array-map :type "function"
                   :function (array-map :name tool-name :description description
                                        :parameters (array-map :type "object"
                                                               :properties (into (array-map)
                                                                                 (map (fn [{p :name d :description}]
                                                                                        [p (array-map :type "string" :description d)]))
                                                                                 params)
                                                               :required (vec (keep #(when (:required? % true) (:name %)) params)))))))

(def ^:private tools-head "# Tools\n\nYou have access to the following functions:\n\n<tools>")

(def ^:private tools-tail
  (str "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
       "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
       "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n"
       "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n"
       "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
       "- Required parameters MUST be specified\n"
       "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
       "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n"
       "</IMPORTANT>"))

(defn- tool-response? [{:keys [role]}] (= :tool role))

(defn- last-query
  "The index of the last user message (tool responses are not one), or -1."
  [messages]
  (or (last (keep-indexed (fn [i {:keys [role]}] (when (= :user role) i)) messages)) -1))

(defn- check-messages [messages]
  (doseq [[i {:keys [role content]}] (map-indexed vector messages)]
    (when-not (#{:system :user :assistant :tool} role)
      (throw (ex-info (str "chat: a message's :role is :system, :user, :assistant or :tool, not " (pr-str role)) {:index i})))
    (when-not (string? content)
      (throw (ex-info "chat: a message's :content is a string" {:index i :content content})))
    (when (and (= :system role) (pos? i))
      (throw (ex-info "chat: a :system message comes first" {:index i})))))

(defn- system-turn [messages tools]
  (let [system (when (= :system (:role (first messages))) (str/trim (:content (first messages))))]
    (cond (seq tools) (str "<|im_start|>system\n" tools-head (str/join (map #(str "\n" (tool-json %)) tools)) tools-tail
                           (when (seq system) (str "\n\n" system)) "<|im_end|>\n")
          system (str "<|im_start|>system\n" system "<|im_end|>\n")
          :else "")))

(defn- turn
  "Message i of messages, rendered (the tool turns' grouping by their
   neighbours)."
  [messages i lq]
  (let [{:keys [role content reasoning]} (messages i)
        prev (when (pos? i) (messages (dec i)))
        nxt (when (< (inc i) (count messages)) (messages (inc i)))]
    (case role
      :system ""
      :user (str "<|im_start|>user\n" (str/trim content) "<|im_end|>\n")
      :assistant (str "<|im_start|>assistant\n"
                      (when (> i lq) (str "<think>\n" (str/trim (or reasoning "")) "\n</think>\n\n"))
                      (str/trim content) "<|im_end|>\n")
      :tool (str (when-not (and prev (tool-response? prev)) "<|im_start|>user")
                 "\n<tool_response>\n" (str/trim content) "\n</tool_response>"
                 (when-not (and nxt (tool-response? nxt)) "<|im_end|>\n")))))

(defn- generation-prompt [thinking?]
  (str "<|im_start|>assistant\n" (if thinking? "<think>\n" "<think>\n\n</think>\n\n")))

(defn render
  "messages in Qwen's chat template. See the namespace's docstring."
  ([messages] (render messages nil))
  ([messages {:keys [tools thinking? generation-prompt?] :or {generation-prompt? true}}]
   (let [messages (vec messages)
         _ (check-messages messages)
         lq (last-query messages)]
     (str (system-turn messages tools)
          (str/join (map #(turn messages % lq) (range (count messages))))
          (when generation-prompt? (generation-prompt thinking?))))))

(defn continuation
  "The text that extends (render messages opts) without its generation
   prompt to (render (into messages new) opts): new may hold tool responses
   only, which change no earlier turn. A new user message re-renders the
   assistant turns before it (they lose their think blocks): render the
   whole conversation then; an LM's cache keeps the prefix they share."
  [messages new opts]
  (when-not (every? tool-response? new)
    (throw (ex-info "chat/continuation: only tool responses extend a rendering; a user message changes the turns before it (render the conversation)"
                    {:roles (mapv :role new)})))
  (let [all (vec (concat messages new))
        before (render messages (assoc opts :generation-prompt? false))
        after (render all opts)]
    (when-not (str/starts-with? after before)
      (throw (ex-info "chat/continuation: the rendering changed before the new messages" {})))
    (subs after (count before))))

(defn tool-calls
  "The tool calls in text, in order: [{:name :params {name value}}], the
   values without the newlines that open and close them."
  [text]
  (vec (for [[_ n body] (re-seq #"(?s)<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>" text)]
         {:name n
          :params (into {} (map (fn [[_ p v]] [p (-> v (str/replace #"^\n" "") (str/replace #"\n$" ""))]))
                        (re-seq #"(?s)<parameter=([^>\s]+)>(.*?)</parameter>" body))})))

(defn split-reasoning
  "[reasoning content] of a reply generated after \"<think>\\n\" (thinking
   on): the text before \"</think>\" and the text after it; [nil text] when
   the reply never closes its think block (cut short)."
  [text]
  (if-let [i (str/index-of text "</think>")]
    [(str/trim (subs text 0 i)) (str/triml (subs text (+ i (count "</think>"))))]
    [nil text]))
