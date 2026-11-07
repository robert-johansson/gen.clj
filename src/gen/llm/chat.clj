(ns gen.llm.chat
  "Qwen's chat template (ChatML), as the Qwen3.5, 3.6 and 3.8 GGUFs carry
   it: messages rendered to the text a model continues, thinking on or
   off, tools offered in the template's format, and the model's tool calls
   parsed back. Qwen3.8's differs (:template :qwen3.8): a reasoning-effort
   sentence opens its system turn when thinking is on, and every assistant
   turn keeps its think block.

     (render messages)                 → text ending in the assistant's turn
     (render messages opts)
     (segments messages opts)          → the same text in segments, each its
                                         message and part (:content, :end, …)
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
     :template            :qwen3.5 (default; Qwen3.5's and 3.6's) or :qwen3.8
     :reasoning-effort    Qwen3.8's, with thinking on: :xhigh (default),
                          :medium or :low; its sentence (none for :medium)
                          opens the system turn, before the tools
     :preserve-thinking?  Qwen3.8's preserve_thinking: true (its default)
                          keeps every assistant turn's think block (empty
                          when it has no :reasoning), false keeps only those
                          after the last user message, as Qwen3.5's

   As the template: an assistant turn after the last user message (one
   that is not tool responses) keeps its think block, an earlier one loses
   it (but under Qwen3.8's preserve-thinking), so a new user message
   re-renders the turns before it; tool turns extend the text
   (continuation). Text holds the special tokens as text:
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

(def ^:private reasoning-sentences
  "Qwen3.8's reasoning-effort sentences, by effort (none for :medium)."
  {:xhigh (str "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions,"
               " consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.")
   :medium nil
   :low (str "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion"
             " without unnecessary elaboration.")})

(defn- template-options
  "opts with the template's defaults, checked: the reasoning sentence (or
   nil) as :reasoning, and :preserve-thinking?."
  [{:keys [template thinking?] :or {template :qwen3.5} :as opts}]
  (case template
    :qwen3.5 (do (when (some #(contains? opts %) [:reasoning-effort :preserve-thinking?])
                   (throw (ex-info "chat: :reasoning-effort and :preserve-thinking? are Qwen3.8's (:template :qwen3.8)"
                                   {:reason :invalid-option :template template})))
                 (assoc opts :reasoning nil :preserve-thinking? false :empty-system-turn? true))
    :qwen3.8 (let [effort (get opts :reasoning-effort :xhigh)]
               (when-not (contains? reasoning-sentences effort)
                 (throw (ex-info (str "chat: :reasoning-effort is :xhigh, :medium or :low, not " (pr-str effort))
                                 {:reason :invalid-option :reasoning-effort effort})))
               (assoc opts
                      :reasoning (when thinking? (reasoning-sentences effort))
                      :preserve-thinking? (get opts :preserve-thinking? true)
                      :empty-system-turn? false))
    (throw (ex-info (str "chat: :template is :qwen3.5 or :qwen3.8, not " (pr-str template))
                    {:reason :invalid-option :template template}))))

(defn- seg
  "A segment of a rendering: text, the part of the template it is, and the
   message it renders, its index i and role (segments's)."
  [text part i role]
  {:text text :part part :message i :role role})

(defn- system-turn
  "The system turn's segments: the reasoning sentence, the tools, the
   system message. An empty system message makes an empty turn where
   empty-turn? (Qwen3.5's rendering here), none otherwise (Qwen3.8's
   template)."
  [messages tools reasoning empty-turn?]
  (let [i (when (= :system (:role (first messages))) 0)
        system (when i (str/trim (:content (first messages))))
        system (when (or empty-turn? (seq system)) system)
        opening (when reasoning (str reasoning "\n\n"))
        turn (fn [head content]
               [(seg head :template i :system) (seg content :content i :system)
                (seg "<|im_end|>" :end i :system) (seg "\n" :template i :system)])]
    (cond (seq tools) (turn (str "<|im_start|>system\n" opening tools-head (str/join (map #(str "\n" (tool-json %)) tools)) tools-tail
                                 (when (seq system) "\n\n"))
                            (or system ""))
          (and reasoning (empty? system)) (turn (str "<|im_start|>system\n" reasoning) "")
          system (turn (str "<|im_start|>system\n" opening) system)
          :else [])))

(defn- turn
  "The segments of message i of messages (the tool turns' grouping by
   their neighbours); an assistant turn keeps its think block after the
   last user message lq, or always when preserve?."
  [messages i lq preserve?]
  (let [{:keys [role content reasoning]} (messages i)
        prev (when (pos? i) (messages (dec i)))
        nxt (when (< (inc i) (count messages)) (messages (inc i)))
        s (fn [text part] (seg text part i role))
        end [(s "<|im_end|>" :end) (s "\n" :template)]]
    (case role
      :system []
      :user (into [(s "<|im_start|>user\n" :template) (s (str/trim content) :content)] end)
      :assistant (into (if (or preserve? (> i lq))
                         [(s "<|im_start|>assistant\n<think>\n" :template) (s (str/trim (or reasoning "")) :reasoning)
                          (s "\n</think>\n\n" :template)]
                         [(s "<|im_start|>assistant\n" :template)])
                       (cons (s (str/trim content) :content) end))
      :tool (cond-> [(s (str (when-not (and prev (tool-response? prev)) "<|im_start|>user") "\n<tool_response>\n") :template)
                     (s (str/trim content) :content)
                     (s "\n</tool_response>" :template)]
              (not (and nxt (tool-response? nxt))) (into end)))))

(defn- generation-prompt [thinking?]
  (seg (str "<|im_start|>assistant\n" (if thinking? "<think>\n" "<think>\n\n</think>\n\n")) :template nil nil))

(defn segments
  "The rendering of messages (render's) in segments, in order: maps
   {:text :part :message :role}, their texts render's text concatenated.
   :message is the index of the message a segment renders and :role its
   role (:message nil for text of no message: the generation prompt, whose
   :role is nil too, and a system turn without a system message), :part
   what the text is:
     :content    a message's content (a system message's, trimmed, in the
                 system turn)
     :reasoning  an assistant turn's think block's text (when it keeps one)
     :end        the end of a turn, <|im_end|> (a token of its own)
     :template   the template's own text: the turn's opening, the newline
                 after its end, the think block's tags, the tools section,
                 the reasoning-effort sentence
   Empty segments are left out. A model is trained on its replies by these
   (gen.llm.sft): an assistant's :content and :end. opts are render's."
  ([messages] (segments messages nil))
  ([messages opts]
   (let [{:keys [tools thinking? generation-prompt? reasoning preserve-thinking? empty-system-turn?] :or {generation-prompt? true}}
         (template-options opts)
         messages (vec messages)
         _ (check-messages messages)
         lq (last-query messages)]
     (into [] (remove #(empty? (:text %)))
           (concat (system-turn messages tools reasoning empty-system-turn?)
                   (mapcat #(turn messages % lq preserve-thinking?) (range (count messages)))
                   (when generation-prompt? [(generation-prompt thinking?)]))))))

(defn render
  "messages in Qwen's chat template. See the namespace's docstring."
  ([messages] (render messages nil))
  ([messages opts] (apply str (map :text (segments messages opts)))))

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
