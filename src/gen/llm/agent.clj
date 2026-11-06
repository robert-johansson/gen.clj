(ns gen.llm.agent
  "A tool-using agent's episodes over a language model (gen.llm.lm): the
   loop of a REPL-driven writer of code, every part but the loop supplied
   by the caller. The loop evaluates nothing itself: what a tool does,
   what a session is and what submitting means are the caller's functions
   (gen.clj's code-evaluating tools live outside the core, in ext/codegen).

     (run lm spec)
     (run lm spec opts)

   spec:
     :system     the system message (the tools are offered before it)
     :task       the first user message
     :tools      {name {:description :params [{:name :description :required?}]
                        :fn (fn [params state] → {:text …})}}, offered in the
                 map's order (an array map keeps the order written); a
                 tool's params are strings by name; its result
                 is a map whose :text the model reads next; its other keys
                 are kept in the log (an outcome's category, say); one
                 holding :done ends the episode with that value as its
                 result. A tool that throws answers \"error: …\".
     :open       (fn [] state): the session, opaque to the loop, given to
                 every tool (nil when omitted)
     :close      (fn [state]): called when the episode ends, however it ends
     :on-budget  (fn [state] → result or nil): the episode's result when its
                 turns or its context run out (a last model, say)
     :notes      (fn [turn turns-left] → text or nil): a note added to a turn's
                 tool responses (a stopping rule's countdown)

   opts:
     :turns        the most turns (40)
     :turn-tokens  the most tokens a reply draws (1024; 8192 with thinking:
                   Qwen3.6's think blocks run to several thousand tokens)
     :thinking?    the model thinks before it answers (false); a reply cut
                   inside its think block by :turn-tokens is a loud error
     :template     gen.llm.chat's: :qwen3.5 (default; Qwen3.5's and 3.6's) or
                   :qwen3.8
     :reasoning-effort, :preserve-thinking?
                   Qwen3.8's (gen.llm.chat's: :xhigh and true by default):
                   options the template refuses are refused as chat
                   refuses them (:invalid-option), before the session opens
     :context      the context's tokens (the LM's :max-cells, else 32768): a
                   reply gets at most the room left; a reply cut by that
                   room (after its complete tool calls run), or a context
                   too full for a reply, ends the episode as :context-full.
                   With thinking, Qwen's template keeps every turn's think
                   block after the task, so a context of thousands of
                   thinking tokens a turn fills in a few turns
     :sampling     gen.llm.sample's (its default-sampling)
     :key          turn t's draws from (fold-in key t)
     :candidates   replies drawn a turn (1); with more, :choose
                   (fn [candidates state] → index) picks the one the episode
                   continues with, candidates [{:text :tokens :logprob}]
     :complete     the completion function, sample/complete's signature
                   (it; a test's scripted replies, say)

   Returns {:result :stop (:done, :turns or :context-full) :turns :log
   :tokens :seconds {:llm :tools}}, :log a turn's {:turn :reply :reasoning
   (with thinking) :calls [{:tool :params :text …the result's other keys}]}.

   Each turn renders the conversation (gen.llm.chat, Qwen's template, the
   one :template names) and encodes it: an LM's cache shares the prefix
   with the turn before, so a turn reads only what is new."
  {:api :experimental}
  (:require [gen.gfi :as gfi]
            [gen.llm.chat :as chat]
            [gen.llm.lm :as lm]
            [gen.llm.sample :as sample]
            [gen.random :as rng]))

(def no-call-text "No tool call in your reply: call one of the tools.")

(defn- tool-list [tools]
  (mapv (fn [[n {:keys [description params]}]] {:name n :description description :params params}) tools))

(defn- call-tool
  "The tool's result for a call, \"error: …\" when there is no such tool or
   it throws."
  [tools state {tool-name :name :keys [params]}]
  (if-let [{f :fn} (get tools tool-name)]
    (try (let [r (f params state)]
           (if (and (map? r) (string? (:text r)))
             r
             (throw (ex-info (str "a tool returns a map with :text, not " (pr-str r)) {}))))
         (catch Throwable e
           {:text (str "error: " (or (ex-message e) (str (class e))))}))
    {:text (str "error: there is no tool " tool-name "; the tools are " (vec (sort (keys tools))))}))

(defn run
  "An agent's episode. See the namespace's docstring."
  ([lm spec] (run lm spec nil))
  ([lm {:keys [system task tools open close on-budget notes]}
    {:keys [turns turn-tokens thinking? context sampling key candidates choose complete]
     :or {turns 40 sampling sample/default-sampling candidates 1 complete sample/complete}
     :as opts}]
   (when-not (and (string? task) (map? tools) (seq tools))
     (throw (ex-info "agent/run: a spec needs :task (a string) and :tools (a map of them)" {:task task :tools (keys tools)})))
   (when (and (> (long candidates) 1) (not choose))
     (throw (ex-info "agent/run: :candidates above 1 needs :choose" {:candidates candidates})))
   (let [turn-tokens (long (or turn-tokens (if thinking? 8192 1024)))
         context (long (or context (:max-cells (lm/info lm)) 32768))
         key (or key (gfi/next-key))
         render-opts (into {:tools (tool-list tools) :thinking? (boolean thinking?)}
                           (select-keys opts [:template :reasoning-effort :preserve-thinking?]))
         ;; the template's options checked by chat, before the session opens
         _ (chat/render [] render-opts)
         state (when open (open))
         seconds (fn [t0] (/ (- (System/nanoTime) t0) 1e9))]
     (try
       (loop [turn 1
              messages (cond-> [] system (conj {:role :system :content system}) true (conj {:role :user :content task}))
              log [] tokens 0 llm-s 0.0 tool-s 0.0]
         (let [finish (fn [stop result] {:result result :stop stop :turns (dec turn) :log log :tokens tokens
                                         :seconds {:llm llm-s :tools tool-s}})
               ctx (when (<= turn turns)
                     (lm/encode lm (chat/render messages render-opts) {:add-special? false :parse-special? true}))
               room (when ctx (- context (count ctx) 16))]
           (cond
             (> turn turns) (finish :turns (when on-budget (on-budget state)))
             (< (long room) 64) (finish :context-full (when on-budget (on-budget state)))
             :else
             (let [t0 (System/nanoTime)
                   drawn (complete lm (vec (repeat candidates ctx))
                                   {:max-tokens (min turn-tokens room) :sampling sampling :logprobs? true
                                    :key (rng/fold-in key turn)})
                   cands (mapv (fn [{:keys [tokens logprob] :as d}] (assoc d :text (lm/decode lm tokens) :logprob logprob)) drawn)
                   chosen (cands (if (= 1 candidates) 0 (choose cands state)))
                   llm-s (+ llm-s (seconds t0))
                   ;; cut by the room left in the context, not by :turn-tokens
                   full? (and (= :max (:stop chosen)) (< (long room) turn-tokens))
                   [reasoning content] (if thinking? (chat/split-reasoning (:text chosen)) [nil (:text chosen)])
                   in-thinking? (and thinking? (nil? reasoning))
                   _ (when (and in-thinking? (not full?))
                       (throw (ex-info (str "agent/run: turn " turn "'s reply was cut inside its think block ("
                                            (count (:tokens chosen)) " tokens): give :turn-tokens room for the thinking")
                                       {:reason :cut-in-thinking :turn turn :tokens (count (:tokens chosen)) :log log})))
                   [reasoning content] (if in-thinking? [(:text chosen) ""] [reasoning content])
                   calls (chat/tool-calls content)
                   t1 (System/nanoTime)
                   results (loop [cs calls acc []]
                             (if-let [c (first cs)]
                               (let [r (call-tool tools state c)
                                     acc (conj acc [c r])]
                                 (if (contains? r :done) acc (recur (rest cs) acc)))
                               acc))
                   tool-s (+ tool-s (seconds t1))
                   entry (cond-> {:turn turn :reply content
                                  :calls (mapv (fn [[c r]] (merge {:tool (:name c) :params (:params c)} (dissoc r :done))) results)}
                           thinking? (assoc :reasoning reasoning))
                   log (conj log entry)
                   tokens (+ tokens (count (:tokens chosen)))
                   done (some (fn [[_ r]] (when (contains? r :done) r)) results)
                   left (- turns turn)
                   note (when notes (notes turn left))
                   responses (cond-> (if (seq calls) (mapv (fn [[_ r]] (:text r)) results) [no-call-text])
                               note (conj note))]
               (cond
                 done {:result (:done done) :stop :done :turns turn :log log :tokens tokens :seconds {:llm llm-s :tools tool-s}}
                 full? {:result (when on-budget (on-budget state)) :stop :context-full :turns turn :log log :tokens tokens
                        :seconds {:llm llm-s :tools tool-s}}
                 :else
                 (recur (inc turn)
                        (-> messages
                            (conj (cond-> {:role :assistant :content content} reasoning (assoc :reasoning reasoning)))
                            (into (map (fn [t] {:role :tool :content t}) responses)))
                        log tokens llm-s tool-s))))))
       (finally (when close (close state)))))))
