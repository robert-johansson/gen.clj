(ns gen.codegen.driver
  "The codegen bake-off's run (genclj-ihc4): a local model (llama.cpp, Qwen's
   chat template) writes each task's model, and a grader in another JVM
   (gen.codegen.server) grades every reply.

   Per task: K one-shot replies to the same prompt, decoded together; then
   the repair loop from the first of them: while a reply fails a hard stage
   (it does not read, is not safe, does not evaluate, simulate or take the
   data), the failure goes back as the next user turn and the model
   replies again, up to R rounds. The explains stage's gap is recorded,
   not fed back (gen.codegen.grade's :gate-explains?, off).

   Tokens are sampled as Qwen's model card says for thinking off
   (default-sampling) unless the options say otherwise, from the Branching protocol's next-token log-probabilities, with
   gen.random keys: the run is a function of its seed (and llama.cpp's
   float noise). The prompt is read once per task: its branch is forked
   for each reply, and a repair turn extends the conversation's tokens
   (earlier replies as generated, without their think block, as the
   template renders them).

   Every reply and its grading are a line of EDN in <out>/results.edn.

     cd ~/code/gen.clj
     tmp/guard.sh 40 3600 -- clojure -J-Dcodegen.model=<gguf> -J-Dcodegen.out=tmp/codegen-36 \\
       -J-Dcodegen.grader='clojure -M:test:codegen -m gen.codegen.server' -M:test -m gen.codegen.driver

   Options, as system properties: codegen.model (a GGUF), codegen.out (a
   directory), codegen.grader (the grader's command, split at spaces),
   codegen.k (4), codegen.rounds (3), codegen.max-tokens (2048),
   codegen.tasks (task ids, comma-separated; all by default), codegen.seed
   (1), codegen.n-ctx (16384), codegen.temp, codegen.top-p and
   codegen.top-k (default-sampling's). codegen.mode repl runs the REPL
   agent (run-repl-task, genclj-t1xh): codegen.episodes (2), codegen.turns
   (30), codegen.turn-tokens (1024), codegen.evidence (true; false leaves
   out the evidence tool and check's log marginal likelihood)."
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [gen.array :as array]
            [gen.codegen.candidate :as candidate]
            [gen.codegen.prompt :as prompt]
            [gen.codegen.tasks :as tasks]
            [gen.llm.branching :as br]
            [gen.llm.llama :as llama]
            [gen.random :as rng]))

(set! *warn-on-reflection* true)

(def default-sampling
  "Qwen's model card's for thinking off: temperature 0.7, top-p 0.8, top-k
   20 (its presence penalty of 1.5 not applied: it lowers every token
   already used, closing parentheses among them). The GGUFs carry its
   thinking mode's (temperature 1, top-p 0.95), under which the 3.6 wrote
   5 running models of 40 where these gave 11 (genclj-ihc4)."
  {:temp 0.7 :top-k 20 :top-p 0.8})

(def hard-stages #{:read :safe :evaluates :simulates :accepts-data})

;; ---------------------------------------------------------------------------
;; Sampling
;; ---------------------------------------------------------------------------

(defn- top-indices
  "The indices of the k largest entries of lp, largest first."
  [^floats lp ^long k]
  (let [idx (long-array k -1)
        val (double-array k Double/NEGATIVE_INFINITY)]
    (dotimes [t (alength lp)]
      (let [v (double (aget lp t))]
        (when (> v (aget val (dec k)))
          (loop [j (dec k)]
            (if (and (pos? j) (> v (aget val (dec j))))
              (do (aset val j (aget val (dec j))) (aset idx j (aget idx (dec j))) (recur (dec j)))
              (do (aset val j v) (aset idx j t)))))))
    (vec (remove neg? idx))))

(defn pick
  "A token from next-token log-probabilities lp: the top-k tokens, the
   fewest of them whose probabilities reach top-p, at temperature temp; u
   a uniform draw in [0, 1)."
  [^floats lp {:keys [temp top-k top-p]} ^double u]
  (let [idx (top-indices lp top-k)
        ws (mapv #(/ (double (aget lp (int %))) (double temp)) idx)
        m (double (first ws))
        ps (mapv #(Math/exp (- (double %) m)) ws)
        z (reduce + ps)
        cum (reductions + (map #(/ % z) ps))
        n (min (count ps) (inc (count (take-while #(< % top-p) cum))))
        kept (subvec ps 0 n)
        target (* u (reduce + kept))]
    (nth idx (min (dec n) (count (take-while #(<= % target) (reductions + kept)))))))

(defn- uniform [key] (double (array/item (rng/uniform key []))))

(defn generate
  "n replies continuing branch base (left live), each up to max-tokens,
   decoded together, reply i's draws from (fold-in (split key n)[i] step):
   [{:tokens (without the end token) :stop (:end or :max)}]."
  [base n max-tokens eog sampling key]
  (let [ks (rng/split key n)]
    (loop [live (mapv vector (range n) (br/fork base n)) toks (vec (repeat n [])) stops {} step 0]
      (if (empty? live)
        (mapv (fn [i] {:tokens (toks i) :stop (stops i)}) (range n))
        (let [picked (mapv (fn [[i b]] [i b (pick (br/logprobs b) sampling (uniform (rng/fold-in (ks i) step)))]) live)
              ended (filter (fn [[i _ t]] (or (eog t) (>= (inc (count (toks i))) max-tokens))) picked)
              going (remove (set ended) picked)
              toks (reduce (fn [ts [i _ t]] (if (eog t) ts (update ts i conj t))) toks picked)
              stops (into stops (map (fn [[i _ t]] [i (if (eog t) :end :max)])) ended)]
          (br/dispose! (mapv second ended))
          (recur (if (seq going)
                   (mapv vector (map first going) (br/advance (mapv second going) (mapv #(nth % 2) going)))
                   [])
                 toks stops (inc step)))))))

;; ---------------------------------------------------------------------------
;; The grader, a process of its own
;; ---------------------------------------------------------------------------

(defn- kill! [^Process p]
  (doseq [^java.lang.ProcessHandle h (iterator-seq (.iterator ^java.util.stream.Stream (.descendants p)))] (.destroyForcibly h))
  (.destroyForcibly p))

(defn- start-grader [command log]
  (let [pb (doto (ProcessBuilder. ^java.util.List command)
             (.redirectError (java.lang.ProcessBuilder$Redirect/appendTo (io/file log))))
        p (.start pb)
        in (io/reader (.getInputStream p))
        out (io/writer (.getOutputStream p))]
    (loop []
      (let [l (.readLine ^java.io.BufferedReader in)]
        (cond (nil? l) (throw (ex-info "the grader exited before it was ready" {:log log}))
              (:ready (try (edn/read-string l) (catch Exception _ nil))) {:process p :in in :out out}
              :else (recur))))))

(defn- request!
  "request's response from the grader process in g (an atom), restarting it
   when the response is tainted, times out (timeout-ms) or the grader dies:
   then :restarted, and a REPL session it held is gone."
  [g command log request timeout-ms]
  (let [id (str (random-uuid))
        {:keys [in out]} @g
        restart! #(do (kill! (:process @g)) (reset! g (start-grader command log)))]
    (.write ^java.io.Writer out (str (pr-str (assoc request :id id)) "\n"))
    (.flush ^java.io.Writer out)
    (let [f (future (loop []
                      (when-let [l (.readLine ^java.io.BufferedReader in)]
                        (let [m (try (edn/read-string l) (catch Exception _ nil))]
                          (if (and (map? m) (= id (:id m))) m (recur))))))
          r (deref f timeout-ms ::timeout)]
      (cond (= r ::timeout) (do (future-cancel f) (restart!)
                                {:error "timed out" :text "error: timed out" :tainted? true :restarted true})
            (nil? r) (do (restart!) {:error "the grader exited" :text "error: the grader exited" :restarted true})
            :else (do (when (:tainted? r) (restart!)) (cond-> (dissoc r :id) (:tainted? r) (assoc :restarted true)))))))

(defn- grade!
  "The grading of code for task by the grader process."
  [g command log task-id code timeout-ms]
  (let [r (request! g command log {:task task-id :code code} timeout-ms)]
    (cond-> (dissoc r :text :restarted) (:error r) (assoc :task task-id))))

;; ---------------------------------------------------------------------------
;; The run
;; ---------------------------------------------------------------------------

(defn- prop [k default] (or (System/getProperty (str "codegen." k)) default))

(defn- failed-stage [grading] (:stage (first (remove :ok (:stages grading)))))

(defn run-task
  "One task's replies and gradings, each passed to record!."
  [e task {:keys [k rounds max-tokens seed eog sampling]} grade record!]
  (let [tok (fn [text] (llama/tokenize e text {:add-special? false :parse-special? true}))
        think-text "<think>\n\n</think>\n\n"
        chat (prompt/chat [[:system (prompt/system-message)] [:user (prompt/task-message task)]])
        ;; the conversation up to the assistant's turn, without its think block
        head (tok (subs chat 0 (- (count chat) (count think-text))))
        think (tok think-text)
        key (rng/fold-in (rng/key seed) (.indexOf ^java.util.List (mapv :id tasks/tasks) (:id task)))
        t0 (System/nanoTime)
        base (br/open e head)
        read-s (/ (- (System/nanoTime) t0) 1e9)
        reply (fn [branch n key]
                (let [t0 (System/nanoTime)
                      g (:branch (br/extend (br/fork branch) think))
                      rs (try (generate g n max-tokens eog sampling key) (finally (br/dispose! g)))]
                  (mapv #(assoc % :seconds (/ (- (System/nanoTime) t0) 1e9)) rs)))
        graded (fn [mode i round r]
                 (let [text (llama/detokenize e (:tokens r))
                       grading (grade (:id task) (candidate/code-of text))
                       entry {:task (:id task) :mode mode :reply i :round round :tokens (count (:tokens r))
                              :prompt-tokens (count head) :sampling sampling :stop (:stop r) :seconds (:seconds r) :text text
                              :grading grading}]
                   (record! entry)
                   (assoc entry :ids (:tokens r))))
        one-shot (mapv #(graded :one-shot %2 0 %1) (reply base k (rng/fold-in key 0)) (range))]
    (println (format "%s: prompt %d tokens (%.1f s), one shot %s" (name (:id task)) (count head) read-s
                     (pr-str (mapv (comp #(or (failed-stage %) :passed) :grading) one-shot))))
    ;; the repair loop, from the first one-shot reply
    (loop [branch (br/fork base) prev (first one-shot) round 1]
      (let [stage (failed-stage (:grading prev))]
        (if (or (> round rounds) (not (hard-stages stage)))
          (br/dispose! branch)
          (let [feedback (prompt/feedback-message (:grading prev))
                ;; the reply as generated, then the next user turn
                turn (into (:ids prev) (tok (str "<|im_end|>\n<|im_start|>user\n" feedback "<|im_end|>\n<|im_start|>assistant\n")))
                branch (:branch (br/extend branch turn))
                [r] (reply branch 1 (rng/fold-in key round))
                entry (graded :repair 0 round r)]
            (println (format "%s: repair %d %s" (name (:id task)) round (or (failed-stage (:grading entry)) :passed)))
            (recur branch entry (inc round))))))
    (br/dispose! base)))

(defn- clip [s n] (if (> (count s) n) (str (subs s 0 n) "…") s))

(defn run-repl-task
  "The REPL agent's episodes for task (genclj-t1xh): each opens a session
   in the grader's process and runs turns until the model submits or the
   turn budget ends (then model, if the session defines it, is submitted).
   A turn: the reply sampled after an empty think block, its tool calls run
   in order, their responses the next user turn. call sends a request to
   the grader process; record! takes each episode's entry."
  [e task {:keys [episodes turns turn-tokens seed eog sampling evidence?]} call record!]
  (let [tok (fn [text] (llama/tokenize e text {:add-special? false :parse-special? true}))
        think (tok "<think>\n\n</think>\n\n")
        head (tok (prompt/repl-head task evidence?))
        base (br/open e head)
        n-ctx (long (:n-ctx (br/info e)))
        task-key (rng/fold-in (rng/key seed) (.indexOf ^java.util.List (mapv :id tasks/tasks) (:id task)))]
    (doseq [episode (range episodes)]
      (let [key (rng/fold-in task-key (+ 1000 episode))
            sid (:session (call {:op :open :task (:id task)}))
            run-tool (fn [{:keys [name params]}]
                       (let [p #(get params %)
                             req (case name
                                   "eval" {:op :eval :code (p "code")}
                                   "simulate" {:op :simulate :name (p "name") :args (p "args")}
                                   "check" {:op :check :name (p "name") :ml? evidence?}
                                   "evidence" (when evidence? {:op :evidence :name (p "name")})
                                   "submit" {:op :submit :name (p "name")}
                                   nil)]
                         (if req
                           (call (assoc req :session sid))
                           {:text (str "error: there is no tool " name)})))
            t0 (System/nanoTime)]
        (loop [conv (br/fork base) turn 1 log [] llm-s 0.0 tool-s 0.0 tokens 0]
          (let [t1 (System/nanoTime)
                ;; a turn's reply gets at most the room left in the context
                budget (min (long turn-tokens) (- n-ctx (br/position conv) (count think) 16))
                [r] (if (< budget 128)
                      [{:tokens [] :stop :context-full}]
                      (let [g (:branch (br/extend (br/fork conv) think))]
                        (try (generate g 1 budget eog sampling (rng/fold-in key turn))
                             (catch Exception _ [{:tokens [] :stop :context-full}])
                             (finally (when (br/live? g) (br/dispose! g))))))
                text (llama/detokenize e (:tokens r))
                t2 (System/nanoTime)
                calls (vec (prompt/tool-calls text))
                responses (loop [cs calls acc []]
                            (if-let [c (first cs)]
                              (let [resp (run-tool c)]
                                (if (or (:grading resp) (:restarted resp)) (conj acc resp) (recur (rest cs) (conj acc resp))))
                              acc))
                t3 (System/nanoTime)
                llm-s (+ llm-s (/ (- t2 t1) 1e9))
                tool-s (+ tool-s (/ (- t3 t2) 1e9))
                tokens (+ tokens (count (:tokens r)))
                log (conj log {:turn turn :reply (clip text 1500)
                               :calls (mapv (fn [c resp] {:tool (:name c) :params (update-vals (:params c) #(clip % 600))
                                                          :response (clip (str (:text resp)) 600)})
                                            calls responses)})
                submitted (some #(when (:grading %) %) responses)
                lost (some :restarted responses)
                finish (fn [submission why]
                         (when (br/live? conv) (br/dispose! conv))
                         (when-not (or (:grading submission) (= why :session-lost))
                           (call {:op :close :session sid}))
                         (let [entry {:task (:id task) :mode :repl :reply episode :round 0 :evidence? evidence?
                                      :turns turn :tokens tokens :stop why
                                      :seconds llm-s :tool-seconds tool-s :wall-seconds (/ (- (System/nanoTime) t0) 1e9)
                                      :prompt-tokens (count head) :sampling sampling
                                      :text (str/join "\n\n" (map :reply log)) :log log
                                      :program (:program submission)
                                      :grading (or (:grading submission) {:task (:id task) :error (str "not submitted: " why)})}]
                           (record! entry)
                           (println (format "%s: episode %d, %s after %d turns (%.0f s LLM, %.0f s tools): %s"
                                            (name (:id task)) episode (clojure.core/name why) turn llm-s tool-s
                                            (let [st (first (remove :ok (:stages (:grading entry))))]
                                              (if (:grading submission) (or (:stage st) :passed) (:error (:grading entry))))))))]
            (cond (= :context-full (:stop r))
                  (let [s (call {:op :submit :session sid :name "model"})]
                    (finish (when (:grading s) s) :context-full))
                  submitted (finish submitted :submitted)
                  lost (finish nil :session-lost)
                  (>= turn turns) (let [s (call {:op :submit :session sid :name "model"})]
                                    (finish (when (:grading s) s) :turns))
                  :else
                  (let [texts (if (seq calls)
                                (mapv :text responses)
                                ["No tool call in your reply: call one of the tools."])
                        next-conv (try (:branch (br/extend conv (-> think (into (:tokens r)) (into (tok (prompt/tool-turn texts))))))
                                       (catch Exception ex {::error (ex-message ex)}))]
                    (if (::error next-conv)
                      (let [s (call {:op :submit :session sid :name "model"})]
                        (finish (when (:grading s) s) :context-full))
                      (recur next-conv (inc turn) log llm-s tool-s tokens))))))))
    (br/dispose! base)))

(defn -main [& _]
  (let [model (prop "model" nil)
        out (io/file (prop "out" "tmp/codegen"))
        command (str/split (prop "grader" "clojure -M:test:codegen -m gen.codegen.server") #" ")
        opts {:k (parse-long (prop "k" "4")) :rounds (parse-long (prop "rounds" "3"))
              :max-tokens (parse-long (prop "max-tokens" "2048")) :seed (parse-long (prop "seed" "1"))
              :sampling {:temp (parse-double (prop "temp" (str (:temp default-sampling))))
                         :top-p (parse-double (prop "top-p" (str (:top-p default-sampling))))
                         :top-k (parse-long (prop "top-k" (str (:top-k default-sampling))))}}
        ids (if-let [t (prop "tasks" nil)] (mapv keyword (str/split t #",")) (mapv :id tasks/tasks))
        _ (.mkdirs out)
        results (io/file out "results.edn")
        log (str (io/file out "grader.log"))
        g (atom (start-grader command log))
        record! (fn [entry] (spit results (str (pr-str (assoc entry :model model)) "\n") :append true))]
    (println "grader ready; model" model)
    (with-open [e (llama/engine {:model model :n-ctx (parse-long (prop "n-ctx" "16384")) :max-branches 8})]
      (let [opts (assoc opts :eog (set (:eog (llama/tokenizer e))))]
        (doseq [id ids]
          (if (= "repl" (prop "mode" "one-shot"))
            (run-repl-task e (tasks/by-id id)
                           (assoc opts :episodes (parse-long (prop "episodes" "2")) :turns (parse-long (prop "turns" "30"))
                                  :turn-tokens (parse-long (prop "turn-tokens" "1024"))
                                  :evidence? (not= "false" (prop "evidence" "true")))
                           #(request! g command log % 900000) record!)
            (run-task e (tasks/by-id id) opts #(grade! g command log %1 %2 900000) record!)))))
    (kill! (:process @g))
    (println "done:" (str results))
    (shutdown-agents)
    (System/exit 0)))
