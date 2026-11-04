(ns gen.codegen.driver
  "The codegen bake-off's run (genclj-ihc4): a local model (llama.cpp, Qwen's
   chat template through gen.llm.chat) writes each task's model, and a
   grader in another JVM (gen.codegen.server, a gen.codegen.process host)
   grades every reply.

   Per task: K one-shot replies to the same prompt, drawn together
   (gen.llm.sample/complete); then the repair loop from the first of them:
   while a reply fails a hard stage (it does not read, is not safe, does
   not evaluate, simulate, take the data or keep the laws), the failure
   goes back as the next user turn and the model replies again, up to R
   rounds. The explains stage's gap is recorded, not fed back
   (gen.codegen.grade's :gate-explains?, off). The REPL mode runs the
   REPL agent's episodes instead (run-repl-task, genclj-t1xh), on
   gen.llm.agent, its tools the grader's sessions (gen.codegen.repl).

   Tokens are sampled as Qwen's model card says for thinking off
   (gen.llm.sample/default-sampling) unless the options say otherwise,
   with gen.random keys: the run is a function of its seed (and llama.cpp's
   float noise). The LM's prefix cache reads a task's prompt once, and a
   turn reads only what the turn before did not.

   Every reply and its grading are a line of EDN in <out>/results.edn.

     clojure -J-Dcodegen.model=<gguf> -J-Dcodegen.out=tmp/codegen-36 -M:test:codegen -m gen.codegen.driver

   (on a shared Mac, under a memory guard: one GPU-heavy JVM at a time).

   Options, as system properties: codegen.model (a GGUF), codegen.out (a
   directory), codegen.grader (the grader's command, split at spaces),
   codegen.k (4), codegen.rounds (3), codegen.max-tokens (2048),
   codegen.tasks (task ids, comma-separated; all by default), codegen.seed
   (1), codegen.n-ctx (16384), codegen.temp, codegen.top-p and
   codegen.top-k (default-sampling's). codegen.mode repl runs the REPL
   agent: codegen.episodes (2), codegen.turns (40; the last five turns
   are told how many are left), codegen.turn-tokens (1024),
   codegen.evidence (true; false leaves out the evidence tool and check's
   log marginal likelihood)."
  {:api :internal}
  (:require [clojure.java.io :as io]
            [clojure.string :as str]
            [gen.codegen.candidate :as candidate]
            [gen.codegen.process :as process]
            [gen.codegen.prompt :as prompt]
            [gen.codegen.tasks :as tasks]
            [gen.llm.agent :as agent]
            [gen.llm.chat :as chat]
            [gen.llm.llama :as llama]
            [gen.llm.lm :as lm]
            [gen.llm.sample :as sample]
            [gen.random :as rng]))

(set! *warn-on-reflection* true)

(def hard-stages #{:read :safe :evaluates :simulates :accepts-data :laws})

(defn- prop [k default] (or (System/getProperty (str "codegen." k)) default))

(defn- failed-stage [grading] (:stage (first (remove :ok (:stages grading)))))

(defn- clip [s n] (if (> (count s) n) (str (subs s 0 n) "…") s))

(defn- task-key
  "The task's key: seed's, folded with the task's position (a hash may be
   negative)."
  [seed task]
  (rng/fold-in (rng/key seed) (.indexOf ^java.util.List (mapv :id tasks/tasks) (:id task))))

(defn- seconds [t0] (/ (- (System/nanoTime) t0) 1e9))

(defn run-task
  "One task's one-shot replies and repair rounds, their gradings (by
   grade, a function of the task id and code) each passed to record!."
  [m task {:keys [k rounds max-tokens seed sampling]} grade record!]
  (let [context (fn [messages] (lm/encode m (chat/render messages) {:add-special? false :parse-special? true}))
        key (task-key seed task)
        replies (fn [ctx n key]
                  (let [t0 (System/nanoTime)
                        rs (sample/complete m (vec (repeat n ctx)) {:max-tokens max-tokens :sampling sampling :key key})]
                    (mapv #(assoc % :seconds (seconds t0)) rs)))
        graded (fn [mode i round ctx r]
                 (let [text (lm/decode m (:tokens r))
                       entry {:task (:id task) :mode mode :reply i :round round :tokens (count (:tokens r))
                              :prompt-tokens (count ctx) :sampling sampling :stop (:stop r) :seconds (:seconds r) :text text
                              :grading (grade (:id task) (candidate/code-of text))}]
                   (record! entry)
                   entry))
        messages [{:role :system :content (prompt/system-message)} {:role :user :content (prompt/task-message task)}]
        ctx (context messages)
        one-shot (mapv #(graded :one-shot %2 0 ctx %1) (replies ctx k (rng/fold-in key 0)) (range))]
    (println (format "%s: prompt %d tokens, one shot %s" (name (:id task)) (count ctx)
                     (pr-str (mapv (comp #(or (failed-stage %) :passed) :grading) one-shot))))
    ;; the repair loop, from the first one-shot reply
    (loop [messages messages prev (first one-shot) round 1]
      (let [stage (failed-stage (:grading prev))]
        (when (and (<= round rounds) (hard-stages stage))
          (let [messages (conj messages
                               {:role :assistant :content (:text prev)}
                               {:role :user :content (prompt/feedback-message (:grading prev))})
                ctx (context messages)
                entry (graded :repair 0 round ctx (first (replies ctx 1 (rng/fold-in key round))))]
            (println (format "%s: repair %d %s" (name (:id task)) round (or (failed-stage (:grading entry)) :passed)))
            (recur messages entry (inc round))))))))

(defn- repl-tools
  "The REPL agent's tools for gen.llm.agent: each a request to the
   grader's session (call), the episode's state an atom of {:session}. A
   submission ends the episode with its response as the result; a lost
   session (the grader restarted) ends it with {:lost true}."
  [call evidence?]
  (let [request {"eval" (fn [p] {:op :eval :code (p "code")})
                 "simulate" (fn [p] {:op :simulate :name (p "name") :args (p "args")})
                 "check" (fn [p] {:op :check :name (p "name") :ml? evidence?})
                 "evidence" (fn [p] {:op :evidence :name (p "name")})
                 "submit" (fn [p] {:op :submit :name (p "name")})}]
    (into (array-map)
          (for [{n :name :as t} (prompt/tools-for evidence?)]
            [n (assoc t :fn (fn [params state]
                              (let [r (call (assoc ((request n) params) :session (:session @state)))]
                                (cond (:restarted r) (do (swap! state assoc :ended true)
                                                         {:text (str (:text r)) :done {:lost true}})
                                      (:grading r) (do (swap! state assoc :ended true)
                                                       {:text (str (:text r)) :done r})
                                      :else {:text (str (:text r))}))))]))))

(defn run-repl-task
  "The REPL agent's episodes for task (genclj-t1xh), on gen.llm.agent:
   each opens a session in the grader's process and runs turns until the
   model submits or the turns or the context run out (then model, if the
   session defines it, is submitted). call sends a request to the
   grader's process; record! takes each episode's entry."
  [m task {:keys [episodes turns turn-tokens seed sampling evidence?]} call record!]
  (let [tkey (task-key seed task)
        system (prompt/repl-system-message evidence?)]
    (doseq [episode (range episodes)]
      (let [t0 (System/nanoTime)
            submit-model (fn [state]
                           (let [r (call {:op :submit :session (:session @state) :name "model"})]
                             (when (:grading r) (swap! state assoc :ended true) r)))
            r (agent/run m {:system system
                            :task (prompt/repl-task-message task)
                            :tools (repl-tools call evidence?)
                            :open #(atom {:session (:session (call {:op :open :task (:id task)}))})
                            :close #(when-not (:ended @%) (call {:op :close :session (:session @%)}))
                            :on-budget submit-model
                            :notes (fn [_ left]
                                     (when (<= left 5)
                                       (str left " turn" (when (not= 1 left) "s") " left: submit your best model "
                                            (if (<= left 2) "now." "soon."))))}
                         {:turns turns :turn-tokens turn-tokens :sampling sampling
                          :key (rng/fold-in tkey (+ 1000 episode))})
            result (:result r)
            why (cond (:lost result) :session-lost (= :done (:stop r)) :submitted :else (:stop r))
            entry {:task (:id task) :mode :repl :reply episode :round 0 :evidence? evidence?
                   :turns (:turns r) :tokens (:tokens r) :stop why
                   :seconds (:llm (:seconds r)) :tool-seconds (:tools (:seconds r)) :wall-seconds (seconds t0)
                   :sampling sampling
                   :text (str/join "\n\n" (map :reply (:log r)))
                   :log (mapv (fn [e] (-> e
                                          (update :reply clip 1500)
                                          (update :calls (partial mapv (fn [c] (-> c (update :params update-vals #(clip % 600))
                                                                                   (update :text clip 600)))))))
                              (:log r))
                   :program (:program result)
                   :grading (or (:grading result) {:task (:id task) :error (str "not submitted: " (name why))})}]
        (record! entry)
        (println (format "%s: episode %d, %s after %d turns (%.0f s LLM, %.0f s tools): %s"
                         (name (:id task)) episode (name why) (:turns r) (:seconds entry) (:tool-seconds entry)
                         (if (:grading result)
                           (or (failed-stage (:grading result)) :passed)
                           (:error (:grading entry)))))))))

(defn -main [& _]
  (let [model (prop "model" nil)
        out (io/file (prop "out" "tmp/codegen"))
        command (str/split (prop "grader" "clojure -M:test:codegen -m gen.codegen.server") #" ")
        sampling {:temp (parse-double (prop "temp" (str (:temp sample/default-sampling))))
                  :top-p (parse-double (prop "top-p" (str (:top-p sample/default-sampling))))
                  :top-k (parse-long (prop "top-k" (str (:top-k sample/default-sampling))))}
        opts {:k (parse-long (prop "k" "4")) :rounds (parse-long (prop "rounds" "3"))
              :max-tokens (parse-long (prop "max-tokens" "2048")) :seed (parse-long (prop "seed" "1"))
              :sampling sampling}
        ids (if-let [t (prop "tasks" nil)] (mapv keyword (str/split t #",")) (mapv :id tasks/tasks))
        _ (.mkdirs out)
        results (io/file out "results.edn")
        host (process/start command {:log (str (io/file out "grader.log"))})
        record! (fn [entry] (spit results (str (pr-str (assoc entry :model model)) "\n") :append true))]
    (println "grader ready; model" model)
    (with-open [e (llama/engine {:model model :n-ctx (parse-long (prop "n-ctx" "16384")) :max-branches 8})
                m (lm/lm e (llama/tokenizer e) {:max-branches 8})]
      (doseq [id ids]
        (if (= "repl" (prop "mode" "one-shot"))
          (run-repl-task m (tasks/by-id id)
                         (assoc opts :episodes (parse-long (prop "episodes" "2")) :turns (parse-long (prop "turns" "40"))
                                :turn-tokens (parse-long (prop "turn-tokens" "1024"))
                                :evidence? (not= "false" (prop "evidence" "true")))
                         #(process/call host % 900000) record!)
          (run-task m (tasks/by-id id) opts
                    (fn [task-id code]
                      (let [r (process/call host {:task task-id :code code} 900000)]
                        (cond-> (dissoc r :text :restarted) (:error r) (assoc :task task-id))))
                    record!))))
    (process/stop host)
    (println "done:" (str results))
    (shutdown-agents)
    (System/exit 0)))
