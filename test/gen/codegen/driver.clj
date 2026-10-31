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

   Tokens are sampled as Qwen's GGUFs say (temperature 1, top-k 20, top-p
   0.95), from the Branching protocol's next-token log-probabilities, with
   gen.random keys: the run is a function of its seed (and llama.cpp's
   float noise). The prompt is read once per task: its branch is forked
   for each reply, and a repair turn extends the conversation's tokens
   (earlier replies as generated, without their think block, as the
   template renders them).

   Every reply and its grading are a line of EDN in <out>/results.edn.

     cd ~/code/gen.clj
     tmp/guard.sh 40 3600 -- clojure -J-Dcodegen.model=<gguf> -J-Dcodegen.out=tmp/codegen-36 \\
       -J-Dcodegen.grader='clojure -M:test -m gen.codegen.server' -M:test -m gen.codegen.driver

   Options, as system properties: codegen.model (a GGUF), codegen.out (a
   directory), codegen.grader (the grader's command, split at spaces),
   codegen.k (4), codegen.rounds (3), codegen.max-tokens (1024),
   codegen.tasks (task ids, comma-separated; all by default), codegen.seed
   (1), codegen.n-ctx (16384)."
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

(def sampling {:temp 1.0 :top-k 20 :top-p 0.95})

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
  [base n max-tokens eog key]
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

(defn- grade!
  "The grading of code for task by the grader in g (an atom), restarting it
   when the grading is tainted, times out (timeout-ms) or the grader dies."
  [g command log task-id code timeout-ms]
  (let [id (str (random-uuid))
        {:keys [in out]} @g
        restart! #(do (kill! (:process @g)) (reset! g (start-grader command log)))]
    (.write ^java.io.Writer out (str (pr-str {:id id :task task-id :code code}) "\n"))
    (.flush ^java.io.Writer out)
    (let [f (future (loop []
                      (when-let [l (.readLine ^java.io.BufferedReader in)]
                        (let [m (try (edn/read-string l) (catch Exception _ nil))]
                          (if (and (map? m) (= id (:id m))) m (recur))))))
          r (deref f timeout-ms ::timeout)]
      (cond (= r ::timeout) (do (future-cancel f) (restart!) {:task task-id :error "the grading timed out" :tainted? true})
            (nil? r) (do (restart!) {:task task-id :error "the grader exited"})
            :else (do (when (:tainted? r) (restart!)) (dissoc r :id))))))

;; ---------------------------------------------------------------------------
;; The run
;; ---------------------------------------------------------------------------

(defn- prop [k default] (or (System/getProperty (str "codegen." k)) default))

(defn- failed-stage [grading] (:stage (first (remove :ok (:stages grading)))))

(defn run-task
  "One task's replies and gradings, each passed to record!."
  [e task {:keys [k rounds max-tokens seed eog]} grade record!]
  (let [tok (fn [text] (llama/tokenize e text {:add-special? false :parse-special? true}))
        think-text "<think>\n\n</think>\n\n"
        chat (prompt/chat [[:system (prompt/system-message)] [:user (prompt/task-message task)]])
        ;; the conversation up to the assistant's turn, without its think block
        head (tok (subs chat 0 (- (count chat) (count think-text))))
        think (tok think-text)
        key (rng/fold-in (rng/key seed) (hash (:id task)))
        t0 (System/nanoTime)
        base (br/open e head)
        read-s (/ (- (System/nanoTime) t0) 1e9)
        reply (fn [branch n key]
                (let [t0 (System/nanoTime)
                      g (:branch (br/extend (br/fork branch) think))
                      rs (try (generate g n max-tokens eog key) (finally (br/dispose! g)))]
                  (mapv #(assoc % :seconds (/ (- (System/nanoTime) t0) 1e9)) rs)))
        graded (fn [mode i round r]
                 (let [text (llama/detokenize e (:tokens r))
                       grading (grade (:id task) (candidate/code-of text))
                       entry {:task (:id task) :mode mode :reply i :round round :tokens (count (:tokens r))
                              :prompt-tokens (count head) :stop (:stop r) :seconds (:seconds r) :text text
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

(defn -main [& _]
  (let [model (prop "model" nil)
        out (io/file (prop "out" "tmp/codegen"))
        command (str/split (prop "grader" "clojure -M:test -m gen.codegen.server") #" ")
        opts {:k (parse-long (prop "k" "4")) :rounds (parse-long (prop "rounds" "3"))
              :max-tokens (parse-long (prop "max-tokens" "1024")) :seed (parse-long (prop "seed" "1"))}
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
          (run-task e (tasks/by-id id) opts #(grade! g command log %1 %2 900000) record!))))
    (kill! (:process @g))
    (println "done:" (str results))
    (shutdown-agents)
    (System/exit 0)))
