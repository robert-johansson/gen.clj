(ns gen.codegen.server
  "The codegen bake-off's grader as a process (genclj-ihc4): it reads
   requests from stdin, one EDN map a line, {:id :task :code}, and writes
   each grading to stdout, one EDN map a line, the grading
   (gen.codegen.grade/grade) with :id. A request with :op another than
   :grade is a REPL session's (gen.codegen.repl, genclj-t1xh): :open
   {:task}, then :eval {:session :code}, :simulate {:session :name :args},
   :check, :evidence and :submit {:session :name}, and :close; its result
   is {:text} (and :id, :grading, :program as the operation gives them). A candidate's own printing goes to
   :out in the result, never to the protocol. Run in a JVM of its own,
   never the LLM's (the driver starts it, and starts another when a
   grading is tainted):

     clojure -M:test -m gen.codegen.server

   It prints {:ready true} when it can take requests."
  (:require [clojure.edn :as edn]
            [gen.codegen.candidate :as candidate]
            [gen.codegen.grade :as grade]
            [gen.codegen.repl :as repl]
            [gen.codegen.tasks :as tasks]))

(defn- session-op
  "A REPL session's operation (gen.codegen.repl) within timeout-ms: its
   result, or {:text \"error: …\"} (with :tainted? when it timed out)."
  [{:keys [op session task code name args ml?] :or {ml? true}} timeout-ms]
  (try
    (candidate/within timeout-ms
                      #(case op
                         :open (repl/open! task)
                         :eval (repl/eval! session code)
                         :simulate (repl/simulate! session name args)
                         :check (repl/check! session name ml?)
                         :evidence (repl/evidence! session name)
                         :submit (repl/submit! session name)
                         :close (repl/close! session)))
    (catch Throwable e
      {:text (str "error: " (->> (take-while some? (iterate ex-cause e))
                                 (map #(or (ex-message %) (str (class %))))
                                 distinct
                                 (clojure.string/join " Caused by: ")))
       :tainted? (boolean (::candidate/timeout (ex-data e)))})))

(defn- respond [m]
  (locking *out*
    (println (pr-str m))
    (flush)))

(defn -main [& _]
  (let [out *out*]
    (binding [*out* out]
      (respond {:ready true}))
    (loop []
      (when-let [line (read-line)]
        (let [{:keys [id op task code] :as request} (try (edn/read-string line) (catch Exception _ nil))
              t (tasks/by-id task)
              captured (java.io.StringWriter.)
              result (binding [*out* captured *err* captured]
                       (cond (and op (not= :grade op)) (session-op request 300000)
                             t (try (grade/grade t code)
                                    (catch Throwable e {:task task :error (str (ex-message e))}))
                             :else {:task task :error (str "unknown task or unreadable request: "
                                                           (subs line 0 (min 200 (count line))))}))]
          (binding [*out* out]
            (respond (assoc result :id id :out (let [s (str captured)] (subs s 0 (min 2000 (count s)))))))
          (recur))))
    (shutdown-agents)
    (System/exit 0)))
