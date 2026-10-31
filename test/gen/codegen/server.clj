(ns gen.codegen.server
  "The codegen bake-off's grader as a process (genclj-ihc4): it reads
   requests from stdin, one EDN map a line, {:id :task :code}, and writes
   each grading to stdout, one EDN map a line, the grading
   (gen.codegen.grade/grade) with :id. A candidate's own printing goes to
   :out in the result, never to the protocol. Run in a JVM of its own,
   never the LLM's (the driver starts it, and starts another when a
   grading is tainted):

     clojure -M:test -m gen.codegen.server

   It prints {:ready true} when it can take requests."
  (:require [clojure.edn :as edn]
            [gen.codegen.grade :as grade]
            [gen.codegen.tasks :as tasks]))

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
        (let [{:keys [id task code]} (try (edn/read-string line) (catch Exception _ nil))
              t (tasks/by-id task)
              captured (java.io.StringWriter.)
              result (if t
                       (binding [*out* captured *err* captured]
                         (try (grade/grade t code)
                              (catch Throwable e {:task task :error (str (ex-message e))})))
                       {:task task :error (str "unknown task or unreadable request: " (subs line 0 (min 200 (count line))))})]
          (binding [*out* out]
            (respond (assoc result :id id :out (let [s (str captured)] (subs s 0 (min 2000 (count s)))))))
          (recur))))
    (shutdown-agents)
    (System/exit 0)))
