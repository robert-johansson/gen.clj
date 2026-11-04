(ns gen.codegen.server
  "The codegen bake-off's grader as a process (genclj-ihc4), a
   gen.codegen.process host: a request {:task :code} is graded
   (gen.codegen.grade/grade), its response the grading. A request with
   :op another than :grade is a REPL session's (gen.codegen.repl,
   genclj-t1xh): :open {:task}, then :eval {:session :code}, :simulate
   {:session :name :args}, :check, :evidence and :submit {:session :name},
   and :close; its response is {:text} (and :session, :grading, :program
   as the operation gives them), within 300 s (else :tainted?). Run in a
   JVM of its own, never the LLM's (the driver starts it, and starts
   another when a response is tainted):

     clojure -M:test:codegen -m gen.codegen.server"
  {:api :internal}
  (:require [gen.codegen.grade :as grade]
            [gen.codegen.process :as process]
            [gen.codegen.repl :as repl]
            [gen.codegen.session :as session]
            [gen.codegen.tasks :as tasks]))

(defn handle
  "A request's response."
  [{:keys [op session task code name args ml?] :or {ml? true}}]
  (if (or (nil? op) (= :grade op))
    (if-let [t (tasks/by-id task)]
      (try (grade/grade t code)
           (catch Throwable e {:task task :error (str (ex-message e))}))
      {:task task :error (str "unknown task " (pr-str task))})
    (session/within 300000
                    #(case op
                       :open (repl/open! task)
                       :eval (repl/eval! session code)
                       :simulate (repl/simulate! session name args)
                       :check (repl/check! session name ml?)
                       :evidence (repl/evidence! session name)
                       :submit (repl/submit! session name)
                       :close (repl/close! session)))))

(defn -main [& _]
  (process/serve handle)
  (shutdown-agents)
  (System/exit 0))
