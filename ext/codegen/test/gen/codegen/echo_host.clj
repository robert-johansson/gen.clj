(ns gen.codegen.echo-host
  "A gen.codegen.process host for gen.codegen.process-test: :echo answers
   its :value, :print prints, :throw throws, :sleep sleeps :ms (past the
   host's 1 s limit, tainted), :exit ends the process."
  (:require [gen.codegen.process :as process]))

(defn- handle [{:keys [op value ms]}]
  (case op
    :echo {:value value :pid (.pid (java.lang.ProcessHandle/current))}
    :print (do (println "printed") (binding [*out* *err*] (println "to err")) {:value :printed})
    :throw (throw (ex-info "it broke" {}))
    :sleep (do (Thread/sleep (long ms)) {:value :slept})
    :exit (System/exit 3)
    :bare 42))

(defn -main [& _]
  (process/serve handle {:timeout-ms 1000})
  (shutdown-agents)
  (System/exit 0))
