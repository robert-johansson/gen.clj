;; Verifier: rerun the P2 reviewer's scripts in one JVM, each isolated by try/catch.
(def dir "/private/tmp/claude-501/-Users-robert-code-gen-clj/826ae76f-f183-43cd-bbe4-ed91ce6aaf6f/scratchpad/review-post-m5/scripts/")
(doseq [id ["P2-09" "P2-01" "P2-02" "P2-03" "P2-04" "P2-05" "P2-06" "P2-07" "P2-08" "P2-11"]]
  (println "\n=====" id)
  (let [t0 (System/nanoTime)]
    (try (load-file (str dir id ".clj"))
         (catch Throwable e (println "SCRIPT THREW:" (class e) (ex-message e))))
    (println "-----" id "took" (format "%.1f s" (/ (- (System/nanoTime) t0) 1e9)))))
(shutdown-agents)
