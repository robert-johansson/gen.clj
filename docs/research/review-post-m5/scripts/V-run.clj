;; Verifier: run the independent checks in one JVM, each isolated.
(def dir "/private/tmp/claude-501/-Users-robert-code-gen-clj/826ae76f-f183-43cd-bbe4-ed91ce6aaf6f/scratchpad/review-post-m5/scripts/")
(doseq [id ["V-P2-09" "V-P2-01" "V-P2-02" "V-P2-06" "V-P2-03" "V-P2-04" "V-P2-11"]]
  (println "\n=====" id)
  (let [t0 (System/nanoTime)]
    (try (load-file (str dir id ".clj"))
         (catch Throwable e (println "SCRIPT THREW:" (class e) (ex-message e)) (.printStackTrace e (java.io.PrintWriter. *out*))))
    (println "-----" id "took" (format "%.1f s" (/ (- (System/nanoTime) t0) 1e9)))))
(shutdown-agents)
