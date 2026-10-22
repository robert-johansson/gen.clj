;; Verifier's check of I1-02's drift claim (the JVM's solve-triangular lacks
;; solve's rank guard on the right-hand side): through gen.array.linalg a 1-d
;; right-hand side never reaches the backend (with-matrix-rhs expands it), and
;; a 0-d one reaches both solves, where the JVM backend reads (peek shape)
;; before any guard. What error does each give?
(require '[gen.array :as array] '[gen.array.linalg :as la])
(doseq [[label f] [["solve, 1-d rhs" #(la/solve (array/eye 2) (array/array [1.0 2.0]))]
                   ["solve-triangular, 1-d rhs" #(la/solve-triangular (array/eye 2) (array/array [1.0 2.0]))]
                   ["solve, 0-d rhs" #(la/solve (array/eye 2) 1.0)]
                   ["solve-triangular, 0-d rhs" #(la/solve-triangular (array/eye 2) 1.0)]
                   ["solve, 1-d matrix" #(la/solve (array/array [1.0 2.0]) (array/array [1.0 2.0]))]]]
  (try (println (format "%-28s -> %s" label (pr-str (array/->clj (f)))))
       (catch Throwable e
         (println (format "%-28s -> %s: %s %s" label (.getName (class e)) (ex-message e) (pr-str (ex-data e)))))))
(flush)

;; where the NPE comes from
(try (la/solve (array/eye 2) 1.0)
     (catch NullPointerException e
       (doseq [f (take 8 (.getStackTrace e))] (println "  at" (str f)))))
(flush)
