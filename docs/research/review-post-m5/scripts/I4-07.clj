;; I4-07: trie/remove-entry of tokens the trie does not hold recurses until the stack overflows.
;; Run from the snapshot worktree: nice -n 10 clojure -J-Xmx1g -M:test -i OUT/scripts/I4-07.clj
(require '[gen.llm.trie :as trie])
(println "remove-entry of an absent path:"
         (try (trie/remove-entry (trie/insert trie/empty-trie [1 2] 0 [0.0 0.0] :x) [3 4]) :no-error
              (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))
              (catch Throwable t (.getName (class t)))))
(System/exit 0)
