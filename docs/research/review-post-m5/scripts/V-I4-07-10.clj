;; V-I4-07 and V-I4-10 (verifier's checks):
;; - trie/remove-entry off the trie's paths: an absent first token, and a path that ends inside an edge.
;; - vswap! tags its target clojure.lang.Volatile itself (so the hints I4-10 names are redundant).
;; - a tag on a fn form in argument position is never resolved (so ^Callable does nothing; a class
;;   that does not exist compiles too).
;; Run from the snapshot worktree: nice -n 10 clojure -J-Xmx1g -M:test -i OUT/scripts/V-I4-07-10.clj
(require '[gen.llm.trie :as trie])

(defn reason [f]
  (try (f) :no-error
       (catch clojure.lang.ExceptionInfo e (:reason (ex-data e) :no-reason))
       (catch Throwable t (.getName (class t)))))

(def t (trie/insert trie/empty-trie [1 2 3] 0 [0.0 0.0 0.0] :x))
(println "remove-entry [3 4] (absent):         " (reason #(trie/remove-entry t [3 4])))
(println "remove-entry [1 2] (inside an edge): " (reason #(trie/remove-entry t [1 2])))
(println "remove-entry [1 2 3] (held):         " (reason #(trie/remove-entry t [1 2 3])))
(println "vswap! expansion:" (binding [*print-meta* true] (pr-str (macroexpand-1 '(vswap! v assoc :a 1)))))
(set! *warn-on-reflection* true)
(println "a fn tagged with a class that does not exist, in argument position:"
         (reason #(eval '(mapv (fn [x] ^NoSuchClass (fn [] x)) [1 2]))))
(System/exit 0)
