;; I4-10: the six LLM namespaces that do not set *unchecked-math* :warn-on-boxed, loaded with it on.
;; Run from the snapshot worktree: nice -n 10 clojure -J-Xmx1g -M:test -i OUT/scripts/I4-10.clj
(require 'gen.llm.cells 'gen.llm.trie 'gen.dist.core 'gen.gfi 'gen.array 'gen.random 'gen.choicemap
         'gen.inference.weights 'gen.selection 'gen.trace 'gen.diff 'gen.array.batch)
(set! *unchecked-math* :warn-on-boxed)
(doseq [ns '[gen.llm.branching gen.llm.slots gen.llm.llama.recurrent gen.llm.llama gen.llm.lm gen.llm.potential]]
  (println "== loading" ns)
  (require ns))
(println :done)
