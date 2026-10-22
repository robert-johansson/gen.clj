;; I4 (checked and sound): load every gen.llm namespace, printing any reflection or boxed-math warning.
(doseq [ns '[gen.llm.branching gen.llm.slots gen.llm.llama.recurrent gen.llm.llama gen.llm.cells gen.llm.trie
             gen.llm.cache gen.llm.lm gen.llm.dfa gen.llm.regex gen.llm.bytes gen.llm.potential gen.llm.grammar
             gen.llm.constrained gen.llm.token gen.llm.smc]]
  (println "loading" ns)
  (require ns))
(println :done)
