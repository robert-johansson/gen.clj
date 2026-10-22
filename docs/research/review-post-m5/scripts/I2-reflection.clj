(doseq [n '[gen.choicemap gen.selection gen.trace gen.diff gen.gfi gen.dynamic.core gen.dynamic.static gen.dynamic gen.batched gen.custom gen.combinators]]
  (require n :reload))
(println "loaded")
