;; Load layer I3's namespaces (dependencies first, so their warnings are
;; not counted) and print reflection warnings from the layer's files.
(require 'gen.array 'gen.dynamic 'gen.batched 'gen.combinators 'gen.custom)
(println "=== loading layer I3")
(require 'gen.dist.core 'gen.dist 'gen.dist.dsl 'gen.dist.host 'gen.dist.commons-math
         'gen.inference.weights 'gen.inference.importance 'gen.inference.mh 'gen.inference.kernel
         'gen.inference.mcmc 'gen.inference.particle-filter 'gen.inference.proposal
         'gen.inference.enumerative 'gen.inference.transform 'gen.inference.trace-kernel
         'gen.inference.vi 'gen.gradients 'gen.learning)
(binding [*warn-on-reflection* true] (doseq [n (quote [gen.dist.core gen.dist gen.dist.dsl gen.dist.host gen.dist.commons-math gen.inference.weights gen.inference.importance gen.inference.mh gen.inference.kernel gen.inference.mcmc gen.inference.particle-filter gen.inference.proposal gen.inference.enumerative gen.inference.transform gen.inference.trace-kernel gen.inference.vi gen.gradients gen.learning])] (require n :reload)))
(println "=== done")
