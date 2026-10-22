;; Load layer I1's JVM-side namespaces with reflection and boxed-math warnings
;; on, so that files that do not set the flags themselves (gen.array.dtype)
;; report too. Files that set only *warn-on-reflection* inherit :warn-on-boxed
;; from here; their boxed warnings are informative, not a rule.
(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)
(println "=== loading gen.array.dtype (sets no flags of its own)")
(require 'gen.array.dtype)
(println "=== loading the rest")
(require 'gen.array 'gen.array.batch 'gen.array.tape 'gen.array.linalg 'gen.array.memory
         'gen.random 'gen.backend.mlx.exact)
(println "=== done")
