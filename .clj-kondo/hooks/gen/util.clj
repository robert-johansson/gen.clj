(ns hooks.gen.util
  "Shared by the hooks: a binding a macro makes that its body need not read.")

(defn used
  "Symbol s as a binding clj-kondo counts as used: a name the macro binds
   for the body (gen's trace, a defdist parameter in a clause), which the
   body is free to ignore."
  [s]
  (vary-meta s assoc :clj-kondo/mark-used true))
