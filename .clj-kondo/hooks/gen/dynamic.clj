(ns hooks.gen.dynamic
  "gen: a fn of its parameters (so a call's arity is checked against them),
   its body in the scope of the names it binds itself (trace, splice,
   param, untraced, untraced-key), as the macro's body fn has them
   (gen.dynamic/gen). They are an inner fn's parameters, not a let's, so a
   body that is a let is not a redundant one."
  (:require [hooks.gen.util :refer [used]]))

(def ^:private own-names '[trace splice param untraced untraced-key])

(defmacro gen [params & body]
  `(fn [~@params]
     ((fn [~@(map used own-names)] ~@body) ~@(map (constantly nil) own-names))))
