(ns hooks.gen.laws
  "gen.laws' deflaws (and gen.laws.potential's) and defspecs: the specs'
   names come from the law set, evaluated when the macro expands, so the
   hooks define none; the arguments are analyzed as the expressions they
   are.")

(defmacro deflaws
  ([_prefix models opts] `(do ~models ~opts))
  ([_prefix models opts laws] `(do ~models ~opts ~laws)))

(defmacro defspecs [_prefix cases-form opts laws]
  `(do ~cases-form ~opts ~laws))
