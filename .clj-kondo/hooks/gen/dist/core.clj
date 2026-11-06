(ns hooks.gen.dist.core
  "defdist: the constructor, a defn of the parameters whose body is the
   validate clause and whose value is the Distribution record, and each
   other clause a fn of its arguments with the parameters in scope (as the
   macro binds them all in every clause, used or not). The constructor's
   value is the record's, not a type clj-kondo infers: in a dist body
   (gen.dist.dsl) a constructor's call stands for its draw."
  (:require [hooks.gen.util :refer [used]]))

(defmacro defdist [dist-name & body]
  (let [[doc [params & clauses]] (if (string? (first body)) [(first body) (rest body)] [nil body])
        validate (for [[head & forms] clauses :when (= 'validate head) form forms] form)]
    `(do (defn ~dist-name ~@(when doc [doc]) ~params
           ~@validate
           (gen.dist.core/->Distribution nil ~(into {} (map (fn [p] [(keyword p) p])) params)))
         ~@(for [[head args & forms] clauses :when (not= 'validate head)]
             `(fn [~@(map used params) ~@args] ~@forms)))))
