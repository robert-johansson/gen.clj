(ns hooks.gen.gfi
  "defgf: a defrecord (its IFn methods left out), the :call function, when
   given, referenced.")

(defmacro defgf [name fields & impls]
  (let [[call impls] (if (= :call (first impls)) [(second impls) (nnext impls)] [nil impls])]
    `(do ~@(when call [call])
         (defrecord ~name ~fields ~@impls))))
