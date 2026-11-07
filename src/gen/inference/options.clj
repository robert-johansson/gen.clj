(ns gen.inference.options
  "The inference functions' options maps, checked: a function refuses a
   key it does not read, which it would otherwise ignore, so a misspelt
   option ({:burnin 50} for :burn) cannot run with its default unnoticed.
   One map shared by functions that read different keys is refused too:
   give each its own keys, (select-keys opts ks), ks the keys the error
   names as :accepted (DESIGN §9, genclj-vux3). gen.learning and
   gen.gradients, in inference's layer, check theirs here too
   (genclj-yrb6)."
  {:api :internal}
  (:require [clojure.string :as str]))

(defn check
  "opts, an options map or nil, for the function who (its qualified
   symbol): loud for anything else, and for a key outside accepted (a set
   of keywords), the error's data naming the function (:fn), the keys it
   refused (:unknown, in opts' order) and the ones it takes (:accepted,
   sorted)."
  [who opts accepted]
  (when-not (or (nil? opts) (map? opts))
    (throw (ex-info (str who " takes an options map, not " (pr-str opts))
                    {:reason :invalid-options :fn who :value opts})))
  (when-let [unknown (seq (remove accepted (keys opts)))]
    (let [ks (vec (sort accepted))]
      (throw (ex-info (str who ": unknown options " (vec unknown) "; the options are " (str/join " " ks))
                      {:reason :unknown-options :fn who :unknown (vec unknown) :accepted ks}))))
  opts)
