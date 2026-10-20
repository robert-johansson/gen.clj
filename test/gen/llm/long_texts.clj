(ns gen.llm.long-texts
  "Long, varied, deterministic texts (several thousand tokens each) for
   branches longer than a sliding window: gen.llm.llama-test and
   bench/gen/llama_long_noise.clj, which measured its band on them."
  (:require [clojure.string :as str]))

(def texts
  "Four long texts, each of several thousand tokens, by name."
  {:list (str/join (for [i (range 220)]
                     (str " Item " i " of the list weighs " (* 7 i) " grams and costs " (mod (* 13 i) 97) " crowns.")))
   :prose (let [ss ["The river turned north past the old mill." "Nobody in the village remembered who built it."
                    "In winter the ice reached the far bank." "Children crossed it on their way to school."
                    "The miller's daughter kept the accounts in a green book." "Prices rose every year after the war."
                    "A storm in March took the roof off the barn." "They rebuilt it with timber from the hills."]]
            (str/join " " (for [i (range 400)] (ss (mod (* 5 i) (count ss))))))
   :code (str/join "\n" (for [i (range 260)]
                          (str "def f" i "(x, y):\n    return x * " i " + y // " (inc (mod (* 31 i) 17)))))
   :numbers (str/join ", " (for [i (range 900)] (mod (* i i 7919) 10007)))})

