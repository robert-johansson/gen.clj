(ns gen.book.part2-test
  "Part II, Inference: its chapters run, their ;=> values and figures
   checked (gen.book). ^:slow: minutes on the JVM backend."
  (:require [clojure.test :refer [deftest]]
            [gen.book :as book]))

(deftest ^:slow inference-runs
  (book/check-chapters (book/part-chapters "02")))
