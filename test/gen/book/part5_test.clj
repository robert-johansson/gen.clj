(ns gen.book.part5-test
  "Part V, Applications: its chapters run, their ;=> values and figures
   checked (gen.book). ^:slow: minutes on the JVM backend."
  (:require [clojure.test :refer [deftest]]
            [gen.book :as book]))

(deftest ^:slow applications-runs
  (book/check-chapters (book/part-chapters "05")))
