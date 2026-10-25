(ns gen.book.part4-test
  "Part IV, The System: its chapters run, their ;=> values and figures
   checked (gen.book). ^:slow: minutes on the JVM backend."
  (:require [clojure.test :refer [deftest]]
            [gen.book :as book]))

(deftest ^:slow system-runs
  (book/check-chapters (book/part-chapters "04")))
