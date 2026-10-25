(ns gen.book.part3-test
  "Part III, Theory: its chapters run, their ;=> values and figures
   checked (gen.book). ^:slow: minutes on the JVM backend."
  (:require [clojure.test :refer [deftest]]
            [gen.book :as book]))

(deftest ^:slow theory-runs
  (book/check-chapters (book/part-chapters "03")))
