(ns gen.inference.cache-test
  "The compiled-code cache that mcmc's compiled steps and the device's
   programs share (genclj-2j7v): bounded, the least recently used out
   first. Planted bugs (2026-11-06): no eviction fails the bound; eviction
   by age of storing, not of use, fails the second."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.inference.cache :as cache]))

(deftest the-cache-is-bounded-and-keeps-what-is-used
  (cache/clear!)
  (try
    (dotimes [i 64] (cache/store! [::k i] i))
    (is (= 64 (cache/size)))
    (testing "past the bound, the least recently used goes"
      (is (= 0 (cache/lookup [::k 0])) "entry 0 used: entry 1 is now the least recent")
      (cache/store! [::k 64] 64)
      (is (= 64 (cache/size)))
      (is (nil? (cache/lookup [::k 1])))
      (is (= 0 (cache/lookup [::k 0])))
      (is (= 64 (cache/lookup [::k 64]))))
    (testing "cached makes a value once"
      (let [made (atom 0)
            make #(do (swap! made inc) :v)]
        (is (= :v (cache/cached [::made] make)))
        (is (= :v (cache/cached [::made] make)))
        (is (= 1 @made))))
    (finally (cache/clear!))))
