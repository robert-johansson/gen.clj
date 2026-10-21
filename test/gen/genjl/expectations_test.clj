(ns gen.genjl.expectations-test
  "gen.genjl.expectations, the ports' expectation maps."
  (:require [clojure.test :refer [deftest is]]
            [gen.diff :as diff]
            [gen.genjl.expectations :as expectations]))

(deftest each-entry-is-checked-by-its-key
  (let [calls (atom [])
        checks {:a (fn [shown want] (swap! calls conj [:a shown want]))
                :b (fn [shown want] (swap! calls conj [:b shown want]))}]
    (expectations/check checks :result {:a 1 :b 2})
    (is (= #{[:a :result 1] [:b :result 2]} (set @calls)))
    (reset! calls [])
    (expectations/check checks :result {:b 3})
    (is (= [[:b :result 3]] @calls) "a key not given is not checked")))

(deftest an-unknown-key-is-loud
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"no assertion is named :scroe"
                        (expectations/check {:score (fn [_ _])} :result {:scroe 1.0}))))

(deftest diff-kinds
  (is (= :no-change (expectations/diff-kind diff/no-change)))
  (is (= :unknown-change (expectations/diff-kind diff/unknown-change)))
  (is (= :vector-diff (expectations/diff-kind (diff/vector-diff 2 2 {}))))
  (is (= (diff/int-diff 3) (expectations/diff-kind (diff/int-diff 3))) "another diff is itself"))
