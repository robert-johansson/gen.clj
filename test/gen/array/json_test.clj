(ns gen.array.json-test
  "gen.array.json, the host's JSON reader (safetensors headers, their
   indexes, config.json): values, escapes, numbers, and the loud errors
   of a strict reader. Planted mutations, each failing a test here before
   it was trusted (2026-11-08): a \\u escape's digits read in reverse; a
   key twice keeping the last value; a trailing comma accepted; an
   integer past a long's range read as a double."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array.json :as json]))

(defn- reason [s]
  (try (json/read-str s) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(deftest values
  (is (= {"a" [1 -2500.0 true false nil "x"] "b" {} "c" [] "d" {"e" [[]]}}
         (json/read-str " {\"a\": [1, -2.5e3, true, false, null, \"x\"], \"b\": {}, \"c\": [], \"d\": {\"e\": [[]]}}\n")))
  (testing "numbers: integers as longs, past a long's range as BigInts, the rest doubles"
    (is (= [0 -7 9223372036854775807 123456789012345678901234N 0.5 1.0E-6 -0.0 1.0E300]
           (json/read-str "[0, -7, 9223372036854775807, 123456789012345678901234, 0.5, 1e-06, -0.0, 1E+300]")))
    (is (instance? Long (json/read-str "10000000")))
    (is (instance? clojure.lang.BigInt (json/read-str "-9223372036854775809"))))
  (testing "NaN and the infinities, as Python's json module writes them"
    (let [[n i j] (json/read-str "[NaN, Infinity, -Infinity]")]
      (is (Double/isNaN n))
      (is (= [Double/POSITIVE_INFINITY Double/NEGATIVE_INFINITY] [i j]))))
  (testing "strings: every escape, \\u's surrogate pairs, UTF-8"
    (is (= "\"\\/\b\f\n\r\tAé😀ö" (json/read-str "\"\\\"\\\\\\/\\b\\f\\n\\r\\t\\u0041\\u00E9\\ud83d\\ude00ö\"")))
    (is (= "Stockholm är" (json/read-bytes (.getBytes "\"Stockholm är\"" "UTF-8"))))))

(deftest strict
  (doseq [s ["{\"a\": 1,}" "[1,]" "{\"a\": 1, \"a\": 2}" "01" "-" "1." "1e" ".5" "[1] 2" "" "  " "{a: 1}" "'a'"
             "\"a\tb\"" "\"abc" "\"\\x\"" "\"\\u12\"" "\"\\u12G4\"" "[1 2]" "{\"a\" 1}" "tru" "nul" "// c\n1" "[" "{"]]
    (is (= :invalid-json (reason s)) (pr-str s))))

(deftest files
  (is (= :no-file (:reason (try (json/read-file "test/fixtures/no-such.json") (catch clojure.lang.ExceptionInfo e (ex-data e))))))
  (is (= "qwen3" (get (json/read-file "test/fixtures/safetensors/qwen3-tiny/config.json") "model_type"))))
