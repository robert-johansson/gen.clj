(ns gen.trace-test
  "How a trace prints (genclj-qvnu): as its maker, then its arguments,
   score and choices, not every field of its record (a 500-step Unfold's
   trace printed its 500 subtraces); a batched trace as its maker,
   arguments and batch shape. The printed form is for reading:
   gen.serialize, which reads #gen/trace as write-trace writes it, refuses
   it loudly."
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [gen.batched :as batched]
            [gen.choicemap :as cm]
            [gen.combinators :as comb]
            [gen.dist :as d]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.random :as rng]
            [gen.serialize :as serialize]
            [gen.trace :as trace]))

(def ^:private unit (gen [] (trace :u (d/uniform 0 1))))

(def ^:private sensor (gen [mu] (trace :value (d/normal mu 0.5))))

(def ^:private fusion
  (gen [position]
    [(splice :a sensor position) (splice :b sensor position)]))

(defn- printed
  "A trace's printed form, from its maker and the map it shows, each
   printed as itself: what the print method writes, on every backend."
  [maker m]
  (str "#gen/trace[" (pr-str maker) " " (pr-str m) "]"))

;; Planted (2026-10-25): without the print method, every test here fails;
;; one that prints the trace's :gen-fn in place of its maker fails
;; "the maker".
(deftest a-trace-prints-its-maker-arguments-score-and-choices
  (testing "a dynamic trace"
    (let [tr (:trace (gfi/generate unit [] (cm/choicemap :u 0.25) (rng/key 1)))]
      (is (= (printed unit (array-map :args [] :score (:score tr) :choices {:u 0.25})) (pr-str tr)))))
  (testing "its calls' traces are not printed, only their choices"
    (let [tr (gfi/simulate fusion [5.0] (rng/key 2))]
      (is (= (printed fusion (array-map :args [5.0] :score (:score tr) :choices (cm/to-map (:choices tr))))
             (pr-str tr)))
      (is (= 1 (count (re-seq #"#gen/trace" (pr-str tr)))))))
  (testing "a distribution's trace, whose choices are one value"
    (let [tr (gfi/simulate (d/normal 0 1) [] (rng/key 3))]
      (is (= (printed (d/normal 0 1) (array-map :args [] :score (:score tr) :choices (tr [])))
             (pr-str tr)))))
  (testing "the maker, not a :gen-fn given since"
    (let [tr (gfi/simulate unit [] (rng/key 4))]
      (is (str/starts-with? (pr-str (assoc tr :gen-fn sensor)) "#gen/trace[#gen/fn[] ")))))

(deftest a-combinator-trace-prints-its-choices-not-its-elements-traces
  (let [walk (comb/unfold (gen [_t x] (trace :x (d/normal x 1))))
        tr (gfi/simulate walk [500 0.0] (rng/key 5))
        s (pr-str tr)]
    (is (str/starts-with? s "#gen/trace[#gen/unfold[#gen/fn[_t x]] {:args [500 0.0], :score "))
    (is (= 1 (count (re-seq #"#gen/trace" s))) "500 steps, one trace printed")
    (is (= 500 (count (re-seq #"\{:x " s))) "and every step's choice")
    (is (= (printed (trace/maker tr) (array-map :args [500 0.0] :score (:score tr) :choices (cm/to-map (:choices tr))))
           s))))

(deftest a-batched-trace-prints-its-batch-shape
  ;; its values hold every particle's: the shape says how many
  (let [ps (batched/simulate fusion [5.0] 100 (rng/key 6))]
    (is (batched/batched? ps))
    (is (= "#gen/trace[#gen/fn[position] {:args [5.0], :batch-shape [100]}]" (pr-str (:trace ps))))))

(deftest the-printed-form-is-for-reading
  (let [tr (gfi/simulate unit [] (rng/key 7))]
    (is (= (str "[" (pr-str tr) "]") (pr-str [tr])) "inside other data")
    (is (thrown? clojure.lang.ExceptionInfo (serialize/read (pr-str tr))))
    (is (thrown? clojure.lang.ExceptionInfo (serialize/read-trace unit (pr-str tr))))
    (let [parts (juxt :args :choices :retval :score)]
      (is (= (parts tr) (parts (serialize/read-trace unit (serialize/write-trace tr))))
          "gen.serialize is the round trip"))))
