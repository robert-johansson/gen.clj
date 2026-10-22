(ns gen.llm.host-test
  "The LLM layer's host pools (gen.llm.host), which the llama.cpp engine's
   rows and token SMC's particles share the code of (the post-M5 review's
   I4-04: each had a copy): the default size, the pools, and the ordered
   map, whose results and errors are the calling thread's.

   Mutations checked (2026-10-22, genclj-egmg), with the tests that caught
   them: the results in the order the calls end, not their inputs'
   (in-order-is-mapv-on-a-pool); an error wrapped in its
   ExecutionException (a-call-s-error-is-raised-as-itself); no bindings on
   the pool (a-call-sees-the-calling-thread-s-bindings)."
  (:require [clojure.test :refer [deftest is]]
            [gen.llm.branching-contract :as bc]
            [gen.llm.host :as host])
  (:import [java.util.concurrent ExecutorService ThreadPoolExecutor]))

(deftest the-default-size-is-a-positive-integer-or-loud
  (let [threads #'host/threads]
    (is (= 4 (threads nil 10)) "half the processors")
    (is (= 4 (threads nil 32)) "up to 4")
    (is (= 1 (threads nil 1)))
    (is (= 3 (threads "3" 10)))
    (is (= 1 (threads "1" 10)) "the off switch")
    (doseq [bad ["four" "4 " "" "0" "-2" "2.5"]]
      (is (= :invalid-option (bc/reason #(threads bad 10))) (pr-str bad))))
  (is (pos? (host/default-threads))))

(deftest a-pool-has-daemon-threads-named-for-it
  (with-open [^ExecutorService pool (host/pool "gen.llm.host-test-" 2)]
    (let [threads (host/in-order pool (fn [_] (Thread/currentThread)) (range 8))]
      (is (every? #(.isDaemon ^Thread %) threads))
      (is (every? #(.startsWith (.getName ^Thread %) "gen.llm.host-test-") threads))
      (is (= 2 (.getMaximumPoolSize ^ThreadPoolExecutor pool)))))
  (doseq [n [0 -1 2.0 nil]]
    (is (= :invalid-args (bc/reason #(host/pool "p" n))) (pr-str n))))

(deftest in-order-is-mapv-on-a-pool
  ;; later inputs finish first, the results are still in the inputs' order
  (with-open [^ExecutorService pool (host/pool "gen.llm.host-test-" 4)]
    (let [f (fn [i] (Thread/sleep (long (- 40 (* 5 i)))) (* i i))]
      (is (= (mapv f (range 8)) (host/in-order pool f (range 8))))
      (is (pos? (.getCompletedTaskCount ^ThreadPoolExecutor pool)) "the pool ran them")))
  (let [caller (Thread/currentThread)]
    (is (= [caller caller] (host/in-order false (fn [_] (Thread/currentThread)) [1 2])) "no pool: the calling thread")
    (with-open [^ExecutorService pool (host/pool "gen.llm.host-test-" 2)]
      (is (= [caller] (host/in-order pool (fn [_] (Thread/currentThread)) [1])) "one call: the calling thread"))))

(deftest a-call-s-error-is-raised-as-itself
  (with-open [^ExecutorService pool (host/pool "gen.llm.host-test-" 2)]
    (is (= :planted (bc/reason #(host/in-order pool (fn [i] (if (= i 3) (throw (ex-info "planted" {:reason :planted})) i))
                                               (range 6)))))))

(def ^:dynamic *bound* :root)

(deftest a-call-sees-the-calling-thread-s-bindings
  (with-open [^ExecutorService pool (host/pool "gen.llm.host-test-" 2)]
    (binding [*bound* :caller]
      (is (= [:caller :caller :caller] (host/in-order pool (fn [_] *bound*) (range 3)))))))
