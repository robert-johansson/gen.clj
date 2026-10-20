(ns gen.llm.branching-test
  "The Branching protocol and gen.llm.slots on the pure-JVM toy engine
   (gen.llm.toy-engine), whose distributions are exact functions of the
   token sequence: the contract (gen.llm.branching-contract) holds with
   band 0, and each planted bug in the toy engine fails the check meant to
   catch it.

   Mutations of gen.llm.slots checked by hand (2026-10-20, each applied,
   the contract run, and reverted), with the checks that caught them:
   - resample giving every child its parent's slot, no copies: resample
     (the toy engine refuses a step whose slots repeat);
   - resample giving child j parent j's state: resample (a step away from
     the end of its slot);
   - freeing a slot without clearing it: every check (a run into a slot
     that is not empty);
   - advance not consuming its branches: resample (the live count),
     misuse, gc-frees;
   - score reading the continuation into the branch's own slot:
     score-matches-steps (the branch was advanced);
   - no GC when slots run short: gc-frees."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.llm.branching :as br]
            [gen.llm.branching-contract :as contract]
            [gen.llm.slots :as slots]
            [gen.llm.toy-engine :as toy]))

(def ^:private opts
  {:prompt [3 1 4 1 5 9 2 6] :continuation [5 3 5 8 9 7] :k 4 :band 0.0 :top 97})

(deftest the-contract-holds-exactly
  (with-open [e (toy/engine {})]
    (doseq [[k check] contract/checks]
      (testing k
        (is (= [] (contract/run-check check e opts)))))
    (is (zero? (:live (br/info e))) "every check disposes what it makes")))

(deftest planted-bugs-fail-their-checks
  ;; the power of the contract: each mutation of a Slots operation is caught
  (doseq [[bug caught-by] {:copy-perturbs [:fork-matches-fresh :resample]
                           :step-rotates [:fork-matches-fresh :batch-matches-single :resample]
                           :step-ignores [:fork-matches-fresh :score-matches-steps]
                           :run-shifts [:score-matches-steps]
                           :run-stale [:fork-matches-fresh :score-matches-steps]}]
    (with-open [e (toy/engine {:bug bug})]
      (doseq [k caught-by]
        (testing [bug k]
          (is (seq (contract/run-check (contract/checks k) e opts))))))))

(deftest a-branch-is-a-value
  (with-open [e (toy/engine {})]
    (let [b (br/open e [1 2 3])
          lp (br/logprobs b)]
      (aset lp 0 (float 99))
      (is (not= 99.0 (br/logprob b 0)) "logprobs is a copy")
      (is (= [1 2 3] (br/tokens b)))
      (is (= 3 (br/position b)))
      (is (identical? e (br/engine b)))
      (let [[b2] (br/advance [b] [7])]
        (is (= [1 2 3] (br/tokens b)) "a consumed branch keeps its value")
        (is (= (vec (toy/toy-logprobs 97 [1 2 3])) (vec (br/logprobs b))))
        (is (not (br/live? b)))
        (is (= [1 2 3 7] (br/tokens b2)))
        (is (= :invalid-token (:reason (ex-data (try (br/logprob b2 97) (catch Exception x x))))))
        (br/dispose! b2)))))

(deftest a-closed-engine-refuses-everything
  (let [e (toy/engine {})
        b (br/open e [1 2])]
    (.close e)
    (.close e)
    (is (:closed? (br/info e)))
    (is (not (br/live? b)))
    (doseq [f [#(br/open e [1]) #(br/fork b) #(br/advance [b] [1]) #(br/score b [1 2]) #(br/dispose! b)
               #(slots/call e (fn [_] :ran))]]
      (is (= :closed (:reason (ex-data (try (f) (catch Exception x x)))))))
    (is (= [1 2] (br/tokens b)) "its branches keep their values")))

(deftest branches-of-two-engines-do-not-mix
  (with-open [e1 (toy/engine {}) e2 (toy/engine {})]
    (let [a (br/open e1 [1]) b (br/open e2 [1])]
      (is (= :engines-differ (:reason (ex-data (try (br/advance [a b] [1 1]) (catch Exception x x))))))
      (is (= :engines-differ (:reason (ex-data (try (br/resample [a b] [0]) (catch Exception x x))))))
      (is (and (br/live? a) (br/live? b)))
      (br/dispose! a)
      (br/dispose! b))))

(deftest an-engine-calls-itself-without-deadlock
  ;; a call made on the engine's thread runs there at once
  (with-open [e (toy/engine {})]
    (let [b (slots/call e (fn [_] (br/open e [1 2 3])))]
      (is (= [1 2 3] (br/tokens b)))
      (br/dispose! b))))

(deftest a-failed-construction-is-loud-and-leaks-no-thread
  (let [before (count (filter #(.startsWith (.getName ^Thread %) "gen.llm") (.keySet (Thread/getAllStackTraces))))]
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"no model"
                          (slots/engine #(throw (ex-info "no model" {})) "failing")))
    (Thread/sleep 50)
    (is (= before (count (filter #(.startsWith (.getName ^Thread %) "gen.llm")
                                 (.keySet (Thread/getAllStackTraces))))))))
