(ns gen.llm.sample-test
  "Sampling in lockstep (gen.llm.sample) on a toy engine: pick keeps to the
   top-k tokens and the top-p mass; complete is a function of its key,
   honours each completion's :max-tokens and end tokens, reports each
   completion's log-probability as the LM scores its tokens (so its rows
   are the completion's own: planted, a completion reading another's row
   fails it), and refuses loudly, before reading, completions that do not
   fit the LM's cells or branches."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.llm.byte-tokenizer :as bt]
            [gen.llm.lm :as lm]
            [gen.llm.sample :as sample]
            [gen.llm.toy-engine :as toy]
            [gen.random :as rng]))

(def ^:private tok (bt/tokenizer (mapv #(byte-array [(unchecked-byte %)]) (range 256))))

(defn- with-lm [opts f]
  (with-open [e (toy/engine {:n-vocab (:n-vocab tok) :max-branches 8 :n-ctx 512})
              m (lm/lm e tok opts)]
    (f m)))

(deftest pick-keeps-to-top-k-and-top-p
  (let [lp (float-array (map #(Math/log %) [0.5 0.3 0.1 0.05 0.05]))]
    (is (= #{0 1 2 3} (set (map #(sample/pick lp {:temp 1.0 :top-k 20 :top-p 0.95} %) (map #(/ % 100.0) (range 100))))))
    (is (= #{0 1} (set (map #(sample/pick lp {:temp 1.0 :top-k 2 :top-p 1.0} %) (map #(/ % 100.0) (range 100))))))
    (is (= 0 (sample/pick lp {:temp 1.0 :top-k 20 :top-p 0.95} 0.52)))
    (is (= 1 (sample/pick lp {:temp 1.0 :top-k 20 :top-p 0.95} 0.54)))
    (is (thrown? clojure.lang.ExceptionInfo (sample/pick lp {:temp 0 :top-k 20 :top-p 0.9} 0.5)))))

(deftest complete-in-lockstep
  (with-lm {:max-branches 8}
    (fn [m]
      (let [prompt (lm/encode m "Once upon a time" {:add-special? false})
            ctxs [prompt prompt (lm/encode m "Elsewhere" {:add-special? false})]
            opts {:max-tokens [5 7 9] :key (rng/key 1) :logprobs? true :sampling {:temp 1.0 :top-k 50 :top-p 0.95}}
            r (sample/complete m ctxs opts)]
        (testing "a function of its key"
          (is (= r (sample/complete m ctxs opts)))
          (is (not= (map :tokens r) (map :tokens (sample/complete m ctxs (assoc opts :key (rng/key 2)))))))
        (testing "each completion's own budget, or its end"
          (doseq [[c n] (map vector r [5 7 9])]
            (is (case (:stop c) :max (= n (count (:tokens c))) :end (< (count (:tokens c)) n)))))
        (testing "each completion's log-probability is the LM's score of its tokens (and its end token)"
          (doseq [[ctx c] (map vector ctxs r)]
            (let [scored (into (:tokens c) (when (= :end (:stop c)) [(:end-token c)]))]
              (is (< (Math/abs (- (:logprob c) (reduce + (lm/score m ctx scored)))) 1e-9)))))))))

(deftest end-tokens-end-a-completion
  (with-lm {:max-branches 8}
    (fn [m]
      (let [ctx (lm/encode m "abc" {:add-special? false})
            ;; every token but a few ends it: the first draw ends it, unless it is one of them
            r (first (sample/complete m [ctx] {:max-tokens 20 :end-tokens (set (range 0 250)) :key (rng/key 3)
                                               :sampling {:temp 1.0 :top-k 300 :top-p 1.0}}))]
        (is (or (= :end (:stop r)) (every? #(>= % 250) (:tokens r))))
        (when (= :end (:stop r)) (is (< (:end-token r) 250)))))))

(deftest completions-that-do-not-fit-are-refused
  (with-lm {:max-branches 4 :max-cells 100}
    (fn [m]
      (let [ctx (lm/encode m (apply str (repeat 30 "x")) {:add-special? false})]
        (testing "more completions than branches"
          (is (thrown-with-msg? clojure.lang.ExceptionInfo #"at most 4 branches"
                                (sample/complete m (vec (repeat 5 ctx)) {:max-tokens 2 :key (rng/key 1)}))))
        (testing "contexts and completions past the cells (a shared prefix counted once)"
          (is (some? (sample/complete m [ctx ctx] {:max-tokens 10 :key (rng/key 1)})) "30 + 2 × 10 fits 100")
          (is (thrown-with-msg? clojure.lang.ExceptionInfo #"need 110 cells"
                                (sample/complete m [ctx ctx] {:max-tokens 40 :key (rng/key 1)}))))))))
