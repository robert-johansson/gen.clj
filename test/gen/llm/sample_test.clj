(ns gen.llm.sample-test
  "Sampling in lockstep (gen.llm.sample) on a toy engine: pick keeps to the
   top-k tokens and the top-p mass, and draws from the tempered
   distribution at temperatures 0.5 and 2; complete is a function of its key,
   honours each completion's :max-tokens and end tokens, reports each
   completion's log-probability as the LM scores its tokens (so its rows
   are the completion's own: planted, a completion reading another's row
   fails it), and refuses loudly, before reading, completions that do not
   fit the LM's cells or branches."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
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

(def ^:private jvm (array/backend-named :jvm))

(deftest pick-draws-from-the-tempered-distribution
  ;; genclj-w867: at temperature T (top-k and top-p wide open), pick draws
  ;; token t with probability p(t)^(1/T) / Σ p^(1/T). Each token's frequency
  ;; over 20,000 uniforms from a fixed key is within 5 standard errors,
  ;; sqrt(p (1 - p) / n), of it. Planted (2026-11-06): the division by the
  ;; temperature dropped fails both temperatures, on every token.
  (let [ps [0.5 0.3 0.1 0.05 0.05]
        lp (float-array (map #(Math/log %) ps))
        n 20000
        us (array/->clj (rng/uniform (rng/key 12) [n] {:backend jvm :dtype :float64}))]
    (doseq [temp [0.5 2.0]]
      (let [ws (map #(Math/pow % (/ 1.0 temp)) ps)
            tempered (mapv #(/ % (reduce + ws)) ws)
            counts (frequencies (map #(sample/pick lp {:temp temp :top-k 20 :top-p 1.0} (double %)) us))]
        (doseq [t (range (count ps))]
          (let [p (double (tempered t))
                f (/ (double (get counts t 0)) n)
                band (* 5 (Math/sqrt (/ (* p (- 1.0 p)) n)))]
            (is (<= (Math/abs (- f p)) band)
                (str "temperature " temp ", token " t ": frequency " f ", tempered " p ", band " band))))))))

(deftest pick-draws-from-the-whole-distribution
  ;; a :top-k of nil (gen.llm.grpo's policy sampling): every token, at
  ;; temperatures 1 and 0.5, each token's frequency over 20,000 uniforms
  ;; within 5 standard errors of p(t)^(1/T) / Σ p^(1/T). Planted
  ;; (2026-11-07): the temperature dropped fails at 0.5; the inverse CDF
  ;; read one token late fails at both.
  (let [ps [0.4 0.0 0.3 0.1 0.15 0.05]
        lp (float-array (map #(Math/log %) ps))
        n 20000
        us (array/->clj (rng/uniform (rng/key 13) [n] {:backend jvm :dtype :float64}))]
    (doseq [temp [1.0 0.5]]
      (let [ws (map #(Math/pow % (/ 1.0 temp)) ps)
            tempered (mapv #(/ % (reduce + ws)) ws)
            counts (frequencies (map #(sample/pick lp {:temp temp :top-k nil :top-p 1.0} (double %)) us))]
        (is (zero? (get counts 1 0)) "a token of probability 0 is never drawn")
        (doseq [t (range (count ps))]
          (let [p (double (tempered t))
                f (/ (double (get counts t 0)) n)
                band (* 5 (Math/sqrt (/ (* p (- 1.0 p)) n)))]
            (is (<= (Math/abs (- f p)) band)
                (str "temperature " temp ", token " t ": frequency " f ", tempered " p ", band " band))))))
    (is (= 5 (sample/pick lp {:temp 1.0 :top-k nil :top-p 1.0} 0.9999999)) "the last token with mass at the end")
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"top-p of 1" (sample/pick lp {:temp 1.0 :top-k nil :top-p 0.9} 0.5))
        "every token takes no nucleus")))

(deftest tempered-is-the-tempered-log-probability
  (let [ps [0.5 0.3 0.2]
        lp (float-array (map #(Math/log %) ps))
        z (reduce + (map #(Math/pow % 2.0) ps))]
    (is (= (double (aget lp 1)) (sample/tempered lp 1.0 1)))
    (is (< (Math/abs (- (sample/tempered lp 0.5 1) (Math/log (/ (Math/pow 0.3 2.0) z)))) 1e-6))))

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
              (is (< (Math/abs (- (:logprob c) (reduce + (lm/score m ctx scored)))) 1e-9)))))
        (testing "each token's log-probability at the sampling temperature, the end token's last"
          (doseq [temp [1.0 0.7]]
            (let [opts (assoc opts :token-logprobs? true :sampling {:temp temp :top-k 50 :top-p 0.95})
                  r' (sample/complete m ctxs opts)]
              (is (= (map :tokens r') (map :tokens (sample/complete m ctxs (dissoc opts :token-logprobs?))))
                  "the same draws")
              (doseq [[ctx c] (map vector ctxs r')]
                (let [scored (into (:tokens c) (when (= :end (:stop c)) [(:end-token c)]))
                      want (map-indexed (fn [k t] (sample/tempered (lm/next-logprobs m (into ctx (subvec scored 0 k))) temp t))
                                        scored)]
                  (is (= (count scored) (count (:token-logprobs c))))
                  (is (every? #(< (Math/abs (double %)) 1e-9) (map - want (:token-logprobs c)))
                      (str "temperature " temp)))))))))))

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
