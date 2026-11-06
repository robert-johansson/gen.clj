(ns gen.llm.potential-combinators-test
  "Potential combinators (gen.llm.potential): temper, normalize, prompted,
   coerce and after, each under the potential contract's laws
   (gen.laws.potential), and each against what it must equal: tempered
   weights and kept zeros, a temperature on an LM's own completion, a
   normalized LM unchanged, an LM's rows and scores, a coerced byte
   grammar and the grammar over the same vocabulary, a soft byte potential
   summed along a token's bytes, a scope's weights before and after its
   close; misuse."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.laws.potential :as potential-laws]
            [gen.llm.branching-contract :as bc]
            [gen.llm.constrained :as constrained]
            [gen.llm.grammar :as grammar]
            [gen.llm.lm :as lm]
            [gen.llm.potential :as potential]
            [gen.llm.toy-engine :as toy])
  (:import (java.nio.charset StandardCharsets)))

(def ^:private n 6)

(defn- repeats? [ctx] (boolean (some (fn [[a b]] (= a b)) (partition 2 1 ctx))))

(def ^:private no-repeats
  "Sequences whose tokens never repeat the one before, complete at even
   lengths."
  (potential/potential n {:prefix (fn [ctx] (if (repeats? ctx) ##-Inf 0.0))
                          :complete (fn [ctx] (if (and (even? (count ctx)) (not (repeats? ctx))) 0.0 ##-Inf))}))

(defn- cost [ctx] (* -0.1 (reduce + 0 ctx)))

(def ^:private by-length
  "A soft potential: each token costs 0.1 per unit of its value."
  (potential/potential n {:prefix cost :complete (fn [ctx] (- (cost ctx) 1.0))}))

(def ^:private ended
  "no-repeats read with the end token 5."
  (potential/potential n {:prefix #(potential/prefix no-repeats %) :complete #(potential/complete no-repeats %)
                          :end-tokens #{5}}))

(def ^:private soft
  "The laws of a potential without zero weights."
  (dissoc potential-laws/all :a-zero-prefix-has-zero-extensions))

(defn- close-to? [^double a ^double b ^double tol]
  (or (= a b) (<= (Math/abs (- a b)) tol)))

;; ---------------------------------------------------------------------------
;; temper
;; ---------------------------------------------------------------------------

(potential-laws/deflaws tempered
  [[:no-repeats-squared (potential/temper no-repeats 2.0) n]
   [:indicator (potential/temper (potential/product no-repeats by-length) 0.0) n]]
  100)

(potential-laws/deflaws tempered-soft [[:by-length-inverse-root (potential/temper by-length -0.5) n]] 100 soft)

(deftest tempering-scales-weights-and-keeps-zeros
  (let [both (potential/product no-repeats by-length)
        p (potential/temper both 2.5)]
    (is (= (* 2.5 (potential/prefix both [1 2])) (potential/prefix p [1 2])))
    (is (= (* 2.5 (potential/complete both [1 2])) (potential/complete p [1 2])))
    (is (= (mapv #(unchecked-float (* 2.5 (double %))) (potential/logw-next both [1])) (vec (potential/logw-next p [1])))
        "the next weights, a zero among them")
    (is (= ##-Inf (potential/prefix p [3 3])))
    (is (= ##-Inf (potential/complete (potential/temper no-repeats -1.0) [1])) "a negative beta keeps a zero")
    (is (= 0.0 (potential/prefix (potential/temper by-length 0.0) [1 2 3])) "beta 0: the indicator")
    (is (= #{5} (potential/end-tokens (potential/temper ended 2.0))))))

;; ---------------------------------------------------------------------------
;; normalize
;; ---------------------------------------------------------------------------

(potential-laws/deflaws normalized
  [[:no-repeats (potential/normalize no-repeats) n]
   [:product (potential/normalize (potential/product no-repeats by-length)) n]
   [:ended (potential/normalize ended) n]]
  100)

(potential-laws/deflaws normalized-soft [[:by-length (potential/normalize by-length) n]] 100 soft)

(defn- log-sum-exp [xs]
  (let [m (reduce max ##-Inf xs)]
    (if (= ##-Inf m) m (+ m (Math/log (reduce + (map #(Math/exp (- (double %) m)) xs)))))))

(deftest normalized-next-weights-are-a-distribution
  (doseq [[what p skip] [[:product (potential/product no-repeats by-length) #{}] [:ended ended #{5}]]
          ctx [[] [1] [1 2] [0 3 1]]
          :let [w (vec (potential/logw-next (potential/normalize p) ctx))]]
    (is (close-to? 0.0 (log-sum-exp (keep-indexed (fn [t x] (when-not (skip t) x)) w)) 1e-6)
        (str what " at " ctx ": the tokens and the end, but the end tokens")))
  (testing "a prefix weight is the sum along the context"
    (let [q (potential/normalize by-length)
          along (reduce + (map (fn [i t] (aget ^floats (potential/logw-next q (subvec [2 0 4] 0 i)) (int t))) (range) [2 0 4]))]
      (is (close-to? along (potential/prefix q [2 0 4]) 1e-12))
      (is (= ##-Inf (potential/prefix (potential/normalize no-repeats) [1 1 2]))))))

;; ---------------------------------------------------------------------------
;; prompted: an LM as a potential (the toy engine's LM)
;; ---------------------------------------------------------------------------

(def ^:private lm-vocab 5)
(def ^:private end 4)

(def ^:private tokenizer
  {:n-vocab lm-vocab :bos nil :eos end :eog #{end} :add-bos? false
   :token-bytes (fn [t] (byte-array [(byte t)])) :encode (fn [_ _] []) :decode (fn [ts] (apply str ts))})

(def ^:private m (lm/lm (toy/engine {:n-vocab lm-vocab :max-branches 128 :n-ctx 64}) tokenizer {:max-cells 8192}))
(def ^:private prompt [0 1])

(potential-laws/deflaws lm-weights
  [[:prompted (potential/prompted m prompt) lm-vocab {:max-length 5}]
   [:two-end-tokens (potential/prompted m prompt {:end-tokens #{3 4}}) lm-vocab {:max-length 5}]
   [:two-prompts (potential/product (potential/prompted m prompt) (potential/prompted m [2 3])) lm-vocab {:max-length 5}]]
  100 soft)

(defn- toy-row ^floats [ctx] (toy/toy-logprobs lm-vocab ctx))

(deftest an-lm-potential-is-its-rows-and-scores
  (let [p (potential/prompted m prompt)
        row (toy-row [0 1 2])]
    (is (= #{end} (potential/end-tokens p)))
    (is (= 0.0 (potential/prefix p [])))
    (is (= (reduce + 0.0 (lm/score m prompt [2 3])) (potential/prefix p [2 3])))
    (is (= (vec row) (vec (take lm-vocab (potential/logw-next p [2])))) "the LM's row")
    (is (= (aget row end) (aget ^floats (potential/logw-next p [2]) lm-vocab)) "the end: its one end token's")
    (is (close-to? (+ (potential/prefix p [2]) (aget row end)) (potential/complete p [2]) 1e-12))
    (testing "two end tokens: the end is either"
      (let [q (potential/prompted m prompt {:end-tokens #{3 4}})]
        (is (close-to? (log-sum-exp [(aget row 3) (aget row 4)]) (aget ^floats (potential/logw-next q [2]) lm-vocab) 1e-6))))))

(deftest a-tempered-lm-potential-sets-a-completions-temperature
  ;; under the LM's own completion after the same prompt, (temper p (1/T -
  ;; 1)) makes each site's distribution the row at temperature T (one end
  ;; token, so the end's weight is its own)
  (doseq [temperature [0.5 2.0]
          ctx [[] [2] [3 1]]]
    (let [p (potential/temper (potential/prompted m prompt) (- (/ 1.0 temperature) 1.0))
          row (toy-row (into prompt ctx))
          q (constrained/site-row p #{end} row ctx)
          z (log-sum-exp (map #(/ (double %) temperature) row))]
      (doseq [t (range lm-vocab)]
        (is (close-to? (- (/ (aget row t) temperature) z) (aget q t) 1e-5) (str "T " temperature " at " ctx ", token " t))))))

(deftest a-normalized-lm-potential-is-unchanged
  ;; its rows sum to 1 over the tokens, the end token's mass the end's
  (let [p (potential/prompted m prompt)
        q (potential/normalize p)]
    (doseq [ctx [[] [2] [3 1 2]]]
      (is (every? true? (map #(close-to? %1 %2 1e-5) (potential/logw-next p ctx) (potential/logw-next q ctx))) (str ctx))
      (is (close-to? (potential/prefix p ctx) (potential/prefix q ctx) 1e-5)))))

;; ---------------------------------------------------------------------------
;; coerce: a potential over bytes, read through a vocabulary
;; ---------------------------------------------------------------------------

(defn- utf8 ^bytes [^String s] (.getBytes s StandardCharsets/UTF_8))

(def ^:private byte-tokens (mapv #(byte-array [(unchecked-byte %)]) (range 256)))

(def ^:private text-vocab
  "Tokens of a, b and é: every byte alone (é's two halves too), and
   longer pieces, some splitting é."
  (into [(utf8 "a") (utf8 "b") (byte-array [(unchecked-byte 0xC3)]) (byte-array [(unchecked-byte 0xA9)])]
        (map utf8 ["ab" "é" "ba" "abé" "éa" "bb"])))

(def ^:private language "(ab|é)+b?")

(def ^:private coerced (potential/coerce (grammar/grammar byte-tokens language) text-vocab))

(def ^:private byte-cost
  "A soft potential over bytes: each byte costs 0.01 per unit of its value."
  (let [c (fn [bs] (* -0.01 (reduce + 0 bs)))]
    (potential/potential 256 {:prefix c :complete (fn [bs] (- (c bs) 1.0))})))

(potential-laws/deflaws coerced-weights
  [[:coerced-grammar coerced (count text-vocab)]
   [:coerced-with-an-end-token (potential/coerce (grammar/grammar byte-tokens language) (conj text-vocab (utf8 "<end>"))
                                                 {:end-tokens #{(count text-vocab)}})
    (inc (count text-vocab))]]
  100)

(potential-laws/deflaws coerced-soft [[:coerced-cost (potential/coerce byte-cost text-vocab) (count text-vocab)]] 100 soft)

(deftest a-coerced-byte-grammar-is-the-grammar-over-the-vocabulary
  ;; every byte the language uses is a token, so the vocabulary's tokens
  ;; complete whatever its bytes complete, and the two agree exactly
  (let [direct (grammar/grammar text-vocab language)]
    (doseq [ctx [[] [0] [0 1] [5] [4] [7 2] [2 3 1] [1] [0 0]]]
      (is (= (potential/prefix direct ctx) (potential/prefix coerced ctx)) (str "prefix " ctx))
      (is (= (potential/complete direct ctx) (potential/complete coerced ctx)) (str "complete " ctx))
      (when (< ##-Inf (potential/prefix direct ctx))
        (is (= (vec (potential/logw-next direct ctx)) (vec (potential/logw-next coerced ctx))) (str "logw-next " ctx))))))

(deftest a-coerced-soft-potential-sums-its-bytes
  (let [p (potential/coerce byte-cost text-vocab)
        w (potential/logw-next p [0])]
    (doseq [t (range (count text-vocab))]
      (is (close-to? (* -0.01 (reduce + 0 (map #(bit-and (long %) 0xff) (text-vocab t)))) (aget w (int t)) 1e-6) (str "token " t)))
    (is (close-to? -1.0 (aget w (count text-vocab)) 1e-6) "the end: complete less prefix")
    (is (close-to? (* -0.01 (+ 97 98 97)) (potential/prefix p [4 0]) 1e-12) "a context: its bytes' sum")))

(deftest a-coerced-potential-reads-its-end-tokens
  (let [k (count text-vocab)
        p (potential/coerce byte-cost (conj text-vocab (utf8 "<end>")) {:end-tokens #{k}})]
    (is (= #{k} (potential/end-tokens p)))
    (is (= ##-Inf (aget ^floats (potential/logw-next p [0]) k)) "an end token is no text")
    (is (= ##-Inf (potential/prefix p [0 k])))
    (is (= :zero-prefix (bc/reason #(potential/logw-next p [k]))))))

;; ---------------------------------------------------------------------------
;; after: a scope past a close token
;; ---------------------------------------------------------------------------

(def ^:private close 5)

(def ^:private shifted
  "by-length less 1 at the empty context: its close weighs -1."
  (potential/potential n {:prefix #(- (cost %) 1.0) :complete #(- (cost %) 2.0)}))

;; two closes and longer walks, so that most walks have an answer to test
(potential-laws/deflaws scoped [[:after-no-repeats (potential/after #{4 close} no-repeats) n {:max-length 12}]] 100)

(potential-laws/deflaws scoped-soft
  [[:after-by-length (potential/after #{close} by-length) n]
   [:after-shifted (potential/after #{close 4} shifted) n]]
  100 soft)

(deftest a-scope-frees-the-think-block-and-holds-the-answer
  (let [p (potential/after #{close} shifted)]
    (testing "before the close: free, and incomplete"
      (is (= 0.0 (potential/prefix p [1 1 3])))
      (is (= ##-Inf (potential/complete p [1 1 3])))
      (is (= [0.0 0.0 0.0 0.0 0.0 -1.0 ##-Inf] (mapv double (potential/logw-next p [1 1 3]))) "the close weighs p's empty context"))
    (testing "after it: p over the answer, a later close an ordinary token"
      (is (= (potential/prefix shifted [2 3]) (potential/prefix p [1 1 close 2 3])))
      (is (= (potential/complete shifted [2 close]) (potential/complete p [0 close 2 close])))
      (is (= (vec (potential/logw-next shifted [2])) (vec (potential/logw-next p [3 close 2])))))
    (is (= #{5} (potential/end-tokens (potential/after #{0} ended))) "its end tokens are p's")))

;; ---------------------------------------------------------------------------
;; Misuse
;; ---------------------------------------------------------------------------

(deftest misuse-is-loud
  (testing "temper"
    (is (= :invalid-potential (bc/reason #(potential/temper by-length ##Inf))))
    (is (= :invalid-potential (bc/reason #(potential/temper by-length "2"))))
    (is (= :invalid-potential (bc/reason #(potential/temper :p 2.0)))))
  (is (= :invalid-potential (bc/reason #(potential/normalize nil))))
  (is (= :invalid-potential (bc/reason #(potential/product by-length :p))))
  (testing "prompted"
    (is (= :invalid-potential (bc/reason #(potential/prompted by-length prompt))))
    (is (= :invalid-token (bc/reason #(potential/prompted m []))))
    (is (= :invalid-end-tokens (bc/reason #(potential/prompted m prompt {:end-tokens #{9}})))))
  (testing "coerce"
    ;; genclj-kdau: refused when made, where only logw-next refused it and
    ;; prefix and complete read the bytes as by-length's tokens
    (is (= :invalid-potential (bc/reason #(potential/coerce by-length text-vocab))) "not over bytes")
    (is (= :invalid-potential (bc/reason #(potential/coerce (potential/potential 300 {:prefix (constantly 0.0)
                                                                                       :complete (constantly 0.0)})
                                                             text-vocab)))
        "more tokens than bytes")
    (is (= ##-Inf (potential/prefix (potential/coerce (potential/potential 7 {:prefix (constantly ##-Inf)
                                                                               :complete (constantly ##-Inf)})
                                                      text-vocab)
                                    [0]))
        "a potential of no weight at all is zero over any vocabulary: not checked")
    (is (= :mismatched-end-tokens (bc/reason #(potential/coerce (potential/potential 256 {:prefix (constantly 0.0) :complete (constantly 0.0)
                                                                                           :end-tokens #{0}})
                                                                 text-vocab))))
    (is (= :invalid-vocab (bc/reason #(potential/coerce byte-cost ["a"]))))
    (is (= :invalid-token (bc/reason #(potential/prefix coerced [99])))))
  (testing "after"
    (is (= :invalid-token (bc/reason #(potential/after #{} by-length))))
    (is (= :invalid-token (bc/reason #(potential/after [close] by-length))))
    ;; genclj-kdau: a close past the vocabulary never closes (Z = 0 before it)
    (is (= :invalid-token (bc/reason #(potential/after #{n} by-length))) "a close past the vocabulary")
    (is (= :invalid-token (bc/reason #(potential/after #{close (+ n 3)} by-length))) "one of the closes past it")
    (is (= :invalid-potential (bc/reason #(potential/after #{close} (potential/potential n {:prefix (constantly ##-Inf)
                                                                                           :complete (constantly ##-Inf)})))))))
