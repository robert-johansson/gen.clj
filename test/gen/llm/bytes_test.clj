(ns gen.llm.bytes-test
  "A text's probability over its tokenizations (gen.llm.bytes), on the toy
   engine with a vocabulary whose tokens overlap and split a character
   (é as one token, and as its two bytes).

   - Exact: prefix-logprob and text-logprob equal a brute-force recursion
     over the vocabulary (bytes compared as strings, no trie), for every
     text of up to four bytes over the vocabulary's.
   - next-byte-logprobs is the prefixes' ratios, and sums to 1.
   - The completion's own frequencies (gen.llm.token, 4,000 keys) of texts
     beginning with a string agree with prefix-logprob within 5 standard
     errors: the marginal is the token generative function's.
   - A beam gives lower bounds, rising with its width to the exact value.
   - Misuse is loud.

   Mutations checked: see docs/research/llm-grammar.md, Tests."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.gfi :as gfi]
            [gen.llm.branching-contract :as bc]
            [gen.llm.byte-tokenizer :as bt]
            [gen.llm.bytes :as bytes]
            [gen.llm.lm :as lm]
            [gen.llm.token :as token]
            [gen.llm.toy-engine :as toy]
            [gen.random :as rng])
  (:import (java.nio.charset StandardCharsets)))

(def ^:private pieces ["a" "b" "ab" "ba" "aba" "é" [0xC3] [0xA9] "bé" "c"])
(def ^:private tok (bt/tokenizer pieces))
(def ^:private n-vocab (:n-vocab tok))
(def ^:private bos (:bos tok))
(def ^:private eos (:eos tok))
(def ^:private m (lm/lm (toy/engine {:n-vocab n-vocab :max-branches 200 :n-ctx 512}) tok {:max-cells 20000}))
(def ^:private bm (bytes/byte-lm m))

(defn- utf8 [s] (vec (.getBytes ^String s StandardCharsets/UTF_8)))
(defn- piece [t] (vec ((:token-bytes tok) t)))
(defn- row ^floats [ctx] (toy/toy-logprobs n-vocab ctx))

(defn- begins? [xs prefix] (= (take (count prefix) xs) (seq prefix)))

(defn- brute-prefix
  "P(the text after ctx begins with b): each token that b begins with, or
   that begins with b."
  ^double [ctx b]
  (if (empty? b)
    1.0
    (reduce + (for [t (range n-vocab)
                    :when (not= t eos)
                    :let [tb (piece t)
                          p (Math/exp (aget (row ctx) t))]]
                (cond (begins? tb b) p
                      (begins? b tb) (* p (brute-prefix (conj ctx t) (drop (count tb) b)))
                      :else 0.0)))))

(defn- brute-text
  "P(the text after ctx is b, then the end)."
  ^double [ctx b]
  (+ (if (empty? b) (Math/exp (aget (row ctx) (int eos))) 0.0)
     (reduce + (for [t (range n-vocab)
                     :when (not= t eos)
                     :let [tb (piece t)]
                     :when (and (seq b) (begins? b tb))]
                 (* (Math/exp (aget (row ctx) t)) (brute-text (conj ctx t) (drop (count tb) b)))))))

(def ^:private texts
  "Every text of up to four bytes over the vocabulary's bytes, and a few
   no token spells."
  (let [alphabet (distinct (mapcat piece (range bos)))]
    (concat (mapcat (fn [k] (reduce (fn [xs _] (for [x xs b alphabet] (conj x b))) [[]] (range k))) (range 5))
            [(utf8 "z") (utf8 "abz") (utf8 "<s>a")])))

(defn- close? [^double a ^double b]
  (or (= a b) (<= (Math/abs (- a b)) 1e-9)))

(deftest marginals-equal-brute-force
  (doseq [prompt [[bos] [bos 0 1]]]
    (is (every? true? (for [b texts]
                        (and (close? (bytes/prefix-logprob bm prompt (byte-array b)) (Math/log (brute-prefix prompt b)))
                             (close? (bytes/text-logprob bm prompt (byte-array b)) (Math/log (brute-text prompt b)))))))))

(deftest the-next-byte-is-the-prefixes-ratio
  (doseq [b (take 40 (filter #(pos? (brute-prefix [bos] %)) texts))]
    (let [^doubles r (bytes/next-byte-logprobs bm [bos] (byte-array b))
          base (bytes/prefix-logprob bm [bos] (byte-array b))]
      ;; within the rows' rounding: each log-probability is a float, off by
      ;; at most 2^-24 of its size (under 8 on the toy), so a sum by 4e-6
      (is (<= (Math/abs (Math/log (reduce + (map #(Math/exp %) r)))) 4e-6) "a distribution")
      (is (every? #(close? (aget r (int %)) (- (bytes/prefix-logprob bm [bos] (byte-array (conj b (unchecked-byte %)))) base))
                  (range 256)))
      (is (close? (aget r 256) (- (bytes/text-logprob bm [bos] (byte-array b)) base))))))

(deftest the-marginal-is-the-completions
  ;; frequencies of texts that begin with each string, over 4,000
  ;; completions of the token generative function, within 5 standard errors
  (let [gf (token/completion m)
        k 4000
        text-of (fn [tokens] (vec (mapcat piece (remove #{eos} tokens))))
        samples (mapv #(text-of (:retval (gfi/simulate gf [[bos] 6] (rng/key %)))) (range k))]
    (doseq [s ["a" "ab" "é" "bé" "aba"]]
      (let [b (utf8 s)
            p (Math/exp (bytes/prefix-logprob bm [bos] s))
            hits (count (filter #(begins? % b) samples))]
        (is (<= (Math/abs (- hits (* k p))) (* 5 (Math/sqrt (* k p (- 1 p))))) s)))))

(deftest a-beam-gives-lower-bounds
  (let [b "abab"
        exact (bytes/prefix-logprob bm [bos] b)
        by-width (mapv #(bytes/prefix-logprob (bytes/byte-lm m {:beam %}) [bos] b) [1 2 4 8 64])]
    (is (every? #(<= % (+ exact 1e-12)) by-width) "lower bounds")
    (is (apply <= by-width) "rising with the width")
    (is (close? exact (peek by-width)) "exact when every tokenization fits")
    (is (< (first by-width) (- exact 1e-4)) "one tokenization misses mass: the test can tell")))

(deftest end-tokens-hold-no-text
  (let [no-ends (bytes/byte-lm m {:end-tokens #{}})]
    (is (close? (bytes/prefix-logprob bm [bos] "</s>") Double/NEGATIVE_INFINITY) "the end token ends the text")
    (is (< Double/NEGATIVE_INFINITY (bytes/prefix-logprob no-ends [bos] "</s>")) "unless it is no end")
    (is (= Double/NEGATIVE_INFINITY (bytes/text-logprob no-ends [bos] "ab")) "no end, no complete text")))

(deftest misuse-is-loud
  (is (= :zero-prefix (bc/reason #(bytes/next-byte-logprobs bm [bos] "z"))))
  (is (= :invalid-text (bc/reason #(bytes/prefix-logprob bm [bos] 3))))
  (is (= :invalid-token (bc/reason #(bytes/prefix-logprob bm [] "a"))))
  (is (= :invalid-end-tokens (bc/reason #(bytes/byte-lm m {:end-tokens #{n-vocab}}))))
  (is (= :invalid-end-tokens (bc/reason #(bytes/byte-lm m {:end-tokens [1]}))))
  (is (= :invalid-option (bc/reason #(bytes/byte-lm m {:beam 0})))))
