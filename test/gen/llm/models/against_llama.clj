(ns gen.llm.models.against-llama
  "Qwen3's forward over gen.array (gen.llm.models.qwen3) against the
   llama.cpp engine (gen.llm.llama) on the same GGUF, for the test
   (gen.llm.models.qwen3-llama-test) and its band's bench
   (bench/gen/qwen3_llama_noise.clj): the next-token log-probabilities of
   one prompt, by both, compared as the Branching contract compares two
   computations (gen.llm.branching-contract's max-diff: max |Δ log p| over
   the reference's 64 most likely tokens, llama.cpp's here), with the
   argmax and the top 5 (as sets; a swap across the boundary between
   tokens llama.cpp holds within a tie of each other is noise).

   Each prompt is read three ways by the forward:
   - :whole, a prefill of the prompt;
   - :chunk, a prefill of its first half and a step of the rest;
   - :step i, the prompt's cache stepped i times by greedy tokens
     (llama.cpp's argmax), against llama.cpp's branch advanced so.

   And the planted convention errors (mutations), as with-redefs-fn's
   redefinitions of gen.array's operations, each self-consistent:
   - :kv-tiled: query head h reads KV head h mod KV, where Qwen3 (and
     gen.array/sdpa) groups them, h / (H/KV);
   - :rope-traditional: RoPE's consecutive pairs for NEOX's halves;
   - :qk-norms-skipped: q and k not normalized per head."
  (:require [gen.array :as array]
            [gen.llm.branching :as br]
            [gen.llm.branching-contract :refer [max-diff top-tokens]]
            [gen.llm.models.qwen3 :as qwen3]))

(def ^:private sdpa array/sdpa)
(def ^:private rope array/rope)
(def ^:private rms-norm array/rms-norm)

(def mutations
  "Planted convention errors: name to with-redefs-fn's map."
  {:kv-tiled {#'array/sdpa (fn sdpa-tiled
                             ([q k v] (sdpa-tiled q k v nil))
                             ([q k v opts]
                              (let [rep (quot (long (second (array/shape q))) (long (second (array/shape k))))
                                    tile (fn [x] (array/concatenate (vec (repeat rep x)) 1))]
                                (sdpa q (tile k) (tile v) opts))))}
   :rope-traditional {#'array/rope (fn rope-traditional
                                     ([x dims] (rope-traditional x dims nil))
                                     ([x dims opts] (rope x dims (assoc opts :traditional? true))))}
   :qk-norms-skipped {#'array/rms-norm (fn [x w eps] (if (= 4 (array/ndim x)) x (rms-norm x w eps)))}})

(defn- logprobs
  "The log-probabilities of logits' last row (logits [1 T vocab]), a
   float[]."
  ^floats [logits]
  (let [t (long (second (array/shape logits)))]
    (float-array (flatten (array/->clj (array/log-softmax (array/slice logits 1 (dec t) t)))))))

(defn- same-top?
  "Are lp's k most likely tokens ref's, but for a swap across the k-th
   place between tokens ref holds within tie of its k-th (a tie inside
   the noise)?"
  [^floats ref ^floats lp k tie]
  (let [r (top-tokens ref k)
        l (top-tokens lp k)
        kth (aget ref (int (peek r)))
        tied? (fn [t] (<= (Math/abs (- (aget ref (int t)) kth)) (double tie)))]
    (and (every? tied? (remove (set r) l))
         (every? tied? (remove (set l) r)))))

(defn- verdict
  "{:max-diff :argmax? :top5?} of lp against the reference ref: the
   argmax and the top 5 (as sets) the same, but for swaps across the
   boundary between tokens ref holds within tie of each other."
  [^floats ref ^floats lp tie]
  {:max-diff (max-diff ref lp 64)
   :argmax? (same-top? ref lp 1 tie)
   :top5? (same-top? ref lp 5 tie)})

(defn compare-prompt
  "The forward of model m against llama.cpp engine e on the prompt's
   tokens (a vector of at least 2): a vector of {:what :max-diff :argmax?
   :top5?}, :what :whole, :chunk or [:step i] (i from 1 to steps); tie as
   verdict's (0: exactly the same tokens)."
  [e m tokens steps tie]
  (let [n (count tokens)
        root (br/open e tokens)]
    (try
      (let [ref (br/logprobs root)
            [whole cache] (qwen3/prefill m [tokens] {:logits :last})
            [_ half] (qwen3/prefill m [(subvec tokens 0 (quot n 2))])
            [chunk] (qwen3/step m half [(subvec tokens (quot n 2))] {:logits :last})]
        (loop [i 1 b (br/fork root) c cache
               out [(assoc (verdict ref (logprobs whole) tie) :what :whole)
                    (assoc (verdict ref (logprobs chunk) tie) :what :chunk)]]
          (if (> i steps)
            (do (br/dispose! b) out)
            (let [t (first (top-tokens (br/logprobs b) 1))
                  [b'] (br/advance [b] [t])
                  [lg c'] (qwen3/step m c [[t]])]
              (recur (inc i) b' c' (conj out (assoc (verdict (br/logprobs b') (logprobs lg) tie) :what [:step i])))))))
      (finally (br/dispose! root)))))

(defn mutated
  "The worst :max-diff of the forward under mutation (a key of mutations)
   on the prompt's tokens: its :whole and :chunk readings, and one step."
  [e m tokens mutation]
  (with-redefs-fn (mutations mutation)
    #(reduce max (map :max-diff (compare-prompt e m tokens 1 0)))))
