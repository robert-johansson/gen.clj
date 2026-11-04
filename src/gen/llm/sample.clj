(ns gen.llm.sample
  "Sampling from a language model (gen.llm.lm) as chat models are run:
   temperature, top-k and top-p, and n completions advanced in lockstep.

     (pick logprobs sampling u)          → a token: the top-k tokens, the fewest
                                           of them whose probabilities reach top-p,
                                           at temperature temp; u uniform in [0, 1)
     (complete lm contexts)              → one completion of each context
     (complete lm contexts opts)

   complete advances its completions together: each step is one
   lm/batch-next-logprobs over the live ones, so contexts that share a
   prefix (one prompt, K replies) share the LM cache's cells, and the
   engine reads a step of all of them in one decode. opts:
     :sampling    {:temp :top-k :top-p} (default-sampling: Qwen's for
                  thinking off)
     :max-tokens  the most tokens a completion draws (256), or a vector of
                  them, one per completion
     :end-tokens  the tokens that end a completion (the LM's :eog), or a
                  vector of sets, one per completion; the end token is not
                  part of :tokens
     :logprobs?   each completion's log-probability under the LM itself
                  (unfiltered), summed over its tokens and its end token
     :key         completion i draws from (rng/split key n)[i], its step s
                  from fold-in s; the implicit key source without one
   Returns, in order, [{:tokens :stop (:end or :max) :end-token :logprob}].

   Before it reads anything, complete checks that the contexts and their
   completions fit the LM: their distinct tokens (shared prefixes counted
   once) plus every completion's :max-tokens within its :max-cells, and
   the completions within its :max-branches. Past either, the LM's cache
   would evict and re-read prefixes at every step, a run that crawls; it is
   a loud error instead."
  {:api :experimental}
  (:require [gen.array :as array]
            [gen.gfi :as gfi]
            [gen.llm.lm :as lm]
            [gen.random :as rng]))

(set! *warn-on-reflection* true)

(def default-sampling
  "Qwen's model card's for thinking off (temperature 0.7, top-p 0.8, top-k
   20); its GGUFs carry its thinking mode's (temperature 1, top-p 0.95)."
  {:temp 0.7 :top-k 20 :top-p 0.8})

(defn- top-indices
  "The indices of the k largest entries of lp, largest first."
  [^floats lp ^long k]
  (let [idx (long-array k -1)
        val (double-array k Double/NEGATIVE_INFINITY)]
    (dotimes [t (alength lp)]
      (let [v (double (aget lp t))]
        (when (> v (aget val (dec k)))
          (loop [j (dec k)]
            (if (and (pos? j) (> v (aget val (dec j))))
              (do (aset val j (aget val (dec j))) (aset idx j (aget idx (dec j))) (recur (dec j)))
              (do (aset val j v) (aset idx j t)))))))
    (vec (remove neg? idx))))

(defn pick
  "A token from next-token log-probabilities lp: the top-k tokens, the
   fewest of them whose probabilities reach top-p, at temperature temp; u
   a uniform draw in [0, 1)."
  [^floats lp {:keys [temp top-k top-p]} ^double u]
  (when-not (and (pos? (double temp)) (pos-int? top-k) (< 0.0 (double top-p) 1.0000001))
    (throw (ex-info "pick: temp > 0, top-k a positive integer, 0 < top-p <= 1"
                    {:temp temp :top-k top-k :top-p top-p})))
  (let [idx (top-indices lp top-k)
        ws (mapv #(/ (double (aget lp (int %))) (double temp)) idx)
        m (double (first ws))
        ps (mapv #(Math/exp (- (double %) m)) ws)
        z (reduce + ps)
        cum (reductions + (map #(/ % z) ps))
        n (min (count ps) (inc (count (take-while #(< % top-p) cum))))
        kept (subvec ps 0 n)
        target (* u (reduce + kept))]
    (nth idx (min (dec n) (count (take-while #(<= % target) (reductions + kept)))))))

(defn- distinct-tokens
  "The tokens of contexts, a prefix shared by several counted once: the
   sum of their lengths less the common prefixes of neighbours in sorted
   order (the nodes of their trie)."
  [contexts]
  (let [sorted (sort (fn [a b] (compare (vec a) (vec b))) contexts)
        lcp (fn [a b] (count (take-while true? (map = a b))))]
    (- (reduce + (map count sorted))
       (reduce + (map lcp sorted (rest sorted))))))

(defn- per
  "x for each of n completions: x itself, or x's i-th when it is a vector."
  [x n what]
  (if (vector? x)
    (do (when-not (= n (count x))
          (throw (ex-info (str "complete: " what " has " (count x) " entries for " n " completions") {what x})))
        x)
    (vec (repeat n x))))

(defn- check-fit!
  "A loud error when the completions do not fit lm's cache (when its info
   tells its bounds)."
  [lm contexts max-tokens]
  (let [{:keys [max-cells max-branches]} (lm/info lm)
        n (count contexts)
        cells (+ (distinct-tokens contexts) (reduce + max-tokens))]
    (when (and max-branches (> n (long max-branches)))
      (throw (ex-info (str "complete: " n " completions, but the LM holds at most " max-branches
                           " branches; past it, its cache would evict and re-read at every step")
                      {:reason :too-many-completions :completions n :max-branches max-branches})))
    (when (and max-cells (> cells (long max-cells)))
      (throw (ex-info (str "complete: the contexts and their completions need " cells " cells (shared prefixes counted"
                           " once, every completion's :max-tokens included), but the LM holds " max-cells
                           "; its cache would evict and re-read prefixes at every step. Fewer completions, shorter"
                           " contexts or completions, or a larger context (:n-ctx)")
                      {:reason :contexts-do-not-fit :cells cells :max-cells max-cells :completions n})))))

(defn complete
  "One completion of each of contexts (token vectors), in lockstep. See
   the namespace's docstring."
  ([lm contexts] (complete lm contexts nil))
  ([lm contexts {:keys [sampling max-tokens end-tokens logprobs? key]
                 :or {sampling default-sampling max-tokens 256}}]
   (let [contexts (mapv vec contexts)
         n (count contexts)
         max-tokens (per max-tokens n ":max-tokens")
         end-tokens (mapv set (per (or end-tokens (:eog (lm/vocab lm))) n ":end-tokens"))
         ks (rng/split (or key (gfi/next-key)) n)
         _ (check-fit! lm contexts max-tokens)
         uniform (fn [k] (double (array/item (rng/uniform k))))]
     (loop [live (vec (range n)) toks (vec (repeat n [])) out (vec (repeat n nil)) lps (vec (repeat n 0.0)) step 0]
       (if (empty? live)
         (mapv (fn [i] (cond-> (assoc (out i) :tokens (toks i)) logprobs? (assoc :logprob (lps i)))) (range n))
         (let [rows (lm/batch-next-logprobs lm (mapv #(into (contexts %) (toks %)) live))
               picked (mapv (fn [i ^floats row] [i (pick row sampling (uniform (rng/fold-in (ks i) step))) row]) live rows)
               lps (reduce (fn [l [i t ^floats row]] (update l i + (double (aget row (int t))))) lps picked)
               ended (into {} (keep (fn [[i t _]]
                                      (cond ((end-tokens i) t) [i {:stop :end :end-token t}]
                                            (>= (inc (count (toks i))) (long (max-tokens i))) [i {:stop :max}])))
                           picked)
               toks (reduce (fn [ts [i t _]] (if (= :end (:stop (ended i))) ts (update ts i conj t))) toks picked)]
           (recur (filterv #(not (ended %)) live) toks (reduce-kv assoc out ended) lps (inc step))))))))
