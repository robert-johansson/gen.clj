(ns gen.llama-branches-bench
  "Branches over llama.cpp, by their number: tokens a second of n
   sequences decoded together past one prompt, as the engine's batch alone
   (lm/batch-next-logprobs, n contexts a step), lockstep sampling
   (gen.llm.sample/complete, n completions of one context) and token SMC
   (gen.llm.smc/run, n particles, resampling at ESS < n/2), each of 64
   tokens past a short and a long prompt. Written to run unchanged on
   gen.clj 1.1.0 (llama.cpp v0.7.0) and 1.2.0 (v0.9.0), so that the two
   builds compare (genclj-ajse: v0.9.0's unified KV cache for
   multi-sequence work): copy it into a checkout of v1.1.0's bench/ to run
   it there.

   Before every run, untimed, the prompt is read: a live branch at it,
   which the LM's cache forks the run's contexts from (it forks only from
   a live branch whose context is a prefix; it keeps a branch for every
   context it is asked, so a run's steps evict the prompt's, and the next
   run reads it again), so a run reads only its own tokens. Its contexts are new (a token of the run's
   own after the prompt), so the cache answers none of them from an
   earlier run. The batch alone reads n distinct contexts (the prompt, then
   a token of each sequence's own), so every step decodes n sequences;
   lockstep sampling's n completions and SMC's particles start from one
   context and part as they sample (the cache reads a context they share
   once), so their columns also say the tokens the engine read per token
   made. Each measurement has an LM of its own: its cache starts empty, so
   no earlier measurement's cells sit in the KV cache that every decode
   reads. Every measurement is warmed once, then timed three times; the
   table shows the median. No sequence ends (no end tokens).

   Run: clojure -J-Dgen.llama.lib=<lib> -M:bench -m gen.llama-branches-bench <gguf>
   (-J-Dgen.bench.branches=1,4,8,16 for other branch counts)"
  (:require [clojure.string :as str]
            [gen.llm.lm :as lm]
            [gen.llm.sample :as sample]
            [gen.llm.smc :as smc]
            [gen.llm.token :as token]
            [gen.random :as rng]))

(set! *warn-on-reflection* true)

(def ^:private steps 64)

(def ^:private branch-counts
  "1, 4, 8, 16 and 32, or -Dgen.bench.branches=1,4,… ."
  (mapv parse-long (str/split (System/getProperty "gen.bench.branches" "1,4,8,16,32") #",")))

(def ^:private short-prompt "Once upon a time, in a small village by the sea,")

(def ^:private long-prompt
  (str (str/join " " (for [i (range 40)]
                       (str "Entry " i ": the committee weighed the harbour's accounts, the quotas and the price of salt.")))
       "\n\nNow something else entirely. Twenty random words, one per line:\n"))

(defn- measure
  "f's median time over three runs after a warm one (ms), and the median
   tokens the engine read in a run, (setup) before each run, untimed."
  [m setup f]
  (setup)
  (f)
  (let [runs (sort-by first (for [_ (range 3)]
                              (let [_ (setup)
                                    r0 (:read-tokens (lm/info m))
                                    t0 (System/nanoTime)]
                                (f)
                                [(/ (- (System/nanoTime) t0) 1e6) (- (long (:read-tokens (lm/info m))) (long r0))])))]
    (nth runs 1)))

(defn- row
  "For n branches past prompt: tokens a second of the batch alone,
   lockstep sampling and token SMC, and the tokens read per token made by
   the latter two. Each measurement on an LM of its own over e."
  [e tokenizer prompt ^long n]
  (let [run (volatile! 0)
        own (fn [^long i] (+ 1000 (mod (+ (* 97 (long @run)) i) 20000)))
        made (* n steps)
        rate (fn [[ms]] (/ made (/ (double ms) 1e3)))
        per-token (fn [[_ read]] (/ (double read) made))
        ;; before each run, untimed: the prompt read (a live branch at it,
        ;; read again when the cache evicted it), and the run's own token
        on-lm (fn [f] (with-open [^java.io.Closeable m (lm/lm e tokenizer)]
                        (measure m #(do (lm/next-logprobs m prompt) (vswap! run inc)) (f m))))
        raw (on-lm (fn [m] (fn []
                             (loop [ctxs (vec (for [i (range n)] (conj prompt (own i)))) t 0]
                               (when (< t steps)
                                 (lm/batch-next-logprobs m ctxs)
                                 (recur (mapv #(conj % (own (count %))) ctxs) (inc t)))))))
        lockstep (on-lm (fn [m] #(sample/complete m (repeat n (conj prompt (own 0)))
                                                  {:max-tokens steps :end-tokens #{} :key (rng/key @run)})))
        particles (on-lm (fn [m] (let [gf (token/completion m {:end-tokens #{}})]
                                   #(smc/run gf [(conj prompt (own 0)) steps] n {:key (rng/key @run)}))))]
    [(rate raw) (rate lockstep) (per-token lockstep) (rate particles) (per-token particles)]))

(defn -main [& [gguf]]
  (let [engine @(requiring-resolve 'gen.llm.llama/engine)
        tokenizer @(requiring-resolve 'gen.llm.llama/tokenizer)
        tokenize @(requiring-resolve 'gen.llm.llama/tokenize)
        info @(requiring-resolve 'gen.llm.branching/info)]
    (with-open [^java.io.Closeable e (engine {:model gguf :max-branches 160 :n-ctx 8192})]
      (println (format "## %s, %s: tokens a second (n branches × %d tokens, median of 3)"
                       (:desc (:model (info e))) (System/getProperty "gen.llama.lib") steps))
      (doseq [[label text] [["short" short-prompt] ["long" long-prompt]]]
        (let [prompt (vec (tokenize e text))]
          (println (format "\n### A %s prompt (%d tokens)\n" label (count prompt)))
          (println "| branches | the batch alone | lockstep sampling (read a token) | token SMC (read a token) |")
          (println "|---:|---:|---:|---:|")
          (doseq [n branch-counts]
            (let [[raw lockstep l-read particles p-read] (row e (tokenizer e) prompt n)]
              (println (format "| %d | %.0f | %.0f (%.2f) | %.0f (%.2f) |" n (double raw) (double lockstep) (double l-read)
                               (double particles) (double p-read)))
              (flush)))))))
  (shutdown-agents))
