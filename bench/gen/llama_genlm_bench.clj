(ns gen.llama-genlm-bench
  "1.2.0's GenLM tools end to end on a real model (genclj-e3q9), over
   llama.cpp:

   (a) A JSON object asked for in Qwen's chat template (thinking off), its
       language a JSON grammar by Earley (gen.llm.cfg): the LM alone; a
       completion under the grammar's potential (each token's row times
       its mask, gen.llm.constrained); and token SMC under it by the
       :masked proposal (a mask a particle and step) and by :awrs (only
       the tokens drawn are checked, gen.llm.awrs). For each: ms a token,
       how many ended within :max-tokens, and how many texts are JSON
       objects by a parser that shares nothing with gen.llm.cfg (an
       instaparse grammar, java.util.regex for strings and numbers).
   (b) Token SMC with a critic (Loula et al., 2025: cheap potentials in
       the proposal, expensive ones as critics): a sentence about the sea,
       its critic the same LM after a prompt for a sentence about a cat
       (potential/prompted: genlm's prompt intersection, the target the
       two prompts' product), against plain SMC (the LM alone) and the
       prompted LM as the :potential, its twist a row a particle and step.
       For each: ms a step, log Ẑ, and how many texts mention a cat.

   Run: clojure -J-Dgen.llama.lib=<lib> -M:test:bench -m gen.llama-genlm-bench <gguf>"
  (:require [clojure.string :as str]
            [gen.gfi :as gfi]
            [gen.llm.cfg :as cfg]
            [gen.llm.chat :as chat]
            [gen.llm.lm :as lm]
            [gen.llm.potential :as potential]
            [gen.llm.smc :as smc]
            [gen.llm.token :as token]
            [gen.random :as rng]
            [instaparse.core :as insta]))

(set! *warn-on-reflection* true)

;; ---------------------------------------------------------------------------
;; JSON
;; ---------------------------------------------------------------------------

(def ^:private string-re "\"(?:[^\"\\\\\\x00-\\x1F]|\\\\[\"\\\\/bfnrt]|\\\\u[0-9a-fA-F]{4})*\"")
(def ^:private number-re "-?(?:0|[1-9][0-9]*)(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")

(def json
  "A JSON object (RFC 8259's values, its whitespace spaces, newlines and
   tabs), by Earley: strings and numbers lexemes."
  (cfg/cfg {:start :object
            :rules {:value [[:object] [:array] [:string] [:number] ["true"] ["false"] ["null"]]
                    :object [["{" :ws "}"] ["{" :ws :members :ws "}"]]
                    :members [[:pair] [:pair :ws "," :ws :members]]
                    :pair [[:string :ws ":" :ws :value]]
                    :array [["[" :ws "]"] ["[" :ws :elements :ws "]"]]
                    :elements [[:value] [:value :ws "," :ws :elements]]
                    :string [[(cfg/regex string-re)]]
                    :number [[(cfg/regex number-re)]]
                    :ws [[(cfg/regex "[ \\n\\t]*")]]}}))

(def ^:private parser
  "An independent JSON object parser (instaparse)."
  (insta/parser (str "object = <'{'> ws (pair (ws <','> ws pair)*)? ws <'}'>
                      pair = string ws <':'> ws value
                      value = object | array | string | number | 'true' | 'false' | 'null'
                      array = <'['> ws (value (ws <','> ws value)*)? ws <']'>
                      string = #'" (str/replace string-re "'" "\\'") "'
                      number = #'" number-re "'
                      <ws> = <#'[ \\n\\t]*'>")))

(defn- json-object? [^String text] (not (insta/failure? (insta/parse parser text))))

(defn- ms ^double [f] (let [t0 (System/nanoTime)] (f) (/ (- (System/nanoTime) t0) 1e6)))

(defn- text-of [m tokens] (lm/decode m (vec (remove (:eog (lm/vocab m)) tokens))))

(defn- report [label results]
  (let [n (count results)
        tokens (reduce + (map :tokens results))
        ms (reduce + (map :ms results))]
    (if (zero? n)
      (println (format "| %s | 0 | | | | |" label))
      (println (format "| %s | %d | %.1f | %d of %d | %d of %d | %.0f |" label n (/ ms (max 1 tokens))
                       (count (filter :ended? results)) n (count (filter :valid? results)) n
                       (double (/ tokens n)))))
    (flush)))

(defn- json-section [e m]
  (let [tokenize @(requiring-resolve 'gen.llm.llama/tokenize)
        prompt (tokenize e (chat/render [{:role :user :content "Describe a fictional person as a JSON object with the keys name, age, city and hobbies. Reply with the JSON only."}])
                         {:add-special? false :parse-special? true})
        max-tokens 160
        ends (:eog (lm/vocab m))
        t0 (System/nanoTime)
        p (cfg/potential m json)
        made (/ (- (System/nanoTime) t0) 1e6)
        plain (token/completion m)
        under (token/completion m {:potential p})
        sample (fn [gf k]
                 (let [tr (volatile! nil)
                       t (ms #(vreset! tr (gfi/simulate gf [prompt max-tokens] (rng/key k))))
                       tokens (:retval @tr)
                       text (text-of m tokens)]
                   {:ms t :tokens (count tokens) :ended? (contains? ends (peek tokens))
                    :valid? (json-object? (str/trim text)) :text text}))
        particles (fn [opts k]
                    (let [s (volatile! nil)
                          t (ms #(vreset! s (smc/run plain [prompt max-tokens] 8 (assoc opts :potential p :key (rng/key k)))))
                          live (remove :dead? (:particles @s))
                          steps (:t @s)]
                      (println (format "  %s, key %d: %d steps, log Ẑ %.2f, %d of 8 live" (name (:proposal opts)) k steps
                                       (double (smc/log-ml-estimate @s)) (count live)))
                      (for [{:keys [tokens]} live]
                        {:ms (/ t (count live)) :tokens (count tokens) :ended? (contains? ends (peek tokens))
                         :valid? (json-object? (str/trim (text-of m tokens))) :text (text-of m tokens)})))]
    (println (format "\n## (a) A JSON object, %d prompt tokens, at most %d tokens; the grammar's potential made in %.0f ms\n"
                     (count prompt) max-tokens made))
    (let [unconstrained (mapv #(sample plain %) (range 8))
          constrained (mapv #(sample under %) (range 8))
          masked (vec (mapcat #(particles {:proposal :masked} %) [1 2]))
          awrs (vec (mapcat #(particles {:proposal :awrs} %) [1 2]))]
      (println "\n| How | texts | ms a token | ended | JSON objects (instaparse) | tokens a text |")
      (println "|---|---:|---:|---:|---:|---:|")
      (report "the LM alone" unconstrained)
      (report "a completion under the grammar (a mask a token)" constrained)
      (report "token SMC, 8 particles, :masked (a mask a particle and step)" masked)
      (report "token SMC, 8 particles, :awrs (the tokens drawn checked)" awrs)
      (println "\nThe LM alone, its first text:" (pr-str (:text (first unconstrained))))
      (println "Under the grammar, its first text:" (pr-str (:text (first constrained))))
      (println "By :awrs, a particle's text:" (pr-str (:text (first awrs))))
      (println "Every constrained text in the grammar's language:"
               (every? #(cfg/recognizes? json (str/trim (:text %))) (filter :ended? (concat constrained masked awrs)))))))

;; ---------------------------------------------------------------------------
;; A critic
;; ---------------------------------------------------------------------------

(defn- critic-section [e m]
  (let [tokenize @(requiring-resolve 'gen.llm.llama/tokenize)
        chat-prompt (fn [s] (tokenize e (chat/render [{:role :user :content s}]) {:add-special? false :parse-special? true}))
        prompt (chat-prompt "Write one short sentence about the sea.")
        other (chat-prompt "Write one short sentence about a cat.")
        completion (token/completion m)
        critic (potential/prompted m other)
        n 16
        max-tokens 32
        run (fn [opts k]
              (let [s (volatile! nil)
                    t (ms #(vreset! s (smc/run completion [prompt max-tokens] n (assoc opts :key (rng/key k)))))
                    texts (mapv #(text-of m (:tokens %)) (remove :dead? (:particles @s)))]
                {:ms-step (/ t (:t @s)) :log-z (smc/log-ml-estimate @s)
                 :cats (count (filter #(re-find #"(?i)\bcats?\b|kitten" %) texts)) :n (count texts) :texts texts}))]
    (println (format "\n## (b) Token SMC with a critic: %d particles, at most %d tokens, the sea's prompt (%d tokens) and the cat's (%d)\n"
                     n max-tokens (count prompt) (count other)))
    (println "| How | key | ms a step | log Ẑ | texts that mention a cat |")
    (println "|---|---:|---:|---:|---:|")
    (doseq [[label opts] [["plain SMC (the LM alone)" {}]
                          ["the cat's prompt as :critic" {:critic critic}]
                          ["the cat's prompt as :potential, :masked" {:potential critic :proposal :masked}]]
            k [1 2]]
      (let [{:keys [ms-step log-z cats n texts]} (run opts k)]
        (println (format "| %s | %d | %.1f | %.2f | %d of %d |" label k (double ms-step) (double log-z) cats n))
        (when (= k 1) (println "|  | e.g. |" (pr-str (first texts)) "| | |"))
        (flush)))))

(defn -main [& [gguf]]
  (let [engine @(requiring-resolve 'gen.llm.llama/engine)
        tokenizer @(requiring-resolve 'gen.llm.llama/tokenizer)
        info @(requiring-resolve 'gen.llm.branching/info)]
    (with-open [^java.io.Closeable e (engine {:model gguf :max-branches 64 :n-ctx 8192})
                ^java.io.Closeable m (lm/lm e (tokenizer e))]
      (println (format "# %s (%d tokens), %s" (:desc (:model (info e))) (:n-vocab (lm/vocab m))
                       (System/getProperty "gen.llama.lib")))
      (json-section e m)
      (critic-section e m)))
  (shutdown-agents))
