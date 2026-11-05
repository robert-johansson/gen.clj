(ns gen.instaparse-bench
  "instaparse grammars as constraints (gen.llm.instaparse, ext/instaparse)
   at Qwen3's vocabulary (151,936 tokens, read from its GGUF as
   gen.grammar-bench reads it; llama.cpp is not loaded): the term, the DFA
   and every mask once per grammar, then the per-token costs, a mask row
   (logw-next) and the site's distribution (the row times the mask,
   renormalized). And a recursive grammar (JSON values) read by Earley
   (gen.llm.instaparse/cfg, gen.llm.cfg): a prefix weight of a text, and a
   mask (logw-next's trie sweep), at the start and inside a string.

   Run: clojure -M:test:bench -m gen.instaparse-bench [Qwen3 GGUF]"
  (:require [criterium.core :as crit]
            [gen.grammar-bench]
            [gen.llm.constrained :as constrained]
            [gen.llm.cfg :as cfg]
            [gen.llm.dfa :as dfa]
            [gen.llm.grammar :as grammar]
            [gen.llm.instaparse :as ip]
            [gen.llm.potential :as potential]
            [gen.llm.regex :as regex])
  (:import (java.io File)))

(set! *warn-on-reflection* true)

(def grammars
  [["a tool call" "call = <'{\"name\": \"'> name <'\", \"arguments\": {'> args <'}}'>
                   name = 'get_weather' | 'search' | 'send_email'
                   args = (pair (<', '> pair)*)?
                   pair = <'\"'> key <'\": '> value
                   key = #'[a-z_]{1,12}'
                   value = #'\"[^\"]{0,40}\"' | #'-?[0-9]{1,6}' | 'true' | 'false'" {}]
   ["an ISO date-time" "dt = date <'T'> time zone?
                        date = #'[0-9]{4}' '-' #'(0[1-9]|1[0-2])' '-' #'(0[1-9]|[12][0-9]|3[01])'
                        time = #'([01][0-9]|2[0-3])' ':' #'[0-5][0-9]' (':' #'[0-5][0-9]')?
                        zone = 'Z' | #'[+-]([01][0-9]|2[0-3]):[0-5][0-9]'" {}]
   ["nested lists, 3 deep" "list = '[' (item (', ' item)*)? ']'
                            item = list | #'[0-9]+'" {:max-depth 3}]])

(defn- ms ^double [f] (let [t0 (System/nanoTime)] (f) (/ (- (System/nanoTime) t0) 1e6)))

(defn- us [f] (* 1e6 (first (:mean (crit/quick-benchmark* f {})))))

(defn -main [& [gguf]]
  (let [path (or gguf (str (System/getProperty "user.home") "/gen-clj-models/Qwen3-0.6B-Q8_0.gguf"))
        f (File. ^String path)
        vocab (if (.exists f)
                (#'gen.grammar-bench/gguf-vocab f)
                (do (println "no GGUF at" (str f) "- a synthetic vocabulary of 151,936 tokens")
                    (#'gen.grammar-bench/synthetic-vocab)))
        n (count vocab)
        row (let [r (float-array n (float (- (Math/log n))))] r)]
    (println "##" n "tokens\n")
    (println "| Grammar | DFA states | term, DFA, grammar (every mask), ms | logw-next, µs | site's distribution, µs |")
    (println "|---|---:|---|---:|---:|")
    (doseq [[label text opts] grammars]
      (let [t (volatile! nil) d (volatile! nil) g (volatile! nil)
            term-ms (ms #(vreset! t (ip/term text opts)))
            dfa-ms (ms #(vreset! d (regex/dfa @t)))
            grammar-ms (ms #(vreset! g (grammar/grammar vocab @d {:end-tokens #{}})))
            ctx (vec (take 2 (grammar/allowed @g (dfa/start @d))))
            ctx (if (zero? (potential/prefix @g ctx)) ctx [])]
        (println (format "| %s | %d | %.0f, %.0f, %.0f | %.0f | %.0f |" label (dfa/size @d) term-ms dfa-ms grammar-ms
                         (us #(potential/logw-next @g ctx))
                         (us #(constrained/site-row @g #{} row ctx))))))
    (let [json "value = object | array | string | number | 'true' | 'false' | 'null'
                object = <'{'> (pair (<', '> pair)*)? <'}'>
                pair = string <': '> value
                array = <'['> (value (<', '> value)*)? <']'>
                string = #'\"[a-z ]*\"'
                number = #'-?[0-9]+'"
          g (ip/cfg json)
          p (cfg/potential vocab g {:end-tokens #{}})
          encode (fn [^String text]
                   ;; a text as single-byte tokens: Qwen3's first 256 are the bytes
                   (let [index (into {} (keep-indexed (fn [t ^bytes bs] (when (== 1 (alength bs)) [(aget bs 0) t]))) vocab)]
                     (mapv #(index (byte %)) (.getBytes text "UTF-8"))))
          short-text (encode "{\"a\": [1, 2, {\"b\": ")
          long-text (encode (apply str "[" (repeat 20 "{\"key\": [1, -2, \"text here\"]}, ")))]
      (println "\n## A recursive grammar (JSON values) by Earley\n")
      (println "| Context | prefix, µs | logw-next (every token), µs |")
      (println "|---|---:|---:|")
      (doseq [[label ctx] [["empty" []] ["22 bytes, inside a value" short-text] ["661 bytes, inside the array" long-text]]]
        (println (format "| %s | %.0f | %.0f |" label (us #(potential/prefix p ctx)) (us #(potential/logw-next p ctx))))))))
