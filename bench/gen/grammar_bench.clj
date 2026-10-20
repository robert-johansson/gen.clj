(ns gen.grammar-bench
  "Grammar costs (gen.llm.regex, gen.llm.grammar) at real vocabulary sizes:
   the DFA's construction, the trie's, each state's mask, and the
   per-token costs of decoding under a grammar.

   The vocabularies are read from GGUF files (their tokenizer.ggml.tokens,
   by a reader of the header here; llama.cpp is not loaded): Qwen3's
   byte-level BPE (151,936 tokens) and Gemma 3's SentencePiece (262,144).
   Without the files, a synthetic vocabulary of 151,936 tokens stands in.

   Decoding under a grammar runs on the toy engine at a vocabulary's size,
   its tokens' bytes the GGUF's: what a constraint adds to a token.

   Run: clojure -M:bench -m gen.grammar-bench [dir-of-gguf-files]"
  (:require [criterium.core :as crit]
            [gen.gfi :as gfi]
            [gen.llm.bytes :as bytes]
            [gen.llm.constrained :as constrained]
            [gen.llm.dfa :as dfa]
            [gen.llm.grammar :as grammar]
            [gen.llm.lm :as lm]
            [gen.llm.potential :as potential]
            [gen.llm.regex :as regex]
            [gen.llm.token :as token]
            [gen.llm.toy-engine :as toy]
            [gen.random :as rng])
  (:import (java.io File RandomAccessFile)
           (java.nio ByteBuffer ByteOrder)
           (java.nio.channels FileChannel$MapMode)
           (java.nio.charset StandardCharsets)))

(set! *warn-on-reflection* true)

;; ---------------------------------------------------------------------------
;; A GGUF file's vocabulary
;; ---------------------------------------------------------------------------

(defn- gguf-string ^String [^ByteBuffer b]
  (let [n (.getLong b)
        bs (byte-array n)]
    (.get b bs)
    (String. bs StandardCharsets/UTF_8)))

(declare gguf-value)

(defn- gguf-array [^ByteBuffer b]
  (let [type (.getInt b)
        n (.getLong b)]
    (if (= type 8)
      (vec (repeatedly n #(gguf-string b)))
      (vec (repeatedly n #(gguf-value b type))))))

(defn- gguf-value [^ByteBuffer b ^long type]
  (case type
    (0 1) (.get b) (2 3) (.getShort b) (4 5) (.getInt b) 6 (.getFloat b) 7 (.get b)
    8 (gguf-string b) 9 (gguf-array b) (10 11) (.getLong b) 12 (.getDouble b)))

(defn- gguf-metadata
  "The key-value metadata of a GGUF file."
  [^File f]
  (with-open [raf (RandomAccessFile. f "r")]
    (let [ch (.getChannel raf)
          b (doto (.map ch FileChannel$MapMode/READ_ONLY 0 (min (.size ch) (* 256 1024 1024)))
              (.order ByteOrder/LITTLE_ENDIAN))]
      (when-not (= 0x46554747 (.getInt b)) (throw (ex-info "not a GGUF file" {:file f})))
      (.getInt b)
      (.getLong b)
      (let [n (.getLong b)]
        (into {} (repeatedly n (fn [] (let [k (gguf-string b)] [k (gguf-value b (.getInt b))]))))))))

(def ^:private gpt2-byte-of
  "GPT-2's printable stand-in for each byte, inverted: char → byte."
  (let [direct (concat (range 33 127) (range 161 173) (range 174 256))
        others (remove (set direct) (range 256))]
    (merge (zipmap (map char direct) direct)
           (zipmap (map #(char (+ 256 %)) (range)) others))))

(defn- token-bytes
  "A GGUF token's bytes: byte-level BPE's stand-ins decoded, or
   SentencePiece's ▁ as a space and <0xXX> as the byte."
  ^bytes [model ^String piece]
  (if (= model "gpt2")
    (if (every? gpt2-byte-of piece)
      (byte-array (map #(unchecked-byte (gpt2-byte-of %)) piece))
      (.getBytes piece StandardCharsets/UTF_8))
    (if-let [[_ hex] (re-matches #"<0x([0-9A-Fa-f]{2})>" piece)]
      (byte-array [(unchecked-byte (Integer/parseInt hex 16))])
      (.getBytes (.replace piece "▁" " ") StandardCharsets/UTF_8))))

(defn- gguf-vocab [^File f]
  (let [md (gguf-metadata f)
        model (md "tokenizer.ggml.model")]
    (mapv #(token-bytes model %) (md "tokenizer.ggml.tokens"))))

(defn- synthetic-vocab
  "151,936 tokens: the 256 bytes, then pieces of English-like text."
  []
  (let [rnd (java.util.Random. 1)
        letters "etaoinshrdlucmfwypvbgkjqxz"
        word (fn [] (apply str (repeatedly (+ 1 (.nextInt rnd 8)) #(.charAt letters (min 25 (int (Math/abs (* 6 (.nextGaussian rnd)))))))))]
    (into (mapv #(byte-array [(unchecked-byte %)]) (range 256))
          (comp (map (fn [_] (.getBytes (str (when (.nextBoolean rnd) " ") (word)) StandardCharsets/UTF_8)))
                (take (- 151936 256)))
          (range))))

;; ---------------------------------------------------------------------------
;; Measurements
;; ---------------------------------------------------------------------------

(defn- ms ^double [f] (let [t0 (System/nanoTime)] (f) (/ (- (System/nanoTime) t0) 1e6)))

(defn- us [f] (* 1e6 (first (:mean (crit/quick-benchmark* f {})))))

(def ^:private str-re "\"(?:[^\"\\\\\\x00-\\x1F]|\\\\[\"\\\\/bfnrt]|\\\\u[0-9a-fA-F]{4})*\"")
(def ^:private num-re "-?(?:0|[1-9][0-9]*)(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")

(def ^:private languages
  [["phone number" "[0-9]{3}-[0-9]{4}"]
   ["yes or no" " ?(?:yes|no)"]
   ["JSON string" str-re]
   ["JSON number" num-re]
   ["a JSON object" (str "\\{\"name\": " str-re ", \"age\": " num-re ", \"tags\": \\[(?:" str-re "(?:, " str-re
                         "){0,4})?\\]\\}")]
   ["tool calls (prose and blocks)"
    (str "(?:[^<]|\\n)*(?:<tool_call>\\n<function=(?:get_weather|search|calc)>\\n(?:<parameter=(?:location|unit|query)>"
         "\\n[^<]*\\n</parameter>\\n)*</function>\\n</tool_call>(?:[^<]|\\n)*)*")]
   ["any text" ".*"]])

(defn- report [name vocab]
  (println (str "\n## " name ": " (count vocab) " tokens"))
  (let [trie-ms (ms #(bytes/byte-trie vocab))
        trie (bytes/byte-trie vocab)]
    (println (format "trie: %.0f ms, %s" trie-ms trie))
    (println "\n| language | DFA states | DFA ms | reached by tokens | live | masks ms | per reached state ms |")
    (println "|---|---|---|---|---|---|---|")
    (doseq [[lname re] languages]
      (let [d (regex/dfa re)
            dfa-ms (ms #(regex/dfa re))
            g (grammar/grammar vocab d)
            {:keys [reachable live mask-ns]} (grammar/info g)]
        (println (format "| %s | %d | %.1f | %d | %d | %.0f | %.2f |" lname (dfa/size d) dfa-ms reachable live
                         (/ mask-ns 1e6) (/ mask-ns 1e6 reachable)))))
    ;; the sweep alone, from the most and the least permissive states
    (let [g (grammar/grammar vocab ".*")
          d (grammar/automaton g)
          ok (boolean-array (dfa/size d) true)]
      (println (format "\nthe sweep from '.*''s start (every token but a few): %.0f µs"
                       (us #(bytes/sweep trie (dfa/table d) 0 ok nil)))))
    (let [g (grammar/grammar vocab "[0-9]{3}-[0-9]{4}")
          d (grammar/automaton g)
          ok (boolean-array (dfa/size d) true)]
      (println (format "the sweep from a digit state (%d tokens): %.1f µs" (count (grammar/allowed g 0))
                       (us #(bytes/sweep trie (dfa/table d) 0 ok nil)))))
    ;; decoding: a token's state advance, a step's weights
    (let [g (grammar/grammar vocab (second (languages 4)))
          ctx (loop [ctx [] s 0]
                (if (== 12 (count ctx))
                  ctx
                  (let [t (first (grammar/allowed g s))] (recur (conj ctx t) (grammar/advance g s t)))))
          s (grammar/state g (subvec ctx 0 1))
          t (first (grammar/allowed g s))]
      (println (format "advance by a token: %.3f µs; the state of a 12-token context: %.2f µs; logw-next (a dense row): %.0f µs"
                       (us #(grammar/advance g s t)) (us #(grammar/state g ctx)) (us #(potential/logw-next g []))))))
  nil)

(defn- tokenizer
  "A tokenizer (gen.llm.lm's map) of a byte vocabulary, its last token the
   end, its first BOS."
  [vocab]
  (let [n (count vocab)]
    {:n-vocab n :bos 0 :eos (dec n) :eog #{(dec n)} :add-bos? true
     :token-bytes (fn [t] (aclone ^bytes (vocab t)))
     :encode (fn [_ _] (throw (ex-info "the bench tokenizer does not encode" {})))
     :decode (fn [tokens] (String. (byte-array (mapcat #(seq ^bytes (vocab %)) tokens)) StandardCharsets/UTF_8))}))

(defn- decoding [name vocab]
  (println (str "\n## Decoding under a grammar, toy engine at " name "'s " (count vocab) " tokens, µs"))
  (with-open [^java.io.Closeable e (toy/engine {:n-vocab (count vocab) :max-branches 64 :n-ctx 1024})
              m (lm/lm e (tokenizer vocab))]
    (doseq [[lname re] [["string content, [^\"]*" "[^\"]*"] ["digits, [0-9]*" "[0-9]*"]]]
      (let [g (grammar/grammar m re)
            plain (token/completion m {:end-tokens #{}})
            under (token/completion m {:end-tokens #{} :potential g})
            prompt [0]
            seed (volatile! 0)
            row (lm/next-logprobs m prompt)
            ws (potential/logw-next g [])
            n 16
            sim-plain (us #(gfi/simulate plain [prompt n] (rng/key (vswap! seed inc))))
            sim-under (us #(gfi/simulate under [prompt n] (rng/key (vswap! seed inc))))]
        (println (str "\n" lname ": " (count (grammar/allowed g 0)) " tokens allowed at the start\n"))
        (println "| | µs |")
        (println "|---|---|")
        (println (format "| the site's distribution (constrained/weighted: row times weights, renormalized) | %.0f |"
                         (us #(constrained/weighted #{} row ws))))
        (println (format "| the grammar's weights (potential/logw-next, a dense row) | %.0f |" (us #(potential/logw-next g []))))
        (println (format "| simulate, per token, unconstrained (the toy's row is most of it) | %.0f |" (/ sim-plain n)))
        (println (format "| simulate, per token, under the grammar | %.0f |" (/ sim-under n)))
        (println (format "| what the grammar adds to a token | %.0f |" (/ (- sim-under sim-plain) n)))))))

(defn -main [& [dir]]
  (let [dir (File. (str (or dir "/Users/robert/gen-clj-models")))
        files (filter #(.endsWith (.getName ^File %) ".gguf") (or (seq (.listFiles dir)) []))]
    (if (seq files)
      (doseq [f (sort files)]
        (let [vocab (gguf-vocab f)]
          (report (.getName ^File f) vocab)
          (decoding (.getName ^File f) vocab)))
      (let [vocab (synthetic-vocab)]
        (report "synthetic" vocab)
        (decoding "synthetic" vocab)))
    (shutdown-agents)))
