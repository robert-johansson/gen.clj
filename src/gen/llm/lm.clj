(ns gen.llm.lm
  "The LM protocol (DESIGN.md §6): a language model as a function from
   token contexts to next-token log-probabilities, shaped like
   genlm-backend's AsyncLM. The layer under the token generative function,
   grammar-constrained decoding and token SMC; above the Branching protocol
   (gen.llm.branching).

     (lm engine tokenizer)                   → an LM over a Branching engine (close it)
     (vocab m)                               → {:n-vocab :bos :eos :eog :add-bos?}
     (encode m text), (decode m tokens)      → a token vector, a string
     (token-bytes m token), (byte-vocab m)   → a token's bytes, every token's
     (next-logprobs m context)               → log p(t | context) for every token t
     (batch-next-logprobs m contexts)        → the same for each context, read together
     (score m context continuation)          → [log p(c_i | context, c_<i)]
     (info m)                                → sizes and counts

   Contexts are token vectors: an LM is content-addressed, so callers hold
   tokens, never engine state, and the same context always has the same
   log-probabilities (on llama.cpp, within its float noise: they depend on
   how a context was read; see docs/research/llm-lm-protocol.md). A context
   holds at least one token (a BOS for an unconditional start).
   Log-probabilities are natural logs, normalized over the vocabulary,
   returned as new float arrays; scores are doubles.

   The LM over an engine keeps what the engine has read in a trie of token
   prefixes that owns every branch, bounded in branches and KV cells, and
   reads what it lacks from the deepest prefix it holds (gen.llm.cache).
   batch-next-logprobs reads the contexts one token past a held prefix in
   one advance: batching is explicit, so a batch's composition, and with it
   every result, is a function of the program, never of thread timing.
   Calls are synchronous; concurrent callers are served one at a time, in
   no promised order: their order, and with it the cache's choices (and
   on llama.cpp the results, bit for bit), depends on their timing.

   A tokenizer is a map: :n-vocab, :bos, :eos, :eog (a set), :add-bos?,
   and the functions :token-bytes (token → a new byte array), :encode
   ((text opts) → tokens) and :decode (tokens → text). gen.llm.llama's
   tokenizer gives the GGUF's own."
  (:require [gen.llm.branching :as br]
            [gen.llm.cache :as cache]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

(defprotocol LM
  "What an LM implements, called through this namespace's functions, which
   check the arguments' shapes."
  (-tokenizer [m] "The tokenizer map.")
  (-next-logprobs [m contexts] "A vector of float arrays, one for each context (non-empty token vectors).")
  (-score [m context continuation] "A vector of doubles, one for each token of continuation.")
  (-info [m] "A map of sizes and counts."))

(deftype CachedLM [cache tokenizer lock]
  LM
  (-tokenizer [_] tokenizer)
  (-next-logprobs [_ contexts] (locking lock (cache/next-logprobs! cache contexts)))
  (-score [_ context continuation] (locking lock (cache/score! cache context continuation)))
  (-info [_] (locking lock (cache/info cache)))

  java.io.Closeable
  (close [_] (locking lock (cache/close! cache)))

  Object
  (toString [_]
    (let [{:keys [live max-branches cells max-cells]} (locking lock (cache/info cache))]
      (str "#<LM " live "/" max-branches " branches, " cells "/" max-cells " cells>"))))

(defmethod print-method CachedLM [m ^java.io.Writer w]
  (.write w (str m)))

(defn- check-tokenizer [engine {:keys [n-vocab eog token-bytes encode decode] :as tokenizer}]
  (let [engine-vocab (:n-vocab (br/info engine))]
    (when-not (and (map? tokenizer) (fn? token-bytes) (fn? encode) (fn? decode))
      (throw (ex-info "lm: a tokenizer is a map with the functions :token-bytes, :encode and :decode"
                      {:reason :invalid-tokenizer :keys (when (map? tokenizer) (keys tokenizer))})))
    (when-not (= n-vocab engine-vocab)
      (throw (ex-info (str "lm: the tokenizer's " n-vocab " tokens are not the engine's " engine-vocab)
                      {:reason :invalid-tokenizer :n-vocab n-vocab :engine-n-vocab engine-vocab})))
    (when-not (and (set? eog) (every? #(br/index? % n-vocab) eog))
      (throw (ex-info "lm: the tokenizer's :eog is a set of tokens" {:reason :invalid-tokenizer :eog eog})))
    tokenizer))

(defn lm
  "An LM over engine's branches (java.io.Closeable: closing it disposes of
   them; close the engine yourself). tokenizer is the model's (see the
   namespace's docstring). opts:
     :max-branches  live branches at most (default: every branch the engine
                    has free; the LM should be the engine's only user)
     :max-cells     KV cells its branches hold at most (default: the
                    engine's :n-ctx, all of llama.cpp's unified cache)"
  (^java.io.Closeable [engine tokenizer] (lm engine tokenizer {}))
  (^java.io.Closeable [engine tokenizer opts]
   (CachedLM. (cache/cache engine opts) (check-tokenizer engine tokenizer) (Object.))))

;; ---------------------------------------------------------------------------
;; The vocabulary
;; ---------------------------------------------------------------------------

(defn vocab
  "The vocabulary's size and special tokens: {:n-vocab :bos :eos :eog
   (the tokens that end generation, a set) :add-bos? (does encode add a
   BOS)}."
  [m]
  (select-keys (-tokenizer m) [:n-vocab :bos :eos :eog :add-bos?]))

(defn encode
  "The tokens of text, by the model's tokenizer: a vector. opts are the
   tokenizer's (llama.cpp's: :add-special?, :parse-special?)."
  ([m text] (encode m text {}))
  ([m text opts] ((:encode (-tokenizer m)) text opts)))

(defn decode
  "The text of tokens, by the model's tokenizer."
  [m tokens]
  ((:decode (-tokenizer m)) tokens))

(defn- check-token [m t what]
  (let [n (long (:n-vocab (-tokenizer m)))]
    (when-not (br/index? t n)
      (throw (br/invalid-token what t n)))
    t))

(defn end-tokens-option
  "The end tokens opts gives, a set of tokens of a vocabulary of n-vocab
   (:end-tokens), or default, as a set of longs; what names the caller in
   the error (:invalid-end-tokens). The reader of every :end-tokens option
   that defaults to an LM's :eog (a completion's, a byte LM's, a
   grammar's)."
  [what opts default n-vocab]
  (let [ts (if (contains? opts :end-tokens) (:end-tokens opts) default)
        n (long n-vocab)]
    (when-not (and (set? ts) (every? #(br/index? % n) ts))
      (throw (ex-info (str what ": :end-tokens is a set of tokens of the vocabulary, not " (pr-str ts))
                      {:reason :invalid-end-tokens :end-tokens ts})))
    (into #{} (map long) ts)))

(defn token-bytes
  "The bytes of token, raw (a token may hold part of a UTF-8 character): a
   new byte array."
  ^bytes [m token]
  ((:token-bytes (-tokenizer m)) (check-token m token "token-bytes")))

(defn byte-vocab
  "Every token's bytes, by token: a vector of new byte arrays (genlm's
   byte_vocab)."
  [m]
  (let [{:keys [n-vocab token-bytes]} (-tokenizer m)]
    (mapv token-bytes (range n-vocab))))

;; ---------------------------------------------------------------------------
;; Log-probabilities
;; ---------------------------------------------------------------------------

(defn- context-vector
  "A context as a vector, checked to be a non-empty sequence (its tokens
   are checked when read)."
  [context what]
  (when-not (and (sequential? context) (seq context))
    (throw (ex-info (str what ": a context is a non-empty sequence of tokens (a BOS for an unconditional"
                         " start), not " (pr-str context))
                    {:reason (if (sequential? context) :no-tokens :invalid-token) :context context})))
  (vec context))

(defn next-logprobs
  "The log-probability of every token as the next after context: a new
   float array, by token."
  ^floats [m context]
  (first (-next-logprobs m [(context-vector context "next-logprobs")])))

(defn batch-next-logprobs
  "next-logprobs of each of contexts, read together: a vector of float
   arrays, in order. The contexts one token past a context the LM holds
   are read in one batch (on llama.cpp, one decode); those further are
   read first, each from the deepest prefix the LM holds."
  [m contexts]
  (when-not (sequential? contexts)
    (throw (ex-info (str "batch-next-logprobs: contexts is a sequence of contexts, not " (pr-str contexts))
                    {:reason :invalid-token :contexts contexts})))
  (-next-logprobs m (mapv #(context-vector % "batch-next-logprobs") contexts)))

(defn score
  "The log-probability of each token of continuation given context and the
   tokens before it, teacher-forced: a vector of doubles, whose sum is the
   continuation's log-probability ([] for an empty one). Along what the LM
   has read it reads nothing; otherwise it reads the whole sequence and
   keeps it, so a next-logprobs of it is read already."
  [m context continuation]
  (when-not (sequential? continuation)
    (throw (ex-info (str "score: the continuation is a sequence of tokens, not " (pr-str continuation))
                    {:reason :invalid-token :continuation continuation})))
  (-score m (context-vector context "score") (vec continuation)))

(defn info
  "The LM's sizes and counts: for an LM over an engine, :live branches,
   :cells, :tokens in its trie, :max-branches, :max-cells, :limit (the
   most tokens a context, or a context and the continuation scored after
   it, may hold: the engine's :n-ctx or :max-cells, the lesser), :closed?,
   and since it was made :requests (contexts asked for), :hits (answered
   with no read), :reads (engine reads), :read-tokens, :forks,
   :evictions."
  [m]
  (-info m))
