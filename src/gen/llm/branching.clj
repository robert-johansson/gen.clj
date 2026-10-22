(ns gen.llm.branching
  "The Branching protocol (DESIGN.md §6): a language model's states as
   branches, the layer under the LM protocol. An engine (gen.llm.llama;
   an MLX-native one later) holds the model; a branch is one token sequence
   the model has read, with the distribution of its next token.

     (open engine tokens)            → branch, after reading tokens
     (fork branch n)                 → n new branches with branch's state
     (advance branches tokens)       → branches, each one token further, in one decode
     (extend branch tokens)          → {:branch :scores}: many tokens, teacher-forced
     (score branch tokens)           → the log-probabilities of a continuation
     (resample branches ancestors)   → the ancestors' states, the rest dropped
     (dispose! branches)             → frees them

   A branch is a value: its tokens and its next-token log-probabilities
   never change (tokens, position, logprobs, logprob). It also holds a
   state in the engine (a sequence of the KV cache), which advance, extend
   and resample consume: they return new branches that take over the
   state, and the branches given to them are no longer live. Every
   operation needs live branches; a consumed or disposed branch, or one of
   a closed engine, is a loud error, never a silent use of another state.
   fork and score leave their branch live and unchanged.

   A live branch holds one of the engine's sequences, which are few
   (llama.cpp's limit is 256). dispose! frees one at once; a live branch
   that becomes unreachable is freed by the engine later, when it needs
   the sequence (gen.llm.slots). Close an engine (with-open) to free the
   model.

   Log-probabilities are natural logs, normalized over the vocabulary in
   double precision and stored as floats; scores are doubles. Tokens are
   integers in [0, vocabulary size).

   Engines are confined to one thread each: every operation runs on the
   engine's thread, called from any thread (the caller waits), so
   concurrent callers are serialized."
  (:refer-clojure :exclude [extend]))

(set! *warn-on-reflection* true)

(defprotocol Branching
  "What an engine implements. Called through the functions of this
   namespace, which check the arguments that need no engine state (token
   ranges, one engine, distinct branches) before calling it; the engine
   checks liveness and capacity."
  (-info [engine] "A map: :n-vocab, :max-branches, :n-ctx (the most tokens a branch holds), :live, :closed?, …")
  (-open [engine tokens])
  (-fork [engine branch n])
  (-advance [engine branches tokens])
  (-extend [engine branch tokens])
  (-score [engine branch tokens])
  (-resample [engine branches ancestors])
  (-dispose [engine branches])
  (-live? [engine branch]))

(deftype Branch [engine ^long id tokens ^floats logprobs]
  Object
  (toString [_] (str "#<Branch " id ", " (count tokens) " tokens>")))

(defmethod print-method Branch [b ^java.io.Writer w]
  (.write w (str b)))

(defn branch?
  [x]
  (instance? Branch x))

(defn- as-branch
  "b, checked to be a branch."
  ^Branch [b]
  (if (instance? Branch b)
    b
    (throw (ex-info (str "Not a branch: " (pr-str b)) {:reason :not-a-branch :value b}))))

(defn index?
  "Is x an integer in [0, n), a token of a vocabulary of n tokens, say?
   (Quick for a long; a big integer is out of range, not an error.)"
  [x ^long n]
  (if (instance? Long x)
    (< -1 (.longValue ^Long x) n)
    (and (integer? x) (not (neg? (compare x 0))) (neg? (compare x n)))))

(defn invalid-token
  "The error for t, not a token of a vocabulary of n-vocab tokens
   (:invalid-token, index?'s rule), what naming the caller: the one
   message of the LLM layer's token checks."
  [what t n-vocab]
  (ex-info (str what ": not a token of the vocabulary (0 to " (dec (long n-vocab)) "): " (pr-str t))
           {:reason :invalid-token :token t :n-vocab n-vocab}))

(defn clean-up
  "t, a failure, after each of the cleanup functions fs ran, each even if
   one before it threw: what they throw goes onto t as suppressed, so that
   it is kept and t stays the failure raised."
  ^Throwable [^Throwable t fs]
  (doseq [f fs]
    (try (f) (catch Throwable e (.addSuppressed t e))))
  t)

;; ---------------------------------------------------------------------------
;; A branch's value
;; ---------------------------------------------------------------------------

(defn tokens
  "The tokens the branch has read, a vector."
  [b]
  (.-tokens (as-branch b)))

(defn position
  "The number of tokens the branch has read (the position of its next token)."
  ^long [b]
  (count (.-tokens (as-branch b))))

(defn logprobs
  "The log-probabilities of the branch's next token, a new float array (one
   entry per token of the vocabulary)."
  ^floats [b]
  (aclone ^floats (.-logprobs (as-branch b))))

(defn logprob
  "The log-probability of token as the branch's next token."
  ^double [b token]
  (let [b (as-branch b)
        ^floats lp (.-logprobs b)]
    (when-not (index? token (alength lp))
      (throw (invalid-token "logprob" token (alength lp))))
    (double (aget lp (int token)))))

(defn engine
  "The engine holding the branch's state."
  [b]
  (.-engine (as-branch b)))

;; ---------------------------------------------------------------------------
;; Checks that need no engine state
;; ---------------------------------------------------------------------------

(defn info
  "The engine's sizes and counts: :n-vocab, :max-branches, :n-ctx (the most
   tokens a branch holds), :live (live branches), :closed?, and what the
   engine adds."
  [engine]
  (-info engine))

(defn- vocabulary-tokens
  "tokens as a vector of longs, each below n-vocab, or a loud error
   (:invalid-token) naming what was given them."
  [n-vocab tokens what]
  (when-not (sequential? tokens)
    (throw (ex-info (str what ": the tokens are a sequence of integers, not " (pr-str tokens))
                    {:reason :invalid-token :tokens tokens})))
  (mapv (fn [t]
          (if (index? t n-vocab)
            (long t)
            (throw (invalid-token what t n-vocab))))
        tokens))

(defn check-tokens
  "tokens as a vector of longs, each a token of engine's vocabulary, or a
   loud error (:invalid-token) naming what was given them."
  [engine tokens what]
  (vocabulary-tokens (:n-vocab (-info engine)) tokens what))

(defn- branch-tokens
  "check-tokens against b's vocabulary, which its log-probabilities span
   (no call of the engine)."
  [^Branch b tokens what]
  (vocabulary-tokens (alength ^floats (.-logprobs b)) tokens what))

(defn- non-empty [ts what]
  (when (empty? ts)
    (throw (ex-info (str what " needs at least one token") {:reason :no-tokens})))
  ts)

(defn- branch-vector
  "branches as a vector of distinct branches of one engine."
  [branches what]
  (when-not (and (sequential? branches) (seq branches))
    (throw (ex-info (str what " takes a non-empty sequence of branches, not " (pr-str branches))
                    {:reason :not-a-branch :value branches})))
  (let [bs (mapv as-branch branches)
        e (.-engine ^Branch (first bs))]
    (when-not (every? #(identical? e (.-engine ^Branch %)) bs)
      (throw (ex-info (str what ": the branches belong to different engines") {:reason :engines-differ})))
    (when-not (apply distinct? (map #(.-id ^Branch %) bs))
      (throw (ex-info (str what ": a branch appears more than once (fork it to have two)")
                      {:reason :repeated-branch :branches bs})))
    bs))

;; ---------------------------------------------------------------------------
;; Operations
;; ---------------------------------------------------------------------------

(defn open
  "A new branch that has read tokens (at least one: a BOS token for an
   unconditional start)."
  [engine tokens]
  (-open engine (non-empty (check-tokens engine tokens "open") "open")))

(defn fork
  "A new branch with branch's state ((fork b)), or a vector of n of them.
   branch stays live and unchanged. Each fork holds a sequence of its own
   (with llama.cpp's unified KV cache, sharing the prefix's cells)."
  ([branch] (first (fork branch 1)))
  ([branch n]
   (let [b (as-branch branch)]
     (when-not (index? n (inc Integer/MAX_VALUE))
       (throw (ex-info (str "fork: n is a non-negative integer, not " (pr-str n)) {:reason :invalid-count :n n})))
     (-fork (.-engine b) b (long n)))))

(defn advance
  "Each branch advanced by its token, all in one decode: a vector of new
   branches, in order. The branches given are consumed."
  [branches tokens]
  (let [bs (branch-vector branches "advance")
        e (.-engine ^Branch (first bs))
        ts (branch-tokens (first bs) tokens "advance")]
    (when-not (= (count bs) (count ts))
      (throw (ex-info (str "advance: " (count bs) " branches but " (count ts) " tokens (one each)")
                      {:reason :count-mismatch :branches (count bs) :tokens (count ts)})))
    (-advance e bs ts)))

(defn extend
  "branch advanced by the tokens in turn, teacher-forced (one decode per
   batch of tokens): {:branch the new branch, :scores the log-probability
   of each token given those before it}. branch is consumed."
  [branch tokens]
  (let [b (as-branch branch)
        e (.-engine b)]
    (-extend e b (non-empty (branch-tokens b tokens "extend") "extend"))))

(defn score
  "The log-probability of each token of a continuation given branch and the
   tokens before it, teacher-forced: a vector of doubles (their sum is the
   continuation's log-probability). branch stays live and unchanged."
  [branch tokens]
  (let [b (as-branch branch)
        e (.-engine b)]
    (-score e b (non-empty (branch-tokens b tokens "score") "score"))))

(defn resample
  "The branches' states, one for each ancestor index (into branches): a
   vector of new branches, the j-th with the state of (branches
   (ancestors j)). Every branch given is consumed; those no ancestor names
   are freed. A state is copied only for its second and later children."
  [branches ancestors]
  (let [bs (branch-vector branches "resample")
        n (count bs)]
    (when-not (and (sequential? ancestors) (every? #(index? % n) ancestors))
      (throw (ex-info (str "resample: the ancestors are indices of the " n " branches, not " (pr-str ancestors))
                      {:reason :invalid-ancestors :ancestors ancestors :branches n})))
    (-resample (.-engine ^Branch (first bs)) bs (mapv long ancestors))))

(defn dispose!
  "Free a branch, or a sequence of them, at once. Each must be live."
  [branches]
  (let [bs (if (sequential? branches) branches [branches])]
    (when (seq bs)
      (let [bs (branch-vector bs "dispose!")]
        (-dispose (.-engine ^Branch (first bs)) bs)))
    nil))

(defn live?
  "Does the branch still hold its state (not consumed, disposed or freed,
   its engine open)?"
  [branch]
  (let [b (as-branch branch)]
    (boolean (-live? (.-engine b) b))))
