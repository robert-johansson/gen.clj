(ns gen.llm.potential
  "Potentials: weights of token sequences, after genlm-control's Potential,
   for twisted and grammar-constrained token SMC and constrained decoding.
   A potential weighs a context (a token vector) as a prefix and as a
   complete sequence; the next token's weights follow from those.

     (prefix p context)      → log weight of context as a prefix (-Inf: no
                               sequence it begins has weight)
     (complete p context)    → log weight of context as a whole sequence
     (logw-next p context)   → the next token's log weights: a float array
                               of n-vocab + 1 entries, entry t for token t,
                               the last (index n-vocab) for the end
     (potential n-vocab {:prefix f :complete g :logw-next h :end-tokens ts})
     (end-tokens p)          → the end tokens p reads contexts with, or nil
     (potential? x)

   Combinators (genlm-control's), each a potential of the same contract:

     (product p q …)         → the sum of their weights: every one must hold
     (temper p beta)         → p's weights times beta (p^beta); zero stays zero
     (normalize p)           → p's next weights normalized at every context
     (prompted m prompt)     → an LM's log-probabilities of what follows prompt
     (coerce p vocab)        → a potential over bytes read through vocab's tokens
     (after closes p)        → p over the tokens after the first of closes (a
                               think block's close), the tokens before it free

   With these, an ensemble of LMs of one tokenizer is a product of tempered
   prompted ones (weights the powers), a temperature T on an LM's own
   completion (temper (prompted m prompt) (- (/ 1 T) 1)), and the tempered
   sampler as a potential (normalize (temper (prompted m prompt) (/ 1 T))).

   Tokens are those of an LM's vocabulary, 0 to n-vocab - 1; the end is
   not a token, and contexts never hold it (an LM's end-of-generation token
   that ends a completion stands for it: the end's weight is the context's
   as a complete sequence).

   A potential is consistent when, for a context of finite prefix weight,
   (logw-next p ctx) holds prefix(ctx ++ [t]) - prefix(ctx) at t and
   complete(ctx) - prefix(ctx) at the end. A potential computes it as it
   can (a grammar from its automaton's state, a mask); (potential …)
   computes it densely from prefix and complete when given no :logw-next.

   End tokens. A potential that reads its contexts with particular end
   tokens declares them (the optional EndTokens protocol, end-tokens): a
   grammar leaves its end tokens out of the text, so it means its language
   only under a completion whose end tokens they are. A potential that
   declares none has no opinion: a token-level one, whose contexts never
   hold an end token. (potential …) declares :end-tokens when given them;
   a product declares its factors' set, and two factors that disagree are
   an error when it is made. check-end-tokens is the one check, where a
   potential meets a completion's end tokens (token/completion, and token
   SMC's :potential)."
  {:api :experimental}
  (:require [gen.llm.branching :as br]
            [gen.llm.bytes :as bytes]
            [gen.llm.lm :as lm]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

(defprotocol Potential
  "What a potential implements, called through this namespace's functions."
  (-prefix [p context] "The log weight of context as a prefix, a double.")
  (-complete [p context] "The log weight of context as a whole sequence, a double.")
  (-logw-next [p context] "The next token's log weights: a float array of n-vocab + 1 entries, the end last."))

(defprotocol EndTokens
  "What a potential that reads its contexts with particular end tokens
   implements (optional: one that does not has no opinion)."
  (-end-tokens [p] "The end tokens p reads its contexts with, a set, or nil for no opinion."))

(defn end-tokens
  "The end tokens p reads its contexts with (a set: a grammar's, which it
   leaves out of the text), or nil when it has no opinion."
  [p]
  (when (satisfies? EndTokens p) (-end-tokens p)))

(defn potential?
  "Is x a potential (does it implement the Potential protocol)?"
  [x]
  (satisfies? Potential x))

(defn- check-potential
  "p, or a loud error naming what (the caller) when it is no potential."
  [what p]
  (when-not (potential? p)
    (throw (ex-info (str what ": not a potential: " (pr-str p)) {:reason :invalid-potential :potential p})))
  p)

(defn check-end-tokens
  "p, checked against the end tokens of the completion it meets: what
   names the caller in the error, :mismatched-end-tokens when p declares
   other end tokens. A potential with no opinion passes."
  [what p completion-ends]
  (when-let [own (end-tokens p)]
    (when-not (= own (set completion-ends))
      (throw (ex-info (str what ": the potential's end tokens " own " are not the completion's " (set completion-ends)
                           " (a grammar leaves its end tokens out of the text: make it with the same :end-tokens)")
                      {:reason :mismatched-end-tokens :potential own :completion (set completion-ends)}))))
  p)

(defn- context-vector [what context]
  (when-not (sequential? context)
    (throw (ex-info (str what ": a context is a sequence of tokens, not " (pr-str context))
                    {:reason :invalid-token :context context})))
  (vec context))

(defn prefix
  "The log weight of context as a prefix of a sequence: -Inf when no
   sequence it begins has weight."
  ^double [p context]
  (double (-prefix p (context-vector "prefix" context))))

(defn complete
  "The log weight of context as a whole sequence."
  ^double [p context]
  (double (-complete p (context-vector "complete" context))))

(defn logw-next
  "The next token's log weights after context: a float array of n-vocab + 1
   entries, entry t for token t and the last for the end."
  ^floats [p context]
  (-logw-next p (context-vector "logw-next" context)))

(defn- dense-logw-next
  "logw-next from prefix and complete, one call for each token."
  [^long n-vocab prefix complete context]
  (let [base (double (prefix context))]
    (when (Double/isInfinite base)
      (throw (ex-info "logw-next: the context has no weight as a prefix" {:reason :zero-prefix :context context})))
    (let [out (float-array (inc n-vocab))]
      (dotimes [t n-vocab]
        (aset out t (unchecked-float (- (double (prefix (conj context t))) base))))
      (aset out n-vocab (unchecked-float (- (double (complete context)) base)))
      out)))

(defn potential
  "A potential over a vocabulary of n-vocab tokens from functions of a
   context (a token vector): :prefix and :complete, log weights, and
   optionally :logw-next, the next token's (computed densely from the
   other two when absent: n-vocab + 2 calls). :end-tokens, a set of
   tokens, declares the end tokens it reads contexts with (none: no
   opinion)."
  [n-vocab {:keys [prefix complete logw-next] declared :end-tokens :as fns}]
  (when-not (and (pos-int? n-vocab) (ifn? prefix) (ifn? complete) (or (nil? logw-next) (ifn? logw-next)))
    (throw (ex-info "potential: a vocabulary size and the functions :prefix and :complete (and :logw-next, optionally)"
                    {:reason :invalid-potential :n-vocab n-vocab})))
  (when (and (contains? fns :end-tokens) (not (and (set? declared) (every? #(br/index? % n-vocab) declared))))
    (throw (ex-info (str "potential: :end-tokens is a set of tokens of the vocabulary, not " (pr-str declared))
                    {:reason :invalid-end-tokens :end-tokens declared})))
  (let [next-weights (or logw-next #(dense-logw-next n-vocab prefix complete %))
        ends (some->> declared (into #{} (map long)))]
    (reify
      Potential
      (-prefix [_ context] (prefix context))
      (-complete [_ context] (complete context))
      (-logw-next [_ context] (next-weights context))
      EndTokens
      (-end-tokens [_] ends))))

(defn- product*
  "The product of ps, which read contexts with the end tokens ends (or
   have no opinion: nil)."
  [ps ends]
  (reify
    EndTokens
    (-end-tokens [_] ends)
    Potential
    (-prefix [_ context] (reduce + 0.0 (map #(-prefix % context) ps)))
    (-complete [_ context] (reduce + 0.0 (map #(-complete % context) ps)))
    (-logw-next [_ context]
      (let [ws (mapv #(-logw-next % context) ps)
            n (alength ^floats (first ws))]
        (when-not (every? #(= n (alength ^floats %)) ws)
          (throw (ex-info "product: the potentials' vocabularies differ"
                          {:reason :invalid-potential :sizes (mapv #(alength ^floats %) ws)})))
        (let [out (float-array n)]
          (doseq [^floats w ws]
            (dotimes [t n] (aset out t (unchecked-float (+ (aget out t) (aget w t))))))
          out)))))

(defn product
  "The potential whose log weights are the sums of ps's: each must hold for
   a sequence to (as genlm-control's product). It reads contexts with its
   factors' end tokens, and factors that declare different ones are
   :mismatched-end-tokens."
  [& ps]
  (when (empty? ps)
    (throw (ex-info "product: at least one potential" {:reason :invalid-potential})))
  (run! #(check-potential "product" %) ps)
  (let [declared (distinct (keep end-tokens ps))]
    (when (next declared)
      (throw (ex-info (str "product: the potentials read contexts with different end tokens, " (vec declared))
                      {:reason :mismatched-end-tokens :end-tokens (vec declared)})))
    (product* ps (first declared))))

;; ---------------------------------------------------------------------------
;; Tempering and normalization
;; ---------------------------------------------------------------------------

(defn- scale
  "beta times a log weight w, a zero weight (-Inf) staying zero."
  ^double [^double beta ^double w]
  (if (== w Double/NEGATIVE_INFINITY) w (* beta w)))

(defn temper
  "p's weights times beta, a finite number of either sign (p^beta, as
   genlm-control's power): a zero weight (-Inf) stays zero, so p's support
   is kept, and beta 0 is its indicator. Its end tokens are p's. Under a
   completion of an LM m after prompt, (temper (prompted m prompt) (- (/ 1
   T) 1)) draws each token from the LM's row at temperature T (the end,
   for the end tokens together: see prompted)."
  [p beta]
  (check-potential "temper" p)
  (when-not (and (number? beta) (Double/isFinite (double beta)))
    (throw (ex-info (str "temper: beta is a finite number, not " (pr-str beta)) {:reason :invalid-potential :beta beta})))
  (let [b (double beta)]
    (reify
      EndTokens
      (-end-tokens [_] (end-tokens p))
      Potential
      (-prefix [_ context] (scale b (double (-prefix p context))))
      (-complete [_ context] (scale b (double (-complete p context))))
      (-logw-next [_ context]
        (let [^floats w (-logw-next p context)
              out (float-array (alength w))]
          (dotimes [t (alength w)] (aset out t (unchecked-float (scale b (aget w t)))))
          out)))))

(defn- excluded? [^booleans ends ^long t] (and (< t (alength ends)) (aget ends t)))

(defn- max-included
  "The largest of w's entries but those of the end tokens ends (a boolean
   array by token)."
  ^double [^floats w ^booleans ends]
  (loop [t 0 m Double/NEGATIVE_INFINITY]
    (if (< t (alength w))
      (recur (inc t) (if (excluded? ends t) m (Math/max m (double (aget w t)))))
      m)))

(defn- sum-exp-included
  "Σ exp(w[t] - m) over w's entries but those of the end tokens ends."
  ^double [^floats w ^booleans ends ^double m]
  (loop [t 0 s 0.0]
    (if (< t (alength w))
      (recur (inc t) (if (excluded? ends t) s (+ s (Math/exp (- (double (aget w t)) m)))))
      s)))

(defn- log-normalizer
  "log Σ exp(w[t]) over w's entries but those of the end tokens ends (a
   boolean array by token), the end (the last entry) included: -Inf when
   every one is."
  ^double [^floats w ^booleans ends]
  (let [m (max-included w ends)]
    (if (== m Double/NEGATIVE_INFINITY) m (+ m (Math/log (sum-exp-included w ends m))))))

(defn normalize
  "p with its next-token weights normalized at every context, as a
   distribution over the next token and the end (genlm-control's local
   normalization): q's next weights are p's less log Z, Z the sum of their
   exponentials over the tokens and the end, but p's end tokens (for which
   a completion's end stands). Where p gives no token or end weight, q
   gives none either. q's prefix weight of a context is the sum of its
   next weights along it, and its complete weight that and the end's:
   computing one reads p's next weights at every context along it. Its end
   tokens are p's."
  [p]
  (check-potential "normalize" p)
  (let [ends (end-tokens p)
        mask-of (memoize (fn [^long n] (let [a (boolean-array n)] (doseq [t ends :when (< (long t) n)] (aset a (long t) true)) a)))
        next-weights (fn ^floats [context]
                       (let [^floats w (-logw-next p context)
                             z (log-normalizer w (mask-of (alength w)))
                             out (float-array (alength w) Float/NEGATIVE_INFINITY)]
                         (when-not (== z Double/NEGATIVE_INFINITY)
                           (dotimes [t (alength w)] (aset out t (unchecked-float (- (double (aget w t)) z)))))
                         out))
        prefix (fn ^double [context]
                 (let [n (count context)]
                   (loop [i 0 acc 0.0]
                     (if (== i n)
                       acc
                       (let [x (double (aget ^floats (next-weights (subvec context 0 i)) (int (context i))))]
                         (if (== x Double/NEGATIVE_INFINITY) x (recur (inc i) (+ acc x))))))))]
    (reify
      EndTokens
      (-end-tokens [_] ends)
      Potential
      (-prefix [_ context] (prefix context))
      (-complete [_ context]
        (let [base (double (prefix context))]
          (if (== base Double/NEGATIVE_INFINITY)
            base
            (let [^floats w (next-weights context)]
              (+ base (double (aget w (dec (alength w)))))))))
      (-logw-next [_ context] (next-weights context)))))

;; ---------------------------------------------------------------------------
;; An LM as a potential
;; ---------------------------------------------------------------------------

(defn- max-at
  "The largest of row's entries at the tokens ts (an int array)."
  ^double [^floats row ^ints ts]
  (loop [i 0 m Double/NEGATIVE_INFINITY]
    (if (< i (alength ts)) (recur (inc i) (Math/max m (double (aget row (aget ts i))))) m)))

(defn- sum-exp-at
  "Σ exp(row[t] - m) over the tokens ts."
  ^double [^floats row ^ints ts ^double m]
  (loop [i 0 s 0.0]
    (if (< i (alength ts)) (recur (inc i) (+ s (Math/exp (- (double (aget row (aget ts i))) m)))) s)))

(defn- log-sum-at
  "log Σ exp(row[t]) over the tokens ts (an int array): -Inf for none."
  ^double [^floats row ^ints ts]
  (let [m (max-at row ts)]
    (if (== m Double/NEGATIVE_INFINITY) m (+ m (Math/log (sum-exp-at row ts m))))))

(defn prompted
  "The LM m's weights of what follows prompt (a non-empty token vector), as
   a potential (genlm-control's PromptedLLM): a context's prefix weight is
   the log-probability of its tokens after prompt, its next weights the
   LM's next-token log-probabilities, and the end's the log-probability of
   ending there: log Σ over the end tokens (opts' :end-tokens, by default
   m's :eog), which are its end tokens. complete is prefix and the end's.
   The weights are m's, read through its cache (gen.llm.lm).

   Under a completion of another LM, or of m after another prompt, it
   weighs each token by this one's probability too: token SMC then targets
   their product (genlm's prompt intersection). Tempered, it sets a
   temperature (temper)."
  ([m prompt] (prompted m prompt {}))
  ([m prompt opts]
   (when-not (satisfies? lm/LM m)
     (throw (ex-info (str "prompted: not an LM: " (pr-str m)) {:reason :invalid-potential :lm m})))
   (when-not (and (sequential? prompt) (seq prompt))
     (throw (ex-info (str "prompted: the prompt is a non-empty sequence of tokens, not " (pr-str prompt))
                     {:reason :invalid-token :prompt prompt})))
   (let [{n :n-vocab eog :eog} (lm/vocab m)
         n (long n)
         ends (lm/end-tokens-option "prompted" opts eog n)
         end-array (int-array (sort ends))
         prompt (vec prompt)
         row-of (fn ^floats [context] (lm/next-logprobs m (into prompt context)))
         prefix (fn ^double [context] (if (empty? context) 0.0 (double (reduce + 0.0 (lm/score m prompt context)))))]
     (reify
       EndTokens
       (-end-tokens [_] ends)
       Potential
       (-prefix [_ context] (prefix context))
       (-complete [_ context] (+ (double (prefix context)) (log-sum-at (row-of context) end-array)))
       (-logw-next [_ context]
         (let [^floats row (row-of context)
               out (java.util.Arrays/copyOf row (int (inc n)))]
           (aset out n (unchecked-float (log-sum-at row end-array)))
           out))))))

;; ---------------------------------------------------------------------------
;; Bytes read through a vocabulary
;; ---------------------------------------------------------------------------

(defn- byte-row
  "p's next weights after the bytes bs: 256 bytes and the end."
  ^floats [p bs]
  (let [^floats w (-logw-next p bs)]
    (when-not (== 257 (alength w))
      (throw (ex-info (str "coerce: the potential is over bytes, 256 tokens and the end, not " (dec (alength w)))
                      {:reason :invalid-potential :n-vocab (dec (alength w))})))
    w))

(defn coerce
  "p, a potential over bytes (its tokens the bytes 0-255: a grammar over a
   byte vocabulary, say), read through vocab's tokens (an LM, or a vector
   of byte arrays by token, as gen.llm.grammar takes): a context weighs as
   its tokens' bytes, concatenated, weigh under p, and a next token's
   weight is that of its bytes after them, summed from p's next-byte
   weights along the token byte trie (gen.llm.bytes/weigh, as
   genlm-control's coerce): a step reads p's next weights once at each
   byte prefix of weight that a token extends, and skips the tokens past a
   byte of weight 0. opts: :end-tokens, the tokens that end a text and
   hold none of it (default an LM's :eog, none for a byte vocabulary),
   which are its end tokens; a context holding one has no weight. p
   declares no end tokens of its own (over bytes they mean nothing). p's
   vocabulary is checked when coerce is made, from its next weights at the
   empty context (256 bytes and the end; :invalid-potential otherwise),
   unless it has no weight there: then it has none anywhere, whatever its
   vocabulary."
  ([p vocab] (coerce p vocab {}))
  ([p vocab opts]
   (check-potential "coerce" p)
   (when (seq (end-tokens p))
     (throw (ex-info (str "coerce: a potential over bytes has no end tokens, not " (end-tokens p))
                     {:reason :mismatched-end-tokens :end-tokens (end-tokens p)})))
   (when-not (== Double/NEGATIVE_INFINITY (double (-prefix p [])))
     (byte-row p []))
   (let [byte-vocab (bytes/vocab-bytes "coerce" vocab)
         n (count byte-vocab)
         ends (lm/end-tokens-option "coerce" opts (if (satisfies? lm/LM vocab) (:eog (lm/vocab vocab)) #{}) n)
         trie (bytes/byte-trie byte-vocab ends)
         ;; the bytes of context's tokens, or nil when it holds an end token
         text (fn [context]
                (reduce (fn [acc t]
                          (cond (not (br/index? t n))
                                (throw (ex-info (str "coerce: " (pr-str t) " is no token of the vocabulary")
                                                {:reason :invalid-token :token t}))
                                (contains? ends t) (reduced nil)
                                :else (into acc (map #(bit-and (long %) 0xff)) ^bytes (byte-vocab t))))
                        [] context))]
     (reify
       EndTokens
       (-end-tokens [_] ends)
       Potential
       (-prefix [_ context] (if-let [bs (text context)] (-prefix p bs) Double/NEGATIVE_INFINITY))
       (-complete [_ context] (if-let [bs (text context)] (-complete p bs) Double/NEGATIVE_INFINITY))
       (-logw-next [_ context]
         (let [bs (or (text context)
                      (throw (ex-info "logw-next: the context holds an end token: it has no weight as a prefix"
                                      {:reason :zero-prefix :context context})))
               root (byte-row p bs)
               out (java.util.Arrays/copyOf (bytes/weigh trie root #(byte-row p (into bs %))) (int (inc n)))]
           (aset out n (aget root 256))
           out))))))

;; ---------------------------------------------------------------------------
;; A scope: after a think block
;; ---------------------------------------------------------------------------

(defn after
  "p over the tokens after the first of closes (a set of tokens: a
   reasoning model's close of its think block, Qwen3's </think>, as
   genlm-control's channel scoping): the tokens before it are free (weight
   1, log 0), a close weighs as p's empty context, and from there p weighs
   the tokens after it. A sequence that never closes has no weight as a
   whole (complete -Inf): p's answer must come. Its end tokens are p's.
   Loud when p has no weight at the empty context (:invalid-potential), or
   when a close is no token of p's vocabulary (:invalid-token): checked
   when it is made, from p's next weights at the empty context."
  [closes p]
  (check-potential "after" p)
  (when-not (and (set? closes) (seq closes) (every? #(and (int? %) (not (neg? (long %)))) closes))
    (throw (ex-info (str "after: closes is a non-empty set of tokens, not " (pr-str closes))
                    {:reason :invalid-token :closes closes})))
  (let [start (double (-prefix p []))
        _ (when (== start Double/NEGATIVE_INFINITY)
            (throw (ex-info "after: the potential has no weight at the empty context" {:reason :invalid-potential})))
        n-vocab (dec (alength ^floats (-logw-next p [])))
        closes (into #{} (map long) closes)
        _ (when-let [past (seq (sort (remove #(< (long %) n-vocab) closes)))]
            (throw (ex-info (str "after: the closes " (vec past) " are no tokens of the potential's vocabulary ("
                                 n-vocab " tokens): a sequence would never close")
                            {:reason :invalid-token :closes closes :n-vocab n-vocab})))
        open-row (fn []
                   (let [out (float-array (inc n-vocab))]
                     (doseq [t closes] (aset out (int t) (unchecked-float start)))
                     (aset out n-vocab Float/NEGATIVE_INFINITY)
                     out))
        ;; the tokens after the first close, or nil when there is none
        answer (fn [context]
                 (when-let [i (first (keep-indexed (fn [i t] (when (contains? closes t) i)) context))]
                   (subvec context (inc (long i)))))]
    (reify
      EndTokens
      (-end-tokens [_] (end-tokens p))
      Potential
      (-prefix [_ context] (if-let [a (answer context)] (-prefix p a) 0.0))
      (-complete [_ context] (if-let [a (answer context)] (-complete p a) Double/NEGATIVE_INFINITY))
      (-logw-next [_ context] (if-let [a (answer context)] (-logw-next p a) (open-row))))))
