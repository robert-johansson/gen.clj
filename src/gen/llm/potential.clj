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
     (product p q …)
     (end-tokens p)          → the end tokens p reads contexts with, or nil

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
  (:require [gen.llm.branching :as br]))

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
  (let [declared (distinct (keep end-tokens ps))]
    (when (next declared)
      (throw (ex-info (str "product: the potentials read contexts with different end tokens, " (vec declared))
                      {:reason :mismatched-end-tokens :end-tokens (vec declared)})))
    (product* ps (first declared))))
