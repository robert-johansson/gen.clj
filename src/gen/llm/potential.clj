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
     (potential n-vocab {:prefix f :complete g :logw-next h})
     (product p q …)

   Tokens are those of an LM's vocabulary, 0 to n-vocab - 1; the end is
   not a token, and contexts never hold it (an LM's end-of-generation token
   that ends a completion stands for it: the end's weight is the context's
   as a complete sequence).

   A potential is consistent when, for a context of finite prefix weight,
   (logw-next p ctx) holds prefix(ctx ++ [t]) - prefix(ctx) at t and
   complete(ctx) - prefix(ctx) at the end. A potential computes it as it
   can (a grammar from its automaton's state, a mask); (potential …)
   computes it densely from prefix and complete when given no :logw-next.")

(set! *warn-on-reflection* true)

(defprotocol Potential
  "What a potential implements, called through this namespace's functions."
  (-prefix [p context] "The log weight of context as a prefix, a double.")
  (-complete [p context] "The log weight of context as a whole sequence, a double.")
  (-logw-next [p context] "The next token's log weights: a float array of n-vocab + 1 entries, the end last."))

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
   other two when absent: n-vocab + 2 calls)."
  [n-vocab {:keys [prefix complete logw-next]}]
  (when-not (and (pos-int? n-vocab) (ifn? prefix) (ifn? complete) (or (nil? logw-next) (ifn? logw-next)))
    (throw (ex-info "potential: a vocabulary size and the functions :prefix and :complete (and :logw-next, optionally)"
                    {:n-vocab n-vocab})))
  (let [next (or logw-next #(dense-logw-next n-vocab prefix complete %))]
    (reify Potential
      (-prefix [_ context] (prefix context))
      (-complete [_ context] (complete context))
      (-logw-next [_ context] (next context)))))

(defn product
  "The potential whose log weights are the sums of ps's: each must hold for
   a sequence to (as genlm-control's product)."
  [& ps]
  (when (empty? ps)
    (throw (ex-info "product: at least one potential" {})))
  (reify Potential
    (-prefix [_ context] (reduce + 0.0 (map #(-prefix % context) ps)))
    (-complete [_ context] (reduce + 0.0 (map #(-complete % context) ps)))
    (-logw-next [_ context]
      (let [ws (mapv #(-logw-next % context) ps)
            n (alength ^floats (first ws))]
        (when-not (every? #(= n (alength ^floats %)) ws)
          (throw (ex-info "product: the potentials' vocabularies differ" {:sizes (mapv #(alength ^floats %) ws)})))
        (let [out (float-array n)]
          (doseq [^floats w ws]
            (dotimes [t n] (aset out t (unchecked-float (+ (aget out t) (aget w t))))))
          out)))))
