(ns gen.llm.token
  "A language model's completions as generative functions (DESIGN.md §6, §9),
   over the LM protocol (gen.llm.lm).

     (completion lm)                → a generative function of [prompt max-tokens]
     (completion lm {:eos tokens})  → one that ends on these tokens (default: the LM's :eog)
     (dist row)                     → the distribution of one token, given its log-probabilities
     (forced tokens)                → constraints: tokens at the positions 0, 1, …
     (forced at tokens)             → at the positions at, at + 1, …
     (from k)                       → the selection of every token from position k on

   A completion's trace holds its tokens at the addresses 0, 1, …: token i
   is drawn from the LM's next-token distribution given the prompt and the
   tokens before it, until an end token (drawn and kept) or max-tokens of
   them. The return value is the vector of tokens. Its operations are the
   GFI's (gen.gfi):

   - generate constrains any tokens: a forced prefix, forced text ((forced
     (lm/encode m text))), single tokens. A run of constrained tokens is
     scored by the LM in one call.
   - update changes tokens or the arguments. The tokens after the first
     change are rescored through the LM, those before keep their scores; a
     sequence that now ends sooner discards the rest, one that goes on draws
     fresh tokens.
   - regenerate resamples the selected tokens and rescores the others after
     the first; (from k) resamples the suffix from position k, the tokens
     that depend on it.
   - project sums the selected tokens' scores; propose is a simulate.

   Score once (DESIGN §9): a token's score is the LM's, computed once and
   kept by its trie, so every operation reads the same number for the same
   token in the same context and the GFI laws hold exactly within one LM (on
   llama.cpp, across LMs or after an evicted row is read again, within its
   noise). A trace keeps its tokens' scores, never a branch.

   Draws are on the host (DESIGN §9, rule 3's one exception): token i is the
   inverse CDF, over the LM's row, of one uniform gen.random derives from
   (fold-in key i), so runs are pure functions of their keys. Under a batch
   tracer (gen.batched) the operations raise the batched-branch error, and
   a batched run falls back to one particle at a time. Tokens are discrete:
   choice-gradients over them is a loud error (gen.gradients)."
  (:refer-clojure :exclude [-])
  (:require [gen.array :as array :refer [-]]
            [gen.array.batch :as batch]
            [gen.choicemap :as cm]
            [gen.diff :as diff]
            [gen.dist.core :as dc]
            [gen.gfi :as gfi]
            [gen.llm.lm :as lm]
            [gen.random :as rng]
            [gen.selection :as sel]
            [gen.trace :as trace]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

;; ---------------------------------------------------------------------------
;; One token: the distribution and its draw
;; ---------------------------------------------------------------------------

(defn- one-particle!
  "Token draws are on the host, one particle at a time: under a batch tracer,
   the batched-branch error, so that gen.batched falls back."
  [what]
  (when (batch/batch-backend? array/*backend*)
    (throw (array/batched-branch (str what ": token draws are on the host, one particle at a time") {}))))

(defn- uniform
  "The uniform in [0, 1) gen.random derives from key, a host double."
  ^double [key]
  (double (rng/uniform key [] {:backend (array/backend-named :jvm) :dtype :float64})))

(defn- inverse-cdf
  "The token at which the cumulative probability of row passes u (the last
   token of positive probability when rounding leaves the total under u)."
  ^long [^floats row ^double u]
  (let [n (alength row)]
    (loop [t 0 total 0.0 last -1]
      (if (< t n)
        (let [lp (aget row t)
              total (+ total (Math/exp lp))]
          (if (< u total)
            t
            (recur (inc t) total (if (Double/isInfinite lp) last t))))
        last))))

(defn- draw
  "A token from row with key."
  ^long [^floats row key]
  (inverse-cdf row (uniform key)))

(def ^:private float-array-class (Class/forName "[F"))

(defn dist
  "The distribution of one token whose log-probabilities, by token, are row
   (a float array, normalized, as gen.llm.lm/next-logprobs gives). A draw is
   the inverse CDF of one uniform gen.random derives from the key, on the
   host; a token's log-probability is its entry in row (-Inf for an integer
   outside the vocabulary). row is read, never written: do not change it
   after."
  [row]
  (when-not (and (instance? float-array-class row) (pos? (alength ^floats row)))
    (throw (ex-info (str "token/dist: the log-probabilities are a non-empty float array, not " (pr-str row))
                    {:reason :invalid-row :row row})))
  (dc/->Distribution ::dist {:row row}))

(defn- index?
  "Is x an integer in [0, n)? (Generic comparison: a big integer is out of
   range, not an error.)"
  [x n]
  (and (integer? x) (not (neg? (compare x 0))) (neg? (compare x n))))

(defn- token-of
  "v as a host integer: v itself, or a concrete 0-d array's item; nil for any
   other number. Not a number: a loud error."
  [what v]
  (let [v (if (and (array/array? v) (not (array/host? v)) (array/concrete? v) (empty? (array/shape v)))
            (array/item v)
            v)]
    (cond (integer? v) v
          (number? v) nil
          :else (throw (ex-info (str what ": a token is an integer, not " (pr-str v)) {:reason :invalid-token :value v})))))

(defmethod dc/-sample ::dist [d key batch-shape]
  (one-particle! "token/dist")
  (when (seq batch-shape)
    (throw (array/batched-branch "token/dist draws one token at a time" {:batch-shape batch-shape})))
  (draw (:row (:params d)) key))

(defmethod dc/-log-prob ::dist [d v]
  (let [^floats row (:row (:params d))
        t (token-of "token/dist" v)]
    (if (and t (index? t (alength row)))
      (double (aget row (int t)))
      Double/NEGATIVE_INFINITY)))

(defmethod dc/-support ::dist [d] (vec (range (alength ^floats (:row (:params d))))))
(defmethod dc/-discrete? ::dist [_] true)
(defmethod dc/-value-shape ::dist [_] [])

;; ---------------------------------------------------------------------------
;; Constraints and selections over positions
;; ---------------------------------------------------------------------------

(defn forced
  "The constraints that force tokens at the positions at, at + 1, … (at 0
   by default): a forced prefix, or text as tokens."
  ([tokens] (forced 0 tokens))
  ([at tokens]
   (when-not (nat-int? at)
     (throw (ex-info (str "token/forced: a position is a non-negative integer, not " (pr-str at)) {:position at})))
   (cm/from-map (zipmap (iterate inc (long at)) tokens))))

(defrecord ^:private From [^long k]
  sel/ISelection
  (selected? [_ addr] (and (integer? addr) (<= k (long addr))))
  (get-subselection [this addr] (if (sel/selected? this addr) sel/all sel/none)))

(defn from
  "The selection of every token from position k on: regenerating it
   resamples the suffix, the tokens that depend on token k."
  [k]
  (when-not (nat-int? k)
    (throw (ex-info (str "token/from: a position is a non-negative integer, not " (pr-str k)) {:position k})))
  (->From k))

;; ---------------------------------------------------------------------------
;; The completion generative function
;; ---------------------------------------------------------------------------

;; A run's state: {:ctx the prompt and the tokens so far, :tokens, :scores
;; (each token's, from the LM), :fresh (the positions drawn)}.

(defn- end?
  "Is the run over: max tokens, or its last token an end token?"
  [gf {:keys [tokens]} ^long max]
  (or (<= max (count tokens))
      (and (seq tokens) (contains? (:eos gf) (peek tokens)))))

(defn- run-of
  "The known tokens from position i on, as far as they go: up to max, and
   up to and with the first end token."
  [gf known ^long i ^long max]
  (loop [j i run []]
    (let [t (when (< j max) (known j))]
      (if (nil? t)
        run
        (let [run (conj run t)]
          (if (contains? (:eos gf) t) run (recur (inc j) run)))))))

(defn- append [state tokens scores]
  (-> state
      (update :ctx into tokens)
      (update :tokens into tokens)
      (update :scores into scores)))

(defn- grow
  "state continued until it ends: the tokens known (known: position → token
   or nil) are scored by the LM, a run of them in one call; the others are
   drawn from the LM's row with their site keys, (fold-in key i), and scored
   by the LM (the score it keeps, which is the row's entry unless it read the
   token before)."
  [gf state known max key]
  (let [m (:lm gf)]
    (loop [state state]
      (if (end? gf state max)
        state
        (let [i (count (:tokens state))
              run (run-of gf known i max)]
          (recur
           (if (seq run)
             (append state run (lm/score m (:ctx state) run))
             (let [t (draw (lm/next-logprobs m (:ctx state)) (rng/fold-in key i))]
               (-> (append state [t] (lm/score m (:ctx state) [t]))
                   (update :fresh conj i))))))))))

(defn- start
  "The state after the first k tokens of (old) tokens and scores, with
   prompt."
  [prompt tokens scores ^long k]
  {:ctx (into prompt (subvec tokens 0 k))
   :tokens (subvec tokens 0 k)
   :scores (subvec scores 0 k)
   :fresh []})

(defn- choices-of [tokens]
  (cm/from-map (zipmap (range) tokens)))

(defn- choices-at
  "The choices of tokens at the given positions."
  [tokens positions]
  (cm/from-map (into {} (map (fn [i] [i (tokens i)])) positions)))

(defn- sum ^double [scores]
  (reduce (fn [^double acc s] (+ acc (double s))) 0.0 scores))

(defn- completion-trace [gf args {:keys [tokens scores]}]
  (assoc (trace/make-trace {:gen-fn gf :args args :choices (choices-of tokens) :retval tokens :score (sum scores)})
         ::scores scores))

(defn- check-args
  "[prompt max-tokens], checked: a non-empty token vector and a
   non-negative integer."
  [op gf args]
  (when-not (= 2 (count args))
    (throw (ex-info (str (name op) ": a completion takes [prompt max-tokens], not " (count args) " arguments")
                    {:op op :args args})))
  (let [[prompt max] args
        n-vocab (long (:n-vocab gf))]
    (when-not (and (sequential? prompt) (seq prompt) (every? #(index? % n-vocab) prompt))
      (throw (ex-info (str (name op) ": the prompt is a non-empty vector of tokens (a BOS for an unconditional start), not "
                           (pr-str prompt))
                      {:op op :prompt prompt :reason :invalid-token})))
    (when-not (nat-int? max)
      (throw (ex-info (str (name op) ": max-tokens is a non-negative integer, not " (pr-str max)) {:op op :max-tokens max})))
    [(vec prompt) (long max)]))

(defn- positions
  "The tokens choices holds, by position (a sorted map), each checked: a
   position below max, a token of the vocabulary."
  [op gf choices ^long max]
  (when (cm/has-value? choices)
    (throw (ex-info (str (name op) ": a completion's choices are tokens by position, not a single value")
                    {:op op})))
  (let [n-vocab (long (:n-vocab gf))]
    (into (sorted-map)
          (map (fn [[i sub]]
                 (when-not (index? i max)
                   (throw (ex-info (str (name op) ": the constraint at " (pr-str i) " was not visited (a completion of "
                                        max " tokens has positions 0 to " (dec max) ")")
                                   {:op op :address i :max-tokens max})))
                 (when-not (cm/has-value? sub)
                   (throw (ex-info (str (name op) ": the choice at " i " is a token, not nested choices")
                                   {:op op :address i})))
                 (let [t (token-of (name op) (cm/get-value sub))]
                   (when-not (and t (index? t n-vocab))
                     (throw (ex-info (str (name op) ": the choice at " i " is not a token of the vocabulary (0 to "
                                          (dec n-vocab) "): " (pr-str (cm/get-value sub)))
                                     {:op op :address i :value (cm/get-value sub) :reason :invalid-token})))
                   [(long i) (long t)])))
          (cm/submaps choices))))

(defn- check-visited
  "Every constrained position is one of the run's (the sequence may have
   ended before it)."
  [op constrained {:keys [tokens]}]
  (when-let [i (first (filter #(<= (count tokens) (long %)) (keys constrained)))]
    (throw (ex-info (str (name op) ": the constraint at " i " was not visited (the completion ended after "
                         (count tokens) " tokens)")
                    {:op op :address i :length (count tokens)}))))

(defn- simulate* [gf args key]
  (one-particle! "simulate")
  (let [[prompt max] (check-args :simulate gf args)]
    (completion-trace gf [prompt max] (grow gf (start prompt [] [] 0) (constantly nil) max key))))

(defn- generate* [gf args constraints key]
  (one-particle! "generate")
  (let [[prompt max] (check-args :generate gf args)
        constrained (positions :generate gf constraints max)
        state (grow gf (start prompt [] [] 0) constrained max key)]
    (check-visited :generate constrained state)
    {:trace (completion-trace gf [prompt max] state)
     :weight (sum (map (:scores state) (keys constrained)))}))

(defn- assess* [gf args choices]
  (one-particle! "assess")
  (let [[prompt max] (check-args :assess gf args)
        given (positions :assess gf choices max)
        tokens (vec (vals given))
        n (count tokens)]
    (when-not (= (vec (keys given)) (vec (range n)))
      (throw (ex-info (str "assess: the tokens are at the positions 0 to " (dec n) " without a gap, not "
                           (vec (keys given)))
                      {:positions (vec (keys given))})))
    (when-let [i (first (filter #(contains? (:eos gf) (tokens %)) (range (dec n))))]
      (throw (ex-info (str "assess: the constraint at " (inc (long i)) " was not visited (the end token at " i
                           " ends the completion)")
                      {:address (inc (long i))})))
    (when-not (or (= n max) (and (pos? n) (contains? (:eos gf) (peek tokens))))
      (throw (ex-info (str "assess: the tokens end at " n ", before an end token or max-tokens (" max
                           "): position " n " is unconstrained")
                      {:length n :max-tokens max})))
    {:weight (if (zero? n) 0.0 (sum (lm/score (:lm gf) prompt tokens)))
     :retval tokens}))

(defn- revised
  "An update's or regenerate's result: the trace of state, the weight, the
   retdiff."
  [gf args tr state weight]
  {:trace (completion-trace gf args state)
   :weight weight
   :retdiff (if (= (:tokens state) (:retval tr)) diff/no-change diff/unknown-change)})

(defn- update* [gf tr args constraints key]
  (one-particle! "update")
  (let [[prompt max] (check-args :update gf args)
        old (:retval tr)
        n-old (count old)
        constrained (positions :update gf constraints max)
        ;; the tokens before the first change keep their contexts and scores
        k (min (long (if (= prompt (first (:args tr))) (reduce min n-old (keys constrained)) 0)) (long max))
        known (fn [i] (or (constrained i) (when (< (long i) n-old) (old i))))
        state (grow gf (start prompt old (::scores tr) k) known max key)
        n-new (count (:tokens state))
        _ (check-visited :update constrained state)
        discard (choices-at old (into (sorted-set) cat [(filter #(< (long %) n-old) (keys constrained))
                                                        (range (min n-new n-old) n-old)]))]
    (assoc (revised gf [prompt max] tr state
                    (- (- (sum (:scores state)) (:score tr)) (sum (map (:scores state) (:fresh state)))))
           :discard discard)))

(defn- regenerate* [gf tr args selection key]
  (one-particle! "regenerate")
  (let [[prompt max] (check-args :regenerate gf args)
        old (:retval tr)
        n-old (count old)
        selected? (fn [i] (sel/selected? selection i))
        k (min (long (if (= prompt (first (:args tr))) (or (first (filter selected? (range n-old))) n-old) 0)) (long max))
        known (fn [i] (when (and (< (long i) n-old) (not (selected? i))) (old i)))
        state (grow gf (start prompt old (::scores tr) k) known max key)
        retained (filter #(and (< (long %) n-old) (not (selected? %))) (range k (count (:tokens state))))]
    (revised gf [prompt max] tr state
             (sum (map #(- (double ((:scores state) %)) (double ((::scores tr) %))) retained)))))

(defn- project* [tr selection]
  (sum (keep-indexed (fn [i s] (when (sel/selected? selection i) s)) (::scores tr))))

(defmacro ^:private defgf
  "defrecord, plus untraced calls as gen.combinators': (gf prompt n) is the
   return value of a simulated run."
  [name fields & impls]
  `(defrecord ~name ~fields
     ~@impls
     clojure.lang.IFn
     ~@(for [n (range 3)]
         (let [as (vec (repeatedly n #(gensym "arg")))]
           `(~'invoke [this# ~@as] (:retval (gfi/simulate this# ~as)))))
     (~'applyTo [this# args#] (:retval (gfi/simulate this# (vec args#))))))

(defgf Completion [lm eos n-vocab]
  gfi/IGenerativeFunction
  (-simulate [gf args key] (simulate* gf args key))
  (-generate [gf args constraints key] (generate* gf args constraints key))
  (-update [gf tr args _argdiffs constraints key] (update* gf tr args constraints key))
  (-regenerate [gf tr args _argdiffs selection key] (regenerate* gf tr args selection key))
  (-assess [gf args choices] (assess* gf args choices))
  (-propose [gf args key]
    (let [tr (simulate* gf args key)]
      {:choices (:choices tr) :weight (:score tr) :retval (:retval tr)}))
  (-project [_ tr selection] (project* tr selection)))

(defmethod print-method Completion [gf ^java.io.Writer w]
  (.write w (str "#gen/completion[" (:lm gf) "]")))

(defn completion
  "The generative function of lm's completions of a prompt: its arguments
   are [prompt max-tokens], a token vector and a non-negative integer, and
   its trace holds the tokens drawn after the prompt at the positions 0, 1,
   … (see the namespace's docstring). opts: :eos, the tokens that end a
   completion (a set; default the LM's end-of-generation tokens, :eog; #{}
   for completions of exactly max-tokens)."
  ([lm] (completion lm {}))
  ([lm {:keys [eos] :as opts}]
   (let [{:keys [n-vocab eog]} (lm/vocab lm)
         eos (if (contains? opts :eos) eos eog)]
     (when-not (and (set? eos) (every? #(index? % n-vocab) eos))
       (throw (ex-info (str "token/completion: :eos is a set of tokens of the vocabulary, not " (pr-str eos))
                       {:eos eos})))
     (->Completion lm (into #{} (map long) eos) (long n-vocab)))))
