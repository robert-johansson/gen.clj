(ns gen.llm.token
  "A language model's completions as generative functions (DESIGN.md §6, §9),
   over the LM protocol (gen.llm.lm).

     (completion lm)                       → a generative function of [prompt max-tokens]
     (completion lm {:end-tokens tokens})  → one that ends on these tokens (default: the LM's :eog)
     (dist row)                            → the distribution of one token, given its log-probabilities
     (forced tokens)                       → constraints: tokens at the positions 0, 1, …
     (forced at tokens)                    → at the positions at, at + 1, …
     (from k)                              → the selection of every token from position k on

   A completion's trace holds its tokens at the addresses 0, 1, …: token i
   is drawn from the LM's next-token distribution given the prompt and the
   tokens before it, until an end token (drawn and kept) or max-tokens of
   them. The return value is the vector of tokens. Its operations are the
   GFI's (gen.gfi):

   - generate constrains any tokens: a forced prefix, forced text ((forced
     (lm/encode m text))), single tokens. A run of constrained tokens is
     scored by the LM in one call.
   - update changes tokens or the arguments. The tokens before the first
     change (a constraint equal to the token there is none) keep their
     scores, those after it are rescored through the LM; a sequence that
     now ends sooner discards the rest, one that goes on draws fresh tokens.
   - regenerate resamples the selected tokens and rescores the others after
     the first change (a token redrawn as it was is none); (from k)
     resamples the suffix from position k, the tokens that depend on it.
   - project sums the selected tokens' scores; propose is a simulate.
   - The retdiff of update and regenerate is a vector-diff of the token
     vector (the positions whose tokens changed), or no-change.

   Score once (DESIGN §9): a token's score is the LM's, computed once and
   kept by its trie, so every operation reads the same number for the same
   token in the same context, and the GFI laws hold exactly within one LM
   while it holds the rows it read (on llama.cpp, across LMs or after an
   evicted row is read again, within its noise). A trace keeps its tokens'
   scores, never a branch, and a revision keeps them for every token whose
   context it leaves unchanged (the same prompt, the same tokens before
   it): one that changes no token weighs exactly 0, whatever the LM has
   evicted since.

   Draws are on the host (DESIGN §9, rule 3's one exception): token i is the
   inverse CDF, over the LM's row, of one uniform gen.random derives from
   (fold-in key i), so the same key draws the same tokens from the same
   rows. (On llama.cpp a row read again after an eviction differs within the
   engine's noise, so a run's scores, and rarely its tokens, depend on the
   LM's history too.) Under a batch tracer (gen.batched) the operations
   raise the batched-branch error, and a batched run falls back to one
   particle at a time: a read for each particle's token, where token SMC
   (gen.llm.smc) reads a step's particles in one advance. Tokens are
   discrete: choice-gradients over them is a loud error (gen.gradients)."
  (:require [gen.array :as array]
            [gen.array.batch :as batch]
            [gen.choicemap :as cm]
            [gen.diff :as diff]
            [gen.dist.core :as dc]
            [gen.gfi :as gfi]
            [gen.llm.branching :as br]
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

(def ^:private ^:const slack
  "How far from 1 a row's probabilities may sum: float rows of 151,936
   tokens sum to 1 within 1.3e-9."
  1e-4)

(defn- unnormalized
  "The error for a row whose probabilities do not sum to 1."
  [^double total]
  (ex-info (str "token/dist: the row's probabilities sum to " total ", not 1 (a row is normalized log-probabilities)")
           {:reason :invalid-row :total total}))

(defn- inverse-cdf
  "The token at which the cumulative probability of row passes u (the last
   token of positive probability when rounding leaves the total under u).
   A row that is not normalized is loud as far as the walk sees it, at no
   cost of its own: a total past 1 where the walk stops, or one short of 1
   (or NaN) at the row's end."
  ^long [^floats row ^double u]
  (let [n (alength row)]
    (loop [t 0 total 0.0 positive -1]
      (if (< t n)
        (let [lp (aget row t)
              total (+ total (Math/exp lp))]
          (if (< u total)
            (if (< total (+ 1.0 slack)) t (throw (unnormalized total)))
            (recur (inc t) total (if (Double/isInfinite lp) positive t))))
        (if (< (- 1.0 slack) total (+ 1.0 slack)) positive (throw (unnormalized total)))))))

(defn- draw
  "A token from row with key."
  ^long [^floats row key]
  (inverse-cdf row (uniform key)))

(defn dist
  "The distribution of one token whose log-probabilities, by token, are row
   (a float array, normalized, as gen.llm.lm/next-logprobs gives). A draw is
   the inverse CDF of one uniform gen.random derives from the key, on the
   host, and is loud when it finds that row's probabilities do not sum to
   1; a token's log-probability is its entry in row (-Inf for an integer
   outside the vocabulary). row is read, never written: do not change it
   after."
  [row]
  (when-not (and (instance? float/1 row) (pos? (alength ^floats row)))
    (throw (ex-info (str "token/dist: the log-probabilities are a non-empty float array, not " (pr-str row))
                    {:reason :invalid-row :row row})))
  (dc/->Distribution ::dist {:row row}))

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
    (if (and t (br/index? t (alength row)))
      (double (aget row (int t)))
      Double/NEGATIVE_INFINITY)))

(defmethod dc/-support ::dist [d] (vec (range (alength ^floats (:row (:params d))))))
(defmethod dc/-printed-params ::dist [d] {:n-vocab (alength ^floats (:row (:params d)))})
(defmethod dc/-discrete? ::dist [_] true)
(defmethod dc/-value-shape ::dist [_] [])

;; ---------------------------------------------------------------------------
;; Constraints and selections over positions
;; ---------------------------------------------------------------------------

(defn- check-position [what at]
  (when-not (nat-int? at)
    (throw (ex-info (str what ": a position is a non-negative integer, not " (pr-str at))
                    {:reason :invalid-position :position at}))))

(defn forced
  "The constraints that force tokens at the positions at, at + 1, … (at 0
   by default): a forced prefix, or text as tokens."
  ([tokens] (forced 0 tokens))
  ([at tokens]
   (check-position "token/forced" at)
   (cm/from-map (zipmap (iterate inc (long at)) tokens))))

(defrecord From [^long k]
  sel/ISelection
  (selected? [_ addr] (and (integer? addr) (<= k (long addr))))
  (get-subselection [this addr] (if (sel/selected? this addr) sel/all sel/none)))

(defn from
  "The selection of every token from position k on: regenerating it
   resamples the suffix, the tokens that depend on token k."
  [k]
  (check-position "token/from" k)
  (->From k))

;; ---------------------------------------------------------------------------
;; The completion generative function
;; ---------------------------------------------------------------------------

;; A run's state: {:ctx the prompt and the tokens so far, :tokens, :scores
;; (each token's), :fresh (the positions drawn), :old (the revised trace's
;; tokens and scores while the run is still that trace's, with its prompt
;; and its tokens so far; else nil)}.

(defn- end?
  "Is the run over: max-tokens of them, or its last token an end token?"
  [gf {:keys [tokens]} ^long max-tokens]
  (or (<= max-tokens (count tokens))
      (and (seq tokens) (contains? (:end-tokens gf) (peek tokens)))))

(defn- run-of
  "The known tokens from position i on, as far as they go: up to
   max-tokens, and up to and with the first end token."
  [gf known ^long i ^long max-tokens]
  (loop [j i run []]
    (let [t (when (< j max-tokens) (known j))]
      (if (nil? t)
        run
        (let [run (conj run t)]
          (if (contains? (:end-tokens gf) t) run (recur (inc j) run)))))))

(defn- append [state tokens scores]
  (-> state
      (update :ctx into tokens)
      (update :tokens into tokens)
      (update :scores into scores)))

(defn- agreeing
  "How many of run's first tokens are the revised trace's at their
   positions (none once the run has left it)."
  ^long [{:keys [old tokens]} run]
  (let [old-tokens (:tokens old)
        i (count tokens)
        n (min (count run) (- (count old-tokens) i))]
    (loop [j 0]
      (if (and (< j n) (= (run j) (old-tokens (+ i j)))) (recur (inc j)) j))))

(defn- scored
  "state with run appended: the tokens that are still the revised trace's
   keep its scores (score once, whatever the LM has evicted since); the
   rest are scored by the LM, in one call."
  [m state run]
  (let [j (agreeing state run)
        i (count (:tokens state))
        state (cond-> state
                (pos? j) (append (subvec run 0 j) (subvec (:scores (:old state)) i (+ i j))))]
    (if (< j (count run))
      (let [unscored (subvec run j)]
        (-> state
            (append unscored (lm/score m (:ctx state) unscored))
            (assoc :old nil)))
      state)))

(defn- grow
  "state continued until it ends: the tokens known (known: position → token
   or nil) are taken, a run of them at a time; the others are drawn from
   the LM's row with their site keys, (fold-in key i). Each token is scored
   by the LM (the score it keeps, which is the row's entry unless it read
   the token before), or keeps the revised trace's while the run is that
   trace's."
  [gf state known max-tokens key]
  (let [m (:lm gf)]
    (loop [state state]
      (if (end? gf state max-tokens)
        state
        (let [i (count (:tokens state))
              run (run-of gf known i max-tokens)]
          (recur
           (if (seq run)
             (scored m state run)
             (let [t (draw (lm/next-logprobs m (:ctx state)) (rng/fold-in key i))]
               (update (scored m state [t]) :fresh conj i)))))))))

(defn- start
  "The state before the first token, after prompt; revising trace tr, if
   given, whose scores its tokens keep while the run is tr's."
  ([prompt] (start prompt nil))
  ([prompt tr]
   {:ctx prompt :tokens [] :scores [] :fresh []
    :old (when (and tr (= prompt (first (:args tr)))) {:tokens (:retval tr) :scores (::scores tr)})}))

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
                    {:op op :reason :invalid-args :args args})))
  (let [[prompt max-tokens] args
        n-vocab (long (:n-vocab gf))]
    (when-not (and (sequential? prompt) (seq prompt) (every? #(br/index? % n-vocab) prompt))
      (throw (ex-info (str (name op) ": the prompt is a non-empty vector of tokens (a BOS for an unconditional start), not "
                           (pr-str prompt))
                      {:op op :reason (if (and (sequential? prompt) (empty? prompt)) :no-tokens :invalid-token)
                       :prompt prompt})))
    (when-not (nat-int? max-tokens)
      (throw (ex-info (str (name op) ": max-tokens is a non-negative integer, not " (pr-str max-tokens))
                      {:op op :reason :invalid-args :max-tokens max-tokens})))
    [(vec prompt) (long max-tokens)]))

(defn- positions
  "The tokens choices holds, by position (a sorted map), each checked: a
   position below max-tokens, a token of the vocabulary."
  [op gf choices ^long max-tokens]
  (when (cm/has-value? choices)
    (throw (ex-info (str (name op) ": a completion's choices are tokens by position, not a single value")
                    {:op op :reason :invalid-choices})))
  (let [n-vocab (long (:n-vocab gf))]
    (into (sorted-map)
          (map (fn [[i sub]]
                 (when-not (br/index? i max-tokens)
                   (throw (ex-info (str (name op) ": the constraint at " (pr-str i) " was not visited (a completion of "
                                        max-tokens " tokens has positions 0 to " (dec max-tokens) ")")
                                   {:op op :reason :not-visited :address i :max-tokens max-tokens})))
                 (when-not (cm/has-value? sub)
                   (throw (ex-info (str (name op) ": the choice at " i " is a token, not nested choices")
                                   {:op op :reason :invalid-choices :address i})))
                 (let [t (token-of (name op) (cm/get-value sub))]
                   (when-not (and t (br/index? t n-vocab))
                     (throw (ex-info (str (name op) ": the choice at " i " is not a token of the vocabulary (0 to "
                                          (dec n-vocab) "): " (pr-str (cm/get-value sub)))
                                     {:op op :reason :invalid-token :address i :value (cm/get-value sub)})))
                   [(long i) (long t)])))
          (cm/submaps choices))))

(defn- check-visited
  "Every constrained position is one of the run's (the sequence may have
   ended before it)."
  [op constrained {:keys [tokens]}]
  (when-let [i (first (filter #(<= (count tokens) (long %)) (keys constrained)))]
    (throw (ex-info (str (name op) ": the constraint at " i " was not visited (the completion ended after "
                         (count tokens) " tokens)")
                    {:op op :reason :not-visited :address i :length (count tokens)}))))

(defn- simulate* [gf args key]
  (one-particle! "simulate")
  (let [[prompt max-tokens] (check-args :simulate gf args)]
    (completion-trace gf [prompt max-tokens] (grow gf (start prompt) (constantly nil) max-tokens key))))

(defn- generate* [gf args constraints key]
  (one-particle! "generate")
  (let [[prompt max-tokens] (check-args :generate gf args)
        constrained (positions :generate gf constraints max-tokens)
        state (grow gf (start prompt) constrained max-tokens key)]
    (check-visited :generate constrained state)
    {:trace (completion-trace gf [prompt max-tokens] state)
     :weight (sum (map (:scores state) (keys constrained)))}))

(defn- assess* [gf args choices]
  (one-particle! "assess")
  (let [[prompt max-tokens] (check-args :assess gf args)
        given (positions :assess gf choices max-tokens)
        tokens (vec (vals given))
        n (count tokens)
        end-tokens (:end-tokens gf)]
    (when-not (= (vec (keys given)) (vec (range n)))
      (throw (ex-info (str "assess: the tokens are at the positions 0 to " (dec n) " without a gap, not "
                           (vec (keys given)))
                      {:op :assess :reason :missing-choice :positions (vec (keys given))})))
    (when-let [i (first (filter #(contains? end-tokens (tokens %)) (range (dec n))))]
      (throw (ex-info (str "assess: the constraint at " (inc (long i)) " was not visited (the end token at " i
                           " ends the completion)")
                      {:op :assess :reason :not-visited :address (inc (long i))})))
    (when-not (or (= n max-tokens) (and (pos? n) (contains? end-tokens (peek tokens))))
      (throw (ex-info (str "assess: the tokens end at " n ", before an end token or max-tokens (" max-tokens
                           "): position " n " is unconstrained")
                      {:op :assess :reason :missing-choice :address n :max-tokens max-tokens})))
    {:weight (if (zero? n) 0.0 (sum (lm/score (:lm gf) prompt tokens)))
     :retval tokens}))

(defn- retdiff
  "How the tokens went from before to after: no-change, or a vector-diff
   naming the retained positions whose tokens differ."
  [before after]
  (if (= before after)
    diff/no-change
    (diff/vector-diff (count after) (count before)
                      (into {} (keep #(when (not= (before %) (after %)) [% diff/unknown-change]))
                            (range (min (count before) (count after)))))))

(defn- revised
  "An update's or regenerate's result: the trace of state, the weight, the
   retdiff."
  [gf args tr state weight]
  {:trace (completion-trace gf args state)
   :weight weight
   :retdiff (retdiff (:retval tr) (:tokens state))})

(defn- update* [gf tr args constraints key]
  (one-particle! "update")
  (let [[prompt max-tokens :as args] (check-args :update gf args)
        old (:retval tr)
        n-old (count old)
        constrained (positions :update gf constraints max-tokens)]
    (if (and (= args (:args tr)) (empty? constrained))
      ;; nothing changes: the trace as it is
      {:trace tr :weight 0.0 :retdiff diff/no-change :discard cm/EMPTY}
      (let [known (fn [i] (or (constrained i) (when (< (long i) n-old) (old i))))
            state (grow gf (start prompt tr) known max-tokens key)
            n-new (count (:tokens state))
            _ (check-visited :update constrained state)
            discard (choices-at old (into (sorted-set) cat [(filter #(< (long %) n-old) (keys constrained))
                                                            (range (min n-new n-old) n-old)]))]
        (assoc (revised gf args tr state
                        (- (sum (:scores state)) (double (:score tr)) (sum (map (:scores state) (:fresh state)))))
               :discard discard)))))

(defn- regenerate* [gf tr args selection key]
  (one-particle! "regenerate")
  (let [[prompt max-tokens :as args] (check-args :regenerate gf args)
        old (:retval tr)
        n-old (count old)
        selected? (fn [i] (sel/selected? selection i))]
    (if (and (= args (:args tr)) (not-any? selected? (range n-old)))
      ;; nothing selected, nothing changes: the trace as it is
      {:trace tr :weight 0.0 :retdiff diff/no-change}
      (let [known (fn [i] (when (and (< (long i) n-old) (not (selected? i))) (old i)))
            state (grow gf (start prompt tr) known max-tokens key)
            retained (filter #(and (< (long %) n-old) (not (selected? %))) (range (count (:tokens state))))]
        (revised gf args tr state
                 (sum (map #(- (double ((:scores state) %)) (double ((::scores tr) %))) retained)))))))

(defn- project* [tr selection]
  (sum (keep-indexed (fn [i s] (when (sel/selected? selection i) s)) (::scores tr))))

(gfi/defgf Completion [lm end-tokens n-vocab]
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
   … (see the namespace's docstring). opts: :end-tokens, the tokens that
   end a completion (a set; default the LM's end-of-generation tokens,
   :eog; #{} for completions of exactly max-tokens)."
  ([lm] (completion lm {}))
  ([lm {:keys [end-tokens] :as opts}]
   (when-not (satisfies? lm/LM lm)
     (throw (ex-info (str "token/completion: a completion is over an LM (gen.llm.lm), not " (pr-str lm))
                     {:reason :invalid-lm :lm lm})))
   (let [{:keys [n-vocab eog]} (lm/vocab lm)
         end-tokens (if (contains? opts :end-tokens) end-tokens eog)]
     (when-not (and (set? end-tokens) (every? #(br/index? % n-vocab) end-tokens))
       (throw (ex-info (str "token/completion: :end-tokens is a set of tokens of the vocabulary, not " (pr-str end-tokens))
                       {:reason :invalid-end-tokens :end-tokens end-tokens})))
     (->Completion lm (into #{} (map long) end-tokens) (long n-vocab)))))
