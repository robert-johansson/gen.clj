;; book: ns
(ns book.constrained-generation
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.string :as str]
            [gen.array :as array :refer [+ - * /]]
            [gen.book.svg :as svg]
            [gen.gfi :as gfi]
            [gen.llm.byte-tokenizer :as bt]
            [gen.llm.bytes :as bytes]
            [gen.llm.constrained :as constrained]
            [gen.llm.dfa :as dfa]
            [gen.llm.grammar :as grammar]
            [gen.llm.lm :as lm]
            [gen.llm.potential :as potential]
            [gen.llm.regex :as regex]
            [gen.llm.smc :as smc]
            [gen.llm.token :as token]
            [gen.llm.toy-engine :as toy]
            [gen.random :as rng]))

;; book: vocabulary
(def tokenizer (bt/tokenizer [" " "y" "es" "yes" "n" "o" "no"]))

(def m
  (lm/lm (toy/engine {:n-vocab (:n-vocab tokenizer)
                      :max-branches 256 :n-ctx 64})
         tokenizer {:max-cells 16384}))

(lm/vocab m)
;=> {:n-vocab 9 :bos 7 :eos 8 :eog #{8} :add-bos? true}

(def n-vocab (:n-vocab (lm/vocab m)))

(def end (:eos (lm/vocab m)))

;; book: tokens
(defn token-text
  "The text of token t."
  [t]
  (lm/decode m [t]))

(mapv token-text (range n-vocab))
;=> [" " "y" "es" "yes" "n" "o" "no" "<s>" "</s>"]

;; book: prompt
(def prompt (lm/encode m ""))

prompt
;=> [7]

(def max-tokens 3)

(def args [prompt max-tokens])

(def free (token/completion m))

(:retval (gfi/simulate free args (rng/key 1)))
;=> [6 3 8] ; float64

;; book: enumerate
(defn completions
  "Every completion of the prompt by m, of at most n tokens, with its log
   probability: [tokens log-p] pairs. A completion ends at the end token
   or after n tokens."
  [n]
  (letfn [(grow [tokens log-p]
            (if (or (= n (count tokens)) (= end (peek tokens)))
              [[tokens log-p]]
              (let [^floats row (lm/next-logprobs m (into prompt tokens))]
                (mapcat #(grow (conj tokens %)
                               (+ log-p (double (aget row (int %)))))
                        (range n-vocab)))))]
    (grow [] 0.0)))

(def every-completion (completions max-tokens))

(count every-completion)
;=> 585

;; book: enumerate-check
(array/allclose? 1.0 (reduce + (map #(Math/exp (second %)) every-completion)))
;=> true

(every? (fn [[x log-p]]
          (array/allclose?
           log-p (:weight (gfi/assess free args (token/forced x)))))
        every-completion)
;=> true

;; book: body
(defn body
  "The tokens of completion x before its end token: its text's."
  [x]
  (if (= end (peek x)) (pop x) x))

(defn text
  "The text of completion x."
  [x]
  (lm/decode m (body x)))

(mapv text [[6 3 8] [3 8] [1 2 8] [0 4 5]])
;=> ["noyes" "yes" "yes" " no"]

;; book: lipogram
(def o-free
  "A lipogram: weight 1 for a text with no o, 0 for any other."
  (let [w (fn [ctx] (if (str/includes? (lm/decode m ctx) "o") ##-Inf 0.0))]
    (potential/potential n-vocab {:prefix w :complete w})))

[(potential/prefix o-free [1 2]) (potential/prefix o-free [1 5])]
;=> [0.0 ##-Inf]

;; book: next-weights
(vec (potential/logw-next o-free []))
;=> [0.0 0.0 0.0 0.0 0.0 ##-Inf ##-Inf 0.0 0.0 0.0]

;; book: brevity
(def brevity
  "Each token of text costs half a nat."
  (let [w (fn [ctx] (* -0.5 (count ctx)))]
    (potential/potential n-vocab {:prefix w :complete w})))

(vec (potential/logw-next brevity [1]))
;=> [-0.5 -0.5 -0.5 -0.5 -0.5 -0.5 -0.5 -0.5 -0.5 0.0]

;; book: exact-z
(defn exact-z
  "Z = Σ p(x) exp(complete(x)) over every completion x of at most n tokens
   (max-tokens by default): the normalizer of the LM's completions weighed
   by the potential pot."
  ([pot] (exact-z pot max-tokens))
  ([pot n]
   (reduce + (for [[x log-p] (if (= n max-tokens)
                               every-completion
                               (completions n))]
               (Math/exp (+ log-p (potential/complete pot (body x))))))))

[(exact-z o-free) (exact-z brevity)
 (exact-z (potential/product o-free brevity))]
;=> [0.5006 0.2579 0.1371]

;; book: regex
(def yes-no-dfa (regex/dfa "yes|no"))

(select-keys (dfa/->data yes-no-dfa) [:start :accept])
;=> {:start 0 :accept #{3}}

(:transitions (dfa/->data yes-no-dfa))
;=> {0 {110 1 121 2} 1 {111 3} 2 {101 4} 3 {} 4 {115 3}}

(mapv #(regex/matches? yes-no-dfa %) ["yes" "no" "ye" "nope"])
;=> [true true false false]

;; book: grammar
(def yes-no (grammar/grammar m "yes|no"))

(select-keys (grammar/info yes-no) [:states :reachable :live])
;=> {:states 5 :reachable 4 :live 4}

(defn allowed-after
  "The texts of the tokens grammar g allows after tokens."
  [g tokens]
  (mapv token-text (grammar/allowed g (grammar/state g tokens))))

[(allowed-after yes-no []) (allowed-after yes-no [1])
 (allowed-after yes-no [3])]
;=> [["y" "yes" "n" "no"] ["es"] []]

;; book: token-graph
(defn token-graph
  "The states of grammar g that its tokens reach from the start, as
   svg/dag's nodes and edges: a state is a node (a double circle when it
   accepts), each token between two states a text node on the edge, and
   an edge to the end from each accepting state."
  [g]
  (let [start (grammar/state g [])
        moves (fn [s] (for [t (grammar/allowed g s)]
                        [s t (grammar/advance g s t)]))
        edges (loop [todo [start] seen #{start} out []]
                (if-let [s (first todo)]
                  (let [ms (moves s)
                        new (remove seen (distinct (map peek ms)))]
                    (recur (into (subvec todo 1) new) (into seen new)
                           (into out ms)))
                  out))
        states (distinct (cons start (map peek edges)))]
    [(concat (for [s states]
               {:id s :label (str s)
                :shape (if (grammar/accepting? g s) :circle :ellipse)})
             (for [[s t] edges] {:id [s t] :label (token-text t) :shape :text})
             [{:id :end :label "end" :shape :box}])
     (concat (mapcat (fn [[s t s']] [[s [s t]] [[s t] s']]) edges)
             (for [s states :when (grammar/accepting? g s)] [s :end]))]))

;; book: grammar-figure
;; figure: figures/044-grammar.svg
(apply svg/dag (token-graph yes-no))

;; book: grammar-potential
[(potential/prefix yes-no [1]) (potential/complete yes-no [1])
 (potential/complete yes-no [1 2])]
;=> [0.0 ##-Inf 0.0]

(mapv #(potential/complete yes-no %) [[3] [1 2] [6] [4 5] [0 6]])
;=> [0.0 0.0 0.0 0.0 ##-Inf]

;; book: consistent
(defn consistent?
  "Does potential pot's logw-next after ctx hold prefix(ctx ++ [t]) -
   prefix(ctx) at each token t, and complete(ctx) - prefix(ctx) at the
   end?"
  [pot ctx]
  (let [base (potential/prefix pot ctx)
        ws (vec (potential/logw-next pot ctx))]
    (and (every? #(== (ws %) (- (potential/prefix pot (conj ctx %)) base))
                 (range n-vocab))
         (== (peek ws) (- (potential/complete pot ctx) base)))))

(every? #(consistent? yes-no %) [[] [1] [3] [4] [6] [1 2] [4 5]])
;=> true

;; book: grammar-check
(every? (fn [[x _]]
          (= (zero? (potential/complete yes-no (body x)))
             (boolean (re-matches #"yes|no" (text x)))))
        every-completion)
;=> true

(def z (exact-z yes-no))

z
;=> 0.01724

;; book: text-z
(let [b (bytes/byte-lm m)
      p #(Math/exp (bytes/text-logprob b prompt %))]
  (array/allclose? z (+ (p "yes") (p "no"))))
;=> true

;; book: yes-nope
(def yes-nope (grammar/grammar m "yes|nope"))

(allowed-after yes-nope [])
;=> ["y" "yes"]

(select-keys (grammar/info yes-nope) [:states :reachable :live])
;=> {:states 7 :reachable 5 :live 3}

;; book: byte-live
(def byte-live
  "yes|nope read byte by byte: a text is a prefix when its bytes begin a
   string of the language, whatever the tokens can spell."
  (let [d (regex/dfa "yes|nope")
        utf-8 (fn [^String s] (.getBytes s "UTF-8"))
        state (fn [ctx] (dfa/walk d (dfa/start d) (utf-8 (lm/decode m ctx))))]
    (potential/potential
     n-vocab {:prefix #(if (neg? (state %)) ##-Inf 0.0)
              :complete #(if (dfa/accepting? d (state %)) 0.0 ##-Inf)})))

[(potential/prefix byte-live [6]) (potential/prefix yes-nope [6])]
;=> [0.0 ##-Inf]

;; book: stranded
;; error: token/completion: the potential allows no token at position 1
#_(gfi/generate (token/completion m {:potential byte-live}) args
                (token/forced [6]) (rng/key 1))

;; book: inexpressible
;; error: grammar: no sequence of the vocabulary's tokens spells
#_(grammar/grammar m "nope")

;; book: constrained
(def yes-or-no (token/completion m {:potential yes-no}))

(mapv #(text (:retval (gfi/simulate yes-or-no args (rng/key %)))) (range 8))
;=> ["no" "no" "yes" "yes" "yes" "yes" "yes" "yes"] ; float64

;; book: site
(defn site
  "The log-probabilities of the next token after tokens under potential
   pot, by their definition: the LM's row plus pot's next-token weights
   (an end token takes the end's), renormalized."
  [pot tokens]
  (let [^floats row (lm/next-logprobs m (into prompt tokens))
        ^floats ws (potential/logw-next pot tokens)
        lw (mapv (fn [t] (+ (double (aget row (int t)))
                            (double (aget ws (int (if (= t end) n-vocab t))))))
                 (range n-vocab))
        log-z (Math/log (reduce + (map #(Math/exp %) lw)))]
    (mapv #(- % log-z) lw)))

(mapv #(Math/exp %) (site yes-no []))
;=> [0.0 0.0258 0.0 0.761 0.006 0.0 0.2072 0.0 0.0]

(let [[row] (constrained/weighted #{end} (lm/next-logprobs m prompt)
                                  (potential/logw-next yes-no []))]
  (array/allclose? (array/array (vec row)) (array/array (site yes-no []))))
;=> true

;; book: local
(defn local-log-q
  "log q(x): the log-probabilities of x's tokens at their sites under
   potential pot, summed; -Inf from the first token pot rules out."
  [pot x]
  (reduce (fn [acc i]
            (let [lq ((site pot (subvec x 0 i)) (x i))]
              (if (Double/isInfinite lq) (reduced lq) (+ acc lq))))
          0.0 (range (count x))))

(every? (fn [[x _]]
          (array/allclose?
           (local-log-q yes-no x)
           (:weight (gfi/assess yes-or-no args (token/forced x)))))
        every-completion)
;=> true

;; book: pi-and-q
(defn conditional
  "The LM's completions weighed by potential pot, exactly: a map from
   each completion x of positive weight to π(x) = p(x) exp(complete(x)) /
   Z."
  [pot]
  (let [z (exact-z pot)]
    (into {} (for [[x log-p] every-completion
                   :let [w (potential/complete pot (body x))]
                   :when (Double/isFinite w)]
               [x (/ (Math/exp (+ log-p w)) z)]))))

(defn local
  "The constrained completion's distribution under potential pot: a map
   from each completion x of positive probability to q(x)."
  [pot]
  (into {} (for [[x _] every-completion
                 :let [lq (local-log-q pot x)]
                 :when (Double/isFinite lq)]
             [x (Math/exp lq)])))

(def pi (conditional yes-no))

(def q (local yes-no))

(def answers [[3 8] [1 2 8] [6 8] [4 5 8]])

[(= (set (keys pi)) (set (keys q)) (set answers))
 (mapv pi answers) (mapv q answers)]
;=> [true [0.1067 0.0032 0.89 0.0001] [0.761 0.0258 0.2072 0.006]]

;; book: why
(defn p-next
  "The LM's probability of token t after the prompt and tokens."
  [tokens t]
  (Math/exp (aget ^floats (lm/next-logprobs m (into prompt tokens)) (int t))))

[(p-next [] 3) (p-next [3] end) (p-next [] 6) (p-next [6] end)]
;=> [0.536 0.0034 0.146 0.105]

;; book: conditional-weight
(defn conditional-weight
  "The importance weight for π of completion x drawn by yes-or-no:
   gen.llm.constrained's, of the trace whose tokens are x."
  [x]
  (constrained/conditional-weight
   (:trace (gfi/generate yes-or-no args (token/forced x) (rng/key 0)))))

(def log-p (into {} every-completion))

(every? (fn [x]
          (array/allclose? (conditional-weight x)
                           (- (+ (log-p x) (potential/complete yes-no (body x)))
                              (Math/log (q x)))))
        (keys q))
;=> true

(array/allclose?
 z (reduce + (for [[x qx] q] (* qx (Math/exp (conditional-weight x))))))
;=> true

;; book: spread
(defn spread
  "The mean of the numbers xs, their standard deviation, and the standard
   error of the mean: {:mean :sd :se}."
  [xs]
  (let [n (count xs)
        mean (/ (reduce + xs) n)
        sd (Math/sqrt (/ (reduce + (map #(let [e (- % mean)] (* e e)) xs))
                         (dec n)))]
    {:mean mean :sd sd :se (/ sd (Math/sqrt n))}))

(defn within-5-se?
  "Is the mean of xs within five standard errors of the value v?"
  [xs v]
  (let [{:keys [mean se]} (spread xs)]
    (< (abs (- mean v)) (* 5 se))))

;; book: importance
(def draws
  (mapv #(gfi/simulate yes-or-no args %) (rng/split (rng/key 10) 2000)))

(def weights
  (mapv #(Math/exp (constrained/conditional-weight %)) draws))

(spread weights)
;=> {:mean 0.0183 :sd 0.0298 :se 0.00067} ; float64

(within-5-se? weights z)
;=> true

;; book: smc-run
(def masked
  (smc/run free args 20 {:potential yes-no :proposal :masked
                         :key (rng/key 1)}))

(frequencies (map text (smc/particles masked)))
;=> {"yes" 16 "no" 4} ; float64

[(smc/log-ml-estimate masked) (Math/log z)]
;=> [-4.09 -4.0606] ; float64

(every? #(array/allclose? (:score %) (log-p (:retval %)))
        (smc/traces masked))
;=> true

;; book: smc-refuses
;; error: smc: the completion has a potential of its own
#_(smc/run yes-or-no args 20 {:key (rng/key 1)})

;; book: smc-weights
(defn normalized
  "The normalized weights of SMC state s's particles, in order."
  [s]
  (let [lws (:log-weights s)
        top (apply max lws)
        ws (mapv #(Math/exp (- % top)) lws)
        total (reduce + ws)]
    (mapv #(/ % total) ws)))

(defn weighted-shares
  "Each text's share of the normalized weights of state s's particles."
  [s]
  (reduce (fn [acc [x w]] (update acc (text x) (fnil + 0.0) w))
          {} (map vector (smc/particles s) (normalized s))))

(sort (distinct (:log-weights masked)))
;=> [-6.025 -2.603] ; float64

(weighted-shares masked)
;=> {"yes" 0.116 "no" 0.884} ; float64

;; book: smc-step
(def first-step
  (smc/step (smc/initialize free args 20 {:potential yes-no
                                          :proposal :masked
                                          :key (rng/key 2)})))

(defn log-mass
  "log Σ_t p(t) exp(w(t)) at the site after tokens, w potential pot's
   next-token weights: for a constraint, the log of the LM's probability
   of the tokens it allows there."
  [pot tokens]
  (let [^floats row (lm/next-logprobs m (into prompt tokens))
        ^floats ws (potential/logw-next pot tokens)
        w (fn [t] (double (aget ws (int (if (= t end) n-vocab t)))))]
    (Math/log (reduce + (for [t (range n-vocab)]
                          (Math/exp (+ (double (aget row (int t))) (w t))))))))

(every? #(array/allclose? % (log-mass yes-no [])) (:log-weights first-step))
;=> true

[(log-mass yes-no []) (log-mass yes-no [3]) (log-mass yes-no [6])]
;=> [-0.35 -5.675 -2.253]

;; book: smc-importance
(def unresampled
  (smc/run free args 20 {:potential yes-no :proposal :masked
                         :ess-threshold 0 :key (rng/key 3)}))

(every? true? (map #(array/allclose? %2 (conditional-weight %1))
                   (smc/particles unresampled) (:log-weights unresampled)))
;=> true

;; book: smc-bootstrap
(def bootstrap
  (smc/run free args 20 {:potential yes-no :proposal :lm
                         :ess-threshold 0 :key (rng/key 4)}))

[(frequencies (:log-weights bootstrap)) (smc/log-ml-estimate bootstrap)]
;=> [{##-Inf 20} ##-Inf] ; float64

;; book: smc-unbiased
(defn estimates
  "The estimates of Z, exp(log-ml-estimate), of SMC runs over free with
   opts and n particles, one run for each of keys."
  [opts n keys]
  (mapv #(Math/exp (smc/log-ml-estimate (smc/run free args n
                                                 (assoc opts :key %))))
        keys))

(def z-masked
  (estimates {:potential yes-no :proposal :masked} 20
             (rng/split (rng/key 5) 200)))

(spread z-masked)
;=> {:mean 0.0172 :sd 0.0067 :se 0.00047} ; float64

(within-5-se? z-masked z)
;=> true

;; book: smc-particles
(defn completion-estimate
  "The estimate of Z π(x) for each completion x from SMC's final state s:
   the run's estimate of Z times x's share of the particles' normalized
   weights. A map from completion to estimate, {} when every particle
   died."
  [s]
  (let [z-hat (Math/exp (smc/log-ml-estimate s))]
    (if (zero? z-hat)
      {}
      (reduce (fn [acc [x w]] (update acc x (fnil + 0.0) (* z-hat w)))
              {} (map vector (smc/particles s) (normalized s))))))

(def smc-estimates
  (mapv #(completion-estimate
          (smc/run free args 20 {:potential yes-no :proposal :masked
                                 :key %}))
        (rng/split (rng/key 6) 200)))

(defn shares
  "The estimates of π(x) from runs' estimates of Z π(x): each divided by
   the exact Z."
  [runs x]
  (mapv #(/ (get % x 0.0) z) runs))

(mapv #(within-5-se? (shares smc-estimates %) (pi %)) answers)
;=> [true true true true]

;; book: masking-biased
(def mask-draws
  (mapv #(:retval (gfi/simulate yes-or-no args %))
        (rng/split (rng/key 7) 200)))

(defn says-no?
  "Is the text of completion x no?"
  [x]
  (= "no" (text x)))

(def pi-no (reduce + (map pi (filter says-no? answers))))

(def mask-no (mapv #(if (says-no? %) 1.0 0.0) mask-draws))

(def smc-no
  (mapv (fn [run] (/ (reduce + (for [[x e] run :when (says-no? x)] e)) z))
        smc-estimates))

[pi-no (:mean (spread mask-no)) (:mean (spread smc-no))]
;=> [0.8901 0.195 0.931] ; float64

(- pi-no (reduce + (map q (filter says-no? answers))))
;=> 0.6769

;; book: closer
(defn closer?
  "Is the mean of xs closer to v than the mean of ys is, by more than five
   standard errors of the difference?"
  [xs ys v]
  (let [a (spread xs)
        b (spread ys)]
    (> (- (abs (- (:mean b) v)) (abs (- (:mean a) v)))
       (* 5 (Math/sqrt (+ (* (:se a) (:se a)) (* (:se b) (:se b))))))))

(closer? smc-no mask-no pi-no)
;=> true

;; book: estimates-figure
;; figure: figures/044-estimates.svg
(let [mask (fn [x] (mapv #(if (= x %) 1.0 0.0) mask-draws))
      marks (fn [color dx estimates]
              (let [at (fn [x] (let [{:keys [mean se]} (spread (estimates x))]
                                 [(+ (pi x) dx) mean se]))]
                [{:paths (for [[px y se] (map at answers)]
                           [[px (- y (* 2 se))] [px (+ y (* 2 se))]])
                  :color color}
                 {:paths (for [[px y] (map at answers)] [[px y]])
                  :color color :width 0 :r 4 :opacity 1}]))]
  (svg/paths (concat [{:paths [[[0 0] [1 1]]] :color "gray" :width 1}]
                     (marks "darkorange" -0.01 mask)
                     (marks "steelblue" 0.01 #(shares smc-estimates %)))
             {:x-range [-0.05 1.05] :y-range [-0.05 1.05]
              :x-label "exact probability π(x)" :y-label "estimate"}))

;; book: solution-1
(def z-bootstrap
  (estimates {:potential yes-no :proposal :lm} 20
             (rng/split (rng/key 20) 200)))

(defn zero-share
  "The fraction of the estimates zs that are 0, and its standard error."
  [zs]
  (let [f (/ (count (filter zero? zs)) (double (count zs)))]
    {:share f :se (Math/sqrt (/ (* f (- 1 f)) (count zs)))}))

(spread z-bootstrap)
;=> {:mean 0.0159 :sd 0.0286 :se 0.002} ; float64

[(zero-share z-bootstrap) (zero-share z-masked)]
;=> [{:share 0.73 :se 0.031} {:share 0.0 :se 0.0}] ; float64

;; book: solution-1-check
[(within-5-se? z-bootstrap z)
 (let [{:keys [share se]} (zero-share z-bootstrap)]
   (> (- share (:share (zero-share z-masked))) (* 5 se)))]
;=> [true true]

;; book: solution-2
(def doubt
  "The whole text no costs a nat; as a prefix it costs nothing."
  (potential/potential
   n-vocab {:prefix (constantly 0.0)
            :complete #(if (= "no" (lm/decode m %)) -1.0 0.0)}))

(def hesitant (potential/product yes-no doubt))

(vec (potential/logw-next hesitant [6]))
;=> [##-Inf ##-Inf ##-Inf ##-Inf ##-Inf ##-Inf ##-Inf ##-Inf ##-Inf -1.0]

(def pi-hesitant (conditional hesitant))

[(exact-z hesitant) (reduce + (map pi-hesitant (filter says-no? answers)))]
;=> [0.00754 0.749]

;; book: solution-2-check
(within-5-se? (estimates {:potential hesitant :proposal :masked} 20
                         (rng/split (rng/key 21) 200))
              (exact-z hesitant))
;=> true

(def maybe-not (token/completion m {:potential hesitant}))

(within-5-se? (mapv #(Math/exp (constrained/conditional-weight
                                (gfi/simulate maybe-not args %)))
                    (rng/split (rng/key 24) 2000))
              (exact-z hesitant))
;=> true

;; book: solution-3
(def z-1 (exact-z yes-no 1))

[z-1 (+ (p-next [] 3) (p-next [] 6)) (/ (p-next [] 6) z-1)]
;=> [0.682 0.682 0.214]

(within-5-se? (mapv #(Math/exp (smc/log-ml-estimate
                                (smc/run free [prompt 1] 20
                                         {:potential yes-no :proposal :masked
                                          :key %})))
                    (rng/split (rng/key 22) 200))
              z-1)
;=> true

;; book: solution-4
(def stranded-runs
  (for [k (rng/split (rng/key 23) 200)]
    (smc/run free args 20 {:potential byte-live :proposal :masked
                           :ess-threshold 0 :key k})))

(defn dead-shares
  "The share of each run's particles whose weight fell to 0."
  [runs]
  (mapv (fn [s] (/ (count (filter #(= ##-Inf %) (:log-weights s)))
                   (double (count (:log-weights s)))))
        runs))

(def strand-probability
  (let [first-site (mapv #(Math/exp %) (site byte-live []))]
    (+ (first-site 4) (first-site 6))))

[strand-probability (:mean (spread (dead-shares stranded-runs)))]
;=> [0.213 0.224] ; float64

;; book: solution-4-check
(let [z-yes (exact-z yes-nope)]
  [(within-5-se? (dead-shares stranded-runs) strand-probability)
   (within-5-se? (mapv #(Math/exp (smc/log-ml-estimate %)) stranded-runs)
                 z-yes)
   (array/allclose? z-yes (exact-z byte-live))])
;=> [true true true]
