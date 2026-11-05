(ns gen.laws.potential
  "The potential contract (gen.llm.potential) as laws: test.check
   properties over potentials, which verify and deflaws run as gen.laws's
   run the GFI's, so that every potential is checked as gen.clj's rule 7
   asks (after genlm-control's assert_logw_next_consistency and
   assert_autoreg_fact).

   A potential weighs a context, a token vector, as a prefix and as a
   complete sequence; its next-token weights follow from those. The laws,
   the set all:

   - next-weights-are-prefix-differences: at a context of finite prefix
     weight, logw-next has n-vocab + 1 entries, entry t is
     prefix(ctx ++ [t]) - prefix(ctx) and the end's is
     complete(ctx) - prefix(ctx);
   - weights-factor-autoregressively: complete(ctx) - prefix([]) is the
     sum of the next-token weights along ctx and the end's;
   - a-zero-prefix-has-zero-extensions: a context whose prefix weight is
     zero (-Inf) has no weight as a sequence, and neither has any context
     it begins;
   - weights-are-pure-functions-of-the-context: prefix, complete and
     logw-next weigh a context the same whatever context they weighed
     before (a stateful potential's state must not leak).

   A soft potential, which weighs every sequence, has no zero prefixes:
   its a-zero-prefix-has-zero-extensions cases are all vacuous, so leave
   that law out of its set (dissoc).

   Cases. (cases potentials) draws cases [entry steps seed] over entries
   [label p n-vocab] or [label p n-vocab opts]. steps walks a context from
   the empty one, each step to a token logw-next weighs finitely (steps'
   values pick among them), so a law reads contexts a sampler reaches,
   which have weight; seed picks the rest (a token of zero weight, the
   tokens compared). opts: :max-length, the walk's longest (8 tokens), and
   :tokens, how many tokens a law compares at a context (every one by
   default; for a large vocabulary a sample, half of it among the tokens
   of finite weight). A potential with no weight at all (prefix([]) is
   -Inf) makes vacuous cases.

   Comparisons are within *tolerance*, relative to the weights compared:
   a potential stores next-token weights as floats.

     (potential-laws/verify [[:no-repeats no-repeats 6]])
     (potential-laws/deflaws grammar [[:g g 40 {:tokens 16}]] 100
                             (dissoc potential-laws/all :weights-are-pure-functions-of-the-context))"
  {:api :experimental}
  (:require [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [gen.laws :as laws]
            [gen.llm.potential :as potential])
  (:import (java.util Arrays SplittableRandom)))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

(def ^:dynamic *tolerance*
  "The relative tolerance of the laws' comparisons of weights. A potential
   stores next-token weights as floats, a weight w within 2^-24 |w| (6e-8)
   of its float; 1e-6 is 16 such roundings. A comparison is relative to
   the magnitude of the weights it came from (a difference's operands, a
   sum's terms), so that a difference of large weights may round as its
   operands do."
  1e-6)

(defn- magnitude
  "|w|, 0 for an infinite w (NaN stays NaN, so that a comparison with it
   fails)."
  ^double [^double w]
  (if (Double/isInfinite w) 0.0 (Math/abs w)))

(defn- close-weights?
  "Are weights want and got equal (infinities exactly), within *tolerance*
   of scale?"
  [^double want ^double got ^double scale]
  (or (= want got) (<= (Math/abs (- want got)) (* (double *tolerance*) scale))))

(defn- live?
  "Has a weight w (a log weight) any weight: above -Inf (NaN has none)?"
  [^double w]
  (> w Double/NEGATIVE_INFINITY))

;; ---------------------------------------------------------------------------
;; Cases: entries, walks and the tokens compared
;; ---------------------------------------------------------------------------

(def ^:private default-max-length 8)

(defn- entry!
  "entry [label p n-vocab opts?], checked: a loud error for anything else."
  [entry]
  (let [[_ p n-vocab opts] entry]
    (when-not (and (vector? entry) (<= 3 (count entry) 4)
                   (satisfies? potential/Potential p) (pos-int? n-vocab)
                   (or (nil? opts) (map? opts)))
      (throw (ex-info (str "potential laws: an entry is [label potential n-vocab] or [label potential n-vocab opts], not "
                           (pr-str entry))
                      {:entry entry})))
    entry))

(defn cases
  "A test.check generator of cases [entry steps seed] over potentials: a
   vector of entries [label p n-vocab] or [label p n-vocab opts], or a
   generator of them. steps (up to opts' :max-length values, 8 by
   default) walks a context from the empty one; seed picks the tokens a
   law compares and a token of zero weight. Potentials that are neither
   are a loud error."
  [potentials]
  (let [entries (cond (gen/generator? potentials) (gen/fmap entry! potentials)
                      (and (sequential? potentials) (seq potentials)) (gen/elements (mapv entry! potentials))
                      :else (throw (ex-info "potential laws: potentials are a non-empty vector of entries [label potential n-vocab], or a generator of them"
                                            {:potentials potentials})))]
    (gen/bind entries
              (fn [[_ _ _ opts :as entry]]
                (gen/tuple (gen/return entry)
                           (gen/vector laws/seeds 0 (:max-length opts default-max-length))
                           laws/seeds)))))

(defn- describe
  "A case as a failure shows it: the potential by its label."
  [[[label] steps seed]]
  {:potential label :steps steps :seed seed})

(defn- live-tokens
  "The tokens (not the end) to which a next-token weight row w gives
   weight."
  [^floats w ^long n-vocab]
  (filterv #(live? (aget w (int %))) (range n-vocab)))

(defn- walk
  "The contexts from the empty one along the walk steps make: each step to
   a token logw-next weighs finitely, steps' values picking among them. It
   stops where no token has weight, or where a step reaches a context of
   no prefix weight (logw-next was wrong: the laws at the context before
   it say so). nil when the empty context has no weight."
  [p ^long n-vocab steps]
  (when (live? (potential/prefix p []))
    (loop [path [[]] [s & more :as steps] steps]
      (if (empty? steps)
        path
        (let [ctx (peek path)
              live (live-tokens (potential/logw-next p ctx) n-vocab)]
          (if (empty? live)
            path
            (let [next-ctx (conj ctx (live (int (Math/floorMod (long s) (long (count live))))))]
              (if (live? (potential/prefix p next-ctx))
                (recur (conj path next-ctx) more)
                path))))))))

(defn- sample
  "k of xs (all of them when there are no more), picked by rnd."
  [^SplittableRandom rnd ^long k xs]
  (let [xs (vec xs)]
    (if (<= (count xs) k)
      xs
      (loop [picked (sorted-set)]
        (if (= k (count picked))
          (vec picked)
          (recur (conj picked (xs (.nextInt rnd (count xs))))))))))

(defn- compared-tokens
  "The tokens a law compares at a context: every one when opts give no
   :tokens, or :tokens of them, half among those of finite weight (in
   live) and the rest from the whole vocabulary, picked by seed."
  [opts ^long n-vocab live ^long seed]
  (if-let [k (:tokens opts)]
    (let [k (long k)
          rnd (SplittableRandom. seed)
          some-live (sample rnd (quot k 2) live)]
      (vec (into (sorted-set) (concat some-live (sample rnd (- k (count some-live)) (range n-vocab))))))
    (vec (range n-vocab))))

;; ---------------------------------------------------------------------------
;; The checks a law makes, each with the values to show when it fails
;; ---------------------------------------------------------------------------

(defn- next-weight-mismatch
  "The first of tokens (then the end) at which p's logw-next at ctx is not
   the difference of prefix (complete, for the end) and ctx's prefix, as a
   map of what was compared; nil when every one matches."
  [p ^long n-vocab ctx tokens]
  (let [base (potential/prefix p ctx)
        ^floats w (potential/logw-next p ctx)]
    (if (not= (inc n-vocab) (alength w))
      {:context ctx :logw-next-entries (alength w) :expected-entries (inc n-vocab)}
      (or (some (fn [t]
                  (let [want (- (potential/prefix p (conj ctx t)) base)
                        got (double (aget w (int t)))]
                    (when-not (close-weights? want got (+ 1.0 (magnitude base) (magnitude want)))
                      {:context ctx :token t :logw-next got :prefix-difference want})))
                tokens)
          (let [want (- (potential/complete p ctx) base)
                got (double (aget w (int n-vocab)))]
            (when-not (close-weights? want got (+ 1.0 (magnitude base) (magnitude want)))
              {:context ctx :token :end :logw-next got :complete-less-prefix want}))))))

(defn- next-weights-match?
  "Is p's logw-next at ctx the difference of its prefix (and complete)
   weights at tokens and the end?"
  {::laws/explain next-weight-mismatch}
  [p n-vocab ctx tokens]
  (nil? (next-weight-mismatch p n-vocab ctx tokens)))

(defn- factorization
  "complete(ctx) - prefix([]) and the sum of the next-token weights along
   path (whose last context is ctx) and the end's, with the terms."
  [p ^long n-vocab path]
  (let [ctx (peek path)
        terms (conj (mapv (fn [before t] (double (aget ^floats (potential/logw-next p before) (int t))))
                          path ctx)
                    (double (aget ^floats (potential/logw-next p ctx) (int n-vocab))))
        start (potential/prefix p [])
        complete (potential/complete p ctx)]
    {:context ctx :complete-less-start (- complete start) :sum-of-next-weights (reduce + 0.0 terms)
     :terms terms :start start :complete complete}))

(defn- factors?
  "Does complete(ctx) - prefix([]) equal the sum of the next-token weights
   along path and the end's, within *tolerance* of their magnitudes?"
  {::laws/explain factorization}
  [p n-vocab path]
  (let [{:keys [complete-less-start sum-of-next-weights terms start complete]} (factorization p n-vocab path)]
    (close-weights? complete-less-start sum-of-next-weights
                    (+ 1.0 (magnitude start) (magnitude complete) (double (reduce + 0.0 (map magnitude terms)))))))

(defn- zero-weights
  "The prefix and complete weights of dead (a context of zero prefix
   weight, by logw-next), and the first of its extensions by tokens that
   has prefix weight."
  [p dead tokens]
  {:context dead :prefix (potential/prefix p dead) :complete (potential/complete p dead)
   :extension-with-weight (first (filter #(live? (potential/prefix p (conj dead %))) tokens))})

(defn- zero-throughout?
  "Has dead no weight as a prefix or as a sequence, and no extension of it
   by tokens weight as a prefix?"
  {::laws/explain zero-weights}
  [p dead tokens]
  (let [{:keys [prefix complete extension-with-weight]} (zero-weights p dead tokens)]
    (and (= ##-Inf prefix) (= ##-Inf complete) (nil? extension-with-weight))))

(defn- weights-of
  "p's weights of ctx: prefix, complete and logw-next (as a vector)."
  [p ctx]
  [(potential/prefix p ctx) (potential/complete p ctx) (vec (potential/logw-next p ctx))])

(defn- weighed-after
  "p's weights of ctx right after those of before, and right after those
   of the empty context, in that order."
  [p ctx before]
  (let [after-before (do (weights-of p before) (weights-of p ctx))
        after-empty (do (weights-of p []) (weights-of p ctx))]
    {:context ctx :after before :weights after-before :after-the-empty-context after-empty}))

(defn- same-weights?
  "Are two of weights-of's results the same, NaN included?"
  [[p1 c1 w1] [p2 c2 w2]]
  (and (zero? (Double/compare p1 p2)) (zero? (Double/compare c1 c2))
       (Arrays/equals (float-array w1) (float-array w2))))

(defn- history-free?
  "Does p weigh ctx the same right after before as right after the empty
   context?"
  {::laws/explain weighed-after}
  [p ctx before]
  (let [{:keys [weights after-the-empty-context]} (weighed-after p ctx before)]
    (same-weights? weights after-the-empty-context)))

(defn- neighbour
  "A context of ctx's length beside it on path (its last token another
   that logw-next weighs finitely, picked by seed), or one token past the
   empty context when ctx is empty; ctx's parent, or ctx, when there is
   none."
  [p ^long n-vocab path ^long seed]
  (let [ctx (peek path)
        parent (if (empty? ctx) ctx (pop ctx))
        others (cond->> (live-tokens (potential/logw-next p parent) n-vocab)
                 (seq ctx) (filterv #(not= % (peek ctx))))]
    (cond (empty? others) parent
          (empty? ctx) [(others (.nextInt (SplittableRandom. seed) (count others)))]
          :else (conj parent (others (.nextInt (SplittableRandom. seed) (count others)))))))

;; ---------------------------------------------------------------------------
;; The laws
;; ---------------------------------------------------------------------------

(defn next-weights-are-prefix-differences
  "At a context of finite prefix weight (the end of a walk), logw-next has
   n-vocab + 1 entries, entry t is prefix(ctx ++ [t]) - prefix(ctx) and the
   end's is complete(ctx) - prefix(ctx)."
  [cases]
  (prop/for-all [[[_ p n-vocab opts] steps seed] cases]
    (if-let [path (walk p n-vocab steps)]
      (let [ctx (peek path)
            tokens (compared-tokens opts n-vocab (live-tokens (potential/logw-next p ctx) n-vocab) seed)]
        (laws/checks (next-weights-match? p n-vocab ctx tokens)))
      laws/vacuous)))

(defn weights-factor-autoregressively
  "complete(ctx) - prefix([]) is the sum of the next-token weights along a
   walk to ctx, and the end's."
  [cases]
  (prop/for-all [[[_ p n-vocab] steps] cases]
    (if-let [path (walk p n-vocab steps)]
      (laws/checks (factors? p n-vocab path))
      laws/vacuous)))

(defn a-zero-prefix-has-zero-extensions
  "A context of zero prefix weight (a walk's end and a token logw-next
   gives no weight) has no weight as a sequence, and no extension of it
   by a token has weight as a prefix. Vacuous where every token has
   weight."
  [cases]
  (prop/for-all [[[_ p n-vocab opts] steps seed] cases]
    (let [path (walk p n-vocab steps)
          ctx (peek path)
          ^floats w (when path (potential/logw-next p ctx))
          dead (when path (filterv #(not (live? (aget w (int %)))) (range n-vocab)))]
      (if (seq dead)
        (let [rnd (SplittableRandom. (long seed))
              t (dead (.nextInt rnd (count dead)))
              tokens (compared-tokens opts n-vocab [] (.nextLong rnd))]
          (laws/checks (zero-throughout? p (conj ctx t) tokens)))
        laws/vacuous))))

(defn weights-are-pure-functions-of-the-context
  "A potential weighs a context the same whatever it weighed before: the
   walk's end right after a context beside it (of its length, another
   last token) as right after the empty context, and the other way
   round. A stateful potential's state must not leak from one context into
   another's weights (one advanced from the last context it weighed, as
   if that were a prefix of the next, fails)."
  [cases]
  (prop/for-all [[[_ p n-vocab] steps seed] cases]
    (if-let [path (walk p n-vocab steps)]
      (let [ctx (peek path)
            other (neighbour p n-vocab path seed)]
        (laws/checks (history-free? p ctx other)
                     (history-free? p other ctx)))
      laws/vacuous)))

(def all
  "The potential contract's laws."
  {:next-weights-are-prefix-differences next-weights-are-prefix-differences
   :weights-factor-autoregressively weights-factor-autoregressively
   :a-zero-prefix-has-zero-extensions a-zero-prefix-has-zero-extensions
   :weights-are-pure-functions-of-the-context weights-are-pure-functions-of-the-context})

(defn verify
  "Runs every law of laws (all by default) over cases of potentials, and
   returns the failures as gen.laws/verify does: a sorted map from each
   law that failed to what it says (the values compared, the smallest
   case as {:potential :steps :seed}, the run's :seed); {} when every law
   holds. opts as gen.laws/verify's.

     (potential-laws/verify [[:g g 6]] {:num-tests 50 :seed 1})"
  ([potentials] (verify potentials nil all))
  ([potentials opts] (verify potentials opts all))
  ([potentials opts laws] (laws/verify-cases (cases potentials) opts laws describe)))

(defmacro deflaws
  "A test.check spec per law of laws (all by default), named prefix-law,
   each over cases of potentials, as gen.laws/deflaws defines the GFI's."
  ([prefix potentials opts] `(deflaws ~prefix ~potentials ~opts all))
  ([prefix potentials opts laws] `(laws/defspecs ~prefix (cases ~potentials) ~opts ~laws)))
