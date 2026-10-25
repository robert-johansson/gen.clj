;; book: ns
(ns book.extending
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.string :as str]
            [gen.array :as array :refer [+ - * /]]
            [gen.book.svg :as svg]
            [gen.choicemap :as cm]
            [gen.custom :as custom]
            [gen.diff :as diff]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dist.dsl :refer [dist]]
            [gen.dist.host :as host]
            [gen.dynamic :as dyn :refer [gen]]
            [gen.gfi :as gfi]
            [gen.gradients :as grad]
            [gen.inference.diagnostics :as diag]
            [gen.inference.mh :as mh]
            [gen.laws :as laws]
            [gen.learning :as learning]
            [gen.random :as rng]
            [gen.selection :as sel]
            [gen.serialize :as serialize]
            [gen.trace :as trace]))

;; book: kumaraswamy
(dc/defdist ^:elementwise ^:pathwise kumaraswamy
  "kumaraswamy(a, b): the Kumaraswamy distribution on (0, 1), whose
   density is a·b·x^(a-1)·(1 - x^a)^(b-1)."
  [a b]
  (validate (dc/check-positive "kumaraswamy" "a" a)
            (dc/check-positive "kumaraswamy" "b" b))
  (sample [key batch-shape]
    ;; the inverse of the CDF 1 - (1 - x^a)^b, at a uniform draw
    (let [u (rng/uniform key (dc/elementwise-shape batch-shape a b))]
      (array/pow (- 1.0 (array/pow (- 1.0 u) (/ 1.0 b))) (/ 1.0 a))))
  (log-prob [x]
    ;; outside (0, 1) the density is 0; a stand-in of 0.5 keeps the other
    ;; branch finite, so that its gradient is not NaN
    (let [inside (array/logical-and (array/> x 0.0) (array/< x 1.0))
          x (array/where inside x 0.5)
          log-1-xa (array/log1p (array/negate (array/pow x a)))]
      (array/where inside
                   (+ (array/log (* a b))
                      (* (- a 1.0) (array/log x))
                      (* (- b 1.0) log-1-xa))
                   ##-Inf))))

;; book: kumaraswamy-density
(defn kumaraswamy-density
  "The Kumaraswamy density at x, on the host."
  [a b x]
  (* a b (Math/pow x (- a 1.0)) (Math/pow (- 1.0 (Math/pow x a)) (- b 1.0))))

(def k25 (kumaraswamy 2.0 5.0))

(every? (fn [x] (array/allclose? (dc/log-prob k25 x)
                                 (Math/log (kumaraswamy-density 2.0 5.0 x))))
        [0.05 0.3 0.5 0.9])
;=> true

(mapv #(dc/log-prob k25 %) [-0.5 0.0 1.0 1.5])
;=> [##-Inf ##-Inf ##-Inf ##-Inf]

;; book: kumaraswamy-normalized
(defn integral
  "The integral of the density of d over (0, 1), by the midpoint rule on m
   intervals."
  [d m]
  (let [dx (/ 1.0 m)
        density (fn [x] (array/item (array/exp (dc/log-prob d x))))]
    (* dx (reduce + (map #(density (* (+ % 0.5) dx)) (range m))))))

(integral k25 10000)
;=> 1.0

;; book: kumaraswamy-validation
;; error: kumaraswamy: a must be positive, got -1
#_(kumaraswamy -1 5.0)

;; book: kumaraswamy-moments
(defn mean-se
  "The mean of the host numbers xs, and its standard error."
  [xs]
  (let [n (count xs)
        m (/ (reduce + xs) n)
        v (/ (reduce + (map #(* (- % m) (- % m)) xs)) (dec n))]
    [m (Math/sqrt (/ v n))]))

(defn within-5-se?
  "Is the mean of the host numbers xs within five standard errors of
   value?"
  [xs value]
  (let [[m se] (mean-se xs)]
    (< (abs (- m value)) (* 5 se))))

(def k25-draws
  (array/->clj (dc/sample k25 (rng/key 1) [20000])))

(defn kumaraswamy-moment
  "E[X^r] of kumaraswamy(a, b): b·B(1 + r/a, b)."
  [a b r]
  (* b (array/item (array/exp (array/log-beta (+ 1.0 (/ r a)) b)))))

[(within-5-se? k25-draws (kumaraswamy-moment 2.0 5.0 1))
 (within-5-se? (map #(* % %) k25-draws) (kumaraswamy-moment 2.0 5.0 2))]
;=> [true true]

[(kumaraswamy-moment 2.0 5.0 1) (kumaraswamy-moment 2.0 5.0 2)]
;=> [0.3694 0.1667]

;; book: kumaraswamy-figure
;; figure: figures/045-kumaraswamy.svg
(svg/histogram [{:values k25-draws
                 :density #(kumaraswamy-density 2.0 5.0 %)}]
               {:range [0.0 1.0] :x-label "x"})

;; book: kumaraswamy-site
(def proportion
  (gen []
    (trace :x (kumaraswamy 2.0 5.0))))

(let [{w :weight} (gfi/generate proportion [] (cm/choicemap :x 0.3)
                                (rng/key 2))]
  (array/allclose? w (dc/log-prob k25 0.3)))
;=> true

;; book: kumaraswamy-pathwise
(defn sample-mean
  "The mean of n draws from kumaraswamy(a, 5) with key."
  [a n key]
  (array/mean (dc/sample (kumaraswamy a 5.0) key [n])))

(let [key (rng/key 3)
      h 0.01
      by-autodiff ((array/grad #(sample-mean % 1000 key)) 2.0)
      by-differences (/ (- (sample-mean (+ 2.0 h) 1000 key)
                           (sample-mean (- 2.0 h) 1000 key))
                        (* 2 h))]
  [(dc/pathwise? k25)
   (array/allclose? by-autodiff by-differences {:rtol 0.01})])
;=> [true true]

;; book: pushforwards
(dist labeled-uniform
  "A label drawn uniformly from labels."
  [labels]
  (nth labels (d/uniform-discrete 0 (dec (count labels)))))

(dist labeled-categorical
  "A label of labels, the i-th with probability (probs i)."
  [labels probs]
  (nth labels (d/categorical probs)))

(dist noisy-bit
  "The bit x, flipped with probability p (Gen.jl's bit_flip)."
  [x p]
  (d/bernoulli (+ (* (- 1 x) p) (* x (- 1 p)))))

(dist sym-binom
  "A binomial count of 2·scale fair trials, centered on mean."
  [mean scale]
  (+ (- (d/binom (* 2 scale) 0.5) scale) mean))

(dist shifted-geometric
  "A geometric count, shifted by shift."
  [p shift]
  (+ (d/geometric p) shift))

(dist log-normal
  "e^X for X normal(mu, sigma)."
  [mu sigma]
  (array/exp (d/normal mu sigma)))

(dist gumbel
  "The Gumbel distribution, location mu and scale beta."
  [mu beta]
  (- mu (* beta (array/log (- 0.0 (array/log (d/uniform 0 1)))))))

;; book: labels
(def answers (labeled-categorical [:yes :no :yes] [0.2 0.3 0.5]))

[(dc/sample answers (rng/key 4)) (dc/support answers)]
;=> [:yes [:yes :no]] ; float64

(array/allclose? (dc/log-prob answers :yes) (Math/log 0.7))
;=> true

(dc/log-prob (labeled-uniform [:a :b :c :d]) :c)
;=> -1.3863

;; book: noisy-bit
[(dc/log-prob (noisy-bit true 0.1) true)
 (dc/log-prob (noisy-bit false 0.1) true)]
;=> [-0.1054 -2.3026]

;; book: discrete-pushforwards
(array/allclose? (dc/log-prob (sym-binom 10 3) 12) (Math/log (/ 6.0 64)))
;=> true

(array/allclose? (dc/log-prob (shifted-geometric 0.3 5) 7)
                 (Math/log (* 0.3 0.7 0.7)))
;=> true

(mapv #(dc/log-prob (shifted-geometric 0.3 5) %) [4 7.5])
;=> [##-Inf ##-Inf]

;; book: continuous-pushforwards
(every? (fn [x] (and (array/allclose? (dc/log-prob (log-normal 0.5 0.8) x)
                                      (dc/log-prob (d/log-normal 0.5 0.8) x))
                     (array/allclose? (dc/log-prob (gumbel 1.0 2.0) x)
                                      (dc/log-prob (d/gumbel 1.0 2.0) x))))
        [0.3 1.0 2.0 7.5])
;=> true

;; book: gumbel-moments
(def euler-gamma 0.5772156649015329)

(within-5-se? (array/->clj (dc/sample (gumbel 1.0 2.0) (rng/key 5) [20000]))
              (+ 1.0 (* 2.0 euler-gamma)))
;=> true

;; book: compile-error
(defn compile-error
  "The message of the error that compiling form raises (the cause the
   compiler reports), or nil when it compiles."
  [form]
  (try (eval form)
       nil
       (catch clojure.lang.Compiler$CompilerException e
         (ex-message (ex-cause e)))))

(str/starts-with? (compile-error '(dist sum-of-two [mu]
                                    (+ (d/normal mu 1) (d/normal mu 1))))
                  "dist sum-of-two: the body must make exactly one random draw")
;=> true

;; book: not-invertible
;; error: dist: (array/square (d/normal mu 1)) is not an invertible step
#_(dist squared-normal [mu] (array/square (d/normal mu 1)))

#_(dc/log-prob (squared-normal 0.0) 1.0)

;; book: beta-binomial
(defn log-rising
  "log of x(x + 1)…(x + m - 1), the rising factorial, on the host."
  [x m]
  (reduce + 0.0 (map #(Math/log (+ x %)) (range m))))

(defn beta-binomial-log-pmf
  "log P(K = k) for K beta-binomial(n, a, b), on the host:
   log C(n, k) + log B(k + a, n - k + b) - log B(a, b)."
  [n a b k]
  (+ (- (log-rising 1.0 n) (log-rising 1.0 k) (log-rising 1.0 (- n k)))
     (log-rising a k) (log-rising b (- n k)) (- (log-rising (+ a b) n))))

(def beta-binomial
  "beta-binomial(n, a, b): the number of successes in n trials whose
   success probability is drawn from beta(a, b)."
  (host/family
   {:name "beta-binomial"
    :discrete? true
    :make (fn [n a b]
            (when-not (and (integer? n) (<= 0 n))
              (throw (IllegalArgumentException. "n must be a count")))
            (when-not (and (pos? a) (pos? b))
              (throw (IllegalArgumentException. "a and b must be positive")))
            {:n n :pmf (mapv #(Math/exp (beta-binomial-log-pmf n a b %))
                             (range (inc n)))})
    ;; the least k whose cumulative probability reaches u
    :quantile (fn [{:keys [n pmf]} u]
                (loop [k 0 c (pmf 0)]
                  (if (or (<= u c) (= k n))
                    k
                    (recur (inc k) (+ c (pmf (inc k)))))))
    :log-density (fn [{:keys [n pmf]} k]
                   (if (<= 0 k n) (Math/log (pmf k)) ##-Inf))
    :support (fn [{:keys [n]}] (range (inc n)))}))

;; book: beta-binomial-check
(def bb (beta-binomial 10 2.0 3.0))

[(dc/discrete? bb) (dc/support bb)]
;=> [true [0 1 2 3 4 5 6 7 8 9 10]]

(reduce + (map #(array/item (array/exp (dc/log-prob bb %))) (dc/support bb)))
;=> 1.0

[(dc/log-prob bb 4) (dc/log-prob bb 4.5) (dc/log-prob bb 11)]
;=> [-1.9671 ##-Inf ##-Inf]

(dc/sample bb (rng/key 6) [8])
;=> [4 7 1 5 5 3 1 6]

;; book: host-validation
;; error: beta-binomial: invalid parameters [10 -2.0 3.0]
#_(beta-binomial 10 -2.0 3.0)

;; book: host-gradient
;; error: A beta-binomial distribution is backed by a host library
#_((array/grad (fn [a] (dc/log-prob (beta-binomial 10 a 3.0) 4))) 2.0)

;; book: urn-math
(defn predictive
  "The distribution of the urn's next draw after i draws, k of them true:
   true with probability (a + k) / (a + b + i)."
  [a b k i]
  (d/bernoulli (/ (+ a k) (+ a b i))))

(defn urn-score
  "The log probability of n draws of the urn, k of them true, the bias
   integrated out: log B(a + k, b + n - k) - log B(a, b). Loud for
   arguments outside their ranges."
  [a b n k]
  (dc/check-positive "urn" "a" a)
  (dc/check-positive "urn" "b" b)
  (dc/check-integer "urn" "n" n 0)
  (- (array/log-beta (+ a k) (+ b (- n k))) (array/log-beta a b)))

(array/allclose? (urn-score 1.0 2.0 3 2)
                 (+ (dc/log-prob (predictive 1.0 2.0 0 0) true)
                    (dc/log-prob (predictive 1.0 2.0 1 1) false)
                    (dc/log-prob (predictive 1.0 2.0 1 2) true)))
;=> true

;; book: urn-walk
(defn as-count
  "1 for a true draw, 0 for a false one."
  [x]
  (array/where x 1 0))

(defn count-true
  "The number of true values in xs."
  [xs]
  (reduce + 0 (map as-count xs)))

(defn total
  "The sum of the numbers xs, 0.0 for none."
  [xs]
  (reduce + 0.0 xs))

(defn walk
  "The urn's draws i0 … n-1 on the arguments [a b n], after k0 true
   draws: draw i is (given i), or, where that is nil, drawn from the
   predictive with its own key, (rng/fold-in key i). Returns the draws and
   their predictive log probabilities, as maps from index, and the count
   of true draws."
  [[a b n] given k0 i0 key]
  (loop [i i0, k k0, draws {}, lps {}]
    (if (>= i n)
      {:draws draws :lps lps :k k}
      (let [p (predictive a b k i)
            x (if (some? (given i))
                (given i)
                (dc/sample p (rng/fold-in key i)))]
        (recur (inc i) (+ k (as-count x)) (assoc draws i x)
               (assoc lps i (dc/log-prob p x)))))))

;; book: urn-choices
(defn draw-at
  "The address of draw i."
  [i]
  [:draw i])

(defn draws-in
  "The draws in the choice map c, as a map from index to value: loud, as
   gen.clj's own generative functions are, for an address that is not
   one of n draws."
  [op c n]
  (let [paths (cm/addresses c)
        draw? (fn [[head i :as p]]
                (and (= 2 (count p)) (= :draw head) (int? i) (< -1 i n)))]
    (when-let [others (seq (remove draw? paths))]
      (throw (gfi/unvisited op others)))
    (into {} (map (fn [[_ i :as p]] [i (cm/get-choice c p)])) paths)))

(defn choices-of
  "The choice map c (empty by default) with the draws, pairs of an index
   and a value."
  ([draws] (choices-of cm/EMPTY draws))
  ([c draws] (reduce (fn [c [i x]] (cm/set-choice c (draw-at i) x)) c draws)))

(defn urn-trace
  "The trace of gf on args with the choices and their count k."
  [gf [a b n :as args] choices k]
  (trace/make-trace {:gen-fn gf :args args :choices choices :retval k
                     :score (urn-score a b n k)}))

(defn log-predictives
  "The predictive log probability of each of tr's draws, by index: a walk
   over its draws in order."
  [tr]
  (let [[_ _ n :as args] (:args tr)]
    (:lps (walk args (draws-in :project (:choices tr) n) 0 0 nil))))

;; book: urn-update
(defn fresh-draws
  "The indices of the added draws that the constraints given do not fix:
   the draws an update makes afresh."
  [added given]
  (remove #(contains? given %) (keys added)))

(defn update-urn
  "The urn's update of tr to args with constraints: the old count moved
   by the draws dropped and the draws changed, then the draws past the
   old ones walked, constrained or fresh. Its work grows with the
   changes, not with the number of draws."
  [gf tr [_ _ n :as args] constraints key]
  (let [n0 (nth (:args tr) 2)
        old #(cm/get-choice (:choices tr) (draw-at %))
        given (draws-in :update constraints n)
        changed (filter #(< % n0) (keys given))
        dropped (range n n0)
        k (- (+ (:retval tr) (count-true (map given changed)))
             (count-true (map old (concat changed dropped))))
        {added :draws lps :lps k' :k} (walk args given k n0 key)
        same? (and (empty? given) (= n n0))
        choices (choices-of (reduce #(cm/remove-path %1 (draw-at %2))
                                    (:choices tr) dropped)
                            (concat (select-keys given changed) added))
        t (urn-trace gf args choices (if same? (:retval tr) k'))]
    {:trace t
     :weight (- (:score t) (:score tr)
                (total (map lps (fresh-draws added given))))
     :retdiff (if same? diff/no-change diff/unknown-change)
     :discard (choices-of (map (juxt identity old) (concat changed dropped)))}))

;; book: urn-regenerate
(defn regenerate-urn
  "The urn's regenerate of tr to args: the selected draws drawn again, in
   order, each from the predictive given the draws before it; the weight
   is the ratio of the predictive probabilities of the draws kept."
  [gf tr [_ _ n :as args] selection key]
  (let [old (draws-in :regenerate (:choices tr) (nth (:args tr) 2))
        kept (into {} (remove (fn [[i]] (sel/selected? selection (draw-at i))))
                   old)
        {:keys [draws lps k]} (walk args kept 0 0 key)
        retained (filter #(< % n) (keys kept))
        old-lps (log-predictives tr)]
    {:trace (urn-trace gf args (choices-of draws) k)
     :weight (- (total (map lps retained)) (total (map old-lps retained)))
     :retdiff diff/unknown-change}))

;; book: urn
(gfi/defgf Urn []
  gfi/IGenerativeFunction
  (-simulate [gf args key]
    (let [{:keys [draws k]} (walk args {} 0 0 key)]
      (urn-trace gf args (choices-of draws) k)))
  (-generate [gf [_ _ n :as args] constraints key]
    (let [given (draws-in :generate constraints n)
          {:keys [draws lps k]} (walk args given 0 0 key)
          t (urn-trace gf args (choices-of draws) k)]
      ;; without constraints, simulate's weight, 0; with them,
      ;; log p(draws) - log q(fresh draws), the score itself when every
      ;; draw is given
      {:trace t
       :weight (if (empty? given)
                 0.0
                 (- (:score t) (total (map lps (fresh-draws draws given)))))}))
  (-update [gf tr args _ constraints key]
    (update-urn gf tr args constraints key))
  (-regenerate [gf tr args _ selection key]
    (regenerate-urn gf tr args selection key))
  (-assess [_ [a b n] choices _]
    (let [given (draws-in :assess choices n)]
      (when-let [i (first (remove #(contains? given %) (range n)))]
        (throw (ex-info (str "assess: no value for the draw at " (draw-at i))
                        {:path (draw-at i)})))
      (let [k (count-true (vals given))]
        {:weight (urn-score a b n k) :retval k})))
  (-propose [gf args key]
    (let [t (gfi/simulate gf args key)]
      {:choices (:choices t) :weight (:score t) :retval (:retval t)}))
  (-project [_ tr selection]
    (total (keep (fn [[i lp]] (when (sel/selected? selection (draw-at i)) lp))
                 (log-predictives tr)))))

(def urn
  "The Pólya urn: on [a b n], n draws, true or false, that are exchangeable,
   with the bias beta(a, b) integrated out; the return value is the count
   of true draws."
  (->Urn))

;; book: urn-run
(def urn-tr (gfi/simulate urn [1.0 2.0 6] (rng/key 7)))

[(:retval urn-tr) (mapv #(urn-tr (draw-at %)) (range 6))]
;=> [1 [false false true false false false]] ; float64

;; book: urn-by-gen
(def urn-by-gen
  (gen [a b n]
    (loop [i 0 k 0]
      (if (= i n)
        k
        (let [x (trace (draw-at i) (predictive a b k i))]
          (recur (inc i) (+ k (as-count x))))))))

(array/allclose? (:score urn-tr)
                 (:weight (gfi/assess urn-by-gen [1.0 2.0 6] (:choices urn-tr)
                                      (rng/key 0))))
;=> true

;; book: urn-beta-binomial
(let [k (long (array/item (:retval urn-tr)))
      log-choose (- (log-rising 1.0 6)
                    (log-rising 1.0 k) (log-rising 1.0 (- 6 k)))]
  (array/allclose? (+ log-choose (:score urn-tr))
                   (dc/log-prob (beta-binomial 6 1.0 2.0) k)))
;=> true

;; book: urn-counts
(def urn-counts
  (mapv #(long (array/item (:retval (gfi/simulate urn [2.0 3.0 10] %))))
        (rng/split (rng/key 8) 4000)))

(defn frequency-within-5-se?
  "Is the share of k among counts within five standard errors of p?"
  [counts k p]
  (let [n (count counts)
        share (/ (count (filter #{k} counts)) (double n))]
    (< (abs (- share p)) (* 5 (Math/sqrt (/ (* p (- 1 p)) n))))))

(every? #(frequency-within-5-se? urn-counts %
                                 (array/item (array/exp (dc/log-prob bb %))))
        (range 11))
;=> true

;; book: urn-figure
;; figure: figures/045-urn.svg
(svg/histogram [{:values urn-counts
                 ;; the probability of the count nearest x
                 :density (fn [x] (array/exp (dc/log-prob bb (Math/round x))))}]
               {:range [-0.5 10.5] :bins 11 :x-label "true draws of 10"
                :y-label "share"})

;; book: urn-laws
(def urn-models
  "The urn as a model entry: its arguments, and others to update and
   regenerate with: other a and b, more draws, fewer draws."
  [[:urn urn [1.0 2.0 4] [[2.0 0.5 4] [1.0 2.0 6] [1.0 2.0 2]]]])

(def few
  "The chapter's runs of a law: 20 cases, from one seed."
  {:num-tests 20 :seed 1})

[(laws/verify urn-models few) (laws/verify urn-models few laws/fixed-support)]
;=> [{} {}]

;; book: urn-deflaws
(laws/deflaws urn-law urn-models 100)

(select-keys (urn-law-updates-are-reversible-with-exact-discards 100 :seed 2)
             [:pass? :num-tests :vacuous])
;=> {:pass? true, :num-tests 100, :vacuous 0}

;; book: urn-bug
(defn fresh-by-lookup
  "fresh-draws with the map given as the predicate: a draw constrained to
   false looks unconstrained."
  [added given]
  (remove given (keys added)))

(= (with-redefs [fresh-draws fresh-by-lookup]
     (set (keys (laws/verify urn-models few))))
   #{:assess-is-generate-with-every-choice
     :generate-weight-is-the-constrained-density
     :generate-weight-is-the-projection
     :generate-with-every-choice-is-deterministic
     :update-weight-is-the-projection-change
     :update-weight-is-the-score-change-less-fresh-choices})
;=> true

;; book: urn-model
(defn urn-model
  "A share m from kumaraswamy(2, 5), and n draws of an urn, by urn-gf,
   whose draws are true a share m of the time on average: the urn on
   10m and 10(1 - m)."
  [urn-gf]
  (gen [n]
    (let [m (trace :m (kumaraswamy 2.0 5.0))]
      (splice :urn urn-gf (* 10.0 m) (* 10.0 (- 1.0 m)) n)
      m)))

(def observed
  "Sixty draws, every fourth true, under :urn."
  (cm/set-submap cm/EMPTY :urn
                 (choices-of (map (fn [i] [i (zero? (mod i 4))]) (range 60)))))

;; book: urn-chain
(def m-walk
  "A proposal of m near the current one, inside (0, 1): a beta of mean
   m."
  (gen [tr]
    (let [m (tr :m)]
      (trace :m (d/beta (* 20.0 m) (* 20.0 (- 1.0 m)))))))

(defn m-chain
  "The values of m, host numbers, along n MH steps with the proposal
   m-walk on (urn-model urn-gf), from a trace generated with the observed
   draws. Each trace is dropped as the chain moves on."
  [urn-gf n key]
  (let [model (urn-model urn-gf)
        [k-start k-chain] (rng/split key)
        start (:trace (gfi/generate model [60] observed k-start))]
    (second
     (reduce (fn [[tr ms] k]
               (let [tr' (:trace (mh/mh tr m-walk [] k))]
                 [tr' (conj ms (array/item (tr' :m)))]))
             [start []]
             (rng/split k-chain n)))))

(= (m-chain urn 2000 (rng/key 11)) (m-chain urn-by-gen 2000 (rng/key 11)))
;=> true ; float64

;; book: urn-posterior
(def m-grid "A grid over (0, 1)." (mapv #(/ (+ % 0.5) 1000.0) (range 1000)))

(defn grid-posterior
  "The posterior mean and standard deviation of m given k true draws of
   n, on m-grid: the prior times the urn's probability of the draws."
  [n k]
  (let [log-post (mapv #(array/item
                         (+ (dc/log-prob k25 %)
                            (urn-score (* 10.0 %) (* 10.0 (- 1.0 %)) n k)))
                       m-grid)
        top (apply max log-post)
        w (mapv #(Math/exp (- % top)) log-post)
        z (reduce + w)
        mean (/ (reduce + (map * m-grid w)) z)
        sq (fn [m] (* (- m mean) (- m mean)))
        variance (/ (reduce + (map #(* (sq %1) %2) m-grid w)) z)]
    {:mean mean :sd (Math/sqrt variance)}))

(def exact-m (grid-posterior 60 15))

exact-m
;=> {:mean 0.3113 :sd 0.1123}

(def urn-chain (m-chain urn 10000 (rng/key 12)))

(def urn-summary (diag/summary [(subvec urn-chain 500)]))

(let [{:keys [mean mcse-mean]} urn-summary]
  [(< (abs (- mean (:mean exact-m))) (* 5 mcse-mean))
   (> (- (kumaraswamy-moment 2.0 5.0 1) (:mean exact-m)) (* 5 mcse-mean))])
;=> [true true]

;; book: lambert-w-host
(defn lambert-w-host
  "W(x) for x >= 0, the w with w·e^w = x: Newton's method, on the host."
  [x]
  (loop [w (Math/log1p x) i 0]
    (let [ew (Math/exp w)
          step (/ (- (* w ew) x) (* ew (+ w 1.0)))]
      (if (or (< (abs step) 1e-15) (= i 50))
        (- w step)
        (recur (- w step) (inc i))))))

(let [w (lambert-w-host 2.0)]
  (array/allclose? (* w (Math/exp w)) 2.0))
;=> true

;; book: host-read
;; error: Cannot read a value being differentiated on the host
#_((array/grad (fn [x] (lambert-w-host (array/item x)))) 2.0)

;; book: lambert-w
(def lambert-w
  "W(x), computed on the host, with its gradient: dW/dx = 1/(e^W (1 + W)),
   by the implicit function theorem."
  (array/custom-vjp
   (fn [x] (lambert-w-host (array/item x)))
   (fn [_args w g] [(/ g (* (array/exp w) (+ 1.0 w)))])))

(defn central-difference
  "(f(x + h) - f(x - h)) / 2h, for h = 1e-5."
  [f x]
  (let [h 1e-5]
    (/ (- (f (+ x h)) (f (- x h))) (* 2 h))))

[(lambert-w 2.0)
 (array/allclose? ((array/grad lambert-w) 2.0)
                  (central-difference lambert-w-host 2.0))]
;=> [0.8526 true]

;; book: my-plus
(def my-plus
  "Gen.jl's MyPlus: a + b, with its gradient written by hand."
  (custom/deterministic
   {:apply (fn [[a b]] (+ a b))
    :gradient (fn [_args _retval retgrad] [retgrad retgrad])}))

(defn summed
  "x and y, and z observed around their sum by plus."
  [plus]
  (gen []
    (let [x (trace :x (d/normal 0 1))
          y (trace :y (d/normal 0 1))]
      (trace :z (d/normal (splice :sum plus x y) 0.5)))))

(defn summed-density
  "log p(x, y, z = 1.2) of (summed my-plus)."
  [x y]
  (array/item (:weight (gfi/assess (summed my-plus) []
                                   (cm/choicemap :x x :y y :z 1.2)
                                   (rng/key 0)))))

(defn summed-gradients
  "The gradient of the log density of (summed plus) in x and y, at
   x = 0.3, y = -0.4 and z = 1.2."
  [plus]
  (let [t (:trace (gfi/generate (summed plus) []
                                (cm/choicemap :x 0.3 :y -0.4 :z 1.2)
                                (rng/key 12)))
        gs (:gradients (grad/choice-gradients t (sel/select :x :y)))]
    (mapv #(cm/get-choice gs %) [:x :y])))

(defn agrees-with-differences?
  "Do the gradients of (summed plus) agree with central differences of
   the density, in x and in y?"
  [plus]
  (let [[gx gy] (summed-gradients plus)]
    [(array/allclose? gx (central-difference #(summed-density % -0.4) 0.3))
     (array/allclose? gy (central-difference #(summed-density 0.3 %) -0.4))]))

[(summed-gradients my-plus) (agrees-with-differences? my-plus)]
;=> [[4.9 5.6] [true true]]

;; book: wrong-gradient
(def my-wrong-plus
  (custom/deterministic
   {:apply (fn [[a b]] (+ a b))
    :gradient (fn [_args _retval retgrad] [retgrad (* 2.0 retgrad)])}))

(agrees-with-differences? my-wrong-plus)
;=> [true false]

;; book: momentum
(defmethod learning/-init ::momentum [_ store]
  (array/tree-map array/zeros-like store))

(defmethod learning/-step ::momentum [{:keys [lr beta]} velocity store grads]
  ;; v' = βv + g, then θ' = θ + lr·v'
  (let [v (learning/update-given (fn [v g] (+ (* beta v) g)) velocity grads)]
    [(learning/update-given (fn [theta _ v] (+ theta (* lr v))) store grads v)
     v]))

;; book: momentum-fit
(def mean-model
  (gen [n]
    (let [mu (param :mu)]
      (doseq [i (range n)]
        (trace [:y i] (d/normal mu 1.0))))))

(def ys [1.2 0.4 2.1 1.7 0.9])

(def example
  "The five numbers as one example: mean-model's arguments and
   constraints."
  [[(count ys)] (apply cm/choicemap (mapcat (fn [i y] [[:y i] y]) (range) ys))])

(defn fit
  "mean-model's store after 200 steps of optimizer from mu = 0."
  [optimizer]
  (:store (learning/train mean-model {:mu 0.0} optimizer (constantly example)
                          grad/param-gradients {:batches-per-epoch 200})))

(def fitted (fit {:kind ::momentum :lr 0.05 :beta 0.5}))

(array/allclose? (:mu fitted) (/ (reduce + ys) (count ys)))
;=> true

;; book: momentum-builtin
(array/allclose? (:mu fitted)
                 (:mu (fit (learning/momentum {:lr 0.05 :momentum 0.5}))))
;=> true

;; book: counts
(def n-rates "The estimator's number of importance samples." 5)

(defn log-estimate
  "The log of the importance estimate of p(y): the mean over the rates of
   y's Poisson probability."
  [y rates]
  (- (array/logsumexp (dc/log-prob (d/poisson rates) y)) (Math/log n-rates)))

(def counts
  "A count y, Poisson with a rate from gamma(2, theta), its density
   estimated by importance sampling over the rate from its prior."
  (custom/estimated
   {:addresses [:y]
    :estimate (fn [[theta] choices key]
                (log-estimate (cm/get-choice choices :y)
                              (dc/sample (d/gamma 2.0 theta) key [n-rates])))
    :simulate (fn [[theta] key]
                ;; y with its own rate, which joins the estimate at a
                ;; uniform position among rates from the prior
                (let [[k-rate k-y k-j k-rates] (rng/split key 4)
                      rate (dc/sample (d/gamma 2.0 theta) k-rate)
                      y (dc/sample (d/poisson rate) k-y)
                      j (rng/randint k-j 0 n-rates)
                      others (dc/sample (d/gamma 2.0 theta) k-rates [n-rates])
                      rates (array/where (array/== (array/arange n-rates) j)
                                         rate others)]
                  {:choices (cm/choicemap :y y)
                   :log-estimate (log-estimate y rates)}))}))

;; book: counts-exact
(defn counts-log-density
  "The exact log p(y; theta): a gamma(2, theta) mixture of Poissons is
   neg-binom(2, 1/(1 + theta))."
  [theta y]
  (dc/log-prob (d/neg-binom 2 (/ 1.0 (+ 1.0 theta))) y))

(def log-estimates
  "10000 estimates of log p(y = 3; theta = 1.5), host numbers."
  (let [y3 (cm/choicemap :y 3)]
    (mapv #(array/item (:weight (gfi/generate counts [1.5] y3 %)))
          (rng/split (rng/key 14) 10000))))

(def exact-count (array/item (array/exp (counts-log-density 1.5 3))))

exact-count
;=> 0.1382

[(within-5-se? (map #(Math/exp %) log-estimates) exact-count)
 (let [[m se] (mean-se log-estimates)]
   (> (- (Math/log exact-count) m) (* 5 se)))]
;=> [true true]

;; book: estimates-figure
;; figure: figures/045-estimates.svg
(defn running-means
  "[i, the mean of the first i of xs] for i from 100 on, every 100th."
  [xs]
  (let [sums (vec (reductions + xs))]
    (for [i (range 100 (inc (count xs)) 100)]
      [i (/ (sums (dec i)) i)])))

(let [xis (mapv #(Math/exp %) log-estimates)
      n (count xis)
      sd (* (second (mean-se xis)) (Math/sqrt n))
      band (fn [sign]
             (for [i (range 100 (inc n) 100)]
               [i (+ exact-count (* sign 5 (/ sd (Math/sqrt i))))]))]
  (svg/paths [{:paths [(band 1) (band -1)] :color "gray" :width 1}
              {:paths [[[100 exact-count] [n exact-count]]]
               :color "black" :width 1}
              {:paths [(running-means xis)]}
              {:paths [(map (fn [[i m]] [i (Math/exp m)])
                            (running-means log-estimates))]
               :color "firebrick"}]
             {:x-label "estimates" :y-label "estimate of p(y = 3)"}))

;; book: counts-laws
(def counts-models [[:counts counts [1.5] [[0.5]]]])

(keys (laws/verify counts-models few laws/encapsulated))
;=> (:score-is-assessed-on-the-same-key)

(laws/verify counts-models few laws/estimated)
;=> {}

;; book: counts-model
(def rate-model
  "theta from gamma(2, 1), and a count given theta by counts."
  (gen []
    (let [theta (trace :theta (d/gamma 2.0 1.0))]
      (splice :obs counts theta)
      theta)))

(def rate-trace
  (:trace (gfi/generate rate-model [] (cm/choicemap [:obs :y] 3) (rng/key 15))))

;; book: counts-gradient
;; error: A gradient needs exact densities
#_(grad/choice-gradients rate-trace (sel/select :theta))

;; book: vector-sum
(def additions "The number of additions vector-sum has made." (atom 0))

(defn add
  "x + y, counted in additions."
  [x y]
  (swap! additions inc)
  (+ x y))

(defn sum-of
  "The sum of xs, by add."
  [xs]
  (reduce add 0.0 xs))

(defn moved-sum
  "The sum of xs', from state, holding xs and its sum, and d, xs''s
   vector-diff from xs: one addition for each element updated, added or
   removed."
  [{:keys [xs sum]} xs' d]
  (let [n (count xs)
        n' (count xs')]
    (as-> sum s
      (reduce (fn [s i] (add s (- (xs' i) (xs i)))) s (keys (:updated d)))
      (reduce (fn [s i] (add s (xs' i))) s (range n n'))
      (reduce (fn [s i] (add s (- (xs i)))) s (range n' n)))))

(def vector-sum
  "Gen.jl's MySum: the sum of a vector, which an update with a vector-diff
   moves by the elements that changed."
  (custom/deterministic
   {:apply-with-state (fn [[xs]] (let [s (sum-of xs)] [s {:xs xs :sum s}]))
    :update-with-state
    (fn [state [xs'] [d]]
      (let [s (if (diff/vector-diff? d) (moved-sum state xs' d) (sum-of xs'))]
        [{:xs xs' :sum s} s diff/unknown-change]))}))

;; book: vector-sum-update
(def xs1000 (mapv #(Math/sin %) (range 1000)))

(def xs1000' (assoc xs1000 3 10.0))

(def one-changed
  "xs1000''s change from xs1000: element 3."
  (diff/vector-diff 1000 1000 {3 diff/unknown-change}))

(defn additions-of
  "[the additions (f) makes, its value]."
  [f]
  (reset! additions 0)
  (let [v (f)]
    [@additions v]))

(def summed-trace (gfi/simulate vector-sum [xs1000] (rng/key 16)))

(let [[n-moved moved] (additions-of #(gfi/update summed-trace [xs1000']
                                                 [one-changed] cm/EMPTY
                                                 (rng/key 17)))
      [n-again again] (additions-of #(gfi/update summed-trace [xs1000']
                                                 [diff/unknown-change]
                                                 cm/EMPTY (rng/key 17)))]
  [n-moved n-again (:retval (:trace moved)) (:retval (:trace again))])
;=> [1 1000 9.845970085481296 9.845970085481321]

(let [xs990 (subvec xs1000 0 990)
      ten-dropped (diff/vector-diff 990 1000 {})
      [n-dropped dropped] (additions-of #(gfi/update summed-trace [xs990]
                                                     [ten-dropped] cm/EMPTY
                                                     (rng/key 18)))]
  [n-dropped (array/allclose? (:retval (:trace dropped)) (reduce + xs990))])
;=> [10 true]

;; book: static-hint
(def static-noisy-sum
  (gen ^:static [xs]
    (let [s (splice :sum vector-sum xs)]
      (trace :y (d/normal s 1.0)))))

(def dynamic-noisy-sum
  (gen [xs]
    (let [s (splice :sum vector-sum xs)]
      (trace :y (d/normal s 1.0)))))

(defn update-additions
  "The additions an update of model's trace from xs1000 to xs1000' makes,
   with their vector-diff as the argdiff, and its weight."
  [model]
  (let [t (:trace (gfi/generate model [xs1000] (cm/choicemap :y 20.0)
                                (rng/key 18)))]
    (additions-of #(:weight (gfi/update t [xs1000'] [one-changed] cm/EMPTY
                                        (rng/key 19))))))

(let [[n-static w-static] (update-additions static-noisy-sum)
      [n-dynamic w-dynamic] (update-additions dynamic-noisy-sum)]
  [n-static n-dynamic (array/allclose? w-static w-dynamic)])
;=> [1 1000 true]

;; book: propagate
(defn scaled
  "xs times c, element by element."
  [xs c]
  (mapv #(* c %) xs))

(def scaled-noisy-sum
  (gen ^:static [xs]
    (let [ys (scaled xs 2.0)
          s (splice :sum vector-sum ys)]
      (trace :y (d/normal s 1.0)))))

(first (update-additions scaled-noisy-sum))
;=> 1000

(defmethod diff/propagate #'scaled [_ _ [d-xs d-c]]
  ;; element i of the result changes only when element i of xs does
  (if (diff/no-change? d-c) d-xs diff/unknown-change))

(first (update-additions scaled-noisy-sum))
;=> 1

;; book: propagate-sound
(diff/consistent? (scaled xs1000 2.0) (scaled xs1000' 2.0)
                  (diff/propagate #'scaled [xs1000' 2.0]
                                  [one-changed diff/no-change]))
;=> true

;; book: propagate-laws
(defn doubled
  "xs times 2, element by element."
  [xs]
  (mapv #(* 2.0 %) xs))

(defmethod diff/propagate #'doubled [_ _ _]
  ;; wrong: the result changes when xs does
  diff/no-change)

(def doubled-noisy-sum
  (gen ^:static [xs]
    (let [ys (doubled xs)
          s (splice :sum vector-sum ys)]
      (trace :y (d/normal s 1.0)))))

(defn entry
  "A model entry of the static model on ten numbers, with one of them
   changed as other arguments."
  [label model]
  (let [xs (mapv double (range 10))]
    [label model [xs] [[(assoc xs 3 7.5)]]]))

[(laws/verify [(entry :scaled scaled-noisy-sum)] few laws/static)
 (keys (laws/verify [(entry :doubled doubled-noisy-sum)] few laws/static))]
;=> [{} (:static-matches-dynamic)]

;; book: save-trace
(def urn-model-gf (urn-model urn))

(def saved
  (:trace (gfi/generate urn-model-gf [60] observed (rng/key 20))))

(def trace-file
  (doto (java.io.File/createTempFile "urn-trace" ".edn") .deleteOnExit))

(spit trace-file (serialize/write-trace saved))

(def loaded (serialize/read-trace urn-model-gf (slurp trace-file)))

(let [parts (juxt :args :choices :retval :score)]
  (= (parts saved) (parts loaded)))
;=> true

;; book: trace-data
(let [{:keys [numerics args choices score]}
      (serialize/read-trace-data (slurp trace-file))]
  [(set (keys numerics)) args (count (cm/addresses choices))
   (= score (:score saved))])
;=> [#{:backend :float} [60] 61 true]

;; book: not-serializable
;; error: write-trace: the trace keeps encapsulated randomness
#_(serialize/write-trace rate-trace)

;; book: solution-1
(dc/defdist ^:elementwise unnormalized-kumaraswamy
  "kumaraswamy's density without its constant a·b."
  [a b]
  (sample [key batch-shape] (dc/sample (kumaraswamy a b) key batch-shape))
  (log-prob [x]
    (- (dc/log-prob (kumaraswamy a b) x) (array/log (* a b)))))

(def u25 (unnormalized-kumaraswamy 2.0 5.0))

(integral u25 10000)
;=> 0.1

(array/allclose? (- (dc/log-prob u25 0.3) (dc/log-prob u25 0.5))
                 (- (dc/log-prob k25 0.3) (dc/log-prob k25 0.5)))
;=> true

[(within-5-se? (array/->clj (dc/sample u25 (rng/key 1) [20000]))
               (kumaraswamy-moment 2.0 5.0 1))
 (every? (fn [x] (array/allclose? (dc/log-prob u25 x)
                                  (Math/log (kumaraswamy-density 2.0 5.0 x))))
         [0.05 0.3 0.5 0.9])]
;=> [true false]

;; book: solution-2
(defn norm
  "The Euclidean norm of x."
  [x]
  (array/sqrt (array/sum (* x x))))

(array/->clj (array/isnan ((array/grad norm) (array/array [0.0 0.0]))))
;=> [true true]

(def safe-norm
  "The norm, with the gradient x/|x|, and 0 at x = 0."
  (array/custom-vjp
   norm
   (fn [[x] n g]
     [(array/where (array/> n 0.0) (* g (/ x (array/maximum n 1e-300))) 0.0)])))

(defn norm-difference
  "The central difference of the norm at x along axis i, with the step
   1e-3."
  [x i]
  (let [e (array/array (assoc [0.0 0.0] i 1e-3))]
    (/ (- (norm (+ x e)) (norm (- x e))) 2e-3)))

(let [x (array/array [3.0 4.0])
      differences (array/array [(norm-difference x 0) (norm-difference x 1)])]
  [((array/grad safe-norm) (array/array [0.0 0.0]))
   ((array/grad safe-norm) x)
   (array/allclose? ((array/grad safe-norm) x) differences {:rtol 1e-3})])
;=> [[0.0 0.0] [0.6 0.8] true]

;; book: solution-3
(def straight-through-round
  "array/round, with the gradient of the identity."
  (array/custom-vjp array/round (fn [_args _out g] [g])))

[((array/grad array/round) 2.3)
 ((array/grad straight-through-round) 2.3)
 (central-difference #(array/item (array/round %)) 2.3)]
;=> [0.0 1.0 0.0]

;; book: solution-4
(def walking-regenerate regenerate-urn)

(defn keeps-every-draw?
  "Does a regenerate of tr to n draws with selection keep all of its
   draws?"
  [tr n selection]
  (and (= n (nth (:args tr) 2))
       (not-any? #(sel/selected? selection (draw-at %)) (range n))))

(defn regenerate-urn-fast
  "regenerate-urn, without the walk when it keeps every draw: its weight
   is then the change of the score."
  [gf tr [_ _ n :as args] selection key]
  (if (keeps-every-draw? tr n selection)
    (let [t (urn-trace gf args (:choices tr) (:retval tr))]
      {:trace t :weight (- (:score t) (:score tr)) :retdiff diff/no-change})
    (walking-regenerate gf tr args selection key)))

(with-redefs [regenerate-urn regenerate-urn-fast]
  (laws/verify urn-models few))
;=> {}

(defn regenerate-urn-stale
  "regenerate-urn-fast whose weight is 0 when it keeps every draw: the
   change of the arguments forgotten."
  [gf tr [_ _ n :as args] selection key]
  (cond-> (regenerate-urn-fast gf tr args selection key)
    (keeps-every-draw? tr n selection) (assoc :weight 0.0)))

(with-redefs [regenerate-urn regenerate-urn-stale]
  [(laws/verify urn-models few)
   (= (set (keys (laws/verify urn-models {:num-tests 100 :seed 1})))
      #{:regenerate-weight-is-the-retained-density-ratio
        :regenerate-weight-with-new-arguments-is-the-projection-ratio})])
;=> [{} true]

;; book: solution-5
(def posterior-traces
  "200 MH steps on urn-model-gf from the saved trace, every trace kept."
  (vec (reductions (fn [tr k] (:trace (mh/mh tr m-walk [] k)))
                   saved
                   (rng/split (rng/key 21) 200))))

(def chain-file
  (doto (java.io.File/createTempFile "urn-chain" ".edn") .deleteOnExit))

(spit chain-file (serialize/write (mapv :choices posterior-traces)))

(def rebuilt
  (mapv #(:trace (gfi/generate urn-model-gf [60] % (rng/key 22)))
        (serialize/read (slurp chain-file))))

[(= (mapv :choices posterior-traces) (mapv :choices rebuilt))
 (every? true? (map #(array/allclose? (:score %1) (:score %2))
                    posterior-traces rebuilt))]
;=> [true true]
