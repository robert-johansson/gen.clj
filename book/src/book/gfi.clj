;; book: ns
(ns book.gfi
  (:refer-clojure :exclude [+ - * /])
  (:require [gen.array :as array :refer [+ - * /]]
            [gen.book.svg :as svg]
            [gen.choicemap :as cm]
            [gen.combinators :as comb]
            [gen.diff :as diff]
            [gen.dist :as d]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.gradients :as grad]
            [gen.inference.diagnostics :as diag]
            [gen.inference.enumerative :as enum]
            [gen.inference.proposal :as prop]
            [gen.laws :as laws]
            [gen.random :as rng]
            [gen.selection :as sel]
            [gen.trace :as trace]))

;; book: dictionaries
(defn dictionaries
  "Every choice dictionary of an address universe (Def. 2.1.2), given the
   finite domain of each address: Clojure maps in which each address is
   absent or has one of its values."
  [domains]
  (reduce (fn [dicts [a values]]
            (for [dict dicts, v (cons ::absent values)]
              (if (= ::absent v) dict (assoc dict a v))))
          [{}] domains))

(def nine (dictionaries {:a [0 1] :b [0 1]}))

(count nine)
;=> 9

(= (set nine)
   #{{} {:a 0} {:a 1} {:b 0} {:b 1}
     {:a 0 :b 0} {:a 0 :b 1} {:a 1 :b 0} {:a 1 :b 1}})
;=> true

;; book: operations
(defn restrict
  "σ|_B: the choice map sigma without its choices outside the addresses B."
  [sigma B]
  (sel/get-selected sigma (apply sel/select B)))

(defn agree?
  "σ ~ τ: do the dictionaries sigma and tau (Clojure maps) have the same
   value at every address they share?"
  [sigma tau]
  (every? #(= (sigma %) (tau %)) (filter #(contains? tau %) (keys sigma))))

(def sigma (cm/choicemap :a 0 :b 1))

(cm/to-map (restrict sigma [:a]))
;=> {:a 0}

[(agree? {:a 0 :b 1} {:a 0 :c 1}) (agree? {:a 0 :b 1} {:b 0})]
;=> [true false]

(cm/to-map (cm/merge (cm/choicemap :a 0) (cm/choicemap :b 1)))
;=> {:a 0 :b 1}

;; book: merge-error
;; error: merge: the choice maps overlap
#_(cm/merge (cm/choicemap :a 0) (cm/choicemap :a 1))

;; book: dict
(defn dict
  "The choices of the trace t as a dictionary: a Clojure map of host
   values (for models whose addresses are not nested)."
  [t]
  (update-vals (cm/to-map (:choices t)) array/->clj))

;; book: burglary
(def burglary-model
  (gen []
    (let [burglary (trace :burglary (d/bernoulli 0.01))
          disabled (if burglary (trace :disabled (d/bernoulli 0.1)) false)
          alarm (if disabled
                  false
                  (trace :alarm (d/bernoulli (if burglary 0.94 0.01))))]
      (trace :calls (d/bernoulli (if alarm 0.7 0.05))))))

;; book: density
(defn density
  "p(τ; x): the probability (or density) of the dictionary tau under gf
   on args, by assess. A dictionary that lacks a choice the run makes, or
   has one it does not, is no run's: assess refuses it, and its
   probability is 0."
  [gf args tau]
  (try (array/item (array/exp (:weight (gfi/assess gf args (cm/from-map tau)))))
       (catch clojure.lang.ExceptionInfo e
         (if (some (ex-data e) [:missing-choice :unvisited]) 0.0 (throw e)))))

(defn table
  "The distribution of gf on args over every dictionary of the universe
   with the given domains: a map from dictionary to probability."
  [gf args domains]
  (into {} (map (juxt identity #(density gf args %)))
        (dictionaries domains)))

(defn support
  "The dictionaries of a table with positive probability."
  [table]
  (filter #(pos? (table %)) (keys table)))

(def universe
  (zipmap [:burglary :disabled :alarm :calls] (repeat [true false])))

(def p (table burglary-model [] universe))

[(count p) (count (support p))]
;=> [81 10]

(array/allclose? (reduce + (vals p)) 1.0)
;=> true

(array/allclose? (p {:burglary false :alarm false :calls false})
                 (* 0.99 0.99 0.95))
;=> true

;; book: factors
(defn bern
  "The probability of the value v under (d/bernoulli prob)."
  [prob v]
  (if v prob (- 1.0 prob)))

(defn factors
  "The burglary model's run that makes the dictionary tau, by hand: a map
   from each address to the probability of its choice given the choices
   before it, or nil when no run makes tau (a choice missing or extra)."
  [{:keys [burglary disabled alarm calls] :as tau}]
  (let [disabled? (and burglary disabled)
        visits (cond-> #{:burglary :calls}
                 burglary (conj :disabled)
                 (not disabled?) (conj :alarm))]
    (when (= visits (set (keys tau)))
      (cond-> {:burglary (bern 0.01 burglary)
               :calls (bern (if (and (not disabled?) alarm) 0.7 0.05) calls)}
        burglary (assoc :disabled (bern 0.1 disabled))
        (not disabled?) (assoc :alarm (bern (if burglary 0.94 0.01) alarm))))))

(defn by-hand
  "p(τ) by hand: the product of the factors of tau's run, or 0."
  [tau]
  (if-let [fs (factors tau)] (reduce * (vals fs)) 0.0))

(every? #(array/allclose? (p %) (by-hand %)) (keys p))
;=> true

;; book: runs-figure
;; figure: figures/031-runs.svg
(defn run-path
  "The choices of the dictionary tau in the order the model makes them,
   each labeled with its address's initial and its value."
  [tau]
  (for [a [:burglary :disabled :alarm :calls]
        :when (contains? tau a)]
    (symbol (str (first (name a)) "=" (if (tau a) "T" "F")))))

(svg/tree (map run-path (sort-by p > (support p))) {:root "run"})

;; book: structured
(defn structured?
  "Def. 2.1.3: are any two dictionaries in the support of the table
   either the same or different at an address they share?"
  [table]
  (let [supp (support table)]
    (every? (fn [[s t]] (or (= s t) (not (agree? s t))))
            (for [s supp, t supp] [s t]))))

(def maybe-a {{} 0.5, {:a true} 0.25, {:a false} 0.25})

(def all-ones {{1 1} 0.5, {1 1, 2 1} 0.25, {1 1, 2 1, 3 1} 0.125})

[(structured? p) (structured? maybe-a) (structured? all-ones)]
;=> [true false false]

;; book: flips
(def flips
  (gen []
    (loop [i 1]
      (if (trace i (d/bernoulli 0.5))
        (recur (inc i))
        i))))

(def flips-table
  "The flips model's dictionaries of up to 40 flips, with their
   probabilities: {1 F}, {1 T, 2 F}, … (the rest of the support has
   probability 2^-40)."
  (into {} (for [k (range 1 41)
                 :let [tau (assoc (zipmap (range 1 k) (repeat true)) k false)]]
             [tau (density flips [] tau)])))

(flips-table {1 true 2 true 3 false})
;=> 0.125

;; book: events
(defn in-event?
  "Is tau in the event E_ρ (Eq. 2.1): does it agree with rho?"
  [rho tau]
  (agree? tau rho))

(defn in-event'?
  "Is tau in the event E'_ρ (Eq. 2.1): does it agree with rho and hold
   all of rho's addresses?"
  [rho tau]
  (and (agree? tau rho) (every? #(contains? tau %) (keys rho))))

(def rho {2 true})

(mapv (juxt #(in-event? rho %) #(in-event'? rho %))
      [{1 false} {1 true 2 false} {1 true 2 true 3 false}])
;=> [[true false] [false false] [true true]]

;; book: sound
(defn sound?
  "Def. 2.1.4: is rho existentially sound under the table: does every
   dictionary in its support hold all of rho's addresses?"
  [table rho]
  (every? (fn [tau] (every? #(contains? tau %) (keys rho))) (support table)))

[(sound? flips-table {1 true}) (sound? flips-table rho)]
;=> [true false]

[(sound? p {:calls true}) (sound? p {:alarm true})]
;=> [true false]

;; book: marginal
(defn marginal
  "p̄(σ), Eq. 2.2: the total probability of the table's dictionaries that
   agree with sigma."
  [table sigma]
  (reduce + 0.0 (map table (filter #(agree? % sigma) (keys table)))))

(array/allclose? (marginal flips-table rho) 0.75)
;=> true

;; book: conditional
(defn subsets
  "Every subset of the collection xs, as sets."
  [xs]
  (reduce (fn [ss x] (concat ss (map #(conj % x) ss))) [#{}] xs))

(defn conditional
  "p(υ | σ), Eq. 2.4: the probability of the dictionary v (on addresses
   outside sigma's) given sigma: the total over the subsets B of sigma's
   addresses of the probability of v merged with σ|_B, over p̄(σ)."
  [table sigma v]
  (/ (reduce + 0.0 (for [B (subsets (keys sigma))]
                     (table (merge v (select-keys sigma B)) 0.0)))
     (marginal table sigma)))

(mapv #(conditional flips-table rho %)
      [{1 false} {1 true} {1 true 3 false} {1 true 3 true 4 false}])
;=> [0.6667 0.0 0.16667 0.08333]

;; book: marginal-sound
(defn marginal-sound
  "p̄(σ) by Eq. 2.3, for a structured table and an existentially sound
   sigma: the total probability of sigma merged with each dictionary of
   the other addresses."
  [table domains sigma]
  (reduce + 0.0 (for [v (dictionaries (apply dissoc domains (keys sigma)))]
                  (table (merge v sigma) 0.0))))

(def calls {:calls true})

(marginal p calls)
;=> 0.06193

(array/allclose? (marginal p calls) (marginal-sound p universe calls))
;=> true

[(marginal p {:alarm true}) (marginal-sound p universe {:alarm true})]
;=> [0.01936 0.01836]

(array/allclose? (- (marginal p {:alarm true})
                    (marginal-sound p universe {:alarm true}))
                 (* 0.01 0.1))
;=> true

;; book: posterior
(def posterior
  "p(· | calls = true), Eq. 2.5: the burglary model's dictionaries without
   :calls, each with its conditional probability."
  (into {} (for [v (dictionaries (dissoc universe :calls))]
             [v (/ (p (merge v calls)) (marginal p calls))])))

[(array/allclose? (reduce + (vals posterior)) 1.0) (structured? posterior)]
;=> [true true]

(def burglary-given-calls
  (reduce + (map posterior (filter :burglary (keys posterior)))))

burglary-given-calls
;=> 0.09686

;; book: rejection
(def runs
  "50000 runs of the burglary model, as dictionaries."
  (mapv #(dict (gfi/simulate burglary-model [] %))
        (rng/split (rng/key 1) 50000)))

(defn within-5-se?
  "Is the frequency of hits among n trials within five standard errors of
   the probability prob?"
  [hits n prob]
  (< (abs (- (/ hits (double n)) prob))
     (* 5 (Math/sqrt (/ (* prob (- 1.0 prob)) n)))))

(within-5-se? (count (filter :calls runs)) (count runs) (marginal p calls))
;=> true

(let [kept (filter :calls runs)]
  (within-5-se? (count (filter :burglary kept)) (count kept)
                burglary-given-calls))
;=> true

;; book: estimate
(defn estimate
  "The average of exp(weight) over n runs of generate on gf, args and the
   constraints (a Clojure map), one key each, and its standard error."
  [gf args constraints n key]
  (let [sigma (cm/from-map constraints)
        weight #(:weight (gfi/generate gf args sigma %))
        ws (mapv #(array/item (array/exp (weight %))) (rng/split key n))
        mean (/ (reduce + ws) n)
        var (/ (reduce + (map #(* (- % mean) (- % mean)) ws)) (dec n))]
    {:mean mean :se (Math/sqrt (/ var n))}))

(let [{:keys [mean se]} (estimate burglary-model [] calls 5000 (rng/key 2))]
  (< (abs (- mean (marginal p calls))) (* 5 se)))
;=> true

;; book: generate-strict
;; error: generate: the constraint at [2] was not visited
#_(mapv #(gfi/generate flips [] (cm/choicemap 2 true) %)
        (rng/split (rng/key 28) 20))

;; book: supportive
(every? #(pos? (marginal p %)) (keys p))
;=> true

(def certain
  (gen []
    (trace :a (d/bernoulli 1.0))))

(marginal (table certain [] {:a [true false]}) {:a false})
;=> 0.0

(:weight (gfi/generate certain [] (cm/choicemap :a false) (rng/key 3)))
;=> ##-Inf

;; book: sum-model
(def sum-model
  (gen []
    (let [n (+ 1 (trace :n (d/geometric 0.5)))
          xs (loop [i 0, xs []]
               (if (array/< i n)
                 (recur (inc i) (conj xs (trace [:x i] (d/normal 0 1))))
                 xs))]
      (trace :y (d/normal (reduce + xs) 1)))))

(defn normal-density
  "The density of (d/normal mu sd) at x, by hand."
  [x mu sd]
  (/ (Math/exp (* -0.5 (Math/pow (/ (- x mu) sd) 2)))
     (* sd (Math/sqrt (* 2 Math/PI)))))

(array/allclose? (density sum-model [] {:n 1 :x {0 0.3 1 -0.2} :y 0.5})
                 (* 0.25 (normal-density 0.3 0 1) (normal-density -0.2 0 1)
                    (normal-density 0.5 0.1 1)))
;=> true

;; book: sum-marginal
(defn sum-marginal
  "The marginal likelihood of the observation y under sum-model, by the
   thesis's series: given n normals, y is normal with mean 0 and variance
   n + 1, and n has probability 0.5^n."
  [y]
  (reduce + (for [n (range 1 200)]
              (* (Math/pow 0.5 n) (normal-density y 0 (Math/sqrt (inc n)))))))

(sum-marginal 1.5)
;=> 0.15573434

;; book: sum-estimate
(defn agrees?
  "Is the estimate of the marginal likelihood of y from n runs of
   generate within five standard errors of the series?"
  [y n key]
  (let [{:keys [mean se]} (estimate sum-model [] {:y y} n key)]
    (< (abs (- mean (sum-marginal y))) (* 5 se))))

(mapv (fn [y] (mapv #(normal-density y 0 (Math/sqrt (inc %))) [1 2 3]))
      [1.5 0.0])
;=> [[0.1607 0.1583 0.1506] [0.2821 0.2303 0.1995]]

[(agrees? 1.5 10000 (rng/key 4)) (agrees? 0.0 10000 (rng/key 5))]
;=> [true true]

;; book: sum-figure
;; figure: figures/031-marginal.svg
(svg/plot [{:curves [sum-marginal]}
           {:points (for [y (range -4.0 4.5 1.0)]
                      [y (:mean (estimate sum-model [] {:y y} 1000
                                          (rng/key 27)))])}]
          {:x-range [-6.0 6.0] :x-label "y" :y-label "marginal likelihood"})

;; book: exceeds-one
(density (gen [] (trace :y (d/normal 0 0.1))) [] {:y 0.0})
;=> 3.989

;; book: tuple-examples
(def twice (gen [x] (* 2 x)))

(let [t (gfi/simulate twice [3.0] (rng/key 6))]
  [(cm/empty? (:choices t)) (:score t) (:retval t)])
;=> [true 0.0 6.0]

(def depends
  (gen [x]
    (if (trace :a (d/bernoulli x)) x 0.0)))

(defn row
  "A dictionary's probability under gf on args, and its return value."
  [gf args tau]
  [(density gf args tau) (:retval (gfi/assess gf args (cm/from-map tau)))])

[(row depends [0.3] {:a true}) (row depends [0.3] {:a false})]
;=> [[0.3 0.3] [0.7 0.0]]

(def choosy
  (gen [x]
    (when x (trace :n (d/geometric 0.5)))
    0))

[(row choosy [false] {}) (mapv #(row choosy [true] {:n %}) [0 1 2])]
;=> [[1.0 0] [[0.5 0] [0.25 0] [0.125 0]]]

;; book: trace-tuple
(def t (gfi/simulate burglary-model [] (rng/key 7)))

[(dict t) (:retval t)]
;=> [{:burglary false :alarm false :calls false} false] ; float64

(array/allclose? (:score t) (Math/log (p (dict t))))
;=> true

;; book: law
(def law-models
  [["burglary" burglary-model [] []]
   ["flips" flips [] []]
   ["sum" sum-model [] []]
   ["depends" depends [0.3] [[0.8]]]])

(defn law
  "Checks a law of gen.laws over the models in n cases (from one seed):
   whether it passed. laws/check fails a law most of whose cases were
   vacuous."
  [law-fn models n]
  (-> (laws/check n (law-fn (laws/cases models)) {:seed 31})
      (select-keys [:pass? :num-tests])))

(law laws/score-is-the-assessed-density law-models 50)
;=> {:pass? true :num-tests 50}

;; book: simulate
(defn frequencies-agree?
  "Is the frequency of each dictionary of the table's support among the
   runs within five standard errors of its probability?"
  [table runs]
  (let [n (count runs)
        counts (frequencies runs)]
    (every? #(within-5-se? (counts % 0) n (table %)) (support table))))

[(every? #(pos? (p %)) runs) (frequencies-agree? p runs)]
;=> [true true]

;; book: generate-complete
(def quiet {:burglary false :alarm false :calls false})

(let [{t :trace w :weight}
      (gfi/generate burglary-model [] (cm/from-map quiet) (rng/key 8))]
  [(= quiet (dict t)) (array/allclose? w (:score t))
   (array/allclose? (:score t) (Math/log (* 0.99 0.99 0.95)))])
;=> [true true true]

(law laws/generate-with-every-choice-is-deterministic law-models 50)
;=> {:pass? true :num-tests 50}

;; book: generate-extra
;; error: generate: the constraint at [:foo] was not visited
#_(gfi/generate burglary-model [] (cm/from-map (assoc quiet :foo false))
                (rng/key 8))

;; book: assess-propose
(let [{c :choices w :weight} (gfi/propose burglary-model [] (rng/key 9))]
  (array/allclose? w (Math/log (p (update-vals (cm/to-map c) array/->clj)))))
;=> true

[(law laws/propose-weight-is-the-assessed-density law-models 50)
 (law laws/assess-is-generate-with-every-choice law-models 50)]
;=> [{:pass? true :num-tests 50} {:pass? true :num-tests 50}]

;; book: h-update
(defn h-update
  "Def. 2.3.1: [τ' σ'] for the dictionary tau and the change sigma under
   the table, where τ', τ|_B merged with sigma, is in the support for
   some B, and σ' is tau at the addresses τ' lacks or sigma sets; nil
   (the thesis's bottom) when no B gives one."
  [table tau sigma]
  (some (fn [B]
          (let [tau' (merge (select-keys tau B) sigma)]
            (when (pos? (table tau' 0.0))
              [tau' (select-keys tau (filter #(or (contains? sigma %)
                                                  (not (contains? tau' %)))
                                             (keys tau)))])))
        (subsets (remove #(contains? sigma %) (keys tau)))))

(def alarmed {:burglary true :disabled false :alarm true :calls true})

(h-update p alarmed {:disabled true})
;=> [{:burglary true :disabled true :calls true} {:disabled false :alarm true}]

(h-update p {:burglary false :alarm true :calls true} {:burglary true})
;=> nil

(every? (fn [[tau sigma]]
          (>= 1 (count (distinct
                        (for [B (subsets (remove #(contains? sigma %)
                                                 (keys tau)))
                              :let [tau' (merge (select-keys tau B) sigma)]
                              :when (pos? (p tau' 0.0))]
                          tau')))))
        (for [tau (support p), sigma (keys p)] [tau sigma]))
;=> true

;; book: update-exhaustive
(defn trace-of
  "The trace of gf on args whose choices are the dictionary tau."
  [gf args tau]
  (:trace (gfi/generate gf args (cm/from-map tau) (rng/key 0))))

(defn updated
  "gfi/update of the burglary model's trace of tau with the change sigma:
   the new dictionary, the discard (a dictionary) and the weight."
  [tau sigma]
  (let [{t :trace w :weight disc :discard}
        (gfi/update (trace-of burglary-model [] tau) (cm/from-map sigma)
                    (rng/key 10))]
    [(dict t) (update-vals (cm/to-map disc) array/->clj) w]))

(def update-cases
  (for [tau (support p), sigma (keys p)
        :let [h (h-update p tau sigma)]
        :when h]
    [tau sigma h]))

(count update-cases)
;=> 384

(every? (fn [[tau sigma [tau' sigma']]]
          (let [[new discard w] (updated tau sigma)]
            (and (= tau' new) (= sigma' discard)
                 (array/allclose? w (Math/log (/ (p tau') (p tau)))))))
        update-cases)
;=> true

;; book: reversible
(every? (fn [[tau sigma [tau' sigma']]]
          (and (= [tau sigma] (h-update p tau' sigma'))
               (= [tau sigma] (pop (updated tau' sigma')))))
        update-cases)
;=> true

(law laws/updates-are-reversible-with-exact-discards law-models 50)
;=> {:pass? true :num-tests 50}

;; book: update-bottom
(defn update-outcome
  "What gfi/update does with a change that Def. 2.3.1 leaves undefined:
   :draws-fresh choices, or throws on an :unvisited constraint."
  [tau sigma]
  (try (updated tau sigma) :draws-fresh
       (catch clojure.lang.ExceptionInfo e
         (if (:unvisited (ex-data e)) :unvisited (throw e)))))

(frequencies (for [tau (support p), sigma (keys p)
                   :when (nil? (h-update p tau sigma))]
               (update-outcome tau sigma)))
;=> {:unvisited 372 :draws-fresh 54} ; float64

;; book: change-hints
(def append
  (gen [xs]
    (conj xs (trace :a (d/normal 0 1)))))

(def appended (gfi/simulate append [[4.2 5.3 3.3 1.1]] (rng/key 11)))

(def negated
  (gfi/update appended [[4.2 -5.3 3.3 1.1]]
              [(diff/vector-diff 4 4 {1 diff/unknown-change})]
              cm/EMPTY (rng/key 12)))

[(:weight negated) (cm/empty? (:discard negated))
 (= (appended :a) ((:trace negated) :a))]
;=> [0.0 true true]

[(diff/unknown-change? (:retdiff negated))
 (diff/consistent? (:retval appended) (:retval (:trace negated))
                   (diff/vector-diff 5 5 {1 diff/unknown-change}))]
;=> [true true]

;; book: gradients
(def sensor
  (gen [x]
    (let [mu (trace :mu (d/normal x 1))]
      (trace :y (d/normal mu 0.5))
      (* mu mu))))

(def sensed
  (trace-of sensor [1.0] {:mu 0.4 :y 1.2}))

(def gradients
  (grad/choice-gradients sensed (sel/select :mu) 2.0 {:argnums [0]}))

(let [[x mu y v] [1.0 0.4 1.2 2.0]]
  [(array/allclose? (cm/get-choice (:gradients gradients) :mu)
                    (+ (- x mu) (/ (- y mu) 0.25) (* 2 mu v)))
   (array/allclose? (first (:arg-grads gradients)) (- mu x))])
;=> [true true]

;; book: q-forward
(defn q-forward
  "q(υ; x, σ) of forward sampling (thesis p. 145) for the burglary model:
   the product of the factors of tau's run at υ's addresses, where tau,
   the dictionary the run makes, is υ merged with σ|_B."
  [tau v-addresses]
  (reduce * 1.0 (map (factors tau) v-addresses)))

(defn completions
  "The dictionaries of the burglary model's support that hold sigma: the
   runs that forward sampling may make from sigma."
  [sigma]
  (filter #(in-event'? sigma %) (support p)))

(def sound-sigmas (filter #(sound? p %) (keys p)))

(count sound-sigmas)
;=> 9

(every? (fn [sigma]
          (array/allclose?
           1.0
           (reduce + (map #(q-forward % (remove (set (keys sigma)) (keys %)))
                          (completions sigma)))))
        sound-sigmas)
;=> true

;; book: generate-weight
(every? (fn [[sigma k]]
          (let [{t :trace w :weight}
                (gfi/generate burglary-model [] (cm/from-map sigma) k)
                tau (dict t)]
            (array/allclose?
             w (- (Math/log (p tau))
                  (Math/log (q-forward tau (remove (set (keys sigma))
                                                   (keys tau))))))))
        (for [sigma sound-sigmas, k (rng/split (rng/key 13) 5)] [sigma k]))
;=> true

;; book: project
(every? (fn [[tau S]]
          (array/allclose?
           (gfi/project (trace-of burglary-model [] tau) (apply sel/select S))
           (reduce + 0.0 (map #(Math/log ((factors tau) % 1.0)) S))))
        (for [tau (support p), S (subsets (keys universe))] [tau S]))
;=> true

(law laws/generate-weight-is-the-projection law-models 50)
;=> {:pass? true :num-tests 50}

;; book: eq-4-1
(defn eq-4-1
  "Regenerate's weight by Eq. 4.1 for the burglary model's forward
   sampling: the move from the dictionary tau to tau' with the selection
   of the addresses S."
  [tau tau' S]
  (let [sigma (apply dissoc tau S)
        sigma' (apply dissoc tau' S)
        v (remove #(contains? sigma %) (keys tau'))
        v' (remove #(contains? sigma' %) (keys tau))]
    (- (+ (Math/log (p tau')) (Math/log (q-forward tau v')))
       (+ (Math/log (p tau)) (Math/log (q-forward tau' v))))))

(def regenerate-cases
  (for [tau (support p), S (subsets (keys universe)),
        k (rng/split (rng/key 14) 3)]
    [tau S k]))

(count regenerate-cases)
;=> 480

(every? (fn [[tau S k]]
          (let [{t' :trace w :weight}
                (gfi/regenerate (trace-of burglary-model [] tau)
                                (apply sel/select S) k)]
            (array/allclose? w (eq-4-1 tau (dict t') S))))
        regenerate-cases)
;=> true

(law laws/regenerate-weight-is-the-retained-density-ratio law-models 50)
;=> {:pass? true :num-tests 50}

;; book: abc
(def abc
  (gen []
    (let [a (trace :a (d/bernoulli 0.5))
          b (when a (trace :b (d/bernoulli 0.5)))]
      (trace :c (d/bernoulli (cond (not a) 0.1 b 0.5 :else 0.8))))))

(mapv #(density abc [] %)
      [{:a false :c false} {:a false :c true} {:a true :b false :c false}
       {:a true :b false :c true} {:a true :b true :c false}
       {:a true :b true :c true}])
;=> [0.45 0.05 0.05 0.2 0.125 0.125]

;; book: abc-q
(defn given?
  "Do the constraints give a value at addr?"
  [constraints addr]
  (cm/has-value? constraints addr))

(def abc-q
  (gen [constraints]
    (let [a (if (given? constraints :a)
              (cm/get-choice constraints :a)
              (trace :a (d/bernoulli 0.6)))]
      (when (and a (not (given? constraints :b)))
        (trace :b (d/bernoulli 0.5)))
      (when-not (given? constraints :c)
        (trace :c (d/bernoulli 0.5))))))

(def abc* (prop/with-internal-proposal abc abc-q))

;; book: abc-generate
(def a-false
  (first (filter #(false? ((:trace %) :a))
                 (map #(gfi/generate abc* [] (cm/choicemap :c true) %)
                      (rng/split (rng/key 15) 30)))))

[(dict (:trace a-false)) (:weight a-false)]
;=> [{:a false :c true} -2.0794]

(array/allclose? (:weight a-false) (Math/log (/ 0.05 0.4)))
;=> true

;; book: abc-regenerate
(def abc-t (trace-of abc* [] {:a true :b false :c true}))

(def to-a-false
  (first (filter #(false? ((:trace %) :a))
                 (map #(gfi/regenerate abc-t (sel/select :a :b) %)
                      (rng/split (rng/key 16) 30)))))

[(dict (:trace to-a-false)) (:weight to-a-false)]
;=> [{:a false :c true} -1.674]

(array/allclose? (:weight to-a-false)
                 (Math/log (/ (* 0.05 0.3) (* 0.2 0.4))))
;=> true

;; book: abc-project
(def abc-u (trace-of abc* [] {:a false :c true}))

[(gfi/project abc-u (sel/select :c))
 (gfi/project (trace-of abc [] {:a false :c true}) (sel/select :c))]
;=> [-2.0794 -2.3026]

;; book: abc-drop
;; error: regenerate leaves an unselected choice unvisited
#_(mapv #(gfi/regenerate abc-t (sel/select :a) %) (rng/split (rng/key 16) 30))

;; book: optimal
(def sprinkler
  (gen []
    (let [rain (trace :rain (d/bernoulli 0.2))
          on (trace :sprinkler (d/bernoulli (if rain 0.01 0.4)))]
      (trace :wet (d/bernoulli (cond (and rain on) 0.99
                                     rain 0.8
                                     on 0.9
                                     :else 0.01))))))

(def sprinkler-table
  (table sprinkler [] (zipmap [:rain :sprinkler :wet] (repeat [true false]))))

(def sprinkler* (enum/with-exact-proposal sprinkler [:rain :sprinkler]))

(def wet (cm/choicemap :wet true))

(mapv #(:weight (gfi/generate sprinkler* [] wet %))
      (rng/split (rng/key 17) 3))
;=> [-0.79147 -0.79147 -0.79147]

(array/allclose? (array/exp (:weight (gfi/generate sprinkler* [] wet
                                                   (rng/key 17))))
                 (marginal sprinkler-table {:wet true}))
;=> true

;; book: gibbs
(def wet-t (:trace (gfi/generate sprinkler* [] wet (rng/key 18))))

(mapv #(:weight (gfi/regenerate wet-t (sel/select :rain) %))
      (rng/split (rng/key 19) 3))
;=> [0.0 0.0 0.0]

;; book: optimal-laws
(def exact-models [["sprinkler*" sprinkler* [] []]])

[(law laws/generate-weight-is-the-projection exact-models 50)
 (law laws/regenerate-weight-is-the-projection-ratio exact-models 50)]
;=> [{:pass? true :num-tests 50} {:pass? true :num-tests 50}]

(:pass? (law laws/generate-weight-is-the-constrained-density exact-models 50))
;=> false

;; book: sequencing
(def p1
  (gen []
    (+ 1 (trace :n (d/geometric 0.5)))))

(def p2
  (gen [x]
    (trace :x (d/bernoulli (/ x (+ x 1.0))))
    0))

(def p3
  (gen []
    (splice p2 (splice p1))))

(mapv #(density p3 [] %)
      [{:n 0 :x true} {:n 1 :x true} {:n 2 :x false}])
;=> [0.25 0.16667 0.03125]

(defn sequenced
  "p3(τ) by the thesis's sequencing: p1 of tau's :n, times p2 of its :x
   given p1's return value."
  [tau]
  (let [tau1 (select-keys tau [:n])
        x (:retval (gfi/assess p1 [] (cm/from-map tau1)))]
    (* (density p1 [] tau1) (density p2 [x] (select-keys tau [:x])))))

(every? #(array/allclose? (density p3 [] %) (sequenced %))
        (for [n (range 6), x [true false]] {:n n :x x}))
;=> true

;; book: p1-at-x
(def p1-at-x
  (gen []
    (+ 1 (trace :x (d/geometric 0.5)))))

;; book: overlap
;; error: Address [:x] is used twice
#_(gfi/simulate (gen [] (splice p2 (splice p1-at-x))) [] (rng/key 20))

;; book: namespaced
(def p3-namespaced
  (gen []
    (splice :k2 p2 (splice :k1 p1-at-x))))

(def t3 (gfi/simulate p3-namespaced [] (rng/key 21)))

(cm/addresses (:choices t3))
;=> [[:k1 :x] [:k2 :x]]

(array/allclose? (:score t3)
                 (+ (:score (trace/subtrace t3 :k1))
                    (:score (trace/subtrace t3 :k2))))
;=> true

;; book: branching
(def p1-or-p2 (comb/switch p1 p2))

[(array/allclose? (density p1-or-p2 [0] {:n 2}) (density p1 [] {:n 2}))
 (array/allclose? (density p1-or-p2 [1 3.0] {:x true})
                  (density p2 [3.0] {:x true}))]
;=> [true true]

;; book: compositional-update
(def at-1-true
  (trace-of p3-namespaced [] {:k1 {:x 1} :k2 {:x true}}))

(def change (cm/choicemap [:k1 :x] 3 [:k2 :x] false))

(def whole (gfi/update at-1-true change (rng/key 22)))

(def first-part
  (gfi/update (trace/subtrace at-1-true :k1) (cm/get-submap change :k1)
              (rng/key 23)))

(def second-part
  (gfi/update (trace/subtrace at-1-true :k2) [(:retval (:trace first-part))]
              [diff/unknown-change] (cm/get-submap change :k2) (rng/key 24)))

(array/allclose? (:weight whole)
                 (+ (:weight first-part) (:weight second-part)))
;=> true

(= (:discard whole)
   (cm/choicemap :k1 (:discard first-part) :k2 (:discard second-part)))
;=> true

;; book: recursion
(def g
  (gen []
    (if (trace :go (d/bernoulli 0.2))
      (+ (splice :L g) (splice :R g) 1)
      1)))

(def g-runs
  [(cm/choicemap :go false)
   (cm/choicemap :go true [:L :go] false [:R :go] false)
   (cm/choicemap :go true [:L :go] true [:R :go] false
                 [:L :L :go] false [:L :R :go] false)])

(mapv #(let [{w :weight rv :retval} (gfi/assess g [] %)]
         [(array/exp w) rv])
      g-runs)
;=> [[0.8 1] [0.128 3] [0.02048 5]]

;; book: recursion-flat
(def g-flat
  (gen [i]
    (if (trace [i :go] (d/bernoulli 0.2))
      (+ (splice g-flat (* 2 i)) (splice g-flat (inc (* 2 i))) 1)
      1)))

(def g-flat-runs
  [(cm/choicemap [1 :go] false)
   (cm/choicemap [1 :go] true [2 :go] false [3 :go] false)
   (cm/choicemap [1 :go] true [2 :go] true [3 :go] false
                 [4 :go] false [5 :go] false)])

(mapv #(let [{w :weight rv :retval} (gfi/assess g-flat [1] %)]
         [(array/exp w) rv])
      g-flat-runs)
;=> [[0.8 1] [0.128 3] [0.02048 5]]

;; book: recursion-figure
;; figure: figures/031-namespaces.svg
(svg/tree (concat (map #(into ['g] %) (cm/addresses (g-runs 2)))
                  (map #(into ['g-flat] %) (cm/addresses (g-flat-runs 2))))
          {:root "two runs"})

;; book: source
(def program (second (:source burglary-model)))

[(first (:source burglary-model)) (first program)
 (take-nth 2 (second program))]
;=> [[] let [burglary disabled alarm]]

(nth program 2)
;=> (trace :calls (d/bernoulli (if alarm 0.7 0.05)))

;; book: toy-semantics
(defn unnest
  "A toy let of several bindings as nested lets of one binding each."
  [[_ bindings body]]
  (let [[x e & more] bindings]
    (list 'let [x e]
          (if (seq more) (unnest (list 'let (vec more) body)) body))))

(defn toy-addrs
  "Addrs⟦e⟧ (Fig. 2-1): the addresses of the random choices in the toy
   program e."
  [e]
  (cond (and (seq? e) (= 'trace (first e))) (conj (toy-addrs (nth e 2))
                                                  (second e))
        (coll? e) (reduce into #{} (map toy-addrs e))
        :else #{}))

(defn toy-value
  "Val⟦e⟧(env)(τ): the value of the toy program e, its variables bound in
   env, its random choices read from the dictionary tau."
  [e env tau]
  (cond (symbol? e) (env e)
        (not (seq? e)) e
        :else (case (first e)
                trace (tau (second e))
                let (let [[_ [x e1] body] (unnest e)]
                      (toy-value body (assoc env x (toy-value e1 env tau)) tau))
                if (let [[_ test then else] e]
                     (toy-value (if (toy-value test env tau) then else)
                                env tau)))))

(defn toy-dist
  "Dist⟦e⟧(env)(τ) (Fig. 2-1): the probability of the dictionary tau under
   the toy program e. A let or an if gives its first part tau's choices at
   that part's addresses, and its second part the rest of tau; printed?
   gives the second part tau's choices at its own addresses instead, as
   Fig. 2-1 is printed. Distributions are bernoulli's."
  [e env tau printed?]
  (let [parts (fn [e1 e2]
                [(select-keys tau (toy-addrs e1))
                 (if printed?
                   (select-keys tau (toy-addrs e2))
                   (apply dissoc tau (toy-addrs e1)))])]
    (if-not (seq? e)
      (if (empty? tau) 1.0 0.0)
      (case (first e)
        trace (let [[_ a [_ prob]] e]
                (if (= [a] (keys tau))
                  (bern (toy-value prob env tau) (tau a))
                  0.0))
        let (let [[_ [x e1] e2] (unnest e)
                  [tau1 tau2] (parts e1 e2)]
              (* (toy-dist e1 env tau1 printed?)
                 (toy-dist e2 (assoc env x (toy-value e1 env tau)) tau2
                           printed?)))
        if (let [[_ test then else] e
                 branch (if (toy-value test env tau) then else)
                 [tau1 tau2] (parts test branch)]
             (* (toy-dist test env tau1 printed?)
                (toy-dist branch env tau2 printed?)))))))

;; book: toy-check
(toy-addrs program)
;=> #{:burglary :disabled :alarm :calls}

(every? #(array/allclose? (toy-dist program {} % false) (p %)) (keys p))
;=> true

(every? #(= (toy-value program {} %)
            (:retval (gfi/assess burglary-model [] (cm/from-map %))))
        (support p))
;=> true

;; book: toy-printed
(def printed (into {} (map (juxt identity #(toy-dist program {} % true)))
                   (keys p)))

[(reduce + (vals printed)) (count (support printed)) (structured? printed)]
;=> [2.982 22 false]

(= (printed {:burglary false :disabled true :alarm false :calls false})
   (printed {:burglary false :alarm false :calls false}))
;=> true

;; book: workshop-1
(def maybe-traced
  (gen []
    (when (untraced (d/bernoulli 0.5))
      (trace :a (d/bernoulli 0.5)))))

;; book: solution-1
(defn assessed
  "assess's probability of the dictionary tau under maybe-traced with the
   key k, or :refused."
  [tau k]
  (try (array/item (array/exp (:weight (gfi/assess maybe-traced []
                                                   (cm/from-map tau) k))))
       (catch clojure.lang.ExceptionInfo _ :refused)))

[(mapv #(assessed {} (rng/key %)) [0 1])
 (mapv #(assessed {:a true} (rng/key %)) [0 1])]
;=> [[:refused 1.0] [0.5 :refused]] ; float64

;; book: solution-2
(mapv #(sound? p {% true}) [:burglary :disabled :alarm :calls])
;=> [true false false true]

(defn refusals
  "How many of n keys make generate refuse the burglary model's
   constraints sigma (a constraint unvisited)."
  [sigma n key]
  (count (filter #(try (gfi/generate burglary-model [] (cm/from-map sigma) %)
                       false
                       (catch clojure.lang.ExceptionInfo e
                         (boolean (:unvisited (ex-data e)))))
                 (rng/split key n))))

(within-5-se? (refusals {:disabled true} 2000 (rng/key 25)) 2000 0.99)
;=> true

;; book: solution-3
(defn gibbs-sweep
  "Regenerate :rain, then :sprinkler: each move's weight is 0, so each is
   accepted, and the sweep is a Gibbs sweep."
  [tr key]
  (let [[k1 k2] (rng/split key)]
    (-> tr
        (gfi/regenerate (sel/select :rain) k1) :trace
        (gfi/regenerate (sel/select :sprinkler) k2) :trace)))

(def gibbs-chain
  "4000 Gibbs sweeps from wet-t: whether each trace has rain, as 1 or 0."
  (mapv #(if (% :rain) 1.0 0.0)
        (reductions gibbs-sweep wet-t (rng/split (rng/key 26) 4000))))

(def rain-given-wet
  (/ (reduce + (map sprinkler-table (filter #(and (:rain %) (:wet %))
                                            (keys sprinkler-table))))
     (marginal sprinkler-table {:wet true})))

rain-given-wet
;=> 0.3539

(let [{:keys [mean mcse-mean]} (diag/summary gibbs-chain)]
  (< (abs (- mean rain-given-wet)) (* 5 mcse-mean)))
;=> true

;; book: solution-4
(def coins
  (gen [prob-a]
    (let [a (trace :a (d/bernoulli prob-a))
          b (if a (trace :b (d/bernoulli 0.6)) true)]
      (trace :c (d/bernoulli (if b 0.9 0.2))))))

(def coins-program (second (:source coins)))

(def coins-table
  (table coins [0.3] (zipmap [:a :b :c] (repeat [true false]))))

(every? #(array/allclose? (toy-dist coins-program '{prob-a 0.3} % false)
                          (coins-table %))
        (keys coins-table))
;=> true
