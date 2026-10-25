;; The code of book/034-design.md. Each region, from a `;; book:` line to
;; the next, is included there by name (gen.book: `clojure -M:test -m
;; gen.book sync` writes the Markdown's copies; gen.book-test checks them).

;; book: ns
(ns book.design
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [gen.array :as array :refer [+ - * /]]
            [gen.batched :as batched]
            [gen.book.svg :as svg]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :as dyn :refer [gen]]
            [gen.dynamic.static :as static]
            [gen.gfi :as gfi]
            [gen.gradients :as grads]
            [gen.inference.conjugate :as conj]
            [gen.laws :as laws]
            [gen.random :as rng]
            [gen.selection :as sel]))

;; book: model
(def noisy-mean
  (gen []
    (let [mu (trace :mu (d/normal 0 2))
          y0 (trace [:y 0] (d/normal mu 1))
          y1 (trace [:y 1] (d/normal mu 1))
          y2 (trace [:y 2] (d/normal mu 1))]
      mu)))

(def tr (gfi/simulate noisy-mean [] (rng/key 1)))

[(tr :mu) (:score tr)]
;=> [1.4318 -5.8904] ; float64

;; book: readings
(let [{g :trace w :weight} (gfi/generate noisy-mean [] (:choices tr)
                                         (rng/key 2))
      {a :weight} (gfi/assess noisy-mean [] (:choices tr) (rng/key 3))]
  [(= (:choices g) (:choices tr))
   (array/allclose? w (:score tr))
   (array/allclose? a (:score tr))])
;=> [true true true]

;; book: closed-form
(defn log-joint
  "log p(mu, ys) when mu is normal around 0 with standard deviation
   prior-sd (noisy-mean's is 2) and each y normal around mu with standard
   deviation 1. mu may be an array of one value per particle."
  [prior-sd mu ys]
  (reduce + (dc/log-prob (d/normal 0 prior-sd) mu)
          (map #(dc/log-prob (d/normal mu 1) %) ys)))

(defn d-log-joint
  "The derivative of (log-joint 2 mu ys) in mu: -mu/4 + the sum of
   (y - mu)."
  [mu ys]
  (reduce + (- (/ mu 4.0)) (map #(- % mu) ys)))

(defn ys-of
  "The three observations of a trace of noisy-mean."
  [t]
  (mapv #(t [:y %]) (range 3)))

(array/allclose? (:score tr) (log-joint 2 (tr :mu) (ys-of tr)))
;=> true

;; book: gradient
(let [{:keys [gradients]} (grads/choice-gradients tr (sel/select :mu))]
  (array/allclose? (cm/get-choice gradients :mu)
                   (d-log-joint (tr :mu) (ys-of tr))))
;=> true

;; book: batched
(def ys [1.2 0.4 2.1])

(def observations
  (cm/choicemap [:y 0] (ys 0) [:y 1] (ys 1) [:y 2] (ys 2)))

(def particles
  (batched/generate noisy-mean [] observations 1000 (rng/key 4)))

(let [ps (:particles particles)
      mus (batched/choice ps :mu)
      {:keys [gradients]} (grads/choice-gradients ps (sel/select :mu))]
  [(batched/batched? ps)
   (array/shape mus)
   (array/allclose? (batched/scores ps) (log-joint 2 mus ys))
   (array/allclose? (:weights particles)
                    (- (log-joint 2 mus ys) (dc/log-prob (d/normal 0 2) mus)))
   (array/allclose? (cm/get-choice gradients :mu) (d-log-joint mus ys))])
;=> [true [1000] true true true]

;; book: static
(def noisy-mean-static
  (gen ^:static []
    (let [mu (trace :mu (d/normal 0 2))
          y0 (trace [:y 0] (d/normal mu 1))
          y1 (trace [:y 1] (d/normal mu 1))
          y2 (trace [:y 2] (d/normal mu 1))]
      mu)))

(let [t (gfi/simulate noisy-mean-static [] (rng/key 1))]
  [(= (:choices t) (:choices tr)) (array/allclose? (:score t) (:score tr))])
;=> [true true]

;; book: dispatcher
(defn closed-form-assess
  "A dispatcher for noisy-mean: assess's weight from log-joint, with the
   prior's standard deviation given, in place of the handler's run; every
   other operation is left to the handler (nil)."
  [prior-sd]
  (fn [_gf op _args {:keys [constraints]}]
    (when (= op :assess)
      (let [mu (cm/get-choice constraints :mu)
            ys (mapv #(cm/get-choice constraints [:y %]) (range 3))]
        {:weight (log-joint prior-sd mu ys) :retval mu}))))

(def fast-mean (dyn/with-dispatch noisy-mean (closed-form-assess 2)))

(array/allclose? (:weight (gfi/assess fast-mean [] (:choices tr) (rng/key 5)))
                 (:score tr))
;=> true

;; book: exact-marginal
(defn log-marginal
  "log p(ys) for noisy-mean: the ys are normal with mean 0 and covariance
   I + 4J (J all ones), whose inverse is I - 4J/(1 + 4n) and determinant
   1 + 4n."
  [ys]
  (let [n (count ys)
        s (reduce + ys)
        ss (reduce + (map #(* % %) ys))
        c (+ 1.0 (* 4.0 n))]
    (* -0.5 (+ (* n (Math/log (* 2.0 Math/PI))) (Math/log c)
               (- ss (* (/ 4.0 c) s s))))))

(log-marginal ys)
;=> -4.93814

;; book: conjugate
(def conjugate-mean
  (conj/with-conjugate-proposal noisy-mean :mu [[:y 0] [:y 1] [:y 2]]))

(let [[r1 r2] (map #(gfi/generate conjugate-mean [] observations (rng/key %))
                   [5 6])]
  [(array/allclose? (:weight r1) (log-marginal ys))
   (array/allclose? (:weight r2) (log-marginal ys))
   (array/equal? ((:trace r1) :mu) ((:trace r2) :mu))])
;=> [true true false]

;; book: plain-weights
(mapv #(:weight (gfi/generate noisy-mean [] observations (rng/key %))) [5 6])
;=> [-16.461 -11.222] ; float64

;; book: sources
(defn source-files
  "The Clojure source files under dir."
  [dir]
  (filter #(str/ends-with? (.getName ^java.io.File %) ".clj")
          (file-seq (io/file dir))))

(defn ns-requires
  "[name requires] of the source file f's ns form: requires, the gen.clj
   namespaces its :require clauses name."
  [f]
  (let [form (with-open [r (java.io.PushbackReader. (io/reader f))]
               (read r))
        [_ name & clauses] form]
    [name (set (for [clause clauses
                     :when (and (seq? clause) (= :require (first clause)))
                     spec (rest clause)
                     :let [lib (if (sequential? spec) (first spec) spec)]
                     :when (str/starts-with? (str lib) "gen.")]
                 lib))]))

(def requires
  "Every namespace of gen.clj's src/, with the gen.clj namespaces it
   requires."
  (into (sorted-map) (map ns-requires) (source-files "src/gen")))

(defn closure
  "The namespaces root loads: itself, what it requires, and so on."
  [root]
  (loop [seen #{} todo [root]]
    (if-let [n (peek todo)]
      (recur (conj seen n) (into (pop todo) (remove seen (requires n))))
      seen)))

(defn under?
  "Is the namespace named s the namespace root, or one inside it?"
  [s root]
  (or (= s root) (str/starts-with? s (str root "."))))

(defn layer
  "The layer of a gen.clj namespace, as CLAUDE.md's table gives it."
  [ns-name]
  (let [s (str ns-name)
        in? (fn [& roots] (some #(under? s %) roots))]
    (cond (in? "gen.array" "gen.backend" "gen.random") :L0
          (in? "gen.choicemap" "gen.selection" "gen.trace" "gen.diff") :L1
          (in? "gen.gfi" "gen.serialize" "gen.dynamic" "gen.batched"
               "gen.partition") :L2
          (in? "gen.dist") :L3
          (in? "gen.combinators" "gen.custom" "gen.state-space") :L4
          (in? "gen.inference" "gen.gradients" "gen.learning") :L5
          (in? "gen.llm") :llm)))

;; book: backend-rule
(set (for [[n libs] requires
           :when (some #(str/starts-with? (str %) "gen.backend") libs)]
       n))
;=> #{gen.array}

;; book: closure
(sort (distinct (map layer (closure 'gen.dynamic))))
;=> (:L0 :L1 :L2 :L3)

(keys (:deps (edn/read-string (slurp "deps.edn"))))
;=> (org.clojure/clojure)

;; book: layers-figure
;; figure: figures/034-layers.svg
(svg/dag
 [{:id :llm-gf :label "LLM generative functions (gen.llm.token)" :shape :box}
  {:id :L5 :label "L5 inference (gen.inference)" :shape :box}
  {:id :llm-rt :label "LLM runtime (gen.llm.lm)" :shape :box}
  {:id :L4 :label "L4 combinators (gen.combinators)" :shape :box}
  {:id :llama :label "llama.cpp"}
  {:id :L3 :label "L3 distributions (gen.dist)" :shape :box}
  {:id :L2 :label "L2 GFI (gen.gfi, gen.dynamic, gen.batched)" :shape :box}
  {:id :L1 :label "L1 data (gen.choicemap, gen.trace)" :shape :box}
  {:id :L0 :label "L0 arrays (gen.array, gen.random)" :shape :box}
  {:id :jvm :label "JVM backend"}
  {:id :mlx :label "MLX backend"}]
 [[:llm-gf :L5] [:llm-gf :llm-rt] [:llm-rt :llama] [:L5 :L4] [:L4 :L3]
  [:L3 :L2] [:L2 :L1] [:L1 :L0] [:L0 :jvm] [:L0 :mlx]])

;; book: update
(def moved (gfi/update tr (cm/choicemap :mu 0.5) (rng/key 8)))

(sort (keys moved))
;=> (:discard :retdiff :trace :weight)

[(tr :mu) ((:trace moved) :mu) (cm/get-choice (:discard moved) :mu)]
;=> [1.4318 0.5 1.4318] ; float64

(array/allclose? (:weight moved) (- (:score (:trace moved)) (:score tr)))
;=> true

;; book: project
(defn counted
  "gf with every site transition counting its calls in the atom runs."
  [gf runs]
  (dyn/with-transitions gf
    (into {} (for [[op transition] dyn/site-transitions]
               [op (fn [state path dist key]
                     (swap! runs inc)
                     (transition state path dist key))]))))

(let [runs (atom 0)
      t (gfi/simulate (counted noisy-mean runs) [] (rng/key 1))
      after-simulate @runs
      prior (gfi/project t (sel/select :mu))]
  [after-simulate
   (- @runs after-simulate)
   (array/allclose? prior (dc/log-prob (d/normal 0 2) (t :mu)))])
;=> [4 0 true]

;; book: lint
(static/could-be-static noisy-mean)
;=> {:static? true}

;; book: jax-keys
(mapv rng/key-data (rng/split (rng/key 0)))
;=> [[1797259609 2579123966] [928981903 3453687069]]

;; book: seed-free
(def implicit-seed-before (gfi/implicit-seed))

(let [run (fn [seed]
            (gfi/seed! seed)
            [(gfi/simulate noisy-mean [] (rng/key 9))
             (gfi/simulate noisy-mean [])])
      [keyed-1 keyless-1] (run 1)
      [keyed-2 keyless-2] (run 2)]
  (gfi/seed! implicit-seed-before)
  [(= (:choices keyed-1) (:choices keyed-2))
   (= (:choices keyless-1) (:choices keyless-2))])
;=> [true false]

;; book: plain-call
(def noise
  (gen []
    (trace :e (d/normal 0 1))))

(def with-noise
  (gen []
    (+ (trace :mu (d/normal 0 1)) (noise))))

(array/equal? (:retval (gfi/simulate with-noise [] (rng/key 10)))
              (:retval (gfi/simulate with-noise [] (rng/key 10))))
;=> true

;; book: error-map
;; error: generate: the constraints must be a choice map
#_(gfi/generate noisy-mean [] {:mu 1.0} (rng/key 11))

;; book: error-bernoulli
;; error: bernoulli: values are booleans
#_(def coin
  (gen []
    (trace :heads (d/bernoulli 0.3))))

#_(gfi/generate coin [] (cm/choicemap :heads 1) (rng/key 11))

;; book: error-lookup
;; error: trace: no choice or call at [:nu]
#_(tr :nu)

;; book: lenient
[(tr :nu nil) (cm/has-value? (:choices tr) :nu)]
;=> [nil false]

;; book: casts
(array/->clj (array/astype (array/array [1e20 -1e20 ##NaN 2.7 -2.7]) :int32))
;=> [2147483647 -2147483648 0 2 -2]

;; book: law
(defn dispatch-law
  "The law dispatch-matches-handler over 30 cases of gf, a dispatched
   noisy-mean, with a fixed seed: test.check's result."
  [gf]
  (laws/quick-check 30
              (laws/dispatch-matches-handler (laws/cases [["mean" gf [] []]]))
              {:seed 1}))

(select-keys (dispatch-law fast-mean) [:pass? :num-tests :vacuous])
;=> {:pass? true :num-tests 30 :vacuous 0}

;; book: law-bug
(def wrong-prior (dispatch-law (dyn/with-dispatch noisy-mean
                                 (closed-form-assess 4))))

[(:pass? wrong-prior) (first (:check (:result-data wrong-prior)))]
;=> [false close?]

;; book: fallback
(def two-paths
  (gen []
    (if (trace :b (d/bernoulli 0.5))
      (trace :x (d/normal 0 1))
      (trace :z (d/normal 0 10)))))

(def mixed (batched/simulate two-paths [] 100 (rng/key 12)))

[(batched/batched? mixed) (:message (batched/fallback-reason mixed))]
;=> [false "Cannot branch on a batched value: it differs across particles"]

(count (batched/traces mixed))
;=> 100

;; book: solution-1
(def noisy-mean-untraced
  (gen []
    (let [z (untraced (d/normal 0 1))
          mu (trace :mu (d/normal 0 2))
          y0 (trace [:y 0] (d/normal mu 1))
          y1 (trace [:y 1] (d/normal mu 1))
          y2 (trace [:y 2] (d/normal mu 1))]
      mu)))

(let [t (gfi/simulate noisy-mean-untraced [] (rng/key 1))]
  [(= (:choices t) (:choices tr)) (array/allclose? (:score t) (:score tr))])
;=> [true true]

;; book: solution-2-error
;; error: with-conjugate-proposal: the choice at [:y 2] depends on the latent
#_(gfi/generate (conj/with-conjugate-proposal noisy-mean :mu [[:y 0] [:y 1]])
              [] observations (rng/key 13))

;; book: solution-2
(def two-children
  (conj/with-conjugate-proposal noisy-mean :mu [[:y 0] [:y 1]]
                                {:check-exact? false}))

(def two-children-weights
  (mapv #(array/item (:weight (gfi/generate two-children [] observations %)))
        (rng/split (rng/key 14) 10000)))

(defn mean-and-se
  "The mean of the host numbers xs and its standard error."
  [xs]
  (let [n (count xs)
        mean (/ (reduce + xs) n)
        variance (/ (reduce + (map #(* (- % mean) (- % mean)) xs)) (dec n))]
    [mean (Math/sqrt (/ variance n))]))

(let [[mean se] (mean-and-se (mapv #(Math/exp %) two-children-weights))]
  ;; the mean of exp(weight) is p(ys), within five standard errors
  [(apply = two-children-weights)
   (< (abs (- mean (Math/exp (log-marginal ys)))) (* 5 se))])
;=> [false true]

;; book: solution-3
(sort (distinct (map layer (closure 'gen.inference.mh))))
;=> (:L0 :L1 :L2 :L5)

(set (for [n (closure 'gen.inference.mh)
           :when (some #(str/starts-with? (str %) "gen.backend") (requires n))]
       n))
;=> #{gen.array}

;; book: solution-4
(def back (gfi/update (:trace moved) (:discard moved) (rng/key 15)))

[(= (:choices (:trace back)) (:choices tr))
 (array/allclose? (:weight back) (- (:weight moved)))
 (cm/get-choice (:discard back) :mu)]
;=> [true true 0.5]
