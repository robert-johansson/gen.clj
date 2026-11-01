(ns gen.codegen.tasks
  "The codegen bake-off's tasks (genclj-ihc4): ten models of the book, each
   a description and data a model is written from, easiest first.

   A task is a map:
     :id :title         its name
     :description       the situation, as someone who needs the model would
                        put it: what is known, what is not
     :interface         the model's arguments and the addresses of the data
     :args-text :args   the arguments, as the prompt shows them and as values
     :observations      the data, a choice map
     :answer            the reference model, as the code of an ideal reply:
                        the prelude's names (gen.codegen.candidate), the model
                        defined as model
     :source            where the book has it
     :tolerance         nats: how far below the reference's log marginal
                        likelihood a model may fall and still explain the data
     :particles         the importance sampler's, when not the grader's 20000:
                        200000 for the sine and the outliers, whose posteriors
                        are far from their priors (at 20000 the sine's estimate
                        is 6 nats low); the changepoint's estimates spread at
                        any affordable number, so it keeps 20000

   The data are the book's, but the sine task's (simulated from the
   reference with key 22, whose period and amplitude are 5.5 and 1.2) and
   the changepoint task's readings (the book's ys-complex, computed once on
   the JVM backend, at three decimals)."
  (:require [gen.array :as array]
            [gen.choicemap :as cm]))

(defn- observed
  "The choice map of ys, ys[i] at (address i)."
  [address ys]
  (reduce (fn [m [i y]] (cm/set-choice m (address i) y)) cm/EMPTY (map-indexed vector ys)))

(defn- at-y [i] [:y i])
(defn- at-t [t] [t :y])

(def ^:private xs-11 [-5.0 -4.0 -3.0 -2.0 -1.0 0.0 1.0 2.0 3.0 4.0 5.0])
(def ^:private xs-20 (mapv #(+ -5.0 (* 10.0 (/ % 19.0))) (range 20)))
(def ^:private xs-50 (mapv #(+ -5.0 (* 10.0 (/ % 49.0))) (range 50)))
(def ^:private controls (into (vec (repeat 8 0.3)) (repeat 8 -0.3)))

(def default-tolerance 5.0)

(def tasks
  [{:id :line
    :title "A straight line"
    :source "book/src/book/gallery.clj, tight-model"
    :description "Ten measurements y, taken at positions x, lie close to a straight line. The line's slope
and intercept are unknown; each is probably within about ±10. The instrument's noise is known: normal, with
standard deviation 0.3."
    :interface "The model takes one argument, xs, the positions. The measurement at position i (from 0) is
observed at the address [:y i]."
    :args-text "[[0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5]]"
    :args [[0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5]]
    :observations (observed at-y [0.62 1.48 2.41 3.55 4.37 5.61 6.49 7.53 8.42 9.57])
    :answer "(def model
  (gen [xs]
    (let [slope (trace :slope (d/normal 0 5))
          intercept (trace :intercept (d/normal 0 5))]
      (doseq [[i x] (map-indexed vector xs)]
        (trace [:y i] (d/normal (+ (* slope x) intercept) 0.3)))
      [slope intercept])))"}

   {:id :spread
    :title "A quantity measured repeatedly"
    :source "book/src/book/gallery.clj, spread-model (vi-model)"
    :description "Eight repeated measurements of one quantity. Its true value is unknown, probably within
about ±10. The measurements' spread is unknown too: their standard deviation is probably between about 0.2
and 5."
    :interface "The model takes one argument, n, the number of measurements. Measurement i (from 0) is
observed at the address [:y i]."
    :args-text "[8]"
    :args [8]
    :observations (observed at-y [2.8 3.1 2.9 3.3 3.0 2.7 3.2 8.5])
    :answer "(def model
  (gen [n]
    (let [mu (trace :mu (d/normal 0 5))
          log-sigma (trace :log-sigma (d/normal 0 1))]
      (dotimes [i n]
        (trace [:y i] (d/normal mu (array/exp log-sigma))))
      mu)))"}

   {:id :sine
    :title "A sine wave"
    :particles 200000
    :source "book/src/book/modeling.clj, sine-model"
    :description "Eleven readings of a periodic signal at evenly spaced positions. The signal is a sine wave
of unknown period, amplitude and phase. The period and the amplitude are positive, typically around 1 but
possibly a few times that; the phase is anywhere in a cycle. The readings' noise is known: normal, with
standard deviation 0.1."
    :interface "The model takes one argument, xs, the positions. The reading at position i (from 0) is
observed at the address [:y i]."
    :args-text "[[-5.0 -4.0 -3.0 -2.0 -1.0 0.0 1.0 2.0 3.0 4.0 5.0]]"
    :args [xs-11]
    :observations (observed at-y [1.059 -0.186 -1.193 -0.841 0.344 1.143 0.498 -0.756 -1.047 -0.034 1.091])
    :answer "(defn wave [period amplitude phase]
  (fn [x] (* amplitude (array/sin (+ (/ (* 2 Math/PI x) period) phase)))))

(def model
  (gen [xs]
    (let [period (trace :period (d/gamma 1 1))
          amplitude (trace :amplitude (d/gamma 1 1))
          phase (trace :phase (d/uniform 0 (* 2 Math/PI)))
          y (wave period amplitude phase)]
      (doseq [[i x] (map-indexed vector xs)]
        (trace [:y i] (d/normal (y x) 0.1)))
      y)))"}

   {:id :walk
    :title "A random walk, read by a sensor"
    :source "book/src/book/gallery.clj, walk (walk-model)"
    :description "A hidden position starts at 0 and takes six random steps, each normal with standard
deviation 0.5. After each step a sensor reads the position, with normal noise of standard deviation 1."
    :interface "The model takes two arguments: n, the number of steps, and z0, the starting position. The
reading after step t (from 0) is observed at the address [t :y]."
    :args-text "[6 0.0]"
    :args [6 0.0]
    :observations (observed at-t [0.5 1.2 1.8 2.5 1.5 0.9])
    :answer "(def step
  (gen [t z]
    (let [z' (trace :z (d/normal z 0.5))]
      (trace :y (d/normal z' 1.0))
      z')))

(def model (comb/unfold step))"}

   {:id :hmm
    :title "A two-state hidden Markov model"
    :source "book/src/book/composition.clj, hmm (hmm-model)"
    :description "A system switches between two hidden states, 0 and 1, once at each of six time steps.
From state 0 it stays with probability 0.9; from state 1, with probability 0.8. Before the first step the state
is 0. After each step we read a value: normal with mean -1 in state 0 and 2 in state 1, standard deviation 1."
    :interface "The model takes four arguments: n, the number of steps; the state before the first step; the
transition matrix (row: the current state, column: the next); the two states' means. The reading after step t
(from 0) is observed at the address [t :y]."
    :args-text "[6 0 (array/array [[0.9 0.1] [0.2 0.8]]) (array/array [-1.0 2.0])]"
    :args [6 0 (array/array [[0.9 0.1] [0.2 0.8]]) (array/array [-1.0 2.0])]
    :observations (observed at-t [-0.6 -1.4 1.7 2.5 0.4 2.2])
    :answer "(def step
  (gen [t state transition means]
    (let [state' (trace :z (d/categorical (array/take transition state)))]
      (trace :y (d/normal (array/take means state') 1.0))
      state')))

(def model (comb/unfold step))"}

   {:id :drift
    :title "A state driven by known controls"
    :source "book/src/book/gallery.clj, drift (scan-model)"
    :description "A hidden state starts at 0. At each of 16 steps it moves by a known control input plus
normal noise of standard deviation 0.5; after each move a sensor reads it, with normal noise of standard
deviation 1."
    :interface "The model takes two arguments: z0, the starting state, and controls, the 16 control inputs.
The reading after step t (from 0) is observed at the address [t :y]."
    :args-text "[0.0 [0.3 0.3 0.3 0.3 0.3 0.3 0.3 0.3 -0.3 -0.3 -0.3 -0.3 -0.3 -0.3 -0.3 -0.3]]"
    :args [0.0 controls]
    :observations (observed at-t [2.94 2.53 2.18 3.9 3.69 3.48 3.76 5.75 4.95 5.68 1.43 1.35 0.98 3.81 1.4 -0.75])
    :answer "(def step
  (gen [z u]
    (let [z' (trace :z (d/normal (+ z u) 0.5))]
      (trace :y (d/normal z' 1.0))
      [z' z'])))

(def model (comb/scan step))"}

   {:id :line-or-quadratic
    :title "A line or a quadratic"
    :source "book/src/book/gallery.clj, linear and quadratic (curves), chosen by a coin"
    :description "Five points follow either a straight line or a quadratic curve; a priori each is as likely
as the other. The coefficients are unknown: a line's within about ±10, a quadratic's within about ±4. The
points' noise is known: normal, with standard deviation 0.5."
    :interface "The model takes one argument, xs, the positions. The point at position i (from 0) is
observed at the address [:y i]."
    :args-text "[[1.0 2.0 3.0 4.0 5.0]]"
    :args [[1.0 2.0 3.0 4.0 5.0]]
    :observations (observed at-y [1.1 3.9 9.2 15.8 25.3])
    :answer "(def noisy-curve
  (gen [f xs]
    (doseq [[i x] (map-indexed vector xs)]
      (trace [:y i] (d/normal (f x) 0.5)))))

(def linear
  (gen [xs]
    (let [b (trace :b (d/normal 0 5))
          m (trace :m (d/normal 0 5))]
      (splice noisy-curve (fn [x] (+ b (* m x))) xs))))

(def quadratic
  (gen [xs]
    (let [c (trace :c (d/normal 0 2))
          b (trace :b (d/normal 0 2))
          a (trace :a (d/normal 0 2))]
      (splice noisy-curve (fn [x] (+ c (* b x) (* a x x))) xs))))

(def model
  (gen [xs]
    (if (trace :quadratic? (d/bernoulli 0.5))
      (splice quadratic xs)
      (splice linear xs))))"}

   {:id :line-or-sine
    :title "A line or a sine wave, noise unknown"
    :source "book/src/book/modeling.clj, combined-model"
    :description "Eleven readings of a curve that is either a straight line or a sine wave; a priori each is
as likely as the other. A line's slope is of order 1 and its intercept of order 2. A sine's period is a few
units (around 5), its amplitude around 1, its phase anywhere in a cycle. The noise is normal, of unknown
standard deviation, typically around 1."
    :interface "The model takes one argument, xs, the positions. The reading at position i (from 0) is
observed at the address [:y i]."
    :args-text "[[-5.0 -4.0 -3.0 -2.0 -1.0 0.0 1.0 2.0 3.0 4.0 5.0]]"
    :args [xs-11]
    :observations (observed at-y [2.89 2.22 -0.612 -0.522 -2.65 -0.133 2.70 2.77 0.425 -2.11 -2.76])
    :answer "(defn wave [period amplitude phase]
  (fn [x] (* amplitude (array/sin (+ (/ (* 2 Math/PI x) period) phase)))))

(def line
  (gen [xs]
    (let [slope (trace :slope (d/normal 0 1))
          intercept (trace :intercept (d/normal 0 2))
          noise (trace :noise (d/gamma 1 1))]
      (doseq [[i x] (map-indexed vector xs)]
        (trace [:y i] (d/normal (+ (* slope x) intercept) noise))))))

(def sine
  (gen [xs]
    (let [period (trace :period (d/gamma 5 1))
          amplitude (trace :amplitude (d/gamma 1 1))
          phase (trace :phase (d/uniform 0 (* 2 Math/PI)))
          noise (trace :noise (d/gamma 1 1))
          y (wave period amplitude phase)]
      (doseq [[i x] (map-indexed vector xs)]
        (trace [:y i] (d/normal (y x) noise))))))

(def model
  (gen [xs]
    (if (trace :is-line (d/bernoulli 0.5))
      (splice line xs)
      (splice sine xs))))"}

   {:id :outliers
    :title "A straight line with outliers"
    :particles 200000
    :source "book/src/book/mcmc_map.clj, regression-with-outliers"
    :description "Twenty measurements y, taken at positions x, follow a straight line with normal noise,
except that some are outliers: corrupted readings that ignore the line and scatter widely around 0 (standard
deviation about 10). The line's slope and intercept are unknown, probably within about ±5; the noise's standard
deviation is unknown, typically around 1; the fraction of outliers is unknown. Each point is an outlier or not,
independently."
    :interface "The model takes one argument, xs, the positions. The measurement at position i (from 0) is
observed at the address [:data i :y]."
    :args-text (pr-str [xs-20])
    :args [xs-20]
    :observations (observed (fn [i] [:data i :y])
                            [6.73 -10.75 1.82 5.11 5.09 4.31 3.80 2.39 2.40 2.67
                             2.07 -2.96 0.65 1.13 -0.48 -1.20 -10.62 -1.48 1.68 -2.92])
    :answer "(def model
  (gen [xs]
    (let [slope (trace :slope (d/normal 0 2))
          intercept (trace :intercept (d/normal 0 2))
          noise (trace :noise (d/gamma 1 1))
          prob-outlier (trace :prob-outlier (d/uniform 0 1))]
      (mapv (fn [i x]
              (let [outlier? (trace [:data i :outlier?] (d/bernoulli prob-outlier))
                    line (+ (* slope x) intercept)]
                (trace [:data i :y]
                       (d/normal (array/where outlier? 0 line)
                                 (array/where outlier? 10 noise)))))
            (range) xs))))"}

   {:id :changepoint
    :title "A piecewise-constant signal"
    :source "book/src/book/modeling.clj, changepoint-model (generate-segments, value-at)"
    :description "Fifty readings, at evenly spaced positions from -5 to 5, of a quantity that is piecewise
constant: it holds a level over an interval, then jumps to another at a change point. The number of segments
is unknown (often one or two, sometimes more), the change points are anywhere, each level is of order 1, and
the readings' noise is normal, of unknown standard deviation."
    :interface "The model takes one argument, xs, the positions. The reading at position i (from 0) is
observed at the address [:y i]."
    :particles 20000
    :args-text (pr-str [xs-50])
    :args [xs-50]
    :observations (observed at-y [0.026 -0.17 0.001 0.075 0.055 -0.161 -0.098 0.29 -0.037 0.076 1.833 2.124
                                  2.077 2.132 1.867 2.049 2.206 2.296 1.923 2.07 2.061 1.941 2.185 1.885
                                  2.078 2.052 1.866 1.801 1.986 2.205 2.117 2.073 2.113 2.147 1.878 2.018
                                  1.967 2.1 2.061 2.054 0.023 -0.069 0.019 -0.028 0.045 -0.001 0.056
                                  -0.057 0.005 0.02])
    :answer "(def segments
  (gen [l u]
    (if (trace :isleaf (d/bernoulli 0.7))
      {:interval [l u] :value (trace :value (d/normal 0 1))}
      (let [frac (trace :frac (d/beta 2 2))
            mid (+ l (* (- u l) frac))]
        {:interval [l u]
         :left (splice :left segments l mid)
         :right (splice :right segments mid u)}))))

(defn value-at [node x]
  (if-let [left (:left node)]
    (array/where (<= x (second (:interval left)))
                 (value-at left x)
                 (value-at (:right node) x))
    (:value node)))

(def model
  (gen [xs]
    (let [node (splice :tree segments (apply min xs) (apply max xs))
          noise (trace :noise (d/gamma 0.5 0.5))]
      (doseq [[i x] (map-indexed vector xs)]
        (trace [:y i] (d/normal (value-at node x) noise)))
      node)))"}])

(def by-id (into {} (map (juxt :id identity)) tasks))

(defn tolerance [task] (or (:tolerance task) default-tolerance))
