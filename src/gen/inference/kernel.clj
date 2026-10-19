(ns gen.inference.kernel
  "MCMC kernels and their combinators.

   A kernel is a function (kernel trace key) → {:trace trace' ...}: Gen.jl's
   (trace, metadata), as a map. The MH kernels add :accepted? and :log-alpha.
   A Kernel value can also carry its reversal, the kernel that undoes it in
   distribution (Gen.jl's reversal). MH-family kernels are reversible, so
   they are their own reversals. Combinators of kernels with reversals have
   reversals too: the reversal of a chain is the chain of the reversals, in
   reverse order.

     (chain k1 k2 k3)      k1, then k2, then k3
     (repeat n k)          k, n times
     (cycle n [k1 k2])     k1, k2, k1, … for n steps in all
     (mix [k1 k2] [w1 w2]) one of the kernels, chosen with probability ∝ w
     (checked k observations)  k, then check that it left the observations alone

   A combinator's result reports the accept decisions of the MH moves it
   made, in order, as :accepted (a vector of booleans) when they all report
   one.

   (run kernel trace opts) runs a chain and collects samples."
  (:refer-clojure :exclude [repeat cycle reversible?])
  (:require [gen.array :as array]
            [gen.choicemap :as cm]
            [gen.gfi :as gfi]
            [gen.inference.mh :as mh]
            [gen.random :as rng]))

(set! *warn-on-reflection* true)

(defrecord Kernel [f reverse-f]
  clojure.lang.IFn
  (invoke [_ trace] (f trace (gfi/next-key)))
  (invoke [_ trace key] (f trace key))
  (applyTo [this args] (clojure.lang.AFn/applyToHelper this args)))

(defn kernel
  "A kernel from (fn [trace key] → {:trace …}), with its reversal when given."
  ([f] (->Kernel f nil))
  ([f reverse-f] (->Kernel f reverse-f)))

(defn symmetric
  "A kernel that is its own reversal."
  [f]
  (->Kernel f f))

(defn reversible? [k] (and (instance? Kernel k) (some? (:reverse-f k))))

(defn reversal
  "The declared reversal of a kernel."
  [k]
  (if (reversible? k)
    (->Kernel (:reverse-f k) (:f k))
    (throw (ex-info "The kernel has no declared reversal" {:kernel k}))))

(defn- step-fn
  "The (fn [trace key]) of a kernel or a plain function."
  [k]
  (if (instance? Kernel k) (:f k) k))

(defn- with-reversals
  "A kernel built by combine from the step fns of ks, reversible when every
   k is: its reversal is combine over the reversals in reverse-order."
  [combine ks reverse-order]
  (->Kernel (combine (mapv step-fn ks))
            (when (every? reversible? ks)
              (combine (mapv (comp step-fn reversal) (reverse-order ks))))))

(defn- accepted-moves
  "The accept decisions a kernel result reports: one for an MH step, a
   combinator's sequence, or nil when it reports none."
  [r]
  (cond (contains? r :accepted) (:accepted r)
        (contains? r :accepted?) [(:accepted? r)]))

(defn- sequence-fn [fs]
  (fn [trace key]
    (let [[t moves] (reduce (fn [[t moves] [f k]]
                              (let [r (f t k)]
                                [(:trace r) (when moves (some->> (accepted-moves r) (into moves)))]))
                            [trace []]
                            (map vector fs (rng/split key (count fs))))]
      (cond-> {:trace t} moves (assoc :accepted moves)))))

(defn chain
  "The kernels applied in sequence."
  [& kernels]
  (with-reversals sequence-fn (vec kernels) reverse))

(defn repeat
  "The kernel applied n times."
  [n kernel]
  (apply chain (clojure.core/repeat n kernel)))

(defn cycle
  "The kernels applied in turn, n steps in all (one round by default)."
  ([kernels] (cycle (count kernels) kernels))
  ([n kernels] (apply chain (take n (clojure.core/cycle kernels)))))

(defn mix
  "One kernel chosen at random with probabilities proportional to weights.
   Its reversal mixes the reversals with the same weights."
  [kernels weights]
  (when-not (= (count kernels) (count weights))
    (throw (ex-info "mix needs one weight per kernel" {:kernels (count kernels) :weights (count weights)})))
  (when-not (and (every? #(and (number? %) (>= % 0)) weights) (some pos? weights))
    (throw (ex-info "mix needs non-negative weights, not all zero" {:weights weights})))
  (let [logits (array/log (array/array (mapv double weights)))]
    (with-reversals (fn [fs]
                      (fn [trace key]
                        (let [[k-choose k-step] (rng/split key)]
                          ((fs (array/item (rng/categorical k-choose logits))) trace k-step))))
                    (vec kernels)
                    identity)))

(defn- check-observations! [trace observations]
  (doseq [addr (cm/addresses observations)]
    (let [observed (cm/get-choice observations addr)
          actual (cm/get-choice (:choices trace) addr)]
      (when-not (and (some? actual) (array/equal? observed actual))
        (throw (ex-info (str "The kernel changed the observation at " addr)
                        {:path addr :observed observed :actual actual}))))))

(defn checked
  "The kernel, checking after each application that the trace still holds
   the observations (Gen.jl's check=true, observations=…)."
  [kernel observations]
  (let [check (fn [f] (fn [trace key]
                        (let [r (f trace key)]
                          (check-observations! (:trace r) observations)
                          r)))]
    (->Kernel (check (step-fn kernel))
              (when (reversible? kernel) (check (:reverse-f kernel))))))

;; ---------------------------------------------------------------------------
;; MH kernels
;; ---------------------------------------------------------------------------

(defn mh
  "The MH kernel of a selection, or of a proposal and its arguments."
  ([selection] (symmetric (fn [trace key] (mh/mh trace selection key))))
  ([proposal proposal-args] (symmetric (fn [trace key] (mh/mh trace proposal proposal-args key)))))

(defn involutive-mh
  "The involutive MCMC kernel of a proposal, its arguments and an
   involution."
  [proposal proposal-args involution]
  (symmetric (fn [trace key] (mh/involutive-mh trace proposal proposal-args involution key))))

;; ---------------------------------------------------------------------------
;; Running a chain
;; ---------------------------------------------------------------------------

(defn run
  "Apply the kernel repeatedly from trace and collect samples. opts:
   :samples (required), :burn (steps discarded first, 0), :thin (steps per
   sample, 1), :key. Returns {:traces [...] :acceptance-rate r}: the rate
   over every MH move of every step (a combinator's steps make several), or
   nil when the kernel does not report its accept decisions."
  [kernel trace {:keys [samples burn thin key] :or {burn 0 thin 1}}]
  (when-not (and (nat-int? samples) (nat-int? burn) (pos-int? thin))
    (throw (ex-info "run needs :samples ≥ 0, :burn ≥ 0 and :thin ≥ 1" {:samples samples :burn burn :thin thin})))
  (let [f (step-fn kernel)
        steps (+ burn (* samples thin))
        keys (if (pos? steps) (rng/split (or key (gfi/next-key)) steps) [])]
    (loop [i 0 t trace traces (transient []) accepted 0 moves 0 reported? true]
      (if (< i steps)
        (let [r (f t (keys i))
              i' (inc i)
              ;; one iteration's graph at a time on a lazy backend
              t' (gfi/eval-trace! (:trace r))
              keep? (and (> i' burn) (zero? (mod (- i' burn) thin)))
              ms (accepted-moves r)]
          (recur i' t' (if keep? (conj! traces t') traces)
                 (+ accepted (count (filter true? ms)))
                 (+ moves (count ms))
                 (and reported? (some? ms))))
        {:traces (persistent! traces)
         :acceptance-rate (when (and reported? (pos? moves)) (/ (double accepted) moves))}))))
