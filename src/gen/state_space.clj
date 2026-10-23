(ns gen.state-space
  "State-space models with their latent states summed out: generative
   functions whose choices are the observations and whose densities are the
   exact marginals, by the Kalman filter (linear-Gaussian models) and the
   forward algorithm (hidden Markov models).

     kalman                    a generative function on (n prior params)
     hmm                       a generative function on (n prior params)
     (sample-states tr)        the latent states, drawn given the observations
     (sample-states tr key)

   Each is an Unfold (gen.combinators) over a filter step whose state is the
   belief about the latent state. Step t's one choice, at address t, is the
   observation y_t, of density p(y_t | y_0 … y_t-1), and its return value
   is the filtered belief, p(x_t | y_0 … y_t). So a trace's score is
   log p(y_0 … y_n-1), the states integrated out exactly: assess's weight,
   and generate's with every observation given, is the marginal likelihood;
   and an update to more steps adds the new observations' log predictive
   densities without revisiting the earlier steps (an Unfold's incremental
   update, its retdiff a vector-diff). An observation left open is drawn
   from its predictive distribution, so these are forward-sampling
   generative functions (every GFI law holds). The return value is the
   vector of the n filtered beliefs.

   kalman, the linear-Gaussian state-space model:

     x_0 ~ N(m, P)                    prior {:mean m :cov P}
     x_t = F x_t-1 + b + w_t,  w_t ~ N(0, Q)      (t ≥ 1)
     y_t = H x_t + c + v_t,    v_t ~ N(0, R)

   with params {:transition F :transition-cov Q :emission H :emission-cov R},
   and :transition-offset b and :emission-offset c, 0 when not given. The
   state is a vector [d], or a number (then F, Q and P are numbers too);
   R's shape gives the observations': a number for scalar observations, a
   [k k] matrix for vectors [k]. H is [k d] (or any shape of k·d elements,
   a vector [d] for scalar observations). A belief is {:mean :cov} in the
   prior's shapes.

   hmm, the hidden Markov model with K states:

     x_0 ~ categorical(prior)                     prior: probabilities [K]
     x_t ~ categorical(A[x_t-1])                  (t ≥ 1)
     y_t ~ e's distribution for state x_t

   with params {:transition A :emission e}: A is [K K], row i the next
   state's probabilities from state i; e is a distribution batched over the
   states on its leading axis, such as (categorical B) for a [K M] matrix
   B, (normal mus sigmas) for vectors of K, or (mvnormal mus covs). A belief
   is the vector of state probabilities. An impossible observation gives
   its step -Inf and leaves the belief at its prediction.

   params may also be a function of the step t that returns the map, for
   parameters that change with time (a regression whose design changes
   with t, say); a vector of maps, one per step, is one.

   sample-states draws the states from their posterior given a trace's
   observations (forward filtering, backward sampling): a function of the
   trace, as the states are no choices of it. kalman's states are in the
   prior's mean's shape; hmm's are state indices."
  (:refer-clojure :exclude [+ - * /])
  (:require [gen.array :as array :refer [+ - * /]]
            [gen.array.dtype :as dt]
            [gen.array.linalg :as la]
            [gen.choicemap :as cm]
            [gen.combinators :as comb]
            [gen.diff :as diff]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.gfi :as gfi]
            [gen.random :as rng]
            [gen.selection :as sel]
            [gen.trace :as trace]))

(set! *warn-on-reflection* true)

;; ---------------------------------------------------------------------------
;; Values
;; ---------------------------------------------------------------------------

(defn- as-floats
  "x (an array, or array data) as an array of floats."
  [x]
  (let [x (dc/->array x)]
    (if (= :float (dt/kind (array/dtype x))) x (array/astype x (:float (array/defaults (array/backend x)))))))

(defn- shaped
  "x, a parameter, as floats of shape s: any shape of as many elements."
  [model what x s]
  (let [x (as-floats x)]
    (when-not (== (array/size x) (reduce clojure.core/* 1 s))
      (throw (ex-info (str (name model) ": " what " has shape " (array/shape x) ", which does not fit " s)
                      {:model model :param what :shape (array/shape x) :expected s})))
    (if (= s (array/shape x)) x (array/reshape x s))))

(defn- matvec [m v] (array/squeeze (array/matmul m (array/expand-dims v -1)) -1))

(defn- t-of
  "The transpose of a matrix (its last two axes)."
  [a]
  (let [r (array/ndim a)]
    (array/transpose a (-> (vec (range r)) (assoc (clojure.core/- r 2) (dec r)) (assoc (dec r) (clojure.core/- r 2))))))

(defn- symmetric
  "a's symmetric part, (a + aᵀ)/2: a product of symmetric matrices is
   symmetric only up to rounding."
  [a]
  (* 0.5 (+ a (t-of a))))

(defn- params-at
  "The parameter map of step t, checked to hold the required keys and no
   others (a misspelled offset would otherwise be 0)."
  [model params t required optional]
  (let [p (if (map? params) params (params t))]
    (when-not (map? p)
      (throw (ex-info (str (name model) ": params is a map, or a function of the step returning one")
                      {:model model :params p})))
    (when-let [missing (seq (remove #(contains? p %) required))]
      (throw (ex-info (str (name model) ": params lack " (vec missing)) {:model model :missing (vec missing)})))
    (when-let [unknown (seq (remove (set (concat required optional)) (keys p)))]
      (throw (ex-info (str (name model) ": params has unknown keys " (vec unknown) "; the keys are "
                           (vec (concat required optional)))
                      {:model model :unknown (vec unknown)})))
    p))

;; ---------------------------------------------------------------------------
;; The filter step: one choice, the observation, and the belief it leaves
;; ---------------------------------------------------------------------------

(defn- total [lp] (if (zero? (array/ndim lp)) lp (array/sum lp)))

(defn- predictive
  "[dist correct] of a step on args (t belief params): the observation's
   predictive distribution, and the function from an observation to the
   filtered belief."
  [{:keys [kind observe]} args]
  (when-not (== 3 (count args))
    (throw (ex-info (str (name kind) ": a step takes (t belief params): the generative function's arguments are"
                         " (n prior params)")
                    {:model kind :args (count args)})))
  (let [[t belief params] args]
    (observe t belief params)))

(defn- step-trace [step args y lp correct]
  (trace/make-trace {:gen-fn step :args (vec args) :choices (cm/->Value y) :retval (correct y) :score lp}))

(defn- drawn
  "The step's trace with an observation drawn from its predictive
   distribution."
  [step args key]
  (let [[dist correct] (predictive step args)
        y (dc/sample dist key)]
    (step-trace step args y (total (dc/log-prob dist y)) correct)))

(defn- given
  "The step's trace with the observation y, checked as a constraint is."
  [step op args y]
  (let [[dist correct] (predictive step args)]
    (dc/check-not-nan op y)
    (dc/check-value-shape dist y)
    (step-trace step args y (total (dc/log-prob dist y)) correct)))

(defn- check-leaf
  "A step's choice is a single value: choices under it are unvisited, not
   silently ignored."
  [op constraints]
  (when-not (or (cm/has-value? constraints) (cm/empty? constraints))
    (throw (gfi/unvisited op (cm/addresses constraints) "a step's choice is its observation, a single value"))))

(defn- unchanged? [tr args argdiffs]
  (and (== (count args) (count (:args tr))) (every? diff/no-change? argdiffs)))

(gfi/defgf FilterStep [kind observe]
  gfi/IGenerativeFunction
  (-simulate [this args key] (drawn this args key))
  (-generate [this args constraints key]
    (check-leaf :generate constraints)
    (if (cm/has-value? constraints)
      (let [t (given this :generate args (cm/get-value constraints))] {:trace t :weight (:score t)})
      {:trace (drawn this args key) :weight 0.0}))
  (-update [this tr args argdiffs constraints _]
    (check-leaf :update constraints)
    (let [constrained? (cm/has-value? constraints)]
      (if (and (not constrained?) (unchanged? tr args argdiffs))
        {:trace tr :weight 0.0 :retdiff diff/no-change :discard cm/EMPTY}
        ;; the observation kept, rescored: the belief may have changed
        (let [t (given this :update args (cm/get-value (if constrained? constraints (:choices tr))))]
          {:trace t :weight (- (:score t) (:score tr)) :retdiff diff/unknown-change
           :discard (if constrained? (:choices tr) cm/EMPTY)}))))
  (-regenerate [this tr args argdiffs selection key]
    (cond (sel/root-selected? selection) {:trace (drawn this args key) :weight 0.0 :retdiff diff/unknown-change}
          (unchanged? tr args argdiffs) {:trace tr :weight 0.0 :retdiff diff/no-change}
          :else (let [t (given this :regenerate args (cm/get-value (:choices tr)))]
                  {:trace t :weight (- (:score t) (:score tr)) :retdiff diff/unknown-change})))
  (-assess [this args choices _]
    (check-leaf :assess choices)
    (when-not (cm/has-value? choices)
      (throw (ex-info (str "assess: a " (name kind) " step needs its observation") {:model kind :missing-choice true})))
    (let [t (given this :assess args (cm/get-value choices))] {:weight (:score t) :retval (:retval t)}))
  (-propose [this args key]
    (let [t (drawn this args key)] {:choices (:choices t) :weight (:score t) :retval (:retval t)}))
  (-project [_ tr selection] (if (sel/root-selected? selection) (:score tr) 0.0)))

(defmethod print-method FilterStep [s ^java.io.Writer w]
  (.write w (str "#gen/" (name (:kind s)) "-step")))

;; ---------------------------------------------------------------------------
;; The Kalman filter
;; ---------------------------------------------------------------------------

(def ^:private kalman-required [:transition :transition-cov :emission :emission-cov])
(def ^:private kalman-optional [:transition-offset :emission-offset])

(defn- kalman-params [params t] (params-at :kalman params t kalman-required kalman-optional))

(defn- state-size
  "d, the size of the state of mean, a number or a vector [d]."
  ^long [mean]
  (when (> (array/ndim mean) 1)
    (throw (ex-info (str "kalman: the state's mean is a number or a vector, not of shape " (array/shape mean))
                    {:shape (array/shape mean)})))
  (array/size mean))

(defn- lifted
  "A belief {:mean :cov} as a vector [d] and a matrix [d d], and the shapes
   it gives them."
  [{:keys [mean cov]}]
  (let [m (as-floats mean)
        c (as-floats cov)
        d (state-size m)]
    [(array/reshape m [d]) (shaped :kalman "the cov" c [d d]) (array/shape m) (array/shape c)]))

(defn- offset [x n what] (if (some? x) (shaped :kalman what x [n]) 0.0))

(defn- kalman-observe
  "Step t of the Kalman filter: the prediction (none at t = 0, whose belief
   is the prior), the observation's predictive distribution, and the update
   in Joseph's form, (I - KH) P (I - KH)ᵀ + K R Kᵀ, symmetric and positive
   semidefinite under rounding."
  [t belief params]
  (let [{F :transition Q :transition-cov b :transition-offset
         H :emission R :emission-cov c :emission-offset} (kalman-params params t)
        [m P mean-shape cov-shape] (lifted belief)
        d (peek (array/shape m))
        [m P] (if (zero? t)
                [m P]
                (let [F (shaped :kalman ":transition" F [d d])]
                  [(+ (matvec F m) (offset b d ":transition-offset"))
                   (symmetric (+ (array/matmul F (array/matmul P (t-of F))) (shaped :kalman ":transition-cov" Q [d d])))]))
        R (as-floats R)
        scalar? (zero? (array/ndim R))
        k (if scalar? 1 (first (array/shape R)))
        R (shaped :kalman ":emission-cov" R [k k])
        H (shaped :kalman ":emission" H [k d])
        y-hat (+ (matvec H m) (offset c k ":emission-offset"))
        S (symmetric (+ (array/matmul H (array/matmul P (t-of H))) R))
        dist (if scalar?
               (let [s (array/reshape S [])]
                 (when (and (array/concrete? s) (not (array/all-true? (array/> s 0.0))))
                   (throw (ex-info "kalman: the observation's predictive variance must be positive" {:step t})))
                 (d/normal (array/reshape y-hat []) (array/sqrt s)))
               (d/mvnormal y-hat S))]
    [dist
     (fn [y]
       (let [gain (t-of (la/solve S (array/matmul H P)))
             m (+ m (matvec gain (- (array/reshape y [k]) y-hat)))
             a (- (array/eye d) (array/matmul gain H))
             P (symmetric (+ (array/matmul a (array/matmul P (t-of a)))
                             (array/matmul gain (array/matmul R (t-of gain)))))]
         {:mean (array/reshape m mean-shape) :cov (array/reshape P cov-shape)}))]))

(def kalman
  "The Kalman filter's generative function, on (n prior params) (see the
   namespace doc)."
  (comb/unfold (->FilterStep :kalman kalman-observe)))

;; ---------------------------------------------------------------------------
;; The forward algorithm
;; ---------------------------------------------------------------------------

(defn- hmm-params [params t] (params-at :hmm params t [:transition :emission] []))

(defn- emission-ll
  "log p(y | state) of each state, [K]: the emission's log density of y,
   summed over the axes of one value."
  [emission y]
  (let [lp (dc/log-prob emission y)
        r (array/ndim lp)]
    (if (> r 1) (array/sum lp {:axis (vec (range 1 r))}) lp)))

(defn- observation
  "The predictive distribution of an observation: the emission's, mixed
   over the states by exp(log-weights)."
  [log-weights emission]
  (dc/->Distribution ::observation {:log-weights log-weights :emission emission}))

(defmethod dc/-sample ::observation [{{:keys [log-weights emission]} :params} key shape]
  (when (seq shape)
    (throw (ex-info "hmm: an observation is drawn one at a time" {:shape shape})))
  (let [[k-state k-draw] (rng/split key)]
    ;; every state's draw, the drawn state's kept: branch-free, so it batches
    (array/take (dc/sample emission k-draw) (rng/categorical k-state log-weights) 0)))

(defmethod dc/-log-prob ::observation [{{:keys [log-weights emission]} :params} v]
  (array/logsumexp (+ log-weights (emission-ll emission v))))

(defmethod dc/-value-shape ::observation [{{:keys [emission]} :params}] (subvec (dc/value-shape emission) 1))

(defmethod dc/-discrete? ::observation [{{:keys [emission]} :params}] (dc/discrete? emission))

(defn- check-emission! [emission n]
  (let [s (when (instance? gen.dist.core.Distribution emission) (dc/value-shape emission))]
    (when-not (and (seq s) (== n (long (first s))))
      (throw (ex-info (str "hmm: the emission is a distribution batched over the " n " states on its leading axis"
                           " (say (categorical B) for a [" n " M] matrix B); its values have shape " (pr-str s))
                      {:states n :value-shape s})))))

(defn- hmm-observe
  "Step t of the forward algorithm: the predicted state probabilities (the
   prior at t = 0), the observation's predictive distribution, and the
   filtered probabilities, the predicted ones times the emission's
   likelihoods, normalized."
  [t probs params]
  (let [{:keys [transition emission]} (hmm-params params t)
        probs (as-floats probs)
        _ (when-not (== 1 (array/ndim probs))
            (throw (ex-info (str "hmm: the state probabilities are a vector, not of shape " (array/shape probs))
                            {:shape (array/shape probs)})))
        n (peek (array/shape probs))
        pred (if (zero? t)
               (do (dc/check-probabilities "hmm" "the prior" probs) probs)
               (let [a (shaped :hmm ":transition" transition [n n])]
                 (dc/check-probabilities "hmm" "each row of :transition" a)
                 (array/matmul probs a)))
        lw (array/log pred)]
    (check-emission! emission n)
    [(observation lw emission)
     (fn [y]
       (let [joint (+ lw (emission-ll emission y))]
         (array/where (array/isfinite (array/logsumexp joint)) (array/softmax joint) pred)))]))

(def hmm
  "The forward algorithm's generative function, on (n prior params) (see
   the namespace doc)."
  (comb/unfold (->FilterStep :hmm hmm-observe)))

;; ---------------------------------------------------------------------------
;; Forward filtering, backward sampling
;; ---------------------------------------------------------------------------

(defn- kalman-states
  "x_n-1 from the last filtered belief; then x_t given x_t+1, from the
   filtered belief (m, P) and the transition into t+1: N(m + J (x_t+1 -
   F m - b), (I - JF) P (I - JF)ᵀ + J Q Jᵀ), J = P Fᵀ (F P Fᵀ + Q)⁻¹."
  [beliefs params ks]
  (loop [t (dec (count beliefs)) after nil states ()]
    (if (neg? t)
      (vec states)
      (let [[m P mean-shape] (lifted (beliefs t))
            d (peek (array/shape m))
            dist (if (nil? after)
                   (d/mvnormal m P)
                   (let [{F :transition Q :transition-cov b :transition-offset} (kalman-params params (inc t))
                         F (shaped :kalman ":transition" F [d d])
                         Q (shaped :kalman ":transition-cov" Q [d d])
                         ahead (symmetric (+ (array/matmul F (array/matmul P (t-of F))) Q))
                         j (t-of (la/solve ahead (array/matmul F P)))
                         a (- (array/eye d) (array/matmul j F))]
                     (d/mvnormal (+ m (matvec j (- after (+ (matvec F m) (offset b d ":transition-offset")))))
                                 (symmetric (+ (array/matmul a (array/matmul P (t-of a)))
                                               (array/matmul j (array/matmul Q (t-of j))))))))
            x (dc/sample dist (ks t))]
        (recur (dec t) x (cons (array/reshape x mean-shape) states))))))

(defn- hmm-states
  "x_n-1 from the last filtered probabilities; then x_t given x_t+1, in
   proportion to the filtered probabilities times A's column x_t+1."
  [beliefs params ks]
  (loop [t (dec (count beliefs)) after nil states ()]
    (if (neg? t)
      (vec states)
      (let [logits (array/log (as-floats (beliefs t)))
            logits (if (nil? after)
                     logits
                     (let [n (peek (array/shape logits))]
                       (+ logits (array/log (array/take (shaped :hmm ":transition" (:transition (hmm-params params (inc t))) [n n])
                                                        after 1)))))
            x (rng/categorical (ks t) logits)]
        (recur (dec t) x (cons x states))))))

(defn sample-states
  "The latent states of a trace of kalman or hmm, a vector of n, drawn from
   their posterior given its observations (forward filtering, backward
   sampling)."
  ([tr] (sample-states tr (gfi/next-key)))
  ([tr key]
   (let [kind (:kind (:kernel (trace/maker tr)))
         [_ _ params] (:args tr)
         beliefs (:retval tr)
         ks (rng/split key (max 1 (count beliefs)))]
     (case kind
       :kalman (kalman-states beliefs params ks)
       :hmm (hmm-states beliefs params ks)
       (throw (ex-info "sample-states takes a trace of gen.state-space/kalman or hmm"
                       {:gen-fn (type (trace/maker tr))}))))))
