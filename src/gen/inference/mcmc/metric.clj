(ns gen.inference.mcmc.metric
  "The inverse metrics of gen.inference.mcmc's gradient kernels (hmc, mala
   and nuts: their :inv-metric), read for the selected values and checked,
   and what a step takes from one. A diagonal metric is its variances, flat
   in the values' order (or one host number for all of them); a dense one,
   {:inv M⁻¹ :chol L :chol-inv L⁻¹}, M⁻¹ = L·Lᵀ. gen.inference.mcmc holds
   the forms' public side (dense-inv-metric, the record a dense metric is,
   and inv-metric) and documents them; this namespace parses them for it."
  {:api :internal}
  (:refer-clojure :exclude [+ - *])
  (:require [gen.array :as array :refer [+ - *]]
            [gen.array.linalg :as la]
            [gen.choicemap :as cm]))

(set! *warn-on-reflection* true)

;; ---------------------------------------------------------------------------
;; Diagonal metrics
;; ---------------------------------------------------------------------------

(defn- one-number?
  "Is x one variance for every value: a host number, or a 0-d array (a
   computed scalar on a lazy backend)?"
  [x]
  (or (number? x) (and (array/backend-array? x) (empty? (array/shape x)))))

(defn- choice-variances
  "The variances s given for the selected choice x at path p, flat: one
   number for each of its elements, or an array (or a vector) of x's shape
   or flat with its size. Loud otherwise."
  [s x p]
  (let [m (if (sequential? s) (array/array s) s)
        n (array/size x)]
    (cond (one-number? m) (array/full [n] m)
          (and (array/backend-array? m) (or (= (array/shape m) (array/shape x)) (= (array/shape m) [n])))
          (array/flatten m)
          :else (throw (ex-info (str ":inv-metric at " p " must be a variance, or an array of the choice's shape "
                                     (array/shape x) ", not " (pr-str s))
                                {:reason :invalid-inv-metric :path p :inv-metric s :shape (array/shape x)})))))

(defn- address-key
  "A path as a map key: its address when it has one, as a map of variances
   writes it."
  [p]
  (if (= 1 (count p)) (first p) p))

(defn- labeled-variances
  "The variances of inv, a choice map or a map of them by address, for the
   selected values (a choice map), flat in the values' order. Loud unless
   inv has a variance at each selected address and at no other."
  [inv values]
  (let [m (cm/from-map inv)
        want (cm/addresses values)
        got (cm/addresses m)
        missing (vec (remove (set got) want))
        unexpected (vec (remove (set want) got))]
    (when (or (seq missing) (seq unexpected))
      (throw (ex-info (str ":inv-metric must hold a variance for each selected choice, at " want
                           (when (seq missing) (str "; it has none at " missing))
                           (when (seq unexpected) (str "; it has some at " unexpected ", which are not selected")))
                      {:reason :invalid-inv-metric :addresses want :missing missing :unexpected unexpected})))
    (array/concatenate (mapv (fn [[p x]] (choice-variances (cm/get-choice m p) x p)) (cm/leaves values)))))

(defn diagonal
  "The diagonal metric inv (nil, a number, variances by address as a
   choice map or a map, or a vector or array for one selected choice) for
   the selected values (a choice map) flat as v: its variances flat as v,
   or one number for every value (the host number 1.0, the identity, for
   nil). Loud unless every variance is positive and finite, and for a
   vector or array when more than one choice is selected, whose order it
   would leave to guess."
  [inv values v]
  (let [m (cond (nil? inv) 1.0
                (one-number? inv) inv
                (map? inv) (labeled-variances inv values)
                (or (sequential? inv) (array/backend-array? inv))
                (let [ls (cm/leaves values)
                      paths (mapv first ls)]
                  (if (= 1 (count ls))
                    (let [[p x] (first ls)] (choice-variances inv x p))
                    (throw (ex-info (str ":inv-metric as a vector or array leaves its order to guess for the "
                                         (count ls) " selected choices " paths
                                         ": give a variance for each by address, as a choice map or a map"
                                         (when (= 2 (count (array/shape (if (sequential? inv) (array/array inv) inv))))
                                           ", or a dense metric with its addresses' order, (mcmc/dense-inv-metric addresses matrix)")
                                         (when (<= (count paths) 4)
                                           (str " such as " (pr-str (apply array-map (mapcat (fn [p] [(address-key p) 1.0]) paths))))))
                                    {:reason :unlabeled-inv-metric :inv-metric inv :addresses paths}))))
                :else (throw (ex-info (str ":inv-metric is a number, or variances by address (a choice map or a map), not "
                                           (pr-str inv))
                                      {:reason :invalid-inv-metric :inv-metric inv})))
        m (if (array/host? m) m (array/astype m (array/dtype v)))]
    (when-not (array/truthy? (array/all (array/logical-and (array/isfinite m) (array/> m 0))))
      (throw (ex-info (str ":inv-metric's variances must be positive and finite, not " (pr-str inv))
                      {:reason :invalid-inv-metric :inv-metric inv})))
    m))

;; ---------------------------------------------------------------------------
;; Dense metrics
;; ---------------------------------------------------------------------------

(defn in-values-order
  "A dense-inv-metric's matrix with its rows and columns in the selected
   values' order (flat as cm/to-array), from its :addresses and :matrix.
   Loud unless its addresses are the selected choices', each once, and its
   size is theirs."
  [{:keys [addresses matrix]} values]
  (let [ls (cm/leaves values)
        want (mapv first ls)
        repeated (vec (keep (fn [[a n]] (when (> n 1) a)) (frequencies addresses)))
        missing (vec (remove (set addresses) want))
        unexpected (vec (remove (set want) addresses))
        _ (when (or (seq repeated) (seq missing) (seq unexpected))
            (throw (ex-info (str "A dense :inv-metric's addresses must be the selected choices " want ", each once"
                                 (when (seq missing) (str "; it lacks " missing))
                                 (when (seq unexpected) (str "; it has " unexpected ", which are not selected"))
                                 (when (seq repeated) (str "; it repeats " repeated)))
                            {:reason :invalid-inv-metric :addresses want :missing missing :unexpected unexpected :repeated repeated})))
        size-of (into {} (map (fn [[p x]] [p (long (array/size x))])) ls)
        d (reduce clojure.core/+ 0 (vals size-of))
        shape (vec (array/shape matrix))
        _ (when-not (= [d d] shape)
            (throw (ex-info (str "A dense :inv-metric for the selected choices " want " is " d "×" d ", not " shape)
                            {:reason :invalid-inv-metric :shape shape :size d})))
        start (zipmap addresses (reductions clojure.core/+ 0 (map size-of addresses)))
        idx (vec (mapcat (fn [[p _]] (range (start p) (clojure.core/+ (long (start p)) (long (size-of p))))) ls))]
    (if (= idx (vec (range d)))
      matrix
      (let [i (array/with-backend (array/backend matrix) (array/array idx {:dtype :int32}))]
        (array/take (array/take matrix i 0) i 1)))))

(defn bare-matrix
  "inv as a bare dense matrix for the selected values (a choice map), or
   nil: when one choice of n elements is selected, a matrix of shape [n n]
   (an array or nested vectors), unless the choice itself has that shape
   (its variances then)."
  [inv values]
  (when (or (sequential? inv) (array/backend-array? inv))
    (let [ls (cm/leaves values)]
      (when (= 1 (count ls))
        (let [[_ x] (first ls)
              n (long (array/size x))
              a (if (sequential? inv) (array/array inv) inv)]
          (when (and (= [n n] (vec (array/shape a))) (not= [n n] (vec (array/shape x))))
            a))))))

(defn- symmetry-tolerance
  "How far a dense metric m's entries may be from its transpose's, relative
   to max(1, its largest |entry|), by m's dtype: Stan's check_symmetric's
   1e-8 in float64, about a hundred rounding steps in float32, where a
   computed metric is off-symmetric by its rounding (owner, 2026-11-04)."
  ^double [m]
  (if (= :float64 (array/dtype m)) 1e-8 1e-5))

(defn dense
  "The dense inverse metric m (a matrix in the selected values' order) as
   the kernels use it, in v's dtype: {:inv M⁻¹ :chol L :chol-inv L⁻¹},
   M⁻¹ = L·Lᵀ, so that momenta from N(0, M) are L⁻ᵀ·z. Computed on m's own
   backend, then arrays of the current one: constants of a compiled,
   staged or batched step. Loud unless m is d×d for the d selected values,
   finite, symmetric (within symmetry-tolerance) and positive definite
   (given, the :inv-metric option, in the error's data); m's symmetric
   part, (m + mᵀ)/2, is the one used."
  [m v given]
  (let [d (long (array/size v))
        dtype (array/dtype v)
        shape (vec (array/shape m))
        refuse (fn [why] (throw (ex-info (str "A dense :inv-metric must be " why) {:reason :invalid-inv-metric :inv-metric given})))
        ;; a factor as an array of the current backend: as it is, or made
        ;; from its values where the step runs elsewhere (a staged constant)
        here (fn [x] (if (identical? (array/backend x) array/*backend*) x (array/array (array/->clj x) {:dtype dtype})))]
    (update-vals
     (array/with-backend (array/backend m)
       (when-not (= [d d] shape) (refuse (str d "×" d " for the " d " selected values, not of shape " shape)))
       (when-not (array/truthy? (array/all (array/isfinite m))) (refuse "finite"))
       (let [eps (symmetry-tolerance m)
             scale (max 1.0 (double (array/item (array/max (array/abs m)))))]
         (when-not (array/truthy? (array/<= (array/max (array/abs (- m (array/transpose m)))) (* eps scale)))
           (refuse (str "symmetric (to " eps " of its largest entry)"))))
       ;; its symmetric part, exactly symmetric (a symmetric m unchanged: (a + a)/2 = a)
       (let [m (array/astype (* 0.5 (+ m (array/transpose m))) dtype)
             l (la/cholesky m)]
         (when-not (array/truthy? (array/all (array/logical-and (array/isfinite l) (array/> (la/diagonal l) 0))))
           (refuse "positive definite"))
         {:inv m :chol l :chol-inv (la/solve-triangular l (array/eye d {:dtype dtype}))}))
     here)))

;; ---------------------------------------------------------------------------
;; What a step takes from a metric
;; ---------------------------------------------------------------------------

(defn dense? "Is inv (diagonal's or dense's) a dense metric?" [inv] (map? inv))

(defn velocity
  "The velocity of momentum (or gradient) p under the inverse metric inv:
   inv ⊙ p for a diagonal one, M⁻¹·p for a dense one."
  [inv p]
  (if (dense? inv) (array/matmul (:inv inv) p) (* inv p)))

(defn dense-momenta
  "Momenta from N(0, M) under a dense inverse metric: L⁻ᵀ·z, z standard
   normal, M⁻¹ = L·Lᵀ."
  [inv z]
  (array/matmul (array/transpose (:chol-inv inv)) z))
