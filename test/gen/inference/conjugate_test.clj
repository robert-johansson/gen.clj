(ns gen.inference.conjugate-test
  "Conjugate proposals against closed forms written here, independently of
   gen.inference.conjugate's posteriors: each pair's evidence (from the
   joint Gaussian of the observations, beta and gamma integrals, the
   Dirichlet-multinomial), its posterior's moments, and Gibbs moves; the
   GFI laws (the general set); and GenMLX's conjugacy bug beans (rmy7,
   94qc, 5ytq, jmk4, su6q, yy8u), whose silent wrong marginals are loud
   errors here."
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array :refer [+ - * /]]
            [gen.batched :as b]
            [gen.choicemap :as cm]
            [gen.combinators :as comb]
            [gen.diff :as diff]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.conjugate :as conj]
            [gen.inference.importance :as imp]
            [gen.inference.mh :as mh]
            [gen.inference.particle-filter :as pf]
            [gen.laws :as laws]
            [gen.random :as rng]
            [gen.selection :as sel]
            [gen.test-support :as ts]
            [gen.trace :as trace]))

(defn- lgamma ^double [x] (ts/number (array/lgamma (double x))))

(defn- mvn-log-density
  "log N(y; mu, cov) of host vectors, by gen.dist's mvnormal."
  [mu cov y]
  (ts/number (dc/log-prob (d/mvnormal (array/array mu) (array/array cov)) (array/array y))))

(defn- spread
  "The largest difference between the weights of generate under keys."
  [gf args obs keys]
  (let [ws (map #(ts/number (:weight (gfi/generate gf args obs %))) keys)]
    (clojure.core/- (apply max ws) (apply min ws))))

(defn- mean [xs] (/ (reduce clojure.core/+ xs) (count xs)))

(defn- within-5-se?
  "Is the mean of the draws xs within 5 standard errors of mu, for draws of
   variance v?"
  [xs mu v]
  (< (Math/abs (clojure.core/- (mean xs) mu)) (* 5 (Math/sqrt (/ v (count xs))))))

;; ---------------------------------------------------------------------------
;; The pairs, with their evidence by hand
;; ---------------------------------------------------------------------------

;; mu ~ N(1, 2); y1 ~ N(mu, 0.5), and y2 two observations of mu, sd 1.5 and 0.8
(def normal-model
  (gen [] (let [mu (trace :mu (d/normal 1.0 2.0))]
            (trace :y1 (d/normal mu 0.5))
            (trace :y2 (d/normal mu (array/array [1.5 0.8])))
            mu)))

(def normal-obs (cm/choicemap :y1 0.3 :y2 (array/array [-0.4 0.9])))

;; (y1, y2) ~ N(1, 4·11ᵀ + diag(0.25, 2.25, 0.64))
(def normal-evidence
  (mvn-log-density [1.0 1.0 1.0] [[4.25 4.0 4.0] [4.0 6.25 4.0] [4.0 4.0 4.64]] [0.3 -0.4 0.9]))

;; an array latent: each element with its own prior, observed by y once
;; and by z twice
(def elementwise-model
  (gen [] (let [mu (trace :mu (d/normal (array/array [0.0 1.0]) (array/array [1.0 2.0])))]
            (trace :y (d/normal mu 0.5))
            (trace :z (d/normal mu (array/array [[0.3 0.4] [0.7 0.2]]))))))

(def elementwise-obs (cm/choicemap :y (array/array [0.2 1.5]) :z (array/array [[-0.1 0.8] [0.4 2.0]])))

(def elementwise-evidence
  (clojure.core/+ (mvn-log-density [0.0 0.0 0.0] [[1.25 1.0 1.0] [1.0 1.09 1.0] [1.0 1.0 1.49]] [0.2 -0.1 0.4])
                  (mvn-log-density [1.0 1.0 1.0] [[4.25 4.0 4.0] [4.0 4.16 4.0] [4.0 4.0 4.04]] [1.5 0.8 2.0])))

;; a latent of shape [2 1], each element observed three times along the
;; axis the child's std broadcasts
(def broadcast-model
  (gen [] (let [mu (trace :mu (d/normal (array/array [[0.0] [1.0]]) 1.0))]
            (trace :y (d/normal mu (array/array [[0.5 0.6 0.7]]))))))

(def broadcast-obs (cm/choicemap :y (array/array [[0.1 -0.2 0.3] [1.2 0.9 1.4]])))

(def broadcast-evidence
  (let [cov [[1.25 1.0 1.0] [1.0 1.36 1.0] [1.0 1.0 1.49]]]
    (clojure.core/+ (mvn-log-density [0.0 0.0 0.0] cov [0.1 -0.2 0.3])
                    (mvn-log-density [1.0 1.0 1.0] cov [1.2 0.9 1.4]))))

(def mv-prior-cov [[1.0 0.3] [0.3 2.0]])
(def mv-cov-1 [[0.5 0.1] [0.1 0.4]])
(def mv-cov-2 [[1.0 -0.2] [-0.2 0.3]])

(def mvnormal-model
  (gen [] (let [theta (trace :theta (d/mvnormal (array/array [0.0 1.0]) (array/array mv-prior-cov)))]
            (trace :y1 (d/mvnormal theta (array/array mv-cov-1)))
            (trace :y2 (d/mvnormal theta (array/array mv-cov-2))))))

(def mvnormal-obs (cm/choicemap :y1 (array/array [0.4 0.2]) :y2 (array/array [-0.3 1.7])))

;; (y1, y2) ~ N([mu; mu], [[S0 + S1, S0], [S0, S0 + S2]])
(def mvnormal-evidence
  (let [m (fn [a b] (mapv #(mapv clojure.core/+ %1 %2) a b))
        s0 mv-prior-cov
        top (mapv into (m s0 mv-cov-1) s0)
        bottom (mapv into s0 (m s0 mv-cov-2))]
    (mvn-log-density [0.0 1.0 0.0 1.0] (into top bottom) [0.4 0.2 -0.3 1.7])))

;; p ~ beta(2, 3); x_i ~ bernoulli(p); k ~ binom(5, p)
(def beta-model
  (gen [n] (let [p (trace :p (d/beta 2.0 3.0))]
             (doseq [i (range n)] (trace [:x i] (d/bernoulli p)))
             (trace :k (d/binom 5 p))
             p)))

(def beta-obs (cm/choicemap [:x 0] true [:x 1] false [:x 2] true :k 4))

;; B(2 + 6, 3 + 2) / B(2, 3) · C(5, 4)
(defn- log-beta-fn ^double [a b] (clojure.core/- (clojure.core/+ (lgamma a) (lgamma b)) (lgamma (clojure.core/+ a b))))
(def beta-evidence (clojure.core/+ (clojure.core/- (log-beta-fn 8.0 5.0) (log-beta-fn 2.0 3.0)) (Math/log 5.0)))

;; lambda ~ gamma(2, 1.5) (shape, scale); y_i ~ poisson(lambda)
(def gamma-model
  (gen [n] (let [lam (trace :lam (d/gamma 2.0 1.5))]
             (doseq [i (range n)] (trace [:y i] (d/poisson lam))))))

(def gamma-ys [3 0 5 2])
(def gamma-obs (apply cm/choicemap (mapcat (fn [i y] [[:y i] y]) (range) gamma-ys)))

;; Π 1/y_i! · Γ(k + S) / (Γ(k) θ^k (n + 1/θ)^(k + S))
(def gamma-evidence
  (let [k 2.0 theta 1.5 s (reduce clojure.core/+ gamma-ys) n (count gamma-ys)]
    (clojure.core/- (clojure.core/+ (lgamma (clojure.core/+ k s)))
                    (reduce clojure.core/+ (map #(lgamma (inc %)) gamma-ys))
                    (lgamma k) (* k (Math/log theta)) (* (clojure.core/+ k s) (Math/log (clojure.core/+ n (/ 1.0 theta)))))))

;; theta ~ dirichlet([1 2 0.5]); c_i ~ categorical(theta)
(def dirichlet-model
  (gen [n] (let [theta (trace :theta (d/dirichlet (array/array [1.0 2.0 0.5])))]
             (doseq [i (range n)] (trace [:c i] (d/categorical theta))))))

(def dirichlet-cs [0 2 1 1 0 1])
(def dirichlet-obs (apply cm/choicemap (mapcat (fn [i c] [[:c i] c]) (range) dirichlet-cs)))

;; Γ(A) / Γ(A + n) · Π Γ(α_k + c_k) / Γ(α_k)
(def dirichlet-evidence
  (let [alpha [1.0 2.0 0.5] counts (frequencies dirichlet-cs)]
    (clojure.core/+ (clojure.core/- (lgamma 3.5) (lgamma (clojure.core/+ 3.5 (count dirichlet-cs))))
                    (reduce clojure.core/+ (map-indexed (fn [i a] (clojure.core/- (lgamma (clojure.core/+ a (get counts i 0))) (lgamma a)))
                                                        alpha)))))

(def pairs
  [[:normal (conj/with-conjugate-proposal normal-model :mu [:y1 :y2]) [] normal-obs normal-evidence]
   [:elementwise (conj/with-conjugate-proposal elementwise-model :mu [:y :z]) [] elementwise-obs elementwise-evidence]
   [:broadcast (conj/with-conjugate-proposal broadcast-model :mu [:y]) [] broadcast-obs broadcast-evidence]
   [:mvnormal (conj/with-conjugate-proposal mvnormal-model :theta [:y1 :y2]) [] mvnormal-obs mvnormal-evidence]
   [:beta (conj/with-conjugate-proposal beta-model :p (sel/select :x :k)) [3] beta-obs beta-evidence]
   [:gamma (conj/with-conjugate-proposal gamma-model :lam (sel/select :y)) [4] gamma-obs gamma-evidence]
   [:dirichlet (conj/with-conjugate-proposal dirichlet-model :theta (sel/select :c)) [6] dirichlet-obs dirichlet-evidence]])

(defn- pairs-of [names] (filter #(names (first %)) pairs))

(deftest the-weight-is-the-exact-evidence
  (doseq [[name gf args obs evidence] pairs]
    (testing (str name)
      (doseq [k (range 4)]
        (let [{tr :trace w :weight} (gfi/generate gf args obs (rng/key k))]
          (is (laws/close? w evidence) "log p(observations), whatever latent is drawn")
          (is (laws/close? (:score tr) (:weight (gfi/assess (:gf gf) args (:choices tr)))) "the model's own density")))
      (is (laws/close? (spread gf args obs (rng/split (rng/key 9) 6)) 0.0) "log weights without variance")
      (is (laws/close? (:log-ml-estimate (imp/importance-sampling gf args obs 1 {:key (rng/key 3)})) evidence)
          "so importance sampling's estimate is exact with one sample"))))

(deftest batched-runs-are-the-same-proposal
  (doseq [[name gf args obs evidence] (pairs-of #{:normal :beta :dirichlet})]
    (testing (str name)
      (is (every? #(laws/close? % evidence) (ts/host (:weights (b/generate gf args obs 30 (rng/key 4)))))))))

;; ---------------------------------------------------------------------------
;; The posterior, and Gibbs
;; ---------------------------------------------------------------------------

(defn- inv2
  "The inverse of a 2×2 host matrix."
  [[[a b] [c e]]]
  (let [det (clojure.core/- (* a e) (* b c))]
    [[(/ e det) (/ (clojure.core/- b) det)] [(/ (clojure.core/- c) det) (/ a det)]]))

(defn- mat+ [x y] (mapv #(mapv clojure.core/+ %1 %2) x y))
(defn- mat*v [m v] (mapv #(reduce clojure.core/+ (map * % v)) m))

;; each latent's posterior moments by hand, [mean variance] per element
(def posterior-moments
  {:normal (let [prec (clojure.core/+ 0.25 4.0 (/ 1 2.25) (/ 1 0.64))]
             [[(/ (clojure.core/+ 0.25 (/ 0.3 0.25) (/ -0.4 2.25) (/ 0.9 0.64)) prec) (/ 1 prec)]])
   :mvnormal (let [p0 (inv2 mv-prior-cov) p1 (inv2 mv-cov-1) p2 (inv2 mv-cov-2)
                   cov (inv2 (mat+ (mat+ p0 p1) p2))
                   m (mat*v cov (mapv clojure.core/+ (mat*v p0 [0.0 1.0]) (mat*v p1 [0.4 0.2]) (mat*v p2 [-0.3 1.7])))]
               [[(m 0) (get-in cov [0 0])] [(m 1) (get-in cov [1 1])]])
   :beta (let [a 8.0 b 5.0 s (clojure.core/+ a b)] [[(/ a s) (/ (* a b) (* s s (inc s)))]])
   :gamma (let [k 12.0 scale (/ 1 (clojure.core/+ (/ 1 1.5) 4))] [[(* k scale) (* k scale scale)]])
   :dirichlet (let [alpha [3.0 5.0 1.5] s 9.5]
                (mapv (fn [a] [(/ a s) (/ (* a (clojure.core/- s a)) (* s s (inc s)))]) alpha))})

(def latent-address {:normal :mu :mvnormal :theta :beta :p :gamma :lam :dirichlet :theta})

(deftest ^:slow ^:heavy the-latent-is-drawn-from-its-posterior
  (doseq [[name gf args obs] (pairs-of (set (keys posterior-moments)))
          :let [draws (mapv #(ts/host ((:trace (gfi/generate gf args obs %)) (latent-address name)))
                            (rng/split (rng/key 20) 2000))]]
    (testing (str name)
      (doseq [[i [m v]] (map-indexed vector (posterior-moments name))]
        (is (within-5-se? (map #(if (number? %) % (% i)) draws) m v) (str "element " i "'s mean"))))))

(deftest regenerating-the-latent-is-gibbs
  (doseq [[name gf args obs] pairs
          :let [tr (:trace (gfi/generate gf args obs (rng/key 5)))]]
    (testing (str name)
      (doseq [k (rng/split (rng/key 6) 4)]
        (let [{t2 :trace w :weight} (gfi/regenerate tr (sel/select (latent-address name :mu)) k)]
          (is (laws/close? w 0.0))
          (is (= (ts/host (sel/get-selected (:choices tr) (sel/from-choicemap obs)))
                 (ts/host (sel/get-selected (:choices t2) (sel/from-choicemap obs))))))))))

;; mu ~ N(0, 3); theta ~ N(mu, 1); y_i ~ N(theta, 0.5): theta's prior
;; depends on another latent, so its posterior is the conditional given mu
(def hierarchy
  (gen [] (let [mu (trace :mu (d/normal 0.0 3.0))
                theta (trace :theta (d/normal mu 1.0))]
            (doseq [i (range 3)] (trace [:y i] (d/normal theta 0.5))))))

(def hierarchy-ys [0.8 1.4 0.5])

(def hierarchy-obs (apply cm/choicemap (mapcat (fn [i y] [[:y i] y]) (range) hierarchy-ys)))

(def hierarchy-posterior
  "[mean cov] of (mu, theta) given the ys, by hand: the prior covariance is
   [[9 9] [9 10]], and the ys add precision 12 and information Σy/0.25 to
   theta."
  (let [cov (inv2 (mat+ (inv2 [[9.0 9.0] [9.0 10.0]]) [[0.0 0.0] [0.0 12.0]]))]
    [(mat*v cov [0.0 (/ (reduce clojure.core/+ hierarchy-ys) 0.25)]) cov]))

(deftest ^:slow ^:heavy gibbs-on-one-latent-keeps-the-joint-posterior
  ;; (mu, theta) drawn from their exact posterior, then theta regenerated
  ;; given mu: the pairs still follow the posterior (the move is
  ;; stationary), checked on theta's mean and variance and the covariance
  (let [g (conj/with-conjugate-proposal hierarchy :theta (sel/select :y))
        [m cov] hierarchy-posterior
        start (d/mvnormal (array/array m) (array/array cov))
        draws (mapv (fn [[k0 k1]]
                      (let [[mu theta] (ts/host (dc/sample start k0))
                            tr (:trace (gfi/generate g [] (cm/merge hierarchy-obs (cm/choicemap :mu mu :theta theta)) k1))
                            {t2 :trace w :weight} (gfi/regenerate tr (sel/select :theta) k1)]
                        (is (laws/close? w 0.0))
                        [mu (ts/number (t2 :theta))]))
                    (partition 2 (rng/split (rng/key 21) 4000)))
        thetas (map second draws)
        n (count draws)
        mt (mean thetas)
        mm (mean (map first draws))
        c (/ (reduce clojure.core/+ (map (fn [[x y]] (* (clojure.core/- x mm) (clojure.core/- y mt))) draws)) (dec n))
        [[vmu c-exact] [_ vtheta]] cov]
    (is (within-5-se? thetas (m 1) vtheta))
    (is (within-5-se? (map #(let [d (clojure.core/- % (m 1))] (* d d)) thetas) vtheta (* 2 vtheta vtheta)))
    (is (< (Math/abs (clojure.core/- c c-exact)) (* 5 (Math/sqrt (/ (clojure.core/+ (* vmu vtheta) (* c-exact c-exact)) n)))))))

(deftest ^:slow ^:heavy a-long-chain-finds-the-posterior
  ;; Gibbs on theta (always accepted) and MH on mu from its prior, in turn:
  ;; the chain's means of mu and theta are the posterior's, within 5
  ;; standard errors measured by batch means
  (let [g (conj/with-conjugate-proposal hierarchy :theta (sel/select :y))
        [m] hierarchy-posterior
        n 6000 burn-in 500 batches 22
        states (loop [tr (:trace (gfi/generate g [] hierarchy-obs (rng/key 40))) ks (rng/split (rng/key 41) n) out []]
                 (if-let [[k] (seq ks)]
                   (let [[k-theta k-mu] (rng/split k)
                         {tr :trace gibbs? :accepted?} (mh/mh tr (sel/select :theta) k-theta)
                         {tr :trace} (mh/mh tr (sel/select :mu) k-mu)]
                     (recur tr (rest ks) (conj out [(ts/number (tr :mu)) (ts/number (tr :theta)) gibbs?])))
                   out))
        kept (drop burn-in states)
        batch-means (fn [xs] (map mean (partition (quot (count xs) batches) xs)))]
    (is (zero? (count (remove #(% 2) states))) "every Gibbs move is accepted")
    (doseq [[i label] [[0 "mu"] [1 "theta"]]
            :let [bm (batch-means (map #(% i) kept))
                  mb (mean bm)
                  se (Math/sqrt (/ (/ (reduce clojure.core/+ (map #(Math/pow (clojure.core/- % mb) 2) bm)) (dec (count bm)))
                                   (count bm)))]]
      (is (< (Math/abs (clojure.core/- (mean (map #(% i) kept)) (m i))) (* 5 se)) label))))

;; ---------------------------------------------------------------------------
;; Other choices
;; ---------------------------------------------------------------------------

;; mu's prior depends on the open w; v is independent of everything
(def shifted
  (gen [] (let [w (trace :w (d/normal 0.0 1.0))
                _ (trace :v (d/normal 0.0 1.0))
                mu (trace :mu (d/normal w 1.0))]
            (let [y1 (trace :y1 (d/normal mu 0.5))]
              (trace :y2 (d/normal mu 0.8))
              ;; depends on an observed child only, not on mu
              (trace :z (d/normal (* 2.0 y1) 1.0))))))

(deftest other-choices-keep-the-forward-proposal
  (let [g (conj/with-conjugate-proposal shifted :mu [:y1 :y2])
        obs (cm/choicemap :y1 0.3 :y2 -0.2 :z 0.1)]
    (doseq [k (range 5)]
      (let [{tr :trace wt :weight} (gfi/generate g [] obs (rng/key k))
            w (ts/number (tr :w))]
        (is (laws/close? wt (clojure.core/+ (mvn-log-density [w w] [[1.25 1.0] [1.0 1.64]] [0.3 -0.2])
                                            (ts/number (dc/log-prob (d/normal 0.6 1.0) 0.1))))
            "exact given the forward draws of the other choices: log p(ys | w) + log p(z)")))))

(deftest unobserved-children-are-drawn-after-the-latent
  (let [g (conj/with-conjugate-proposal normal-model :mu [:y1 :y2])
        evidence (ts/number (dc/log-prob (d/normal 1.0 (Math/sqrt 4.25)) 0.3))]
    (is (laws/close? (spread g [] (cm/choicemap :y1 0.3) (rng/split (rng/key 7) 5)) 0.0))
    (let [{tr :trace w :weight} (gfi/generate g [] (cm/choicemap :y1 0.3) (rng/key 8))]
      (is (laws/close? w evidence) "the evidence of the observed children")
      (is (= [2] (array/shape (tr :y2)))))))

;; ---------------------------------------------------------------------------
;; In an Unfold
;; ---------------------------------------------------------------------------

;; the next state is the latent itself: later steps condition on its draw
(def level-step
  (gen [t x] (let [x (trace :x (d/normal x 1.0))] (trace :y (d/normal x 0.5)) x)))

;; the next state does not depend on the latent
(def counting-step
  (gen [t s] (let [mu (trace :mu (d/normal s 1.0))] (trace :y (d/normal mu 0.5)) (+ s 1.0))))

(def unfold-ys [0.4 1.3 1.9 3.4])
(def unfold-obs (apply cm/choicemap (mapcat (fn [t y] [[t :y] y]) (range) unfold-ys)))

(deftest in-an-unfold-each-step-is-exact-given-its-state
  (let [exact (comb/unfold (conj/with-conjugate-proposal counting-step :mu [:y] {:check-retval? true}))
        ll (reduce clojure.core/+ (map-indexed (fn [t y] (ts/number (dc/log-prob (d/normal (double t) (Math/sqrt 1.25)) y)))
                                               unfold-ys))]
    (testing "a state free of the latent: the total is exact"
      (doseq [k (range 3)]
        (is (laws/close? (:weight (gfi/generate exact [4 0.0] unfold-obs (rng/key k))) ll)))
      (let [at (fn [t] (cm/choicemap [t :y] (unfold-ys t)))
            state (reduce (fn [s t] (pf/step s [(inc t) 0.0] [diff/unknown-change diff/no-change] (at t) {:key (rng/key t)}))
                          (pf/initialize exact [1 0.0] (at 0) 1 {:key (rng/key 9)})
                          (range 1 4))]
        (is (laws/close? (pf/log-ml-estimate state) ll) "so a particle filter's estimate is exact with one particle")))
    (testing "a state that is the latent: the total is an estimate, which changes with the key"
      (let [chain (comb/unfold (conj/with-conjugate-proposal level-step :x [:y]))]
        (is (< 1 (count (set (map #(ts/number (:weight (gfi/generate chain [4 0.0] unfold-obs (rng/key %)))) (range 3))))))))
    (testing ":check-retval? makes that an error"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"return value depends on the latent"
                            (gfi/generate (comb/unfold (conj/with-conjugate-proposal level-step :x [:y] {:check-retval? true}))
                                          [4 0.0] unfold-obs (rng/key 1)))))))

;; ---------------------------------------------------------------------------
;; The GFI laws (the general set: the internal proposal is not forward sampling)
;; ---------------------------------------------------------------------------

(def law-models
  ;; alternative arguments only add choices: fewer would leave an
  ;; unselected choice to regenerate unvisited, a loud error
  (into (mapv (fn [[name gf args]] [name gf args (case name :beta [[4] [5]] :gamma [[5] [7]] :dirichlet [[7] [9]] [])])
              pairs)
        ;; :z depends on mu through y1 when y1 is left open, so exactness
        ;; is not claimed: the laws hold for the importance weight still
        [[:other-choices (conj/with-conjugate-proposal shifted :mu [:y1 :y2] {:check-exact? false}) [] []]
         [:hierarchy (conj/with-conjugate-proposal hierarchy :theta (sel/select :y)) [] []]
         [:unfold (comb/unfold (conj/with-conjugate-proposal counting-step :mu [:y])) [3 0.0] [[4 0.0] [3 1.0]]]
         [:unfold-of-the-latent (comb/unfold (conj/with-conjugate-proposal level-step :x [:y])) [2 0.0] [[3 0.0]]]]))

(laws/deflaws conjugate-proposal law-models 50 laws/general)
(laws/deflaws conjugate-proposal law-models 50 laws/fixed-support)

(deftest updates-stay-the-wrapper-s
  (let [g (conj/with-conjugate-proposal beta-model :p (sel/select :x :k))
        tr (gfi/simulate g [2] (rng/key 3))]
    (is (identical? g (:gen-fn (:trace (gfi/update tr [2] [diff/no-change] (cm/choicemap :k 2) (rng/key 4))))))
    (is (identical? g (:gen-fn (:trace (gfi/update tr [3] [diff/unknown-change] cm/EMPTY (rng/key 5))))))))

;; ---------------------------------------------------------------------------
;; Mismatched structures are loud: GenMLX's silent wrong marginals
;; ---------------------------------------------------------------------------

(defn- generate-error
  "The message of the error generate throws, or nil."
  [gf args obs]
  (try (gfi/generate gf args obs (rng/key 1)) nil
       (catch clojure.lang.ExceptionInfo e (ex-message e))))

;; The checks that read values (a child's other parameters, the other
;; choices' densities) run on an eager backend, the JVM's, as distributions
;; validate their parameters; a lazy one reads nothing there.

(deftest genmlx-94qc-a-rebound-latent-is-not-the-latent
  ;; the latent's name rebound to mu + 1: GenMLX matched the name and
  ;; dropped the +1 (log N(2; 0, 2) for log N(2; 1, 2))
  (let [m (gen [] (let [mu (trace :mu (d/normal 0.0 1.0))
                        mu (+ mu 1.0)]
                    (trace :y (d/normal (+ mu 0.0) 1.0))))]
    (is (re-find #"the mu of the child at \[:y\] is not the latent at \[:mu\] itself"
                 (generate-error (conj/with-conjugate-proposal m :mu [:y]) [] (cm/choicemap :y 2.0))))))

(deftest genmlx-yy8u-rmy7-a-function-of-the-latent-is-not-the-latent
  ;; yy8u: a mean 100·slope + 0.005·slope², which an affine probe accepted;
  ;; rmy7: an offset, which a Kalman elimination dropped. Neither is the
  ;; latent itself; affine children are not conjugate pairs here (a
  ;; linear-Gaussian chain is gen.state-space/kalman's, with offsets)
  (doseq [[label mean-of] [["quadratic" #(+ (* 100.0 %) (* 0.005 (* % %)))]
                           ["offset" #(+ % 5.0)]
                           ["scaled" #(* 2.0 %)]]]
    (let [m (gen [] (let [slope (trace :slope (d/normal 0.0 10.0))]
                      (trace :y (d/normal (mean-of slope) 1.0))))]
      (is (re-find #"is not the latent at \[:slope\] itself"
                   (generate-error (conj/with-conjugate-proposal m :slope [:y]) [] (cm/choicemap :y 300.0)))
          label))))

(deftest genmlx-5ytq-a-noise-that-depends-on-the-latent-is-loud
  ;; GenMLX's weight was log N(3; 2, 2), the noise pinned at the prior
  ;; mean, where the marginal is -1.7992
  (let [m (gen [] (let [mu (trace :mu (d/normal 2.0 1.0))]
                    (trace :y (d/normal mu (* 0.5 mu)))
                    mu))]
    (when (ts/jvm?)
      (is (re-find #"the std of the child at \[:y\] depends on the latent"
                   (generate-error (conj/with-conjugate-proposal m :mu [:y]) [] (cm/choicemap :y 3.0)))))))

(def jmk4-model
  (gen [] (let [mu (trace :mu (d/normal 0.0 10.0))]
            (trace :y0 (d/laplace mu 1.0))
            (trace :y1 (d/normal mu 1.0)))))

(deftest genmlx-jmk4-a-dependent-outside-the-pair-is-loud
  ;; a constrained laplace beside a conjugate normal child: GenMLX
  ;; eliminated mu anyway (-0.19 nats, -397 with more sites)
  (let [m jmk4-model
        obs (cm/choicemap :y0 1.0 :y1 1.2)]
    (is (re-find #"the child at \[:y0\] is a laplace, not conjugate to the latent's normal"
                 (generate-error (conj/with-conjugate-proposal m :mu [:y0 :y1]) [] obs)))
    (when (ts/jvm?)
      (is (re-find #"the choice at \[:y0\] depends on the latent"
                   (generate-error (conj/with-conjugate-proposal m :mu [:y1]) [] obs))))
    (testing "and an unobserved one"
      (let [m (gen [] (let [mu (trace :mu (d/normal 0.0 10.0))]
                        (trace :z (d/normal mu 1.0))
                        (trace :y (d/normal mu 1.0))))]
        (when (ts/jvm?)
          (is (re-find #"the choice at \[:z\] depends on the latent"
                       (generate-error (conj/with-conjugate-proposal m :mu [:y]) [] (cm/choicemap :y 1.2)))))))
    (testing "and one that depends on an unobserved child"
      (when (ts/jvm?)
        (is (re-find #"the choice at \[:z\] depends on the latent"
                     (generate-error (conj/with-conjugate-proposal shifted :mu [:y1 :y2]) [] (cm/choicemap :y2 0.1 :z 0.3))))))))

(defn- normal-pdf ^double [x m s] (/ (Math/exp (* -0.5 (Math/pow (/ (clojure.core/- x m) s) 2))) (* s (Math/sqrt (* 2 Math/PI)))))

(deftest ^:slow ^:heavy without-the-exactness-check-the-weight-is-still-unbiased
  ;; jmk4's model with its laplace child undeclared and the check off: the
  ;; posterior given y1 alone is not mu's, but the weight is still a correct
  ;; importance weight, E[exp(w)] = p(y0, y1), here by quadrature
  (let [g (conj/with-conjugate-proposal jmk4-model :mu [:y1] {:check-exact? false})
        obs (cm/choicemap :y0 1.0 :y1 1.2)
        density (fn [mu] (* (normal-pdf mu 0.0 10.0) (* 0.5 (Math/exp (clojure.core/- (Math/abs (clojure.core/- 1.0 mu)))))
                            (normal-pdf 1.2 mu 1.0)))
        h 1e-3
        evidence (* h (reduce clojure.core/+ (map #(density (* h %)) (range -80000 80001))))
        ratios (mapv #(Math/exp (clojure.core/- (ts/number (:weight (gfi/generate g [] obs %))) (Math/log evidence)))
                     (rng/split (rng/key 30) 4000))
        m (mean ratios)
        v (/ (reduce clojure.core/+ (map #(Math/pow (clojure.core/- % m) 2) ratios)) (dec (count ratios)))]
    (is (< 1e-4 v) "the weights vary: not exact")
    (is (within-5-se? ratios 1.0 v))))

(deftest genmlx-su6q-every-element-of-a-vector-latent-is-checked
  ;; GenMLX probed a vector latent along the diagonal only. Here a child's
  ;; mean is the latent or not; a covariance that depends on it off its
  ;; diagonal only is read whole
  (let [prior (d/mvnormal (array/array [0.0 1.0]) (array/array mv-prior-cov))
        swapped (gen [] (let [theta (trace :theta prior)]
                          (trace :y (d/mvnormal (array/take theta (array/array [1 0]) 0) (array/array mv-cov-1)))))
        coupled (gen [] (let [theta (trace :theta prior)
                              rho (* 0.1 (* (array/take theta 0) (array/take theta 1)))]
                          (trace :y (d/mvnormal theta (+ (array/eye 2) (* rho (array/array [[0.0 1.0] [1.0 0.0]])))))))
        obs (cm/choicemap :y (array/array [0.4 0.2]))]
    (is (re-find #"the mu of the child at \[:y\] is not the latent"
                 (generate-error (conj/with-conjugate-proposal swapped :theta [:y]) [] obs)))
    (when (ts/jvm?)
      (is (re-find #"the cov of the child at \[:y\] depends on the latent"
                   (generate-error (conj/with-conjugate-proposal coupled :theta [:y]) [] obs))))))

(deftest unsupported-structures-are-loud
  (let [obs (cm/choicemap :y 2.0)
        err (fn [re gf] (is (re-find re (or (generate-error gf [] obs) "")) (str re)))
        model (gen [] (let [mu (trace :mu (d/normal 0.0 1.0))] (trace :y (d/normal mu 1.0))))]
    (testing "a structure that depends on the latent"
      (err #"structure depends on the latent"
           (conj/with-conjugate-proposal (gen [] (let [mu (trace :mu (d/normal 0.0 1.0))]
                                                   (when (array/> mu 0.0) (trace :z (d/normal 0.0 1.0)))
                                                   (trace :y (d/normal mu 1.0))))
                                         :mu [:y])))
    (err #"the latent at \[:nu\] was not visited" (conj/with-conjugate-proposal model :nu [:y]))
    (testing "a child of the prior's own mean, not of the latent"
      (err #"the mu of the child at \[:y\] is not the latent at \[:mu\] itself"
           (conj/with-conjugate-proposal (gen [] (let [m 0.5] (trace :mu (d/normal m 1.0)) (trace :y (d/normal m 1.0))))
                                         :mu [:y])))
    (testing "a child before the latent"
      (let [g (conj/with-conjugate-proposal (gen [] (trace :y (d/normal 0.0 1.0)) (trace :mu (d/normal 0.0 1.0))) :mu [:y])]
        (is (re-find #"the child at \[:y\] comes before the latent" (generate-error g [] cm/EMPTY)))
        (err #"the mu of the child at \[:y\] is not the latent" g)))
    (err #"no child of the latent at \[:mu\] was visited" (conj/with-conjugate-proposal model :mu [:w]))
    (err #"the latent at \[:mu\] is a uniform, which has no conjugate children"
         (conj/with-conjugate-proposal (gen [] (let [mu (trace :mu (d/uniform 0.0 1.0))] (trace :y (d/normal mu 1.0))))
                                       :mu [:y]))
    (testing "choices made by a callee at an address"
      (let [inner (gen [] (let [mu (trace :mu (d/normal 0.0 1.0))] (trace :y (d/normal mu 1.0))))
            outer (gen [] (splice :in inner))]
        (is (re-find #"latent at \[:in :mu\] is made by a generative function the model calls"
                     (generate-error (conj/with-conjugate-proposal outer [:in :mu] [[:in :y]]) [] (cm/choicemap [:in :y] 2.0)))))
      (let [inner (gen [mu] (trace :y (d/normal mu 1.0)))
            outer (gen [] (let [mu (trace :mu (d/normal 0.0 1.0))] (splice :in inner mu)))]
        (is (re-find #"child at \[:in :y\] is made by a generative function the model calls"
                     (generate-error (conj/with-conjugate-proposal outer :mu [:in]) [] (cm/choicemap [:in :y] 2.0))))))
    (err #"needs exact densities"
         (conj/with-conjugate-proposal (gen [] (let [mu (trace :mu (d/normal (untraced (d/normal 0.0 1.0)) 1.0))]
                                                 (trace :y (d/normal mu 1.0))))
                                       :mu [:y]))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"needs a dynamic generative function"
                          (conj/with-conjugate-proposal (comb/unfold level-step) [0 :x] [[0 :y]])))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"is among its children"
                          (conj/with-conjugate-proposal model :mu (sel/select :mu :y))))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"needs the children's addresses"
                          (conj/with-conjugate-proposal model :mu [])))))
