(ns gen.laws.potential-test
  "The potential contract's laws (gen.laws.potential): correct potentials
   pass, each law fails its planted bug (one that only it can see where
   there is one), sampled tokens for a large vocabulary, vacuous and
   malformed cases, and verify's failures."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.laws :as laws]
            [gen.laws.potential :as potential-laws]
            [gen.llm.potential :as potential]))

(def ^:private n 6)

(defn- repeats? [ctx] (boolean (some (fn [[a b]] (= a b)) (partition 2 1 ctx))))

(def ^:private no-repeats
  "Sequences whose tokens never repeat the one before: 0 or -Inf as a
   prefix, and complete only at even lengths."
  (potential/potential n {:prefix (fn [ctx] (if (repeats? ctx) ##-Inf 0.0))
                          :complete (fn [ctx] (if (and (even? (count ctx)) (not (repeats? ctx))) 0.0 ##-Inf))}))

(defn- cost [ctx] (* -0.1 (reduce + 0 ctx)))

(def ^:private by-length
  "A soft potential: each token costs 0.1 per unit of its value."
  (potential/potential n {:prefix cost :complete (fn [ctx] (- (cost ctx) 1.0))}))

(def ^:private soft (dissoc potential-laws/all :a-zero-prefix-has-zero-extensions))

(def ^:private few {:num-tests 100 :seed 1})

(deftest correct-potentials-satisfy-the-laws
  (is (= {} (potential-laws/verify [[:no-repeats no-repeats n] [:product (potential/product no-repeats by-length) n]] few)))
  (is (= {} (potential-laws/verify [[:by-length by-length n]] few soft)) "a soft potential, without the zero-prefix law")
  (testing "a soft potential's zero-prefix cases are all vacuous"
    (is (= {:vacuous 100 :of 100} (select-keys (:a-zero-prefix-has-zero-extensions (potential-laws/verify [[:by-length by-length n]] few))
                                               [:vacuous :of])))))

(defn- with-next
  "by-length's prefix and complete, with logw-next f."
  [f]
  (potential/potential n {:prefix cost :complete (fn [ctx] (- (cost ctx) 1.0)) :logw-next f}))

(defn- failing [entry laws] (set (keys (potential-laws/verify [entry] few laws))))

(deftest each-law-fails-its-planted-bug
  (testing "next weights: off by 0.5 at a token, at the end, or a row too short"
    (let [law {:l potential-laws/next-weights-are-prefix-differences}]
      (doseq [[what bad] {"one token" (fn [ctx] (let [^floats w (potential/logw-next by-length ctx)] (aset w 3 (float (+ 0.5 (aget w 3)))) w))
                          "the end" (fn [ctx] (let [^floats w (potential/logw-next by-length ctx)] (aset w n (float 0.0)) w))
                          "too short" (fn [ctx] (java.util.Arrays/copyOf ^floats (potential/logw-next by-length ctx) (int n)))}]
        (is (= #{:l} (failing [:bad (with-next bad) n] law)) what))))
  (testing "the factorization: the end's weight off"
    (is (= #{:l} (failing [:bad (with-next (fn [ctx] (let [^floats w (potential/logw-next by-length ctx)] (aset w n (float 0.0)) w))) n]
                          {:l potential-laws/weights-factor-autoregressively}))))
  (testing "a zero prefix, where only its law looks: a complete weight past a repeat, or a prefix that revives"
    (let [complete-anyway (potential/potential n {:prefix (fn [ctx] (if (repeats? ctx) ##-Inf 0.0))
                                                  :complete (fn [ctx] (if (even? (count ctx)) 0.0 ##-Inf))})
          ends-in-a-repeat (fn [ctx] (if (and (<= 2 (count ctx)) (= (peek ctx) (peek (pop ctx)))) ##-Inf 0.0))
          revives (potential/potential n {:prefix ends-in-a-repeat :complete ends-in-a-repeat})]
      (is (= #{:a-zero-prefix-has-zero-extensions} (failing [:complete-anyway complete-anyway n] potential-laws/all)))
      (is (= #{:a-zero-prefix-has-zero-extensions} (failing [:revives revives n] potential-laws/all)))))
  (testing "purity: next weights advanced from the last context weighed, as if it were a prefix of the next"
    ;; no-repeats' row (-Inf at the last token) from a state: the last
    ;; token of the last context weighed, moved on by the tokens past its
    ;; length; right only when that context is a prefix of this one, as
    ;; it is along every walk the other laws take
    (let [state (atom [])
          advanced (fn [ctx]
                     (let [last-ctx @state
                           last-token (if (<= (count last-ctx) (count ctx))
                                        (or (peek (subvec ctx (count last-ctx))) (peek last-ctx))
                                        (peek ctx))]
                       (reset! state ctx)
                       (let [w (float-array (inc n))]
                         (when last-token (aset w (int last-token) Float/NEGATIVE_INFINITY))
                         (aset w n (if (even? (count ctx)) (float 0.0) Float/NEGATIVE_INFINITY))
                         w)))
          p (potential/potential n {:prefix #(potential/prefix no-repeats %) :complete #(potential/complete no-repeats %)
                                    :logw-next advanced})]
      (is (= #{:weights-are-pure-functions-of-the-context} (failing [:advanced p n] potential-laws/all))))))

(deftest a-large-vocabulary-compares-a-sample
  ;; 2000 tokens, 24 compared at a context, half among those with weight
  (let [big 2000
        low? (fn [ctx] (every? #(< (long %) 40) ctx))
        sparse (potential/potential big {:prefix (fn [ctx] (if (low? ctx) (* -0.01 (count ctx)) ##-Inf))
                                         :complete (fn [ctx] (if (low? ctx) -1.0 ##-Inf))})
        entry [:sparse sparse big {:tokens 24 :max-length 4}]]
    (is (= {} (potential-laws/verify [entry] {:num-tests 30 :seed 2})))
    (testing "a planted bug among the tokens with weight is found"
      (let [bad (potential/potential big {:prefix #(potential/prefix sparse %) :complete #(potential/complete sparse %)
                                          :logw-next (fn [ctx] (let [^floats w (potential/logw-next sparse ctx)]
                                                                 (doseq [t (range 0 40 2)] (aset w (int t) (float 0.0)))
                                                                 w))})]
        (is (contains? (failing [:bad bad big {:tokens 24 :max-length 4}] potential-laws/all)
                       :next-weights-are-prefix-differences))))))

(deftest failures-and-misuse
  (testing "a failure says what it compared, and the case by the potential's label"
    (let [bad (with-next (fn [ctx] (let [^floats w (potential/logw-next by-length ctx)] (aset w 3 (float 9.0)) w)))
          f (:next-weights-are-prefix-differences (potential-laws/verify [[:bad bad n]] few soft))]
      (is (= '(next-weights-match? p n-vocab ctx tokens) (:check f)))
      (is (= 3 (:token f)))
      (is (== 9.0 (:logw-next f)))
      (is (< (Math/abs (+ 0.3 (double (:prefix-difference f)))) 1e-12))
      (is (= :bad (get-in f [:case :potential])))
      (is (= 1 (:seed f)))))
  (testing "a potential with no weight makes every case vacuous"
    (let [nothing (potential/potential n {:prefix (constantly ##-Inf) :complete (constantly ##-Inf)})]
      (is (= #{:vacuous :of :max-vacuous :seed}
             (set (keys (:weights-factor-autoregressively (potential-laws/verify [[:nothing nothing n]] few))))))))
  (testing "malformed entries are loud"
    (doseq [potentials [[] [[:p no-repeats]] [[:p "not a potential" n]] [[:p no-repeats 0]] [[:p no-repeats n :opts]] :p]]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"potential laws: "
                            (potential-laws/verify potentials few))
          (pr-str potentials)))))

(potential-laws/deflaws no-repeats-laws [[:no-repeats no-repeats n]] 50)

(deftest deflaws-defines-a-spec-per-law
  (is (= #{"no-repeats-laws-next-weights-are-prefix-differences" "no-repeats-laws-weights-factor-autoregressively"
           "no-repeats-laws-a-zero-prefix-has-zero-extensions" "no-repeats-laws-weights-are-pure-functions-of-the-context"}
         (set (keep (fn [[s v]] (when (::laws/law (meta v)) (name s))) (ns-interns 'gen.laws.potential-test)))))
  (is (true? (:pass? (no-repeats-laws-next-weights-are-prefix-differences 20 :seed 3)))))
