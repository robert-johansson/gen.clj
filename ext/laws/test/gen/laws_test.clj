(ns gen.laws-test
  "The laws' user API (ext/laws): verify's readable failures, check's
   arities, a reader's own law with checks and vacuous-on-unvisitable,
   deflaws' specs (silent, a fresh seed unless given one), the sets
   (estimated, implementation), and the module's independence from gen.clj's
   test/. The laws themselves are tested by every suite that runs them, each
   against its planted bugs."
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.properties :as prop]
            [gen.array :as array]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.laws :as laws]
            [gen.random :as rng]
            [gen.selection :as sel]))

(def ^:private measurement
  (gen [mu]
    (let [x (trace :x (d/normal mu 1))]
      (trace :y (d/normal x 0.5)))))

(def ^:private models [[:measurement measurement [0.0] [[1.5] [-2.0]]]])

(def ^:private mixture
  "A component drawn untraced: its score is an estimate, so the general
   laws' comparisons of separate runs fail."
  (gen []
    (let [i (untraced (d/categorical [0.2 0.5 0.3]))]
      (trace :a (d/normal (array/take (array/array [-2.0 0.0 3.0]) i) 1)))))

(def ^:private mixtures [[:mixture mixture [] []]])

(def ^:private few {:num-tests 20 :seed 1})

(deftest verify-returns-the-failures
  (testing "none for a correct generative function"
    (is (= {} (laws/verify models few)))
    (is (= {} (laws/verify models few laws/fixed-support))))
  (testing "a failure names its check and the values it compared, and the smallest case"
    (let [fs (laws/verify mixtures few laws/general)
          f (:score-is-the-assessed-density fs)]
      (is (sorted? fs))
      (is (= '(consistent? (gfi/simulate gf args (rng/key seed))) (:check f)))
      (is (every? #(contains? f %) [:score :assessed :retval :assessed-retval]) "the explain's values")
      (is (not (laws/close? (:score f) (:assessed f))))
      (is (= :mixture (get-in f [:case :model])))
      (is (int? (get-in f [:case :key-seed])))
      (is (= 1 (:seed f)))
      (testing "a call's check gives the values of its arguments"
        (let [g (:generate-with-every-choice-is-deterministic fs)]
          (is (= '(close? w (:score tr)) (:check g)))
          (is (= '#{w (:score tr)} (set (keys (:values g)))))))))
  (testing "a run's seed, drawn from the clock, reproduces it"
    (let [s (:seed (first (vals (laws/verify mixtures 20 laws/general))))]
      (is (int? s))
      (is (= (laws/verify mixtures {:num-tests 20 :seed s} laws/general)
             (laws/verify mixtures {:num-tests 20 :seed s} laws/general))))))

(deftest verify-reports-an-error-and-a-vacuous-law
  (testing "an error: its message, data and exception"
    (let [boom (gen [mu]
                 (let [x (trace :x (d/normal mu 1))]
                   (when (< 0.5 (array/item x)) (throw (ex-info "boom" {:x x})))
                   x))
          f (:s (laws/verify [[:boom boom [0.0] []]] {:num-tests 50 :seed 2} {:s laws/score-is-the-assessed-density}))]
      (is (= "boom" (:error f)))
      (is (contains? (:data f) :x))
      (is (instance? clojure.lang.ExceptionInfo (:exception f)))
      (is (= :boom (get-in f [:case :model])))))
  (testing "a law whose every case is vacuous fails with the counts"
    (let [always-vacuous (fn [cases] (prop/for-all [_ cases] laws/vacuous))]
      (is (= {:vacuous 20 :of 20 :max-vacuous laws/max-vacuous :seed 1}
             (:v (laws/verify models few {:v always-vacuous}))))
      (is (= {} (laws/verify models (assoc few :max-vacuous 1.0) {:v always-vacuous}))))))

(deftest misuse-is-loud
  (testing "models that are not model entries"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"models are a non-empty vector of model entries"
                          (laws/verify measurement)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"models are a non-empty vector of model entries"
                          (laws/cases []))))
  (testing "options that are not options: a law set in their place"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"unknown options .*a law set comes after them"
                          (laws/verify models laws/general)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"options are a number of cases or a map"
                          (laws/verify models :many))))
  (testing "a law set that is not a map of laws, to verify or deflaws"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"a law set is a map from a law's name to the law"
                          (laws/verify models few laws/score-is-the-assessed-density)))
    (let [e (try (macroexpand '(gen.laws/deflaws one-law models 10 gen.laws/score-is-the-assessed-density))
                 (catch Exception e e))]
      (is (some #(re-find #"a law set is a map" (or (ex-message %) "")) [e (ex-cause e)])))))

(deftest check-has-arities
  (let [p (laws/score-is-the-assessed-density (laws/cases models))
        seen #(select-keys % [:pass? :num-tests :seed :vacuous])]
    (is (= {:pass? true :num-tests 100 :vacuous 0} (dissoc (seen (laws/check p)) :seed)))
    (is (= 10 (:num-tests (laws/check 10 p))))
    (is (= {:pass? true :num-tests 10 :seed 3 :vacuous 0}
           (seen (laws/check 10 p :seed 3))
           (seen (laws/check 10 p {:seed 3}))))
    (testing "and nothing printed"
      (is (= "" (with-out-str (laws/check 10 p)))))))

(defn- unvisited-constraints
  "An update of a simulated trace with a constraint at an address the
   measurement model does not have."
  [seed]
  (let [tr (gfi/simulate measurement [0.0] (rng/key seed))]
    (gfi/update tr (cm/choicemap :z 1.0) (rng/key (inc seed)))))

(defn- own-law
  "A reader's law: an update with an address the model lacks is vacuous,
   and a trace's projection on every address is its score."
  [cases]
  (prop/for-all [[[_ gf args] seed] cases]
    (let [tr (gfi/simulate gf args (rng/key seed))
          r (laws/vacuous-on-unvisitable (unvisited-constraints seed))]
      (laws/checks r (laws/close? (gfi/project tr sel/all) (:score tr))))))

(deftest a-readers-own-law
  (testing "vacuous-on-unvisitable: gfi/unvisited's error is a vacuous case"
    (is (laws/vacuous? (laws/vacuous-on-unvisitable (unvisited-constraints 1))))
    (is (laws/unvisitable? (gfi/unvisited :update [[:z]])))
    (is (not (laws/unvisitable? (ex-info "update: something else" {}))))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"something else"
                          (laws/vacuous-on-unvisitable (throw (ex-info "something else" {}))))))
  (testing "checks: a case is checked when one of its checks is not vacuous"
    (is (= {} (laws/verify models few {:own own-law}))))
  (testing "checks: a failing check is named, with its values"
    (let [wrong (fn [cases]
                  (prop/for-all [[[_ gf args] seed] cases]
                    (let [tr (gfi/simulate gf args (rng/key seed))]
                      (laws/checks (laws/close? (:score tr) 0.0)))))
          f (:wrong (laws/verify models few {:wrong wrong}))]
      (is (= '(laws/close? (:score tr) 0.0) (:check f)))
      (is (= '[(:score tr)] (keys (:values f))) "constants are not shown"))))

;; deflaws over an expression of sets; its specs print nothing. Planted bug:
;; specs given defspec's options (ctc/process-options, whose reporter prints
;; each run's result map) fail it, called and run as tests.
(laws/deflaws measured models 10 (merge laws/general laws/fixed-support))

(deftest deflaws-defines-silent-specs
  (let [specs (filter (comp ::laws/law meta) (vals (ns-interns 'gen.laws-test)))]
    (is (= (+ (count laws/general) (count laws/fixed-support)) (count specs))))
  (is (str/includes? (:doc (meta #'measured-score-is-the-assessed-density)) "fresh seed"))
  (is (= "" (with-out-str (measured-score-is-the-assessed-density))) "called with no arguments")
  (is (= "" (with-out-str (clojure.test/test-var #'measured-score-is-the-assessed-density))) "run as a test")
  (is (= (select-keys (measured-update-weight-is-the-projection-change 10 :seed 4) [:pass? :num-tests :seed :vacuous])
         (select-keys (measured-update-weight-is-the-projection-change 10 {:seed 4}) [:pass? :num-tests :seed :vacuous])
         {:pass? true :num-tests 10 :seed 4 :vacuous 0})))

(def ^:private forever
  "A return value no identity update can compare in full (an infinite
   seq): the update reports unknown-change, which Def 2.3.2 allows."
  (gen [] (trace :x (d/normal 0 1)) (repeat :x)))

(deftest the-sets
  (testing "estimated is encapsulated without the same-key score"
    (is (= (dissoc laws/encapsulated :score-is-assessed-on-the-same-key) laws/estimated))
    (is (contains? laws/encapsulated :score-is-assessed-on-the-same-key)))
  (testing "the identity update's no-change is gen.clj's implementation property, outside all"
    (is (= [:identity-update-reports-no-change] (keys laws/implementation)))
    (is (not-any? #(contains? % :identity-update-reports-no-change)
                  [laws/general laws/forward-sampling laws/all laws/encapsulated]))
    (is (= {} (laws/verify [[:forever forever [] []]] few)) "a correct generative function passes all")
    (is (= [:identity-update-reports-no-change] (keys (laws/verify [[:forever forever [] []]] few laws/implementation)))
        "and fails the implementation property")))

(deftest the-module-needs-no-test-code
  ;; ext/laws is a dependency of its own: gen.laws requires clojure's,
  ;; test.check's and gen.clj's src/ namespaces only. Planted bug: a require
  ;; of gen.test-support (test/) fails it.
  (let [[_ _ & clauses] (with-open [r (java.io.PushbackReader. (io/reader "ext/laws/src/gen/laws.clj"))] (read r))
        required (for [clause clauses :when (and (seq? clause) (= :require (first clause)))
                       spec (rest clause)]
                   (if (vector? spec) (first spec) spec))
        in-src? #(.exists (io/file "src" (str (-> (name %) (str/replace "." "/") (str/replace "-" "_")) ".clj")))]
    (is (seq required))
    (is (empty? (remove #(or (str/starts-with? (name %) "clojure.") (in-src? %)) required))))
  (testing "its deps.edn names the test.check the :laws and :test aliases give the battery"
    (let [module (edn/read-string (slurp "ext/laws/deps.edn"))
          root (edn/read-string (slurp "deps.edn"))
          version #(get-in % ['org.clojure/test.check :mvn/version])]
      (is (= (version (:deps module))
             (version (get-in root [:aliases :laws :extra-deps]))
             (version (get-in root [:aliases :test :extra-deps]))))
      (is (= {:local/root "../.."} (get-in module [:deps 'gen.clj/gen.clj]))))))
