(ns gen.codegen.grade-test
  "The codegen bake-off's grader (genclj-ihc4): a reply's code is found and
   its forbidden names named; every task's answer passes every stage; a
   candidate with a planted fault stops at the stage that is to catch it
   (an impure one draws law warnings); and the log marginal likelihood
   estimator agrees with exact values: the HMM's by enumerating its 64
   paths, the walk's and the drift's by Kalman filtering (gen.state-space),
   as the book computes them. Planted against the estimator (2026-10-31):
   one observation dropped from the walk's data moves its estimate 1.3
   nats, past the band."
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [gen.choicemap :as cm]
            [gen.codegen.candidate :as c]
            [gen.codegen.grade :as g]
            [gen.codegen.tasks :as tasks]
            [gen.gfi :as gfi]
            [gen.state-space :as ss]))

(deftest a-reply's-code-is-its-last-fenced-block
  (is (= "(def model 2)\n" (c/code-of "Here:\n```clojure\n(def model 1)\n```\nbetter:\n```\n(def model 2)\n```\n")))
  (is (= "(def model 1)" (c/code-of "(def model 1)")))
  (is (= "(def model\n  (gen []\n" (c/code-of "Here it is:\n```clojure\n(def model\n  (gen []\n"))
      "a block cut off before its closing fence runs to the reply's end"))

(deftest forbidden-names-are-named
  (is (= [] (c/forbidden (c/read-forms (:answer (tasks/by-id :changepoint))))))
  (is (= '[.exec Runtime/getRuntime slurp]
         (c/forbidden (c/read-forms "(def x (slurp \"f\")) (.exec (Runtime/getRuntime) \"ls\")"))))
  (is (= '[java.io.File. spit] (c/forbidden (c/read-forms "(spit (java.io.File. \"f\") 1)"))))
  (is (thrown? RuntimeException (c/read-forms "#=(+ 1 2)")) "no read-time evaluation"))

(defn- stopped-at
  "The stage a grading stopped at, :none when every stage passed."
  [r]
  (or (:stage (first (remove :ok (:stages r)))) :none))

(deftest ^:slow every-answer-passes-every-stage
  (doseq [t tasks/tasks]
    (testing (:id t)
      (let [r (g/grade t (:answer t))]
        (is (= 7 (:passed r)) (pr-str (:stages r)))
        (is (not (:tainted? r)))
        (is (= 0.0 (:gap (:log-ml r))) "an answer explains the data as the reference does")))))

(def ^:private line (:answer (tasks/by-id :line)))

(def ^:private planted
  "[stage it stops at, task, code]"
  [[:read :line "(def model (gen [xs] (trace :slope (d/normal 0 5))"]
   [:safe :line (str "(def notes (slurp \"/etc/hosts\"))\n" line)]
   [:evaluates :line (str/replace line "(def model" "(def modle")]
   [:evaluates :line "(def model 42)"]
   [:simulates :line (str/replace line "0.3)))" "-0.3)))")]
   [:accepts-data :line (str/replace line "(trace [:y i]" "(trace [:obs i]")]
   [:explains :line (str/replace line "(+ (* slope x) intercept)" "intercept")]])

(deftest ^:slow a-planted-fault-stops-at-its-stage
  (doseq [[stage id code] planted]
    (testing [stage code]
      (is (= stage (stopped-at (g/grade (tasks/by-id id) code {:gate-explains? true :repair-delimiters? false}))))))
  (testing "an evaluation error names its cause, not only where it happened"
    (let [r (g/grade (tasks/by-id :line) (str/replace line "(d/normal 0 5)" "(d/point 0.3)"))]
      (is (re-find #"Syntax error compiling.*Caused by: .*d/point" (:error (first (remove :ok (:stages r))))))))
  (testing "delimiters rebuilt from the indentation: a closing parenthesis short, or one too many"
    (doseq [code [(subs line 0 (dec (count line))) (str line ")")]]
      (let [r (g/grade (tasks/by-id :line) code)]
        (is (= 7 (:passed r)))
        (is (:repaired-delimiters (first (:stages r)))))))
  (testing "without the repair, they do not read"
    (is (= :read (stopped-at (g/grade (tasks/by-id :line) (subs line 0 (dec (count line))) {:repair-delimiters? false})))))
  (testing "a leading ns form (the prelude copied) is dropped"
    (let [r (g/grade (tasks/by-id :line) (str "(ns candidate (:require [gen.dist :as d]))\n" line))]
      (is (= 7 (:passed r)))
      (is (:dropped-ns (first (:stages r))))))
  (testing "without the gate, a model that explains the data badly passes with a warning"
    (let [r (g/grade (tasks/by-id :line) (str/replace line "(+ (* slope x) intercept)" "intercept"))]
      (is (= 7 (:passed r)))
      (is (some #(re-find #"nats worse" %) (:warnings r)))
      (is (< (:gap (:log-ml r)) -100))))
  (testing "an impure model (clojure.core's rand) stops at the laws"
    (let [t (tasks/by-id :spread)
          r (g/grade t (str/replace (:answer t) "(d/normal mu " "(d/normal (+ mu (rand)) "))]
      (is (= :laws (stopped-at r)))
      (is (re-find #"score-is-the-assessed-density" (:error (first (remove :ok (:stages r))))))))
  (testing "every answer passes the laws, the large-score ones included"
    (doseq [id [:line :sine :line-or-sine :outliers]]
      (is (not= :laws (stopped-at (g/grade (tasks/by-id id) (:answer (tasks/by-id id)) {:law-tests 10}))) id))))

(defn- series [ys] (apply cm/choicemap (interleave (range) ys)))

(defn- ys-of [task]
  (mapv (fn [t] (cm/get-choice (:observations task) [t :y])) (range (count (cm/leaves (:observations task))))))

(def ^:private walk-params {:transition 1.0 :transition-cov 0.25 :emission 1.0 :emission-cov 1.0})

(defn- exact-log-ml [id]
  (let [t (tasks/by-id id)
        ys (ys-of t)]
    (case id
      :walk (array/item (:weight (gfi/assess ss/kalman [6 {:mean 0.0 :cov 0.25} walk-params] (series ys))))
      :drift (let [controls (second (:args t))]
               (array/item (:weight (gfi/assess ss/kalman
                                                [16 {:mean (first controls) :cov 0.25}
                                                 (mapv #(assoc walk-params :transition-offset %) controls)]
                                                (series ys)))))
      :hmm (let [[_ model] (c/evaluate (c/read-forms (:answer t)))
                 paths (reduce (fn [ps _] (for [p ps s [0 1]] (conj p s))) [[]] ys)
                 joint (fn [path] (reduce (fn [m [i s]] (cm/set-choice m [i :z] s)) (:observations t) (map-indexed vector path)))]
             (array/item (array/logsumexp (array/array (mapv #(array/item (:weight (gfi/assess model (:args t) (joint %)))) paths))))))))

(deftest ^:slow the-estimator-agrees-with-exact-values
  (is (< (abs (- (exact-log-ml :drift) -31.9447)) 1e-4) "the book's value")
  (doseq [id [:walk :hmm :drift]]
    (testing id
      (let [t (tasks/by-id id)
            [_ model] (c/evaluate (c/read-forms (:answer t)))
            {:keys [median estimates]} (g/log-ml model t g/defaults)
            m (/ (reduce + estimates) (count estimates))
            sd (Math/sqrt (/ (reduce + (map #(let [d (- % m)] (* d d)) estimates)) (dec (count estimates))))
            band (max 0.05 (* 5 (/ sd (Math/sqrt (count estimates)))))]
        (is (< (abs (- median (exact-log-ml id))) band)
            (pr-str {:exact (exact-log-ml id) :estimates estimates :band band}))))))
