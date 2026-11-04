(ns gen.codegen.explain-test
  "Errors an LLM can act on (gen.codegen.explain): a name that is not there
   gets the alias's closest real names, a wrong arity the arglists, a
   macro's spec failure one line a problem with a hint for defn in let
   bindings, Math's methods the advice to use gen.array's; clj-kondo's
   findings (with clj-kondo on the classpath: the :codegen alias) name
   invented vars without flagging gen.array's and gen.dist's real ones."
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [gen.codegen.candidate :as c]
            [gen.codegen.explain :as x]
            [gen.codegen.session :as session]
            [gen.gfi :as gfi]
            [gen.random :as rng]))

(defn- error-of
  "The explained error of evaluating code and simulating its model, or nil."
  [code]
  (try (let [[_ m] (c/evaluate (session/read-forms code))]
         (gfi/simulate m [] (rng/key 1))
         nil)
       (catch Throwable e (x/explain e nil))))

(deftest a-name-that-is-not-there
  (let [e (error-of "(def model (gen [] (trace :a (d/point 0.3))))")]
    (is (str/includes? e "No such var: d/point"))
    (is (str/includes? e "There is no d/point"))
    (is (re-find #"d/ is gen.dist, whose names are: .*\bnormal\b" e)))
  (testing "a misspelled trace is told it is trace"
    (is (re-find #"Did you mean trace" (error-of "(def model (gen [] (traec :a (d/normal 0 1))))")))))

(deftest a-wrong-arity-gets-the-arglists
  (is (str/includes? (error-of "(def model (gen [] (array/sin 1 2)))") "gen.array/sin takes (sin x).")))

(deftest a-spec-failure-briefly
  (let [e (error-of "(def model (gen [] (let [x (trace :x (d/normal 0 1)) (defn f [] 1)] x)))")]
    (is (str/includes? e "did not conform to spec"))
    (is (re-find #"In \(clojure.core/let …\), at bindings: expected .*even-number-of-forms" e))
    (is (str/includes? e "define functions before the let"))))

(deftest math-methods-point-to-gen-array
  (is (str/includes? (error-of "(def model (gen [] (Math/sin 1 2)))") "use gen.array's: array/sin")))

(deftest clj-kondo-names-invented-vars-only
  (if-not (try (requiring-resolve 'clj-kondo.core/run!) (catch Exception _ nil))
    (println "gen.codegen.explain-test: clj-kondo is not on the classpath (the :codegen alias); lint skips")
    (let [l (x/lint "(def model (gen [xs] (let [a (trace :a (d/point 0.3)) b (array/sin a)] (trace :y (d/normal a b)))))")]
      (is (str/includes? l "line 1: There is no d/point"))
      (is (not (str/includes? l "array/sin")) "gen.array's real names are not flagged")
      (is (not (str/includes? l "d/normal")))
      (testing "a session's definitions are known"
        (is (nil? (x/lint "(helper 1)" "(defn helper [x] x)")))
        (is (str/includes? (x/lint "(helpr 1)" "(defn helper [x] x)") "line 1: Unresolved symbol: helpr"))))))
