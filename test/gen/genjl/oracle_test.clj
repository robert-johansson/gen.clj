(ns gen.genjl.oracle-test
  "The oracle's fixtures (gen.genjl.oracle) carried into gen.clj's terms:
   Gen.jl's scores of the recorded Switch traces are gen.clj's assess of
   their choices on their arguments, so the addresses and the 0-based
   branch index are right; beta's log densities are gen.clj's wherever D6
   does not apply (and the backend's floats hold x); and every statistic
   holds a value per seed, with their mean and sd."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.dist :as d]
            [gen.genjl.oracle :as oracle]
            [gen.genjl.support :as genjl]
            [gen.genjl.switch-test :as switch]
            [gen.gfi :as gfi]
            [gen.test-support :as ts]))

(def ^:private gen-fns
  "The gen.clj port's models of each case's generative function."
  {:bare-same switch/sc :bare-switch switch/sc :bare-doubled switch/sc
   :nested-drop switch/bam2 :nested-drop-z switch/bam2 :nested-mh switch/bam2})

(deftest switch-regenerations
  (is (= (zipmap (keys gen-fns) (repeat 10)) (frequencies (map :case oracle/switch-regenerations))))
  (doseq [{:keys [case draw] :as record} oracle/switch-regenerations
          side [:old :new]]
    (let [{:keys [args choices score]} (side record)]
      (is (genjl/approx? score (:weight (gfi/assess (gen-fns case) args choices)))
          (str case ", draw " draw ", " (name side))))))

(defn- d6?
  "Where gen.clj is exact and Gen.jl finite (D6): at 0 with alpha below 1, or
   at 1 with beta below 1."
  [{:keys [x alpha beta]}]
  (or (and (== x 0.0) (< alpha 1.0)) (and (== x 1.0) (< beta 1.0))))

(defn- representable?
  "Is x a value of the backend's floats? 1 - eps is not a float32."
  [{:keys [x]}]
  (or (not (ts/f32?)) (== x (double (float x)))))

(deftest beta-endpoints
  (is (= 20 (count oracle/beta-endpoints)))
  (doseq [{:keys [x alpha beta log-prob] :as point} (filter representable? (remove d6? oracle/beta-endpoints))]
    (is (genjl/approx? log-prob (genjl/logpdf d/beta x alpha beta)) (pr-str point)))
  (testing "D6's points: finite in Gen.jl, +Inf in gen.clj"
    (doseq [{:keys [x alpha beta log-prob] :as point} (filter d6? oracle/beta-endpoints)]
      (is (and (Double/isFinite log-prob) (= ##Inf (genjl/logpdf d/beta x alpha beta))) (pr-str point)))))

(defn- mean [xs] (/ (reduce + xs) (count xs)))

(defn- sd [xs]
  (let [m (mean xs)]
    (Math/sqrt (/ (reduce + (map #(let [e (- % m)] (* e e)) xs)) (dec (count xs))))))

(deftest statistics
  (is (= (range 1 21) oracle/seeds))
  (doseq [path (concat (for [q [:theta1 :theta2 :theta3 :theta4 :theta5]] ["train.jl:1" q])
                       (for [q [:score :theta :log-std :batched-score]] ["train.jl:126" q])
                       (for [q [:slope-mu :slope-log-std :intercept-mu :intercept-log-std]] ["variational.jl:1" q])
                       [["particle_filter.jl:96" :log-ml] ["particle_filter.jl:146" :log-ml]]
                       (for [op [:generate :simulate] s ["(.aa)" "(-b(-aa)b)"]] ["recurse.jl:66" op s]))]
    (let [{:keys [values] :as s} (apply oracle/statistic path)]
      (is (and (= 20 (count values)) (genjl/approx? (:mean s) (mean values)) (genjl/approx? (:sd s) (sd values)))
          (pr-str path))))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"no statistic" (oracle/statistic "train.jl:1" :theta6)))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"no statistic" (oracle/statistic "recurse.jl:66" :generate))
      "a path that ends above a quantity"))
