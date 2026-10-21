(ns gen.genjl.oracle
  "Gen.jl 0.4.8's values for the ports that compare with Gen.jl (M5's
   oracle, docs/research/genjl-test-suite.md §5), from the fixtures
   test/fixtures/genjl/generate.jl writes, in gen.clj's terms: keyword
   addresses, and Switch's branch index 0-based.

   - switch-regenerations: Gen.jl's Switch regenerate on recorded traces
     (U1), each {:case :line :draw :argdiffs :selection :old :new :weight},
     with :old and :new {:args :choices :score}. Choices are choice maps,
     argdiffs gen.diff's values and the selection a gen.selection, so a
     record's call is (gfi/regenerate tr (:args new) argdiffs selection key).
   - beta-endpoints: Gen.jl's beta log density at and next to 0 and 1
     (D6), each {:x :alpha :beta :log-prob}.
   - statistic: a statistical testset's quantity over the seeds (1 to 20),
     {:values :mean :sd}, by Gen.jl's file and line:
     (statistic \"train.jl:1\" :theta1), (statistic \"recurse.jl:66\"
     :generate \"(.aa)\")."
  (:require [clojure.edn :as edn]
            [gen.choicemap :as cm]
            [gen.diff :as diff]
            [gen.selection :as sel]))

(defn- fixture [name]
  (edn/read-string (slurp (str "test/fixtures/genjl/" name ".edn"))))

(def ^:private argdiffs {:unknown-change diff/unknown-change :no-change diff/no-change})

(defn- with-choicemap [revision] (update revision :choices cm/from-map))

(def switch-regenerations
  (mapv (fn [record]
          (-> record
              (update :old with-choicemap)
              (update :new with-choicemap)
              (update :argdiffs #(mapv argdiffs %))
              (update :selection #(apply sel/select %))))
        (fixture "switch-regenerate")))

(def beta-endpoints (fixture "beta-endpoints"))

(def ^:private statistics (fixture "statistics"))

(def seeds "The seeds of every statistic." (:seeds statistics))

(defn statistic
  "{:values :mean :sd} of a statistical testset's quantity over the seeds,
   by Gen.jl's file and line and the quantity's path."
  [testset & path]
  (let [s (get-in (:testsets statistics) (into [testset] path))]
    (if (contains? s :values)
      s
      (throw (ex-info (str "The oracle has no statistic " (pr-str (into [testset] path)))
                      {:testset testset :path (vec path)
                       :quantities (into (sorted-map) (update-vals (:testsets statistics) (comp vec sort keys)))})))))
