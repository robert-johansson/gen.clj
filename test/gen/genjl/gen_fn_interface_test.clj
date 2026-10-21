(ns gen.genjl.gen-fn-interface-test
  "Gen.jl 0.4.8's test/gen_fn_interface.jl: update and regenerate without
   arguments keep the trace's own, in the dynamic and the static language
   (docs/research/genjl-test-suite.md §2)."
  (:refer-clojure :exclude [+])
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :refer [+]]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.random :as rng]
            [gen.selection :as sel]
            [gen.test-support :as ts]))

(def f-dynamic (gen [x y] (let [z (trace :z (d/normal 0 1))] (+ x y z))))

(def f-static (gen ^:static [x y] (let [z (trace :z (d/normal 0 1))] (+ x y z))))

(def ^:private languages {:dynamic f-dynamic :static f-static})

(deftest update-shorthand-assuming-unchanged-args
  ;; gen_fn_interface.jl:12, once per modeling language
  (doseq [[lang f] languages]
    (testing lang
      (let [trace0 (gfi/simulate f [5 6] (rng/key 1))
            {trace1 :trace discard :discard} (gfi/update trace0 (cm/choicemap :z 0.0) (rng/key 2))]
        (is (= [5 6] (:args trace1)))
        (is (= 0.0 (ts/host (trace1 :z))))
        (is (some #{:z} (map first (cm/values-shallow discard))))))))

(deftest regenerate-shorthand-assuming-unchanged-args
  ;; gen_fn_interface.jl:25, once per modeling language
  (doseq [[lang f] languages]
    (testing lang
      (let [trace0 (gfi/simulate f [5 6] (rng/key 1))
            {trace1 :trace} (gfi/regenerate trace0 (sel/select :z) (rng/key 2))]
        (is (= [5 6] (:args trace1)))
        (is (not= (ts/host (trace0 :z)) (ts/host (trace1 :z))))))))
