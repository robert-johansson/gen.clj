(ns gen.genjl.tilde-sugar-test
  "Gen.jl 0.4.8's test/tilde_sugar.jl. Gen.jl's x ~ dist is (trace :x dist)
   and {:a => :b} ~ gf(args) is (splice [:a :b] gf args); a static body is
   one let (docs/research/genjl-test-suite.md §2)."
  (:refer-clojure :exclude [+ -])
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array :refer [+ -]]
            [gen.dist :as d]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.random :as rng]
            [gen.test-support :as ts]))

(deftest tilde-syntax-smoke-test-dynamic
  ;; tilde_sugar.jl:8. {:coin => i} is [:coin (dec i)]. The key gives two
  ;; heads, so the score tells log p from log(1 - p), and their sum is
  ;; Julia's, the integer 2 (booleans count, D9)
  (let [foo (gen [a b]
              (let [p (trace :p (d/beta a b))]
                (+ (trace [:coin 0] (d/bernoulli p)) (trace [:coin 1] (d/bernoulli p)))))
        tr (gfi/simulate foo [1 1] (rng/key 3))
        coins (ts/host [(tr [:coin 0]) (tr [:coin 1])])
        p (ts/host (tr :p))]
    (is (== (count (filter true? coins)) (ts/host (:retval tr))))
    (is (ts/close? (:score tr) (reduce + (map #(if % (array/log p) (array/log (- 1 p))) coins))))))

(def ^:private bar (gen ^:static [r] (let [a (trace :a (d/normal 0 1))] r)))

(deftest tilde-syntax-smoke-test-static
  ;; tilde_sugar.jl:19. Gen.jl's Float64 is the backend's float dtype
  (let [foo (gen ^:static []
              (let [x (splice :x bar 1)
                    yz (splice [:y :z] bar 2)
                    u (trace :u (d/normal 0 1))
                    _ (trace [:v :w] (d/normal 0 1))
                    retrate 7]
                (trace :ret (d/poisson retrate))))
        tr (gfi/simulate foo [] (rng/key 1))]
    (testing "random choices"
      (doseq [path [:u [:v :w] [:x :a] [:y :z :a]]]
        (is (= (ts/float-dtype) (array/dtype (tr path))) (str path))))
    (testing "auxiliary state: a call's return value"
      (is (= 1 (tr :x)))
      (is (= 2 (tr [:y :z]))))
    (testing "return value"
      (is (= (ts/host (:retval tr)) (ts/host (tr :ret)))))))

(deftest tilde-syntax-preserved-in-quoted-expressions
  ;; tilde_sugar.jl:66, once per modeling language: quoted data is data
  (doseq [[lang tilde-expr] {:dynamic (gen [] '(trace :x (d/normal 0 1)))
                             :static (gen ^:static [] (let [] '(trace :x (d/normal 0 1))))}]
    (testing lang
      (is (= '(trace :x (d/normal 0 1)) (tilde-expr))))))

;; Not ported (n/a, genjl-test-suite.md §2): tilde_sugar.jl:50, the output
;; of desugar_tildes, Julia's macro internals.
