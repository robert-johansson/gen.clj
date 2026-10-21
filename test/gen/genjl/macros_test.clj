(ns gen.genjl.macros-test
  "Gen.jl 0.4.8's test/dsl/macros.jl: user macros in a body, expanding to a
   distribution or to a trace call (docs/research/genjl-test-suite.md §2),
   in the dynamic language (line 38) and the static one (line 122), whose
   analysis expands a binding's head (G11)."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.dist :as d]
            [gen.dynamic :refer [gen]]
            [gen.dynamic.static :as static]
            [gen.gfi :as gfi]
            [gen.random :as rng]
            [gen.test-support :as ts]))

(defmacro gaussian-noise-around
  "macros.jl:2's @gaussian_noise_around: (d/normal x 1.0)."
  [x]
  `(d/normal ~x 1.0))

(defmacro traced-gaussian-noise-around
  "macros.jl:7's @traced_gaussian_noise_around: a choice at addr, made by
   the trace of the body the macro expands in."
  [x addr]
  `(~'trace ~addr (d/normal ~x 1.0)))

(defmacro ^:private foos
  "macros.jl's foo1, foo2 and foo3, {name gen}, each gen on params, whose
   metadata says whether it is static (^:static []). foo1's y ~ @m x and
   foo2's y = @trace(@m(x), :y) are one form here, (trace :y (m x)): foo1
   returns it, foo2 binds it."
  [params]
  (into {} (for [[name body] '{:foo1 (let [x (trace :x (d/exponential 1))]
                                       (trace :y (gaussian-noise-around x)))
                               :foo2 (let [x (trace :x (d/exponential 1))
                                           y (trace :y (gaussian-noise-around x))]
                                       y)
                               :foo3 (let [x (trace :x (d/exponential 1))
                                           y (traced-gaussian-noise-around x :y)]
                                       y)}]
             [name (list `gen params body)])))

(defn- check-foos
  "trace[:y] != trace[:x] in each foo's simulated trace."
  [foos]
  (doseq [[name foo] foos]
    (testing name
      ;; Gen.jl's trace[:y] throws where there is no :y; here both are read
      (let [{:keys [x y]} (ts/host (:choices (gfi/simulate foo [] (rng/key 1))))]
        (is (and (number? x) (number? y) (not= x y)))))))

;; macros.jl:38
(deftest macros-in-dynamic-dsl
  (check-foos (foos [])))

;; macros.jl:122: foo3's macro at a binding's head expands to its trace (G11)
(deftest macros-in-static-dsl
  (let [static-foos (foos ^:static [])]
    (is (every? static/ir (vals static-foos)) "static functions")
    (check-foos static-foos)))

;; Not ported (n/a, genjl-test-suite.md §2): macros.jl:65, :72, :79 and :86,
;; foo4 to foo7, and the static ones at :133, :136, :139 and :142. Their macros bind a name in the body they expand in
;; (@add_gaussian_noise x y is y ~ normal(x, 1)), which a Clojure macro,
;; expanding to one form, cannot do for its caller's let; foo7's
;; Expr(:gentrace) is Julia's syntax tree.
