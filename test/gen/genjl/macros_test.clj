(ns gen.genjl.macros-test
  "Gen.jl 0.4.8's test/dsl/macros.jl: user macros in a body, expanding to a
   distribution or to a trace call (docs/research/genjl-test-suite.md §2).
   Its dynamic testset (line 38) is ported here; the static one (line 122)
   belongs to the static language's port (genclj-2ifo)."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.dist :as d]
            [gen.dynamic :refer [gen]]
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

(deftest macros-in-dynamic-dsl
  ;; macros.jl:38. foo1's y ~ @m x and foo2's y = @trace(@m(x), :y) are
  ;; one form here, (trace :y (m x)): foo1's returns it, foo2's binds it
  (doseq [[name foo] {:foo1 (gen []
                              (let [x (trace :x (d/exponential 1))]
                                (trace :y (gaussian-noise-around x))))
                      :foo2 (gen []
                              (let [x (trace :x (d/exponential 1))
                                    y (trace :y (gaussian-noise-around x))]
                                y))
                      :foo3 (gen []
                              (let [x (trace :x (d/exponential 1))
                                    y (traced-gaussian-noise-around x :y)]
                                y))}]
    (testing name
      ;; Gen.jl's trace[:y] throws where there is no :y; here both are read
      (let [{:keys [x y]} (ts/host (:choices (gfi/simulate foo [] (rng/key 1))))]
        (is (and (number? x) (number? y) (not= x y)))))))

;; Not ported (n/a, genjl-test-suite.md §2): macros.jl:65, :72, :79 and :86,
;; foo4 to foo7. Their macros bind a name in the body they expand in
;; (@add_gaussian_noise x y is y ~ normal(x, 1)), which a Clojure macro,
;; expanding to one form, cannot do for its caller's let; foo7's
;; Expr(:gentrace) is Julia's syntax tree.
