(ns gen.random-models
  "Random generative functions for property tests: a test.check generator of
   model specs (data) and an interpreter that turns a spec into a dynamic
   generative function. Specs mix distribution families, dependencies on
   earlier choices, branches on boolean choices (so structure varies),
   splices of sub-models, shared-namespace splices and Map/Unfold calls.

   A spec is a vector of statements:
     [:site addr family dep]     a choice; dep is the address of an earlier
                                 numeric choice its parameters depend on, or nil
     [:branch addr then else]    a bernoulli choice at addr, then the statements
                                 of the branch it selects
     [:splice addr spec]         a sub-model at addr
     [:inline spec]              a sub-model spliced without an address
     [:map addr n]               a Map of a one-site kernel over n inputs
     [:unfold addr n]            an Unfold of a random walk for n steps
     [:switch addr]              a Switch between two small models, its index
                                 a categorical choice at [addr :i]
     [:select addr]              a categorical choice compared with = to pick a
                                 mean (a host predicate on the choice)"
  (:require [clojure.test.check.generators :as gen]
            [gen.array :as a]
            [gen.array.protocols]
            [gen.combinators :as comb]
            [gen.dist :as d]
            [gen.dynamic :as dyn]))

;; Parameters are transformed from a dependency's value into each family's
;; domain, so any value yields a valid distribution.
(defn- positive [x] (a/+ 0.5 (a/softplus x)))
(defn- probability [x] (a/+ 0.05 (a/* 0.9 (a/sigmoid x))))

(def ^:private families
  {:normal (fn [x] (d/normal x 1.0))
   :gamma (fn [x] (d/gamma (positive x) 1.0))
   :beta (fn [x] (d/beta (positive x) 2.0))
   :exponential (fn [x] (d/exponential (positive x)))
   :laplace (fn [x] (d/laplace x 0.7))
   :uniform (fn [x] (d/uniform x (a/+ x (positive x))))
   :poisson (fn [x] (d/poisson (positive x)))
   :categorical (fn [_] (d/categorical [0.2 0.5 0.3]))
   :mixture (fn [x] ((d/homogeneous-mixture d/normal) [0.4 0.6] (a/stack [x (a/+ x 2.0)] -1) [1.0 0.5]))
   ;; a vector site whose mean is a per-particle scalar plus a shared vector
   :vector (fn [x] (d/normal (a/+ x (a/array [0.0 1.0 -1.0])) 1.0))})

(def numeric-families (vec (keys families)))

(def ^:private map-kernel (dyn/make-gen-fn (fn [trace _ _ x] (trace :v (d/normal x 1.0))) '([x])))

(def ^:private walk
  (dyn/make-gen-fn (fn [trace _ _ _t s] (trace :s (d/normal (a/* 0.9 s) 1.0))) '([t s])))

(def ^:private switched
  "A Switch of a one-site model and a two-site model with a dependency."
  (comb/switch (dyn/make-gen-fn (fn [trace _ _ m] (trace :a (d/normal m 1.0))) '([m]))
               (dyn/make-gen-fn (fn [trace _ _ m] (trace :c (d/normal (trace :b (d/gamma 2.0 1.0)) 1.0))) '([m]))))

(declare build)

(defn- run-statements
  "Execute statements with the env of earlier numeric values (address →
   value); returns the env."
  [statements trace splice env]
  (reduce (fn [env [kind addr & more :as st]]
            (case kind
              :site (let [[family dep] more
                          x (if dep (a/astype (get env dep 0.0) (:float (gen.array.protocols/-defaults a/*backend*))) 0.0)
                          v (trace addr ((families family) x))]
                      ;; a vector value enters later parameters as its sum
                      (assoc env addr (if (= family :vector) (a/sum v) v)))
              :select (let [k (trace [addr :k] (d/categorical [0.3 0.7]))]
                        (assoc env addr (trace [addr :v] (d/normal (if (= k 0) -2.0 2.0) 1.0))))
              :branch (let [[then else] more]
                        ;; truthy? as the gen macro would insert: a batch of values is loud
                        (if (a/truthy? (trace addr (d/bernoulli 0.5)))
                          (run-statements then trace splice env)
                          (run-statements else trace splice env)))
              :splice (let [[spec] more] (splice addr (build spec)) env)
              :inline (let [[spec] st] (splice (build (second st))) env)
              :map (let [[n] more] (assoc env addr (reduce a/+ 0.0 (splice addr (comb/map map-kernel) (vec (repeat n 0.5))))))
              :unfold (let [[n] more] (splice addr (comb/unfold walk) n 0.0) env)
              :switch (let [i (trace [addr :i] (d/categorical [0.5 0.5]))]
                        (assoc env addr (splice [addr :s] switched i 0.5)))))
          env
          statements))

(defn build
  "The dynamic generative function of a spec (no arguments)."
  [spec]
  (dyn/make-gen-fn (fn [trace splice _param] (count (run-statements spec trace splice {}))) (list [] spec)))

;; ---------------------------------------------------------------------------
;; The generator
;; ---------------------------------------------------------------------------

(defn- statements
  "A generator of statements at nesting depth; addresses are drawn from a
   counter in the generated data (made unique when the spec is assembled)."
  [depth]
  (let [site (gen/let [family (gen/elements numeric-families)
                       dep? gen/boolean]
               [:site nil family dep?])
        leaf (gen/frequency [[6 site]
                             [1 (gen/let [n (gen/choose 0 3)] [:map nil n])]
                             [1 (gen/let [n (gen/choose 0 3)] [:unfold nil n])]
                             [1 (gen/return [:switch nil])]
                             [1 (gen/return [:select nil])]])]
    (if (zero? depth)
      (gen/vector leaf 1 3)
      (gen/vector (gen/frequency [[5 leaf]
                                  [2 (gen/let [t (statements (dec depth)) e (statements (dec depth))] [:branch nil t e])]
                                  [1 (gen/let [s (statements (dec depth))] [:splice nil s])]
                                  [1 (gen/let [s (statements (dec depth))] [:inline s])]])
                  1 4))))

(defn- assign
  "Give every statement a unique address and resolve dependencies to earlier
   numeric sites."
  [stmts counter earlier]
  (reduce (fn [[out earlier] [kind _ & more :as st]]
            (let [addr (keyword (str "s" (swap! counter inc)))]
              (case kind
                :site (let [[family dep?] more
                            dep (when (and dep? (seq earlier)) (peek earlier))]
                        [(conj out [:site addr family dep]) (conj earlier addr)])
                :branch (let [[t e] more
                              [t' _] (assign t counter earlier)
                              [e' _] (assign e counter earlier)]
                          [(conj out [:branch addr t' e']) earlier])
                :splice (let [[s] more] [(conj out [:splice addr (first (assign s (atom 0) []))]) earlier])
                ;; an inline sub-model shares this namespace: give it addresses from this counter
                :inline (let [[s'] [(first (assign (second st) counter []))]] [(conj out [:inline s']) earlier])
                :map [(conj out [:map addr (first more)]) (conj earlier addr)]
                :unfold [(conj out [:unfold addr (first more)]) earlier]
                :switch [(conj out [:switch addr]) (conj earlier addr)]
                :select [(conj out [:select addr]) (conj earlier addr)])))
          [[] earlier]
          stmts))

(def gen-spec
  "A random model spec."
  (gen/fmap #(first (assign % (atom 0) [])) (statements 2)))

(def gen-model
  "A random model, as a gen.laws model entry [label gf args]."
  (gen/fmap (fn [spec] [:random (build spec) []]) gen-spec))
