(ns gen.law-instances-test
  "Law sets over pairs of generative functions (gen.laws/across): every
   kind that reads a parameter store, given a trace it made under one store
   and updated or regenerated under another (law audit F5: store-test
   checks such revisions by examples). Each pair [label gf args alt-args gf']
   is a model under {:w 1.0} and under {:w 6.0}: a store that changes the
   density, and, where the model branches on it, the structure."
  (:refer-clojure :exclude [+ >])
  (:require [gen.array :as array :refer [+ >]]
            [gen.choicemap :as cm]
            [gen.combinators :as comb]
            [gen.dist :as d]
            [gen.dynamic :as dyn :refer [gen]]
            [gen.inference.enumerative :as enum]
            [gen.inference.proposal :as prop]
            [gen.laws :as laws]))

(def ^:private chain
  (gen [x] (let [a (trace :a (d/normal (param :w) 1))] (trace :b (d/normal (+ a x) 1)))))

(def ^:private forked
  "The store decides the structure: :x above 5, :y below."
  (gen [] (let [c (trace :c (d/normal 0 1))]
            (if (> (param :w) 5.0) (trace :x (d/normal c 1)) (trace :y (d/normal c 1))))))

(def ^:private forked1 (gen [x] (if (> (param :w) 5.0) (trace :x (d/normal x 1)) (trace :y (d/normal x 1)))))

(def ^:private inner (gen [] (trace :x (d/normal (param :w) 1))))

(def ^:private calls-inner
  "The store reaches a callee."
  (gen [] (let [m (splice :c inner)] (trace :z (d/normal m 1)))))

(def ^:private kernel (gen [x] (trace :y (d/normal (+ x (param :w)) 1))))

(def ^:private maps-kernel (gen [xs] (splice :m (comb/map kernel) xs)))

(def ^:private step (gen [_t s] (trace :s (d/normal (+ s (param :w)) 1))))

(def ^:private static-chain
  (gen ^:static [] (let [x (trace :x (d/normal (param :w) 1))] (trace :y (d/normal x 1)))))

(def ^:private inner-b
  (gen [] (let [b (trace :b (d/bernoulli 0.5))] (trace :x (d/normal (+ (param :w) (array/where b 1.0 0.0)) 1)))))

(def ^:private q-x (gen [cs] (when-not (cm/has-value? (cm/get-submap cs :x)) (trace :x (d/normal 0 2)))))

(defn- pair
  "[label (f {:w 1.0}) args alts (f {:w 6.0})]"
  [label f args alts]
  [label (f {:w 1.0}) args alts (f {:w 6.0})])

(def models
  [(pair :dynamic #(dyn/with-params chain %) [0.5] [[1.5]])
   (pair :store-decides-structure #(dyn/with-params forked %) [] [])
   (pair :callee-reads-the-store #(dyn/with-params calls-inner %) [] [])
   (pair :map-in-a-body #(dyn/with-params maps-kernel %) [[0.0 1.0]] [[[0.0 1.0 2.0]] [[0.5]]])
   (pair :map-of-a-with-params-kernel #(comb/map (dyn/with-params kernel %)) [[0.0 1.0]] [[[0.0 1.0 2.0]]])
   (pair :unfold-of-a-with-params-step #(comb/unfold (dyn/with-params step %)) [2 0.0] [[3 0.0]])
   (pair :switch-of-with-params-branches #(comb/switch (dyn/with-params chain %) (dyn/with-params forked1 %)) [0 0.5] [[1 0.5]])
   (pair :static #(dyn/with-params static-chain %) [] [])
   (pair :internal-proposal #(prop/with-internal-proposal (dyn/with-params inner %) q-x) [] [])
   (pair :exact-proposal #(enum/with-exact-proposal (dyn/with-params inner-b %) [:b]) [] [])])

;; Planted bugs (2026-10-20), each failing the law named: update without the
;; gen-fn check (review 5's N1; update) or regenerate without it (regenerate);
;; no store check at a call (N3; both); a revise's update weighing the old
;; trace by assessing it under the new gen-fn, or a discard that leaves out
;; the choices gone (update); a revise's regenerate projecting the old trace
;; by the new gen-fn (regenerate). Both pass review 6's N5 (store marks
;; never refreshed), which the whole suite passes, and R10 (a static walk's
;; own store check, which a revise never reaches; the fast tier catches it).
(laws/deflaws revised models 100 laws/across)
