(ns gen.dynamic.core
  "Guarded versions of the clojure.core functions that test truthiness or
   look up by key. Inside gen bodies the gen macro uses these in place of
   clojure.core's (as it wraps if tests in gen.array/truthy?).

   On ordinary values each behaves as clojure.core's. A batched value
   (gen.array.batch) is an object, and clojure.core's own test would count it
   as true, or find no key, for every particle at once. The guarded versions
   raise the batched-branch error instead, so the batched run falls back to
   one particle at a time. This is what makes (not b), if-not, (filter pred
   xs) or (get v k) inside a body branch per particle.

   Predicates passed to the higher-order functions are tested with truthy?,
   which is also how gen bodies' if tests branch."
  (:refer-clojure :exclude [not boolean true? false? identical?
                            number? integer? int? float? double? boolean? nat-int? pos-int? neg-int?
                            rational? ratio? decimal?
                            every? some not-any? not-every? filter filterv remove take-while drop-while
                            split-with complement every-pred some-fn
                            get contains? find get-in])
  (:require [gen.array :as a]))

(set! *warn-on-reflection* true)

(defn- batched? [x] (instance? gen.array.protocols.IBatchedValue x))

(defn- branch-error [what]
  (ex-info (str what " of a batched value: it differs across particles") {:gen.array/batched-branch true}))

(defn- unbatched
  "x, or the batched-branch error when it is batched."
  [what x]
  (if (batched? x) (throw (branch-error what)) x))

(defn- guard-pred
  "pred, its result tested with truthy? (returned when true, else nil)."
  [pred]
  (fn [& xs] (let [v (apply pred xs)] (when (a/truthy? v) v))))

;; ---------------------------------------------------------------------------
;; Truthiness
;; ---------------------------------------------------------------------------

(defn not [x] (clojure.core/not (a/truthy? x)))

(defn boolean [x] (a/truthy? x))

(defn true? [x] (clojure.core/true? (unbatched "true?" x)))

(defn false? [x] (clojure.core/false? (unbatched "false?" x)))

(defn identical? [x y] (clojure.core/identical? (unbatched "identical?" x) (unbatched "identical?" y)))

(defmacro ^:private def-type-predicates [& names]
  `(do ~@(for [n names]
           `(defn ~n [~'x] (~(symbol "clojure.core" (name n)) (unbatched ~(str n) ~'x))))))

;; a particle's value is a host number or an array: the batch cannot say which
(def-type-predicates number? integer? int? float? double? boolean? nat-int? pos-int? neg-int?
  rational? ratio? decimal?)

;; ---------------------------------------------------------------------------
;; Predicates of higher-order functions
;; ---------------------------------------------------------------------------

(defn every? [pred coll] (clojure.core/every? (guard-pred pred) coll))

(defn some [pred coll] (clojure.core/some (guard-pred pred) coll))

(defn not-any? [pred coll] (clojure.core/not-any? (guard-pred pred) coll))

(defn not-every? [pred coll] (clojure.core/not-every? (guard-pred pred) coll))

(defn filter
  ([pred] (clojure.core/filter (guard-pred pred)))
  ([pred coll] (clojure.core/filter (guard-pred pred) coll)))

(defn filterv [pred coll] (clojure.core/filterv (guard-pred pred) coll))

(defn remove
  ([pred] (clojure.core/remove (guard-pred pred)))
  ([pred coll] (clojure.core/remove (guard-pred pred) coll)))

(defn take-while
  ([pred] (clojure.core/take-while (guard-pred pred)))
  ([pred coll] (clojure.core/take-while (guard-pred pred) coll)))

(defn drop-while
  ([pred] (clojure.core/drop-while (guard-pred pred)))
  ([pred coll] (clojure.core/drop-while (guard-pred pred) coll)))

(defn split-with [pred coll] (clojure.core/split-with (guard-pred pred) coll))

(defn complement [f] (fn [& args] (not (apply f args))))

(defn every-pred [& ps] (apply clojure.core/every-pred (map guard-pred ps)))

(defn some-fn [& ps] (apply clojure.core/some-fn (map guard-pred ps)))

;; ---------------------------------------------------------------------------
;; Lookups: a vector finds no batched key, silently
;; ---------------------------------------------------------------------------

(defn get
  ([m k] (clojure.core/get (unbatched "get" m) (unbatched "A lookup" k)))
  ([m k not-found] (clojure.core/get (unbatched "get" m) (unbatched "A lookup" k) not-found)))

(defn contains? [coll k] (clojure.core/contains? (unbatched "contains?" coll) (unbatched "A lookup" k)))

(defn find [m k] (clojure.core/find (unbatched "find" m) (unbatched "A lookup" k)))

(defn get-in
  ([m ks] (clojure.core/get-in (unbatched "get-in" m) (mapv #(unbatched "A lookup" %) ks)))
  ([m ks not-found] (clojure.core/get-in (unbatched "get-in" m) (mapv #(unbatched "A lookup" %) ks) not-found)))
