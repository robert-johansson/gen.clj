(ns gen.dynamic.core
  "Guarded versions of the functions that test truthiness, look up by key or
   test a value's representation. Inside gen bodies (and guarded-fn helpers)
   the gen macro uses these in place of the functions each one names in its
   :guards metadata (clojure.core's, and gen.array's host? and array?), as it
   wraps if tests in gen.array/truthy?.

   On ordinary values each behaves as the original. A batched value
   (gen.array.batch) is one object standing for every particle's value, and
   the original would answer for that object: count it true, find no key
   for it, or name its class. The guarded versions answer for the particles
   instead:
   - Truthiness tests and lookups raise the batched-branch error, so the
     batched run falls back to one particle at a time. This is what makes
     (not b), if-not, (filter pred xs) or (get v k) inside a body branch per
     particle.
   - Representation tests (class, instance?, number?, coll?, …) answer from a
     particle's value, whose type the batch's dtype and per-particle shape
     determine.

   Predicates passed to the higher-order functions are tested with truthy?,
   which is also how gen bodies' if tests branch."
  (:refer-clojure :exclude [not boolean true? false? identical?
                            class type instance? number? integer? int? float? double? boolean? nat-int? pos-int?
                            neg-int? rational? ratio? decimal? coll? seqable? sequential? seq? map? vector? list?
                            set? string? keyword? symbol? fn? ifn? char? counted? indexed? associative?
                            reversible? sorted? record?
                            every? some not-any? not-every? filter filterv remove take-while drop-while
                            split-with complement every-pred some-fn
                            get contains? find get-in])
  (:require [gen.array :as a]
            [gen.array.batch :as batch]
            [gen.array.dtype :as dt]
            [gen.array.protocols :as p]))

(set! *warn-on-reflection* true)

(defmacro ^:private defguard
  "defn of a guarded version of the function guarded (a qualified symbol)."
  [fname guarded & fdecl]
  `(defn ~(vary-meta fname assoc :guards (list 'quote guarded)) ~@fdecl))

(defn- batched? [x] (clojure.core/instance? gen.array.protocols.IBatchedValue x))

(defn- branch-error [what]
  (a/batched-branch (str what " of a batched value: it differs across particles") {}))

(defn- unbatched
  "x, or the batched-branch error when it is batched."
  [what x]
  (if (batched? x) (throw (branch-error what)) x))

(defn ^:no-doc lookup-error
  "The batched-branch error of a lookup (used by get's inline expansion)."
  []
  (throw (branch-error "A lookup")))

(defn- tested
  "A predicate's result v as clojure.core's own if will see it, decided by
   truthy?: booleans and nil as they are, other values when truthy? holds,
   false when it does not (a one-element false array)."
  [v]
  (cond (clojure.core/instance? Boolean v) v
        (nil? v) v
        (a/truthy? v) v
        :else false))

(defn- guard-pred
  "pred, its result tested with truthy?."
  [pred]
  (fn ([x] (tested (pred x)))
      ([x y] (tested (pred x y)))
      ([x y & more] (tested (apply pred x y more)))))

;; ---------------------------------------------------------------------------
;; Truthiness
;; ---------------------------------------------------------------------------

(defguard not clojure.core/not [x] (clojure.core/not (a/truthy? x)))

(defguard boolean clojure.core/boolean [x] (a/truthy? x))

(defguard true? clojure.core/true? [x] (clojure.core/true? (unbatched "true?" x)))

(defguard false? clojure.core/false? [x] (clojure.core/false? (unbatched "false?" x)))

(defguard identical? clojure.core/identical? [x y]
  (clojure.core/identical? (unbatched "identical?" x) (unbatched "identical?" y)))

;; ---------------------------------------------------------------------------
;; Representation tests: a particle's value is a host scalar of the batch's
;; dtype when its shape is [] (on a backend whose 0-d arrays are host values),
;; else an array of the wrapped backend
;; ---------------------------------------------------------------------------

(defn- particle
  "x, or for a batched x a value of the same type as each particle's."
  [x]
  (if (batched? x)
    (let [dtype (a/dtype x)]
      (p/-full (p/-inner (p/-backend x)) (if (= [] (a/shape x)) [] [1])
               (if (= :bool (dt/kind dtype)) false 0) dtype))
    x))

(defmacro ^:private def-representation-tests [& fnames]
  `(do ~@(for [n fnames
               :let [core (symbol "clojure.core" (name n))]]
           `(defguard ~n ~core [~'x] (~core (particle ~'x))))))

(def-representation-tests class type number? integer? int? float? double? boolean? rational? ratio?
  decimal? coll? seqable? sequential? seq? map? vector? list? set? string? keyword? symbol? fn? ifn? char?
  counted? indexed? associative? reversible? sorted? record?)

(defguard instance? clojure.core/instance? [c x] (clojure.core/instance? c (particle x)))

(defguard host? gen.array/host? [x] (a/host? (particle x)))

(defguard array? gen.array/array? [x] (a/array? (particle x)))

(defn- int-sign-test [what core-pred x]
  (if (batched? x)
    ;; an integer particle's sign differs across particles
    (if (= :int (dt/kind (a/dtype x)))
      (if (= [] (a/shape x)) (throw (branch-error what)) false)
      false)
    (core-pred x)))

(defguard nat-int? clojure.core/nat-int? [x] (int-sign-test "nat-int?" clojure.core/nat-int? x))
(defguard pos-int? clojure.core/pos-int? [x] (int-sign-test "pos-int?" clojure.core/pos-int? x))
(defguard neg-int? clojure.core/neg-int? [x] (int-sign-test "neg-int?" clojure.core/neg-int? x))

;; ---------------------------------------------------------------------------
;; Predicates of higher-order functions
;; ---------------------------------------------------------------------------

(defguard every? clojure.core/every? [pred coll] (clojure.core/every? (guard-pred pred) coll))

(defguard some clojure.core/some [pred coll] (clojure.core/some (guard-pred pred) coll))

(defguard not-any? clojure.core/not-any? [pred coll] (clojure.core/not-any? (guard-pred pred) coll))

(defguard not-every? clojure.core/not-every? [pred coll] (clojure.core/not-every? (guard-pred pred) coll))

(defguard filter clojure.core/filter
  ([pred] (clojure.core/filter (guard-pred pred)))
  ([pred coll] (clojure.core/filter (guard-pred pred) coll)))

(defguard filterv clojure.core/filterv [pred coll] (clojure.core/filterv (guard-pred pred) coll))

(defguard remove clojure.core/remove
  ([pred] (clojure.core/remove (guard-pred pred)))
  ([pred coll] (clojure.core/remove (guard-pred pred) coll)))

(defguard take-while clojure.core/take-while
  ([pred] (clojure.core/take-while (guard-pred pred)))
  ([pred coll] (clojure.core/take-while (guard-pred pred) coll)))

(defguard drop-while clojure.core/drop-while
  ([pred] (clojure.core/drop-while (guard-pred pred)))
  ([pred coll] (clojure.core/drop-while (guard-pred pred) coll)))

(defguard split-with clojure.core/split-with [pred coll] (clojure.core/split-with (guard-pred pred) coll))

(defguard complement clojure.core/complement [f] (fn [& args] (not (apply f args))))

(defguard every-pred clojure.core/every-pred [& ps] (apply clojure.core/every-pred (map guard-pred ps)))

(defguard some-fn clojure.core/some-fn [& ps] (apply clojure.core/some-fn (map guard-pred ps)))

;; ---------------------------------------------------------------------------
;; Lookups: a vector finds no batched key, silently
;; ---------------------------------------------------------------------------

(defguard get clojure.core/get
  {:inline (fn
             ([m k] `(let [m# ~m k# ~k]
                       (when (or (clojure.core/instance? gen.array.protocols.IBatchedValue m#)
                                 (clojure.core/instance? gen.array.protocols.IBatchedValue k#))
                         (lookup-error))
                       (clojure.lang.RT/get m# k#)))
             ([m k not-found] `(let [m# ~m k# ~k]
                                 (when (or (clojure.core/instance? gen.array.protocols.IBatchedValue m#)
                                           (clojure.core/instance? gen.array.protocols.IBatchedValue k#))
                                   (lookup-error))
                                 (clojure.lang.RT/get m# k# ~not-found))))
   :inline-arities #{2 3}}
  ([m k] (clojure.core/get (unbatched "get" m) (unbatched "A lookup" k)))
  ([m k not-found] (clojure.core/get (unbatched "get" m) (unbatched "A lookup" k) not-found)))

(defguard contains? clojure.core/contains? [coll k]
  (clojure.core/contains? (unbatched "contains?" coll) (unbatched "A lookup" k)))

(defguard find clojure.core/find [m k] (clojure.core/find (unbatched "find" m) (unbatched "A lookup" k)))

(defguard get-in clojure.core/get-in
  ([m ks] (clojure.core/get-in (unbatched "get-in" m) (mapv #(unbatched "A lookup" %) ks)))
  ([m ks not-found] (clojure.core/get-in (unbatched "get-in" m) (mapv #(unbatched "A lookup" %) ks) not-found)))

;; ---------------------------------------------------------------------------
;; Exceptions: whether one is thrown may differ across particles
;; ---------------------------------------------------------------------------

(defn ^:no-doc caught
  "Called first in each catch clause of a guarded body: in a batched run a
   caught exception is a branch that may differ across particles, so it
   raises the batched-branch error (and the fallback) instead."
  [^Throwable e]
  (when (batch/batch-backend? a/*backend*)
    (throw (a/batched-branch (str "An exception was caught in a batched run, where it may be thrown for some"
                                  " particles only: " (.getMessage e))
                             {} e))))
