(ns gen.dynamic.core
  "Guarded versions of the functions that test truthiness, look up by key or
   test a value's representation. Inside gen bodies (and guarded-fn helpers)
   the gen macro uses these in place of the functions each one names in its
   :guards metadata (clojure.core's, and gen.array's host?, array? and
   backend-array?), as it wraps if tests in gen.array/truthy?.

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
   which is also how gen bodies' if tests branch.

   A 0-d backend array (MLX's scalars; the JVM backend's are host values
   already) counts as its host value wherever these branch, compare or look
   up on it: = and not=, case (gen.dynamic.guard/case-key), the lookups,
   nth and the representation tests. So a body written with plain Clojure
   behaves alike on every backend. Reading the value evaluates it, as truthy? does."
  (:refer-clojure :exclude [not boolean true? false? identical? = not= nth
                            class type instance? number? integer? int? float? double? boolean? nat-int? pos-int?
                            neg-int? rational? ratio? decimal? coll? seqable? sequential? seq? map? vector? list?
                            set? string? keyword? symbol? fn? ifn? char? counted? indexed? associative?
                            reversible? sorted? record? satisfies?
                            every? some not-any? not-every? filter filterv remove take-while drop-while
                            split-with complement every-pred some-fn
                            get contains? find get-in])
  (:require [gen.array :as array]
            [gen.array.batch :as batch]
            [gen.array.dtype :as dt]
            [gen.array.tape :as tape]))

(set! *warn-on-reflection* true)

(defmacro ^:private defguard
  "defn of a guarded version of the function guarded (a qualified symbol)."
  [fname guarded & fdecl]
  `(defn ~(vary-meta fname assoc :guards (list 'quote guarded)) ~@fdecl))

(defn- branch-error [what]
  (array/batched-branch (str what " of a batched value: it differs across particles") {}))

(defn- unbatched
  "x, or the batched-branch error when it is batched."
  [what x]
  (if (batch/batched? x) (throw (branch-error what)) x))

(defn- scalar-array?
  "Is x a 0-d backend array (not a host scalar, not a batch)?"
  [x]
  (and (array/backend-array? x) (clojure.core/not (batch/batched? x)) (clojure.core/= [] (array/shape x))))

(defn ^:no-doc host-value
  "x as Clojure code branching, comparing or looking up on it sees it: a 0-d
   backend array as its host value (read), a batched value an error (it
   differs across particles), anything else itself."
  [what x]
  (cond (batch/batched? x) (throw (branch-error what))
        ;; branching, comparing and looking up have no gradient: a value
        ;; being differentiated is read on purpose (gen.array.tape)
        (scalar-array? x) (array/item (cond-> x (tape/tracked? x) array/stop-gradient))
        :else x))

(defn- tested
  "A predicate's result v as clojure.core's own if will see it, decided by
   truthy?: booleans and nil as they are, other values when truthy? holds,
   false when it does not (a one-element false array)."
  [v]
  (cond (clojure.core/instance? Boolean v) v
        (nil? v) v
        (array/truthy? v) v
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

(defguard not clojure.core/not [x] (clojure.core/not (array/truthy? x)))

(defguard boolean clojure.core/boolean [x] (array/truthy? x))

(defguard true? clojure.core/true? [x] (clojure.core/true? (host-value "true?" x)))

(defguard false? clojure.core/false? [x] (clojure.core/false? (host-value "false?" x)))

(defguard = clojure.core/=
  ([_] true)
  ([x y] (clojure.core/= (host-value "=" x) (host-value "=" y)))
  ([x y & more] (clojure.core/apply clojure.core/= (map #(host-value "=" %) (list* x y more)))))

(defguard not= clojure.core/not=
  ([_] false)
  ([x y] (clojure.core/not= (host-value "not=" x) (host-value "not=" y)))
  ([x y & more] (clojure.core/apply clojure.core/not= (map #(host-value "not=" %) (list* x y more)))))

(defguard identical? clojure.core/identical? [x y]
  (clojure.core/identical? (host-value "identical?" x) (host-value "identical?" y)))

;; ---------------------------------------------------------------------------
;; Representation tests: a particle's value is a host scalar of the batch's
;; dtype when its shape is [] (on a backend whose 0-d arrays are host values),
;; else an array of the wrapped backend
;; ---------------------------------------------------------------------------

(defn- host-prototype
  "A host scalar of dtype's kind (the JVM backend's 0-d arrays)."
  [dtype]
  (case (dt/kind dtype) :bool false :int 0 :float 0.0))

(defn- particle
  "x, or for a batched x a value of the same type as each particle's; a 0-d
   value (a particle's, or a 0-d backend array) as a host scalar, as on the
   JVM backend."
  [x]
  (cond (batch/batched? x)
        (let [dtype (array/dtype x)]
          (if (clojure.core/= [] (array/shape x))
            (host-prototype dtype)
            (array/with-backend (batch/inner-backend x)
              (array/full [1] (if (clojure.core/= :bool (dt/kind dtype)) false 0) {:dtype dtype}))))
        (scalar-array? x) (host-prototype (array/dtype x))
        :else x))

(defmacro ^:private def-representation-tests [& fnames]
  `(do ~@(for [n fnames
               :let [core (symbol "clojure.core" (name n))]]
           `(defguard ~n ~core [~'x] (~core (particle ~'x))))))

(def-representation-tests class type number? integer? int? float? double? boolean? rational? ratio?
  decimal? coll? seqable? sequential? seq? map? vector? list? set? string? keyword? symbol? fn? ifn? char?
  counted? indexed? associative? reversible? sorted? record?)

(defguard instance? clojure.core/instance? [c x] (clojure.core/instance? c (particle x)))

(defguard satisfies? clojure.core/satisfies? [protocol x] (clojure.core/satisfies? protocol (particle x)))

(defguard host? gen.array/host? [x] (array/host? (particle x)))

(defguard array? gen.array/array? [x] (array/array? (particle x)))

(defguard backend-array? gen.array/backend-array? [x] (array/backend-array? (particle x)))

(defn- int-sign-test [what core-pred x]
  (if (batch/batched? x)
    ;; an integer particle's sign differs across particles
    (if (clojure.core/= :int (dt/kind (array/dtype x)))
      (if (clojure.core/= [] (array/shape x)) (throw (branch-error what)) false)
      false)
    (core-pred (host-value what x))))

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

(defn ^:no-doc lookup-key
  "A lookup's collection or key k, as host-value (used by get's inline
   expansion)."
  [k]
  (host-value "A lookup" k))

(defguard get clojure.core/get
  {:inline (fn
             ([m k] `(clojure.lang.RT/get (lookup-key ~m) (lookup-key ~k)))
             ([m k not-found] `(clojure.lang.RT/get (lookup-key ~m) (lookup-key ~k) ~not-found)))
   :inline-arities #{2 3}}
  ([m k] (clojure.core/get (lookup-key m) (lookup-key k)))
  ([m k not-found] (clojure.core/get (lookup-key m) (lookup-key k) not-found)))

(defguard contains? clojure.core/contains? [coll k]
  (clojure.core/contains? (unbatched "contains?" coll) (lookup-key k)))

(defguard find clojure.core/find [m k] (clojure.core/find (unbatched "find" m) (lookup-key k)))

(defguard get-in clojure.core/get-in
  ([m ks] (clojure.core/get-in (unbatched "get-in" m) (mapv lookup-key ks)))
  ([m ks not-found] (clojure.core/get-in (unbatched "get-in" m) (mapv lookup-key ks) not-found)))

(defguard nth clojure.core/nth
  ([coll i] (clojure.core/nth (unbatched "nth" coll) (host-value "An index" i)))
  ([coll i not-found] (clojure.core/nth (unbatched "nth" coll) (host-value "An index" i) not-found)))

;; ---------------------------------------------------------------------------
;; Exceptions: whether one is thrown may differ across particles
;; ---------------------------------------------------------------------------

(defn ^:no-doc caught
  "Called first in each catch clause of a guarded body: in a batched run a
   caught exception is a branch that may differ across particles, so it
   raises the batched-branch error (and the fallback) instead."
  [^Throwable e]
  (when (batch/batch-backend? array/*backend*)
    (throw (array/batched-branch (str "An exception was caught in a batched run, where it may be thrown for some"
                                      " particles only: " (.getMessage e))
                                 {} e))))
