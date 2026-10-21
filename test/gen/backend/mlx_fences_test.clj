(ns gen.backend.mlx-fences-test
  "A static check of gen.backend.mlx: every handle passed to MLX stays
   reachable through the downcall that reads it (review finding 10,
   docs/research/review-m2-2026-10-19.md; genclj-2t5f). An MlxArray's
   handle is freed when the array becomes unreachable (MlxRuntime/adopt's
   phantom references, drained by any thread making arrays), so a read of
   (.-h x) followed by a downcall, with nothing keeping x reachable after,
   is a use after free when a collection lands in between. The race is too
   narrow to make fail on demand, so the check is on the source: the file
   is read as data (no MLX library loaded), and

   - every handle read, (.-h x) or (operand b x dtype), is followed in an
     enclosing form by (fence x): x a symbol, or for (map #(.-h %) xs)
     (or mapv) the collection xs (concatenate's operands);
   - but operand's own read of its argument's handle, which its callers
     fence; and the with-operands macro, which calls operand for its
     bindings, fences them in a finally.

   A read whose argument is no symbol, (.-h (as-array b x nil)), cannot be
   fenced, and fails. Mutation checked (2026-10-21): the four sites the
   review found, as they were, fail it (from-values's astype, and the
   retains of custom-vjp, value-and-grad and compile)."
  (:require [clojure.java.io :as io]
            [clojure.test :refer [deftest is]]))

(def ^:private source-file "src/gen/backend/mlx.clj")

(defn- forms
  "The top-level forms of a Clojure source file, read as data."
  [file]
  (with-open [r (java.io.PushbackReader. (io/reader file))]
    (binding [*read-eval* false]
      (doall (take-while #(not= ::end %) (repeatedly #(read {:eof ::end :read-cond :allow} r)))))))

(defn- children [x] (cond (map? x) (mapcat identity x) (coll? x) (seq x) :else nil))

(defn- named? [x n] (and (symbol? x) (= n (name x))))

(defn- call? [x n] (and (seq? x) (named? (first x) n)))

(defn- occurrences
  "Every place in form where (pred x) holds: [x ancestors], the ancestors
   nearest first, each [form index-of-the-child-on-the-way-down]."
  [form pred]
  (letfn [(walk [x ancestors]
            (concat (when (pred x) [[x ancestors]])
                    (mapcat (fn [i c] (walk c (cons [x i] ancestors))) (range) (children x))))]
    (walk form ())))

(defn- read-arg
  "The array whose handle the form reads: (.-h x)'s x, (operand b x d)'s x."
  [h]
  (if (call? h ".-h") (second h) (nth h 2 nil)))

(defn- handle-read? [x] (or (call? x ".-h") (call? x "operand")))

(defn- fenced-symbol
  "The symbol a fence must keep reachable for the handle read h: its
   argument, or, for a read in #(…) mapped over a collection, the
   collection; nil when none can (the argument is no symbol)."
  [h ancestors]
  (let [arg (read-arg h)]
    (cond (not (symbol? arg)) nil
          (.startsWith (name arg) "p1__")
          (let [[[mapped]] (drop-while (fn [[a]] (not (and (or (call? a "map") (call? a "mapv")) (seq? (second a)) (named? (first (second a)) "fn*"))))
                                       ancestors)]
            (when (symbol? (nth mapped 2 nil)) (nth mapped 2)))
          :else arg)))

(defn- fenced-after?
  "Is (fence s) in a later child of one of the ancestors (evaluated after
   the read)?"
  [s ancestors]
  (some (fn [[a i]]
          (some (fn [c] (seq (occurrences c #(and (call? % "fence") (= s (second %))))))
                (drop (inc (long i)) (children a))))
        ancestors))

(defn- unfenced-reads
  "The handle reads of the file's forms not followed by a fence, but in
   operand's own definition: [the form's name, the read]."
  [all]
  (for [form all
        :when (not (and (call? form "defn-") (named? (second form) "operand")))
        [h ancestors] (occurrences form handle-read?)
        :let [s (fenced-symbol h ancestors)]
        :when (not (and s (fenced-after? s ancestors)))]
    [(second form) h]))

(deftest every-handle-passed-to-mlx-is-fenced
  (let [all (forms source-file)]
    (is (< 5 (count (mapcat #(occurrences % (fn [x] (call? x ".-h"))) all))) "the check sees the handle reads")
    (is (empty? (unfenced-reads all)))))

(deftest with-operands-fences-its-operands
  ;; the macro's template calls operand for each binding and fences each
  ;; operand in a finally
  (let [[macro] (filter #(and (call? % "defmacro") (named? (second %) "with-operands")) (forms source-file))]
    (is macro)
    (is (seq (occurrences macro #(named? % "operand"))))
    (is (seq (occurrences macro #(named? % "finally"))))
    (is (seq (occurrences macro #(named? % "fence"))))))

(deftest the-check-finds-an-unfenced-read
  ;; the forms the review found, as they were: none is fenced
  (let [old (map read-string ["(defn- from-values [b values shape dtype]
                                 (let [x (from-values b values shape :float32)]
                                   (wrap b (Mlx/mlx_astype (.-h ^MlxArray x) (code dtype) (stream b)))))"
                               "(defn- handles [b xs] (long-array (map #(MlxRuntime/retain (.-h ^MlxArray (as-array b % nil))) xs)))"
                               "(defn- run [b out] (let [out (as-array b out nil)] (long-array [(MlxRuntime/retain (.-h ^MlxArray out))])))"])]
    (is (= 3 (count (unfenced-reads old))))
    (is (empty? (unfenced-reads (map read-string ["(defn- ok [x] (try (Mlx/f (.-h x)) (finally (fence x))))"
                                                  "(defn- ok2 [xs] (let [v (vectorOf (map #(.-h %) xs))] (try (g v) (finally (fence xs)))))"])))
        "a fenced read and a fenced collection pass")))
