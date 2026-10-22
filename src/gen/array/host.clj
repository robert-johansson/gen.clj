(ns gen.array.host
  "Host data, parsed once for every backend: gen.array/array reads nested
   vectors into a shape and row-major values, checked, and gives the backend
   those (docs/array-api.md §10).")

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

(defn- scalar? [x] (or (instance? Number x) (instance? Boolean x)))

(defn- nested-shape [data]
  (loop [shape [] d data]
    (if (sequential? d)
      (recur (conj shape (count d)) (first d))
      shape)))

(defn- leaves
  "The leaves of nested data in row-major order, checking that every level is
   rectangular and every leaf is a number or boolean."
  [data shape]
  (let [rank (count shape)
        walk (fn walk [d ^long ax]
               (if (== ax rank)
                 (if (scalar? d)
                   [d]
                   (throw (ex-info (str "Array element is not a number or boolean: " (pr-str d))
                                   {:element d})))
                 (if (and (sequential? d) (== (count d) (long (nth shape ax))))
                   (mapcat #(walk % (inc ax)) d)
                   (throw (ex-info "Nested data is not rectangular" {:shape shape :at-axis ax})))))]
    (walk data 0)))

(defn parse
  "Host data as [shape values], values in row-major order: a double[] is
   [[n] itself], and nested vectors their shape and leaves, checked
   rectangular, every leaf a number or boolean. (A number or boolean needs
   no parsing: gen.array/array makes it a 0-d array as it is.)"
  [data]
  (cond (instance? double/1 data) [[(alength ^doubles data)] data]
        (sequential? data) (let [shape (nested-shape data)] [shape (leaves data shape)])
        :else (throw (ex-info (str "Cannot make an array from " (type data)) {:data data}))))

(defn infer-dtype
  "The dtype of parsed values on a backend with these default dtypes: its
   float when any value is a float (or a ratio or decimal), and for no
   values or a double[]; :bool when every value is a boolean; its int
   otherwise."
  [values defaults]
  (cond (instance? double/1 values) (:float defaults)
        (empty? values) (:float defaults)
        (some #(or (instance? Double %) (instance? Float %) (ratio? %) (decimal? %)) values) (:float defaults)
        (every? #(instance? Boolean %) values) :bool
        :else (:int defaults)))
