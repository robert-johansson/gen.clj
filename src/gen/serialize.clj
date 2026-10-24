(ns gen.serialize
  "Choice maps, arrays and traces as EDN text. As in Gen.jl's
   GenSerialization, a trace is written as its arguments and its choices,
   never its generative function, and read back given the generative
   function, by generate with every choice constrained.

     (write x)              → EDN text of x: choice maps, arrays, keys and EDN data
     (read s)               → x again, its arrays on gen.array/*backend*
     (write-trace tr)       → EDN text of a trace: its arguments, choices and score
     (read-trace gf s)      → gf's trace that the text describes, rebuilt exactly
     (read-trace-data s)    → the written trace's parts, not rebuilt

   Save and load with (spit file (write x)) and (read (slurp file)).

   The text is EDN with gen.clj's tags:

     #gen/array {:dtype :float32 :shape [2 3] :data [...]}   a backend array, its data row-major
     #gen/choicemap {addr sub ...}   a choice map: sub is a nested map (a node) or a value
     #gen/value v                    a choice map that is one value, or a leaf whose value is a map
     #gen/key [k0 k1]                a gen.random key
     #gen/trace {:numerics :args :choices :score}   a trace, for read-trace

   Values round-trip exactly:
   - A float is written as the shortest decimal of its double, which every
     float32, float16 and bfloat16 value is exactly, so it reads back bit for
     bit: -0.0, ##Inf and ##-Inf too. NaN reads back as one NaN, without its
     payload.
   - Host numbers and booleans, gen.array's 0-d arrays that belong to no
     backend, are written as EDN numbers and read back as host numbers. An
     Integer, Short or Byte reads back as a Long, and a Float as the Double of
     its exact value, which gen.array treats alike.
   - A backend array is written with its dtype and shape, a 0-d one too, and
     read back onto *backend*. Each backend holds a 0-d array in its own way:
     MLX as a 0-d array, the JVM backend as a host number.
   - Arguments and choice values are EDN data: nil, booleans, numbers,
     strings, characters, keywords, symbols, vectors, lists, maps and sets,
     with arrays, keys and choice maps inside. Sorted collections are not
     allowed, since they would read back unsorted. Anything else (a function,
     a record, a lazy seq, a generative function) is a loud error naming its
     path.

   Reading uses clojure.edn with these tags and EDN's #inst and #uuid: no
   eval. An unknown tag, a malformed tagged value, or text after the value is
   a loud error.

   A trace's written score is generate's score on its arguments and
   choices, which write-trace computes with the trace's own generative
   function, under the backend and float dtype it records (:numerics,
   {:backend :jvm :float :float64}, say). The trace read back is generate's,
   so after updates its score may differ from the original's in the last
   bits: Map and Unfold keep their scores incrementally, as Gen.jl's do.
   read-trace rebuilds the trace and checks that it is exact:
   - it runs under the written backend and float dtype, since log densities
     round differently on each: elsewhere it is a loud error, and
     (gfi/generate gf args choices) on read-trace-data's parts loads the
     trace there instead;
   - the rebuilt score is the written one, bit for bit, so a different
     generative function is a loud error;
   - generate's weight is its score, so no choice was missing and proposed.

   Traces with encapsulated randomness (thesis §4.5) are not serializable
   yet. Their scores, or return values, depend on keys ω kept per call: by
   each callee, each Map or Unfold element, each Switch branch. generate
   draws every ω afresh from its key (a dynamic function's folded from it,
   and fold-in is one-way), so no key given to generate reproduces a
   trace's ω's, least of all after updates have redrawn some of them. write-trace refuses a trace
   with an ω anywhere in it, and read-trace a rebuild that drew one.

   Out of scope: gen.batched particle collections (write the scalar traces
   that (gen.batched/traces ps) gives), LLM completion traces, and generative
   functions themselves."
  (:refer-clojure :exclude [read])
  (:require [clojure.edn :as edn]
            [gen.array :as array]
            [gen.array.dtype :as dt]
            [gen.choicemap :as cm]
            [gen.gfi :as gfi]
            [gen.random :as rng]
            [gen.trace :as trace]))

(set! *warn-on-reflection* true)

;; ---------------------------------------------------------------------------
;; Writing
;; ---------------------------------------------------------------------------

(defn- unprintable
  "The error for x, at path at, which is not EDN data: why says what it is."
  [at x why]
  (ex-info (str "write: the value at " at " is " why " (" (.getName (class x)) ")")
           {:reason :unprintable :path at :class (class x)}))

(defn- double-text
  "The EDN text of a double: the shortest decimal that reads back as it."
  ^String [^double x]
  (cond (Double/isNaN x) "##NaN"
        (== x Double/POSITIVE_INFINITY) "##Inf"
        (== x Double/NEGATIVE_INFINITY) "##-Inf"
        :else (Double/toString x)))

(declare emit!)

(defn- emit-all!
  "The elements xs, separated by spaces, between open and close."
  [^StringBuilder sb ^String open xs ^String close at]
  (.append sb open)
  (reduce (fn [^long i x] (when (pos? i) (.append sb \space)) (emit! sb x (conj at i)) (inc i)) 0 xs)
  (.append sb close))

(defn- emit-map! [^StringBuilder sb m at]
  (.append sb \{)
  (reduce (fn [first? [k v]]
            (when-not first? (.append sb ", "))
            (emit! sb k (conj at k))
            (.append sb \space)
            (emit! sb v (conj at k))
            false)
          true m)
  (.append sb \}))

(defn- emit-number! [^StringBuilder sb x at]
  (cond (instance? Long x) (.append sb (long x))
        (instance? Double x) (.append sb (double-text x))
        (or (instance? Integer x) (instance? Short x) (instance? Byte x)) (.append sb (long x))
        (instance? Float x) (.append sb (double-text (double x)))
        (or (instance? clojure.lang.BigInt x) (instance? BigDecimal x) (ratio? x)) (.append sb (pr-str x))
        :else (throw (unprintable at x "a number of a class EDN does not keep"))))

(def ^:private plain-name
  "A keyword's name or namespace that surely reads back: a letter or mark
   first, then letters, digits and marks."
  #"[a-zA-Z_*!?$%&=<>][a-zA-Z0-9_*!?$%&=<>+\-.]*")

(defn- emit-name!
  "A keyword or symbol, which must read back as itself: (keyword \"a b\")
   would read as two forms. A plain keyword is known to; anything else is
   read back to see."
  [^StringBuilder sb x at]
  (let [s (str x)]
    (when-not (or (and (keyword? x) (re-matches plain-name (name x))
                       (or (nil? (namespace x)) (re-matches plain-name (namespace x))))
                  (= x (try (edn/read-string s) (catch RuntimeException _ nil))))
      (throw (unprintable at x (str "a keyword or symbol that does not read back as itself: " s))))
    (.append sb s)))

(defn- emit-array!
  "A backend array as #gen/array, its elements row-major: a float as the
   shortest decimal of its double, which every float dtype's value is
   exactly."
  [^StringBuilder sb x at]
  (let [dtype (array/dtype x)
        n (array/size x)
        data (array/->clj (array/reshape x [n]))
        text (case (dt/kind dtype)
               :float #(double-text (double %))
               :int #(Long/toString (long %))
               :bool str)]
    (when-not (and (vector? data) (== n (count data)))
      (throw (unprintable at x "an array whose elements are not its shape's (a batched or traced value)")))
    (.append sb "#gen/array {:dtype ")
    (.append sb (str dtype))
    (.append sb ", :shape ")
    (emit-all! sb "[" (array/shape x) "]" at)
    (.append sb ", :data [")
    (reduce (fn [first? v] (when-not first? (.append sb \space)) (.append sb ^String (text v)) false) true data)
    (.append sb "]}")))

(defn- emit-node!
  "A choice map's node as an EDN map in canonical address order: a node
   under it as a nested map, a leaf as its value, tagged #gen/value when
   the value is a map (which would read as a node)."
  [^StringBuilder sb c at]
  (.append sb \{)
  (reduce (fn [first? [addr sub]]
            (let [at (conj at addr)]
              (when-not first? (.append sb ", "))
              (emit! sb addr at)
              (.append sb \space)
              (if (cm/has-value? sub)
                (let [v (cm/get-value sub)]
                  (when (map? v) (.append sb "#gen/value "))
                  (emit! sb v at))
                (emit-node! sb sub at))
              false))
          true (cm/submaps c))
  (.append sb \}))

(defn- emit-choicemap! [^StringBuilder sb c at]
  (if (cm/has-value? c)
    (emit! (.append sb "#gen/value ") (cm/get-value c) at)
    (emit-node! (.append sb "#gen/choicemap ") c at)))

(defn- emit!
  "x's EDN text appended to sb; at is x's path, for errors."
  [^StringBuilder sb x at]
  (cond
    (nil? x) (.append sb "nil")
    (boolean? x) (.append sb (if x "true" "false"))
    (number? x) (emit-number! sb x at)
    (or (string? x) (char? x)) (.append sb (pr-str x))
    (or (keyword? x) (symbol? x)) (emit-name! sb x at)
    (array/backend-array? x) (emit-array! sb x at)
    (rng/key? x) (let [[k0 k1] (rng/key-data x)] (.append sb (str "#gen/key [" k0 " " k1 "]")))
    (cm/choicemap? x) (emit-choicemap! sb x at)
    (trace/trace? x) (throw (unprintable at x "a trace: write it with write-trace"))
    (record? x) (throw (unprintable at x (if (gfi/generative-function? x)
                                           "a generative function, which is never written (read-trace takes one)"
                                           "a record")))
    (instance? clojure.lang.Sorted x) (throw (unprintable at x "a sorted collection, which would read back unsorted"))
    (map? x) (emit-map! sb x at)
    (vector? x) (emit-all! sb "[" x "]" at)
    (list? x) (emit-all! sb "(" x ")" at)
    (set? x) (emit-all! sb "#{" x "}" at)
    (seq? x) (throw (unprintable at x "a seq that is not a list: make it a vector"))
    (fn? x) (throw (unprintable at x "a function"))
    :else (throw (unprintable at x "an object EDN does not hold"))))

(defn- text-of
  "The text (f sb) appends to a new StringBuilder."
  [f]
  (let [sb (StringBuilder.)]
    (binding [*print-readably* true] (f sb))
    (str sb)))

(defn write
  "EDN text of x: a choice map, an array or any EDN data holding them (see
   the namespace doc). A value that is not EDN data is a loud error naming
   its path."
  [x]
  (text-of #(emit! % x [])))

;; ---------------------------------------------------------------------------
;; Reading
;; ---------------------------------------------------------------------------

(defn- malformed [tag why data]
  (ex-info (str "read: #" tag " " why) (merge {:reason :malformed :tag tag} data)))

(defn- fits?
  "Is v, an element of a #gen/array's data, a value of dtype exactly?"
  [dtype v]
  (case dtype
    :float64 (instance? Double v)
    :float32 (and (instance? Double v)
                  (let [d (double v)] (or (Double/isNaN d) (== d (unchecked-float d)))))
    :float16 (and (instance? Double v)
                  (let [d (double v) f (unchecked-float d)]
                    (or (Double/isNaN d) (and (== d f) (== f (Float/float16ToFloat (Float/floatToFloat16 f)))))))
    :bfloat16 (and (instance? Double v)
                   (let [d (double v) f (unchecked-float d)]
                     (or (Double/isNaN d) (and (== d f) (zero? (bit-and (Float/floatToRawIntBits f) 0xFFFF))))))
    :int64 (instance? Long v)
    :int32 (and (instance? Long v) (<= Integer/MIN_VALUE (long v) Integer/MAX_VALUE))
    :uint32 (and (instance? Long v) (<= 0 (long v) 0xFFFFFFFF))
    :bool (boolean? v)))

(defn- read-array [form]
  (let [bad #(throw (malformed 'gen/array %1 %2))]
    (when-not (and (map? form) (= #{:dtype :shape :data} (set (keys form))))
      (bad "is a map of :dtype, :shape and :data" {:keys (when (map? form) (vec (keys form)))}))
    (let [{:keys [dtype shape data]} form
          n (reduce *' 1 shape)]
      (when-not (contains? dt/dtypes dtype)
        (bad (str "has the dtype " (pr-str dtype) ", which is none of " (sort dt/dtypes)) {:dtype dtype}))
      (when-not (and (vector? shape) (every? #(and (instance? Long %) (<= 0 (long %))) shape))
        (bad "has a shape that is not a vector of sizes" {:shape shape}))
      (when-not (and (vector? data) (= (count data) n))
        (bad (str "of shape " shape " needs " n " elements of data") {:shape shape}))
      (when-let [i (first (keep-indexed (fn [i v] (when-not (fits? dtype v) i)) data))]
        (bad (str "holds " (pr-str (data i)) ", not a " dtype " value, at index " i)
             {:dtype dtype :index i :value (data i)}))
      (if (empty? shape)
        (array/array (data 0) {:dtype dtype})
        (cond-> (array/array (if (= :float (dt/kind dtype)) (double-array data) data) {:dtype dtype})
          (< 1 (count shape)) (array/reshape shape))))))

(defn- read-choicemap [form]
  (if (and (map? form) (not (record? form)))
    (cm/from-map form)
    (throw (malformed 'gen/choicemap "is a map of addresses to values and nested maps" {}))))

(defn- read-key [form]
  (if (and (vector? form) (== 2 (count form)) (every? #(and (instance? Long %) (<= 0 (long %) 0xFFFFFFFF)) form))
    (rng/key (bit-or (bit-shift-left (long (form 0)) 32) (long (form 1))))
    (throw (malformed 'gen/key "is two words [k0 k1], each in [0, 2^32)" {:form form}))))

(def ^:private readers
  {'gen/array read-array
   'gen/choicemap read-choicemap
   'gen/value cm/->Value
   'gen/key read-key
   'gen/trace (fn [_] (throw (ex-info "read: a #gen/trace is read with read-trace, given its generative function"
                                      {:reason :trace})))})

(defn- unknown-tag [tag _]
  (throw (ex-info (str "read: unknown tag #" tag) {:reason :unknown-tag :tag tag})))

(defn- read-one
  "The next EDN value of r, or eof; EDN's own errors as ex-info."
  [r opts]
  (try (edn/read opts r)
       (catch clojure.lang.ExceptionInfo e (throw e))
       (catch RuntimeException e
         (throw (ex-info (str "read: the text is not EDN: " (ex-message e)) {:reason :not-edn} e)))))

(defn- read-text
  "The one EDN value in the text s, its tags read by readers."
  [s readers]
  (when-not (string? s)
    (throw (ex-info (str "read: takes EDN text, not a " (.getName (class s))) {:reason :not-text :class (class s)})))
  (let [eof (Object.)
        opts {:readers readers :default unknown-tag :eof eof}
        r (java.io.PushbackReader. (java.io.StringReader. s))
        x (read-one r opts)]
    (when (identical? eof x)
      (throw (ex-info "read: the text holds no value" {:reason :empty})))
    (when-not (identical? eof (read-one r opts))
      (throw (ex-info "read: the text goes on after its value" {:reason :trailing-text})))
    x))

(defn read
  "The value the EDN text s holds: its arrays made on gen.array/*backend*,
   its choice maps and keys as written (see the namespace doc). Malformed
   text, an unknown tag and a #gen/trace (read-trace's) are loud errors."
  [s]
  (read-text s readers))

;; ---------------------------------------------------------------------------
;; Traces
;; ---------------------------------------------------------------------------

(def ^:private rebuild-key
  "generate's key for a rebuild, which draws nothing: every choice is
   constrained, and encapsulated randomness is refused."
  (rng/key 0))

(defn- same-number?
  "Are the 0-d arrays a and b the same number (a NaN the same as a NaN)?"
  [a b]
  (let [x (double (array/item a))
        y (double (array/item b))]
    (or (== x y) (and (Double/isNaN x) (Double/isNaN y)))))

(defn- rebuild
  "gf's trace on args by generate, every choice constrained, checked to
   draw nothing: no encapsulated randomness, and a weight that is its score
   (no choice proposed)."
  [op gf args choices]
  (let [{t :trace w :weight} (gfi/generate gf args choices rebuild-key)]
    (when-let [at (trace/omega-at t)]
      (throw (ex-info (str (name op) ": the rebuilt trace drew encapsulated randomness (:gen.trace/omega) at " at
                           ", which generate draws afresh (thesis §4.5), so it is not the written trace")
                      {:reason :encapsulated-randomness :path at})))
    (when-not (same-number? w (:score t))
      (throw (ex-info (str (name op) ": generate's weight " (array/item w) " is not its score " (array/item (:score t))
                           ": the choices are not all of the trace's, and generate proposed the others")
                      {:reason :choices-missing :weight w :score (:score t)})))
    t))

(defn- numerics
  "What the rounding of a score computed on backend b depends on: b's name
   and its float dtype."
  [b]
  {:backend (array/backend-name b) :float (:float (array/defaults b))})

(defn write-trace
  "EDN text of the trace tr, #gen/trace {:numerics :args :choices :score},
   from which read-trace rebuilds it, given its generative function. The
   score written is generate's, which this computes with tr's :gen-fn and
   checks to draw nothing; :numerics is the backend and float dtype it was
   computed under. A trace that keeps encapsulated randomness anywhere is a
   loud error (see the namespace doc)."
  [tr]
  (when-not (trace/trace? tr)
    (throw (ex-info (str "write-trace: takes a trace, not a " (.getName (class tr)))
                    {:reason :not-a-trace :class (class tr)})))
  (when-let [at (trace/omega-at tr)]
    (throw (ex-info (str "write-trace: the trace keeps encapsulated randomness (:gen.trace/omega) at " at
                         ", which generate would draw afresh (thesis §4.5): such traces are not serializable yet")
                    {:reason :encapsulated-randomness :path at})))
  (let [args (text-of #(emit! % (:args tr) [:args]))
        choices (text-of #(emit-choicemap! % (:choices tr) [:choices]))
        score (:score (rebuild :write-trace (:gen-fn tr) (:args tr) (:choices tr)))]
    (str "#gen/trace {:numerics " (text-of #(emit! % (numerics (array/backend score)) [:numerics]))
         ", :args " args ", :choices " choices ", :score " (text-of #(emit! % score [:score])) "}")))

;; a #gen/trace's form, as read-trace-data reads the tag
(deftype Written [form])

(defn read-trace-data
  "The parts of the written trace s (write-trace's text), not rebuilt:
   {:numerics :args :choices :score}, its arrays on gen.array/*backend*.
   Where read-trace refuses it, on another backend or float dtype,
   (gfi/generate gf args choices) loads it there."
  [s]
  (let [bad #(throw (malformed 'gen/trace % {}))
        ;; tags are read inner first, so a trace inside another is the second
        n (volatile! 0)
        x (read-text s (assoc readers 'gen/trace (fn [form]
                                                   (when (< 1 (long (vswap! n inc))) (bad "holds another #gen/trace"))
                                                   (Written. form))))]
    (when-not (instance? Written x)
      (throw (ex-info "read-trace: the text is not a written trace, #gen/trace {:numerics :args :choices :score}"
                      {:reason :not-a-trace})))
    (let [form (.-form ^Written x)]
      (when-not (and (map? form) (= #{:numerics :args :choices :score} (set (keys form))))
        (bad "is a map of :numerics, :args, :choices and :score"))
      (when-not (let [ns (:numerics form)]
                  (and (map? ns) (= #{:backend :float} (set (keys ns)))
                       (keyword? (:backend ns)) (contains? dt/dtypes (:float ns))))
        (bad "has :numerics that are not {:backend name :float dtype}"))
      (when-not (vector? (:args form)) (bad "has :args that are not a vector"))
      (when-not (cm/choicemap? (:choices form)) (bad "has :choices that are not a choice map"))
      (when-not (array/array? (:score form)) (bad "has a :score that is not a number"))
      form)))

(defn read-trace
  "gf's trace that the text s describes (write-trace's), rebuilt by
   generate with every choice constrained, on the backend and float dtype
   it was written under. Anything else is a loud error: another backend or
   float dtype (log densities round differently there: load the trace's
   choices with read-trace-data and generate instead), a score other than
   the written one (a different generative function), a weight other than
   its score (choices missing), or encapsulated randomness drawn."
  [gf s]
  (let [{:keys [args choices score] written :numerics} (read-trace-data s)
        here (numerics array/*backend*)]
    (when-not (= written here)
      (throw (ex-info (str "read-trace: the trace was written under " written " and is read under " here
                           ": an exact rebuild is per backend and float dtype, whose log densities round"
                           " differently. To load its choices here: (let [{:keys [args choices]}"
                           " (gen.serialize/read-trace-data s)] (gen.gfi/generate gf args choices))")
                      {:reason :other-numerics :written written :reading here})))
    (let [t (rebuild :read-trace gf args choices)]
      (when-not (same-number? (:score t) score)
        (throw (ex-info (str "read-trace: the rebuilt score " (array/item (:score t)) " is not the written "
                             (array/item score) ": a different generative function")
                        {:reason :score-differs :score (:score t) :written score})))
      t)))
