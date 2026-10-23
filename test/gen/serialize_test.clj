(ns gen.serialize-test
  "gen.serialize: choice maps, arrays and traces as EDN text.
   - Arrays round-trip bit for bit in every dtype the backend has, special
     floats included, and keep their kind: a host number stays one, and a
     backend array (MLX's 0-d ones too) stays one.
   - Choice maps and traces of random models round-trip (test.check). A
     fresh trace keeps its score bit for bit. An updated one is rebuilt
     with generate's score, within rounding of its own.
   - The traces of every kind round-trip: Map, Unfold, Switch, Recurse,
     static, call-at, choice-at, a distribution, a deterministic function and
     an internal-proposal override.
   - Non-numeric and map-valued choices, keys and EDN data round-trip, and
     the text of a choice map is a function of its value.
   - Encapsulated randomness is refused at write and at read.
   - Loud errors: unprintable values with their paths, unknown tags,
     malformed tags and text, and rebuilds that are not exact.
   Each property was checked against a planted bug, noted in place."
  (:refer-clojure :exclude [+ *])
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as tcg]
            [clojure.test.check.properties :as prop]
            [gen.array :as array :refer [+ *]]
            [gen.choicemap :as cm]
            [gen.combinators :as comb]
            [gen.custom :as custom]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dist.dsl :refer [dist]]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.proposal :as proposal]
            [gen.random :as rng]
            [gen.random-models :as rm]
            [gen.selection :as sel]
            [gen.serialize :as serialize]
            [gen.test-support :as ts]))

(defn- round-trip [x] (serialize/read (serialize/write x)))

(defn- flat [x] (array/->clj (array/reshape x [(array/size x)])))

(defn- same-double? [^double a ^double b]
  (or (and (Double/isNaN a) (Double/isNaN b))
      (== (Double/doubleToRawLongBits a) (Double/doubleToRawLongBits b))))

(defn- same?
  "Are a and b the same value bit for bit? Arrays of one kind (host number
   or backend array), dtype and shape, their elements' bits the same (a NaN
   as any NaN); doubles by their bits; choice maps, maps and sequences by
   their parts; anything else of one class and equal."
  [a b]
  (cond (array/backend-array? a) (and (array/backend-array? b) (= (array/dtype a) (array/dtype b))
                                      (= (array/shape a) (array/shape b)) (same? (flat a) (flat b)))
        (double? a) (and (double? b) (same-double? a b))
        (cm/choicemap? a) (and (cm/choicemap? b)
                               (if (cm/has-value? a)
                                 (and (cm/has-value? b) (same? (cm/get-value a) (cm/get-value b)))
                                 (let [sa (cm/submaps a) sb (cm/submaps b)]
                                   (and (not (cm/has-value? b)) (= (map first sa) (map first sb))
                                        (every? true? (map #(same? (second %1) (second %2)) sa sb))))))
        (map? a) (and (map? b) (= (set (keys a)) (set (keys b))) (every? #(same? (get a %) (get b %)) (keys a)))
        (sequential? a) (and (sequential? b) (= (vector? a) (vector? b)) (= (count a) (count b))
                             (every? true? (map same? a b)))
        :else (and (= (class a) (class b)) (= a b))))

;; ---------------------------------------------------------------------------
;; Arrays
;; ---------------------------------------------------------------------------

(defn- dtypes
  "The dtypes the backend has."
  []
  (if (ts/jvm?)
    [:bool :int32 :int64 :float32 :float64]
    (cond-> [:bool :int32 :int64 :uint32 :float16 :bfloat16 :float32] (ts/float64?) (conj :float64))))

(def ^:private special-floats
  "Floats whose bits a decimal could lose: signed zeros, the infinities,
   NaN, values float32 rounds (0.1), the extremes and subnormals of float32
   and float64."
  [0.0 -0.0 1.0 -1.5 0.1 (/ 1.0 3.0) ##Inf ##-Inf ##NaN
   3.4028234663852886E38 1.1754943508222875E-38 1.401298464324817E-45 4.9E-324 2.2250738585072014E-308 1.0E300])

(def ^:private special-ints [0 1 -1 7 Integer/MAX_VALUE Integer/MIN_VALUE 4294967295 (long (Math/pow 2 53))])

(defn- values-of
  "Host values of dtype's kind, which array/array makes into its values."
  [dtype]
  (case dtype
    :bool [true false]
    (:int32 :int64 :uint32) special-ints
    special-floats))

;; Planted bugs (2026-10-23), each failing this property: -0.0 written as
;; 0.0; a float32 written as its float's shortest decimal (Float/toString:
;; 0.1 reads back as a double that is no float32 value, a loud error); the
;; data of a 2-d array written column first.
(defn- random-value
  "A value of dtype's kind: one of the special ones half the time."
  [^java.util.Random r dtype]
  (let [pool (values-of dtype)]
    (cond (.nextBoolean r) (nth pool (.nextInt r (count pool)))
          (= :bool dtype) (.nextBoolean r)
          (#{:int32 :int64 :uint32} dtype) (long (.nextInt r))
          :else (.nextGaussian r))))

(defspec arrays-round-trip-bit-for-bit 200
  (prop/for-all [dtype (tcg/elements (dtypes))
                 shape (tcg/vector (tcg/choose 0 3) 0 3)
                 seed tcg/nat]
    (let [r (java.util.Random. seed)
          data (vec (repeatedly (reduce clojure.core/* 1 shape) #(random-value r dtype)))
          x (if (empty? shape)
              (array/array (first data) {:dtype dtype})
              (array/reshape (array/array data {:dtype dtype}) shape))]
      (same? x (round-trip x)))))

(deftest special-floats-round-trip-in-every-float-dtype
  (doseq [dtype (filter #{:float16 :bfloat16 :float32 :float64} (dtypes))]
    (testing (name dtype)
      (let [x (array/array special-floats {:dtype dtype})
            text (serialize/write x)]
        (is (same? x (serialize/read text)))
        (is (re-find #" -0\.0 " text) "-0.0 keeps its sign")
        (is (re-find #" ##Inf ##-Inf ##NaN " text) "the special values are written as EDN's")))))

(deftest a-value-keeps-its-kind
  (testing "host numbers and booleans stay host values, of their class"
    (doseq [x [1.5 -0.0 ##NaN 3 true false (long -7)]]
      (is (same? x (round-trip x)) (pr-str x))))
  (testing "Integer, Short and Byte read back as a Long, a Float as the Double of its value"
    (is (same? 3 (round-trip (int 3))))
    (is (same? 3 (round-trip (short 3))))
    (is (same? 3 (round-trip (byte 3))))
    (is (same? (double (float 0.1)) (round-trip (float 0.1)))))
  (testing "a 0-d array stays one where the backend has them (MLX), and is a host number where it has none (JVM)"
    (doseq [dtype (dtypes)]
      (let [x (array/array (if (= :bool dtype) true 3) {:dtype dtype})
            y (round-trip x)]
        (is (= (array/backend-array? x) (array/backend-array? y)) (name dtype))
        (is (same? x y) (name dtype))
        (is (= (ts/jvm?) (not (array/backend-array? y))) "the JVM backend has no 0-d arrays"))))
  ;; Planted bug (2026-10-23): shape [] read as [1]. The JVM backend never
  ;; writes a 0-d #gen/array, so the round trips above pass it there, and
  ;; this text, as MLX writes it, fails.
  (testing "a 0-d #gen/array, as MLX writes it, reads back as the backend's 0-d array"
    (doseq [[text v] [["#gen/array {:dtype :float32, :shape [], :data [0.10000000149011612]}" (double (float 0.1))]
                      ["#gen/array {:dtype :int32, :shape [], :data [-4]}" -4]
                      ["#gen/array {:dtype :bool, :shape [], :data [true]}" true]]]
      (let [y (serialize/read text)]
        (is (= [] (array/shape y)) text)
        (is (same? v (array/item y)) text)
        (is (= (ts/jvm?) (not (array/backend-array? y))) text)))))

(deftest an-array-is-written-with-its-dtype-and-shape
  (let [x (array/array [[1 2 3] [4 5 6]] {:dtype :int32})]
    (is (= "#gen/array {:dtype :int32, :shape [2 3], :data [1 2 3 4 5 6]}" (serialize/write x))))
  (is (= [0 3] (array/shape (round-trip (array/zeros [0 3])))) "an empty array keeps its shape"))

;; ---------------------------------------------------------------------------
;; Choice maps and EDN data
;; ---------------------------------------------------------------------------

;; Planted bug (2026-10-23): a leaf's map value written without #gen/value
;; reads back as a node, and this fails.
(defspec choice-maps-of-random-models-round-trip 100
  (prop/for-all [[_ gf args] rm/gen-model
                 seed tcg/nat]
    (let [c (:choices (gfi/simulate gf args (rng/key seed)))]
      (same? c (round-trip c)))))

(dist labeled [labels probs] (nth labels (d/categorical probs)))

(deftest non-numeric-choices-round-trip
  (let [model (gen [] (trace :color (labeled [:red :green :blue] [0.2 0.3 0.5]))
                (trace :word (labeled ["one" "two words" "\"quoted\"\n"] [0.3 0.3 0.4]))
                (trace :sym (labeled ['alpha 'beta/gamma] [0.5 0.5])))]
    (doseq [k (range 5)]
      (let [c (:choices (gfi/simulate model [] (rng/key k)))]
        (is (same? c (round-trip c)))
        (is (keyword? (cm/get-choice (round-trip c) :color)))))))

(deftest choice-maps-of-every-shape-round-trip
  (doseq [c [cm/EMPTY
             (cm/choicemap :x 1.0)
             (cm/set-choice cm/EMPTY [] 2.5)
             (cm/choicemap :m (cm/->Value {:a 1 :b [2.0]}))
             (cm/set-choice cm/EMPTY [] {:a 1})
             (cm/choicemap [:y 0] 1 [:y 1] 2 "s" 3 'sym 4 1.5 5)
             (cm/choicemap :pair [(array/array [1.0 2.0]) true] :v (array/array [0.5 -0.0]))
             (cm/set-submap (cm/choicemap :x 1.0) :empty cm/EMPTY)]]
    (is (same? c (round-trip c)) (pr-str c)))
  (testing "a leaf whose value is a map is tagged, so it reads back as a value"
    (is (= "#gen/choicemap {:m #gen/value {:a 1}}" (serialize/write (cm/choicemap :m (cm/->Value {:a 1})))))
    (is (cm/has-value? (cm/get-submap (round-trip (cm/choicemap :m (cm/->Value {:a 1}))) :m)))))

;; Planted bug (2026-10-23): a node's entries in the map's own order, not
;; the canonical one: the two texts differ, and this fails.
(deftest a-choice-map's-text-is-a-function-of-its-value
  (let [c1 (cm/choicemap :b 1.0 :a 2.0 [:c 1] 3.0 [:c 0] 4.0)
        c2 (cm/choicemap [:c 0] 4.0 :a 2.0 [:c 1] 3.0 :b 1.0)]
    (is (= c1 c2))
    (is (= (serialize/write c1) (serialize/write c2)))
    (is (= "#gen/choicemap {:a 2.0, :b 1.0, :c {0 4.0, 1 3.0}}" (serialize/write c1)))))

(deftest edn-data-round-trips
  (let [x {:nil nil :bools [true false] :long -3 :double 2.5 :big 12345678901234567890N :decimal 1.5M :ratio 1/3
           :string "a \"b\"\n\\" :char \a :keyword :ns/name :symbol 'ns/sym :list '(1 (2) ()) :set #{1 :a "b"}
           :nested [{:a [1 {:b #{2}}]}] "string key" 1 [1 2] :vector-key
           :key (rng/key 7) :keys (rng/split (rng/key 1) 3) :choices (cm/choicemap :x 1.0)
           :array (array/array [[1.0 2.0]])}
        y (round-trip x)]
    (is (same? x y))
    (is (list? (:list y)))
    (is (rng/key? (:key y)))
    (is (= (rng/key-data (:key x)) (rng/key-data (:key y))))
    (is (= (rng/split (:key x) 2) (rng/split (:key y) 2)) "a key read back draws as the key written")))

;; ---------------------------------------------------------------------------
;; Traces
;; ---------------------------------------------------------------------------

(defn- rebuilt
  "tr written and read back, by gf."
  ([tr] (rebuilt (:gen-fn tr) tr))
  ([gf tr] (serialize/read-trace gf (serialize/write-trace tr))))

(defn- same-trace?
  "Is t, read back, tr with its choices, arguments, score and return value
   bit for bit?"
  [tr t]
  (and (same? (:choices tr) (:choices t)) (same? (:args tr) (:args t))
       (same? (:score tr) (:score t)) (same? (:retval tr) (:retval t))))

;; Planted bug (2026-10-23): write-trace writing the trace's own score as
;; it is, and read-trace comparing it with its rebuild's, passes this and
;; fails the property of updated traces below.
(defspec traces-of-random-models-round-trip 100
  (prop/for-all [[_ gf args] rm/gen-model
                 seed tcg/nat]
    (let [tr (gfi/simulate gf args (rng/key seed))]
      (same-trace? tr (rebuilt tr)))))

(defspec updated-traces-of-random-models-are-rebuilt-with-generate's-score 50
  ;; Map and Unfold keep their scores incrementally, so after a few moves a
  ;; trace's score is generate's only within rounding
  (prop/for-all [[_ gf args] rm/gen-model
                 seed tcg/nat]
    (let [tr (reduce (fn [tr i]
                       (let [as (cm/addresses (:choices tr))]
                         (if (seq as)
                           (:trace (gfi/regenerate tr (sel/from-paths [(nth as (mod (clojure.core/* 7 i) (count as)))])
                                                   (rng/key (clojure.core/+ seed i 1))))
                           tr)))
                     (gfi/simulate gf args (rng/key seed))
                     (range 5))
          t (rebuilt tr)
          fresh (:trace (gfi/generate gf args (:choices tr) (rng/key 9)))]
      (and (same? (:choices tr) (:choices t)) (same? (:args tr) (:args t))
           (same? (:score fresh) (:score t))
           (ts/close? (:score tr) (:score t))))))

(def ^:private kernel (gen [x] (trace :v (d/normal x 1.0))))
(def ^:private step (gen [_t s drift] (trace :s (d/normal (+ s drift) 1.0))))
(def ^:private grow
  (gen [depth]
    (let [split? (trace :split (d/bernoulli (if (clojure.core/< depth 2) 0.6 0.0)))]
      {:value (trace :v (d/normal depth 1)) :children (if split? [(inc depth) (inc depth)] [])})))
(def ^:private fold (gen [v ws] (let [s (reduce + v ws)] (trace :obs (d/normal s 1)) s)))
(def ^:private one (gen [m] (trace :a (d/normal m 1.0))))
(def ^:private two (gen [m] (trace :c (d/normal (trace :b (d/gamma 2.0 1.0)) 1.0))))
(def ^:private point (gen ^:static [x slope] (let [y (trace :y (d/normal (* slope x) 1))] y)))
(def ^:private linear
  (gen ^:static [xs]
    (let [slope (trace :slope (d/normal 0 1))
          ys (splice :ys (comb/map point) xs (vec (repeat (count xs) slope)))]
      ys)))
(def ^:private doubling (custom/deterministic {:apply (fn [[a]] (* 2.0 a))}))
(def ^:private with-x (gen [m] (let [x (trace :x (d/normal m 2))] (trace :y (d/normal x 1)) x)))

(def ^:private kinds
  "[label gf args] of every kind of generative function."
  [["dynamic" with-x [0.5]]
   ["map" (comb/map kernel) [[0.0 1.0 2.0]]]
   ["unfold" (comb/unfold step) [4 0.0 0.1]]
   ["switch" (comb/switch one two) [1 0.5]]
   ["keyed switch" (comb/switch {:one one :two two}) [:one 0.5]]
   ["recurse" (comb/recurse grow fold 2) [0]]
   ["static" linear [[1.0 2.0 3.0]]]
   ["call-at" (comb/call-at one) [0.5 :here]]
   ["choice-at" (comb/choice-at d/normal) [0.0 1.0 :x]]
   ["distribution" (d/normal 0.0 1.0) []]
   ["vector site" (gen [] (trace :v (d/normal (array/array [0.0 1.0]) 1.0))) []]
   ["product" (gen [] (trace :pair ((d/product d/normal d/bernoulli) [0.0 1.0] [0.3]))) []]
   ["deterministic" doubling [1.5]]
   ["deterministic call" (gen [x] (trace :y (d/normal (splice :f doubling x) 1))) [1.5]]
   ["internal proposal" (proposal/with-internal-proposal with-x
                          (gen [sigma m] (when (cm/empty? (cm/get-submap sigma :x)) (trace :x (d/normal (+ m 0.5) 1.5)))))
    [0.5]]])

(deftest traces-of-every-kind-round-trip
  (doseq [[label gf args] kinds
          k (range 3)]
    (testing label
      (let [tr (gfi/simulate gf args (rng/key k))]
        (is (same-trace? tr (rebuilt tr)))))))

(deftest a-trace-is-written-as-its-arguments-choices-and-score
  (let [tr (gfi/simulate with-x [0.5] (rng/key 1))
        text (serialize/write-trace tr)]
    (is (= (str "#gen/trace {:args [0.5], :choices " (serialize/write (:choices tr)) ", :score " (serialize/write (:score tr)) "}")
           text))
    (is (= {:reason :trace} (ts/thrown-data (serialize/read text))) "read refuses a trace: it needs the generative function")))

;; ---------------------------------------------------------------------------
;; Encapsulated randomness
;; ---------------------------------------------------------------------------

(def ^:private noisy (gen [] (trace :x (d/normal (untraced (d/normal 0 1)) 1))))

(def ^:private estimated
  (custom/estimated {:addresses [:y]
                     :estimate (fn [_ choices key]
                                 (+ (dc/log-prob (d/normal 0 1) (cm/get-choice choices :y)) (rng/normal key)))
                     :simulate (fn [_ key] {:choices (cm/choicemap :y (rng/normal key)) :log-estimate 0.0})}))

;; Planted bug (2026-10-23): omega-at finding nothing. write-trace then
;; writes these traces, and read-trace rebuilds them silently with ω drawn
;; from the rebuild's key, not the trace's: write-trace's own rebuild drew
;; the same, so the written score agrees.
(deftest encapsulated-randomness-is-refused
  (doseq [[label gf args at] [["untraced draws" noisy [] []]
                              ["a draw only the return value reads"
                               (gen [] (trace :x (d/normal 0 1)) (untraced (d/normal 0 1))) [] []]
                              ["a call's" (gen [] (splice :c noisy)) [] [:gen.dynamic/subtraces [:c]]]
                              ["a Map element's" (comb/map (gen [m] (trace :v (d/normal (+ m (untraced (d/normal 0 1))) 1))))
                               [[0.0 1.0]] [:gen.combinators/subtraces 0]]
                              ["an estimated density" estimated [] []]]]
    (testing label
      (let [tr (gfi/simulate gf args (rng/key 1))]
        (is (= {:reason :encapsulated-randomness :path at} (ts/thrown-data (serialize/write-trace tr)))))))
  (testing "a rebuild that draws some is refused too"
    (let [text (serialize/write-trace (gfi/simulate (gen [] (trace :x (d/normal 0 1))) [] (rng/key 1)))]
      (is (= {:reason :encapsulated-randomness :path []} (ts/thrown-data (serialize/read-trace noisy text)))))))

;; ---------------------------------------------------------------------------
;; Loud errors
;; ---------------------------------------------------------------------------

(defrecord Point [x y])

(deftest what-is-not-edn-data-is-a-loud-error-naming-its-path
  (doseq [[x path] [[{:f inc} [:f]]
                    [[1 (->Point 1 2)] [1]]
                    [{:a [1 (map inc [1])]} [:a 1]]
                    [(sorted-map :a 1) []]
                    [[(sorted-set 1)] [0]]
                    [{:gf with-x} [:gf]]
                    [[(gfi/simulate with-x [0.5] (rng/key 1))] [0]]
                    [(Object.) []]
                    [(double-array 2) []]
                    [(keyword "a b") []]
                    [[(symbol "nil")] [0]]
                    [(cm/choicemap :x inc) [:x]]
                    [(cm/choicemap [:y 3] (atom 1)) [:y 3]]]]
    (let [data (ts/thrown-data (serialize/write x))]
      (is (= :unprintable (:reason data)) (pr-str x))
      (is (= path (:path data)) (pr-str x))))
  (testing "in a trace's arguments"
    (let [tr (gfi/simulate (gen [f] (trace :x (d/normal (f 0.0) 1))) [identity] (rng/key 1))]
      (is (= {:reason :unprintable :path [:args 0]} (select-keys (ts/thrown-data (serialize/write-trace tr)) [:reason :path]))))))

(deftest malformed-text-is-a-loud-error
  (doseq [[text reason] [["#foo/bar 1" :unknown-tag]
                         ["#gen/arrays {}" :unknown-tag]
                         ["#gen/array [:float64 [1.0]]" :malformed]
                         ["#gen/array (:dtype :float64 :shape)" :malformed]
                         ["#gen/array {:dtype :float64 :shape [2]}" :malformed]
                         ["#gen/array {:dtype :float128 :shape [1] :data [1.0]}" :malformed]
                         ["#gen/array {:dtype :float64 :shape [-1] :data []}" :malformed]
                         ["#gen/array {:dtype :float64 :shape [2] :data [1.0]}" :malformed]
                         ["#gen/array {:dtype :float64 :shape [] :data [1]}" :malformed]
                         ["#gen/array {:dtype :float32 :shape [1] :data [0.1]}" :malformed]
                         ["#gen/array {:dtype :float16 :shape [1] :data [1.0E-10]}" :malformed]
                         ["#gen/array {:dtype :bfloat16 :shape [1] :data [1.1]}" :malformed]
                         ["#gen/array {:dtype :int32 :shape [1] :data [2147483648]}" :malformed]
                         ["#gen/array {:dtype :int32 :shape [1] :data [1.0]}" :malformed]
                         ["#gen/array {:dtype :uint32 :shape [1] :data [-1]}" :malformed]
                         ["#gen/array {:dtype :int64 :shape [1] :data [99999999999999999999]}" :malformed]
                         ["#gen/array {:dtype :bool :shape [1] :data [1]}" :malformed]
                         ["#gen/choicemap [1 2]" :malformed]
                         ["#gen/key [1]" :malformed]
                         ["#gen/key [1 4294967296]" :malformed]
                         ["" :empty]
                         ["  " :empty]
                         ["1 2" :trailing-text]
                         ["[1 2" :not-edn]
                         ["{:a 1 :a 2}" :not-edn]]]
    (is (= reason (:reason (ts/thrown-data (serialize/read text)))) text))
  (is (= :not-text (:reason (ts/thrown-data (serialize/read 1.0)))))
  (testing "no eval, and no tag of *data-readers*"
    (is (= :not-edn (:reason (ts/thrown-data (serialize/read "#=(+ 1 2)")))))
    (is (= :unknown-tag (:reason (ts/thrown-data (binding [*data-readers* {'gen/extra identity}]
                                                   (serialize/read "#gen/extra 1"))))))))

(deftest a-rebuild-that-is-not-exact-is-a-loud-error
  (let [tr (gfi/simulate with-x [0.5] (rng/key 1))
        text (serialize/write-trace tr)]
    ;; Planted bug (2026-10-23): read-trace not comparing the scores; the
    ;; other model is read back silently.
    (testing "another generative function: the scores differ"
      (is (= :score-differs (:reason (ts/thrown-data (serialize/read-trace (gen [m] (let [x (trace :x (d/normal m 3))] (trace :y (d/normal x 1)) x)) text))))))
    ;; Planted bug (2026-10-23): rebuild not comparing the weight with the
    ;; score; :z is proposed silently.
    (testing "choices missing: generate proposes them"
      (is (= :choices-missing (:reason (ts/thrown-data (serialize/read-trace (gen [m] (let [x (trace :x (d/normal m 2))] (trace :y (d/normal x 1)) (trace :z (d/normal 0 1)) x)) text))))))
    (testing "choices the generative function does not make: generate's own error"
      (is (:unvisited (ts/thrown-data (serialize/read-trace (gen [m] (trace :x (d/normal m 2))) text)))))
    (testing "text that is not one written trace"
      (doseq [[s reason] [["1" :not-a-trace]
                          ["[#gen/trace {:args [] :choices #gen/choicemap {} :score 0.0}]" :not-a-trace]
                          ["#gen/trace {:args [] :choices #gen/choicemap {}}" :malformed]
                          ["#gen/trace (:args [] :choices)" :malformed]
                          ["#gen/trace {:args () :choices #gen/choicemap {} :score 0.0}" :malformed]
                          ["#gen/trace {:args [] :choices {} :score 0.0}" :malformed]
                          ["#gen/trace {:args [] :choices #gen/choicemap {} :score :a}" :malformed]
                          ["#gen/trace {:args [#gen/trace {:args [] :choices #gen/choicemap {} :score 0.0}] :choices #gen/choicemap {} :score 0.0}" :malformed]]]
        (is (= reason (:reason (ts/thrown-data (serialize/read-trace with-x s)))) s)))))
