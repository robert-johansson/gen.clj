(ns gen.array.host-test
  "gen.array.host: host data parsed and checked, dtypes inferred, arrays
   printed as gen.serialize reads them, and the math of host scalars, which
   must be what gen.array computes for arrays of the backend (genclj-w867).
   Planted (2026-11-06): :remainder on longs by rem in place of floorMod
   fails host-scalars-compute-what-arrays-do and integer-edges; parse
   accepting ragged rows fails data-is-parsed-and-checked; print-array
   flattening one level fewer fails arrays-print-as-they-read."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [gen.array :as array]
            [gen.array.host :as host]
            [gen.serialize :as serialize]
            [gen.test-support :as ts]))

;; ---------------------------------------------------------------------------
;; Host data
;; ---------------------------------------------------------------------------

(deftest data-is-parsed-and-checked
  (is (= [[2 3] [1 2 3 4 5 6]] (host/parse [[1 2 3] [4 5 6]])))
  (is (= [[0] []] (host/parse [])))
  (is (= [[2 1] [true false]] (host/parse [[true] [false]])))
  (let [a (double-array [1.0 2.0])]
    (is (= [[2] a] (update (host/parse a) 1 identity)))
    (is (identical? a (second (host/parse a))) "a double[] is its own values"))
  (is (= [[2] [1 2]] (update (host/parse (list 1 2)) 1 vec)) "any sequential data")
  (testing "loud errors"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"not rectangular" (host/parse [[1 2] [3]])))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"not rectangular" (host/parse [[1 2] 3])))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"not a number or boolean" (host/parse [1 :a])))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"Cannot make an array" (host/parse "12")))))

(deftest dtypes-are-inferred
  (let [defaults {:float :float32 :int :int32}]
    (is (= :float32 (host/infer-dtype [1 2.0] defaults)) "any float makes it float")
    (is (= :float32 (host/infer-dtype [1 1/2] defaults)) "as does a ratio")
    (is (= :float32 (host/infer-dtype [1M] defaults)) "or a decimal")
    (is (= :float32 (host/infer-dtype [] defaults)) "no values: the float")
    (is (= :float32 (host/infer-dtype (double-array 2) defaults)))
    (is (= :bool (host/infer-dtype [true false] defaults)))
    (is (= :int32 (host/infer-dtype [1 2] defaults)))))

(deftest arrays-print-as-they-read
  (let [printed (fn [dtype shape data] (with-out-str (host/print-array dtype shape data *out*)))]
    (is (= "#gen/array {:dtype :float64, :shape [2 2], :data [1.0 2.0 3.0 4.0]}"
           (printed :float64 [2 2] [[1.0 2.0] [3.0 4.0]])))
    (is (= "#gen/array {:dtype :int64, :shape [], :data [7]}" (printed :int64 [] 7)) "a 0-d array's one element")
    (is (= "#gen/array {:dtype :bool, :shape [1 2 1], :data [true false]}" (printed :bool [1 2 1] [[[true] [false]]])))
    (is (= "#gen/array {:dtype :float64, :shape [4], :data [0.1 0.2 ...]}"
           (binding [*print-length* 2] (printed :float64 [4] [0.1 0.2 0.3 0.4])))
        "*print-length* shortens the elements"))
  (testing "a backend array's text reads back as it"
    (let [x (array/array [[0.1 2.0] [-3.5 1e-7]])]
      (is (= (ts/host x) (ts/host (serialize/read (pr-str x))))))))

;; ---------------------------------------------------------------------------
;; Host math: every op on host scalars as gen.array computes it on arrays
;; ---------------------------------------------------------------------------

(def ^:private binaries
  {:add array/+ :subtract array/- :multiply array/* :divide array// :power array/pow
   :maximum array/maximum :minimum array/minimum :remainder array/mod :floor-divide array/floor-div
   :arctan2 array/atan2 :less array/< :less-equal array/<= :greater array/> :greater-equal array/>=
   :equal array/== :not-equal array/!= :logical-and array/logical-and :logical-or array/logical-or})

(def ^:private int-binaries
  [:add :subtract :multiply :power :maximum :minimum :remainder :floor-divide
   :less :less-equal :greater :greater-equal :equal :not-equal])

(def ^:private unaries
  {:negative array/negate :abs array/abs :sign array/sign :sqrt array/sqrt :exp array/exp :log array/log
   :expm1 array/expm1 :log1p array/log1p :sin array/sin :cos array/cos :tan array/tan :tanh array/tanh
   :arcsin array/asin :arccos array/acos :arctan array/atan :sinh array/sinh :cosh array/cosh
   :erf array/erf :erfinv array/erfinv :floor array/floor :ceil array/ceil :round array/round
   :isnan array/isnan :isinf array/isinf})

(defn- same?
  "Is the host scalar a the one-element array result b's element: equal,
   both NaN, or close in the backend's float?"
  [a b]
  (let [y (first (ts/host b))]
    (cond (boolean? a) (= a y)
          (Double/isNaN (double a)) (Double/isNaN (double y))
          (Double/isInfinite (double a)) (== a y)
          :else (ts/close? a y))))

(defn- one [x] (array/array [x]))

(def ^:private quarters
  "Doubles in [-4, 4] that every float holds exactly, 0 among them."
  (gen/fmap #(/ % 4.0) (gen/choose -16 16)))

(defspec host-scalars-compute-what-arrays-do 300
  ;; the backend's arrays run its own kernels (the JVM backend's loops,
  ;; MLX's), so this checks host math against an independent computation;
  ;; integer division by zero only on the JVM backend, whose semantics
  ;; (0, as NumPy's) host math copies
  (prop/for-all [op (gen/elements (keys binaries)) x quarters y quarters
                 int-op (gen/elements int-binaries) i (gen/choose -9 9) j (gen/choose -9 9)
                 uop (gen/elements (keys unaries)) u quarters
                 p gen/boolean q gen/boolean
                 bop (gen/elements [:logical-and :logical-or :equal :not-equal :maximum :minimum :less :greater-equal])]
    (and (same? (host/binary op x y) ((binaries op) (one x) (one y)))
         (or (and (zero? j) (#{:remainder :floor-divide} int-op) (not (ts/jvm?)))
             (and (= :power int-op) (neg? j))
             (same? (host/binary int-op i j) ((binaries int-op) (one i) (one j))))
         (same? (host/unary uop u) ((unaries uop) (one u)))
         (same? (host/binary bop p q) ((binaries bop) (one p) (one q)))
         (same? (host/unary :logical-not p) (array/logical-not (one p))))))

(deftest integer-edges
  (is (= 2 (host/binary :remainder -7 3)) "the sign of the divisor")
  (is (= -2 (host/binary :remainder 7 -3)))
  (is (= -3 (host/binary :floor-divide -7 3)))
  (is (= 0 (host/binary :remainder 5 0)) "0 for a zero divisor, as NumPy's")
  (is (= 0 (host/binary :floor-divide 5 0)))
  (is (= Long/MIN_VALUE (host/binary :add Long/MAX_VALUE 1)) "int64 wraps")
  (is (= 2.0 (host/binary :remainder -7.0 3.0)))
  (is (Double/isNaN (host/binary :remainder 5.0 0.0)) "NaN for a zero float divisor")
  (is (= 3.5 (host/binary :add 1 2.5)) "a long and a double promote")
  (is (instance? Long (host/binary :add 1 2)))
  (is (true? (host/binary :maximum true false)) "booleans: or")
  (is (true? (host/binary :less false true)) "and compare as 0 and 1"))

(deftest where-elementwise-and-reductions
  (is (= 1.0 (host/where true 1 2.0)) "the branches' common dtype")
  (is (instance? Long (host/where false 1 2)))
  (is (= 2.0 (host/elementwise :where [false 1.0 2.0])))
  (is (= 3 (host/elementwise :stop-gradient [3])))
  (is (= -2.0 (host/elementwise :negative [2.0])))
  (is (= 5 (host/elementwise :add [2 3])))
  (testing "a reduction of a host scalar: the scalar, or its index"
    (is (= 1 (host/reduction :sum true)))
    (is (= 2.5 (host/reduction :prod 2.5)))
    (is (= 2.5 (host/reduction :max 2.5)))
    (is (= 0 (host/reduction :argmin 2.5)))
    (is (false? (host/reduction :all 0.0)))
    (is (true? (host/reduction :any 3)))))
