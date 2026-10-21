(ns gen.backend.jvm-test
  "The JVM backend's own limits (gen.backend.jvm): a JVM array holds at most
   Integer/MAX_VALUE - 8 cells, so an array of a shape with more is a loud
   error at its allocation (gen.array.shape/cells), naming the shape, its
   cell count and the limit, before anything is allocated. Run on the JVM
   backend whatever the suite's (with-backend).

   The cell count once went through an int: (double-array n) takes n's
   int value, so it wrapped modulo 2^32, and [65536 65537] (2^32 + 65536
   cells) made an array over a buffer of 65,536 (genclj-4qrv). Mutations
   checked (2026-10-21, each planted, these tests run, reverted): -full
   allocating by the unchecked size (the reproductions and every
   allocation, 5 failures); strided-offsets counting by size (the
   broadcast reproduction); size multiplying unchecked (counts-are-exact)."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [gen.array.shape :as sh]
            [gen.random :as rng]))

(def ^:private jvm (array/backend-named :jvm))

(defn- failure
  "The ex-data of what (f) throws on the JVM backend, its message under
   :message, and :ms, how long it took (nil data when nothing is thrown)."
  [f]
  (let [t0 (System/nanoTime)
        data (try (array/with-backend jvm (f)) nil
                  (catch clojure.lang.ExceptionInfo e (assoc (ex-data e) :message (ex-message e))))]
    (some-> data (assoc :ms (/ (- (System/nanoTime) t0) 1e6)))))

(defn- refused?
  "Did (f) fail as an array past the JVM's limit should: :too-many-cells,
   naming the shape and the limit, at once (nothing of gigabytes
   allocated)?"
  [f shape]
  (let [{:keys [reason message ms] :as data} (failure f)]
    (and (= :too-many-cells reason)
         (= shape (:shape data))
         (re-find (re-pattern (str "a JVM array holds at most " sh/max-cells)) message)
         (< ms 1000))))

(deftest the-reproductions-fail-at-the-allocation
  (testing "10^12 cells once wrapped to a negative int: a bare NegativeArraySizeException"
    (is (refused? #(array/zeros [1000 1000 1000 1000]) [1000 1000 1000 1000]))
    (is (= 1000000000000 (:cells (failure #(array/zeros [1000 1000 1000 1000]))))))
  (testing "2^32 + 65536 cells once wrapped to 65,536: a malformed array"
    (is (refused? #(array/zeros [65536 65537]) [65536 65537]))
    (is (= 4295032832 (:cells (failure #(array/zeros [65536 65537]))))))
  (testing "10^10 cells once wrapped to a positive but wrong size: OutOfMemoryError"
    (is (refused? #(array/broadcast-to (array/array [1.0]) [100000 100000]) [100000 100000]))))

(deftest every-allocation-by-a-shape-is-checked
  (let [col (array/with-backend jvm (array/zeros [65536 1]))
        row (array/with-backend jvm (array/zeros [1 65537]))]
    (is (refused? #(array/+ col row) [65536 65537]) "a broadcasting binary op")
    (is (refused? #(array/where (array/> col 0.0) col row) [65536 65537]) "where")
    (is (refused? #(array/matmul col row) [65536 65537]) "matmul's output")
    (is (refused? #(rng/uniform (rng/key 1) [65536 65537]) [65536 65537]) "random draws")
    (is (refused? #(array/ones [65536 65537]) [65536 65537]) "full")
    (is (refused? #(array/arange 0 3000000000) [3000000000]) "arange")))

(deftest counts-are-exact
  (is (= :too-many-cells (:reason (failure #(array/zeros [4294967296 4294967296]))))
      "2^64 cells wrapped to 0 in a long")
  (is (= [4294967296 4294967296 0] (array/with-backend jvm (array/shape (array/zeros [4294967296 4294967296 0]))))
      "a zero dimension has no cells, whatever the others")
  (is (= 0 (sh/size [4294967296 4294967296 0])))
  (is (= sh/max-cells (sh/cells [sh/max-cells])) "the limit itself fits")
  (is (= :too-many-cells (:reason (failure #(sh/cells [(inc sh/max-cells)]))))))
