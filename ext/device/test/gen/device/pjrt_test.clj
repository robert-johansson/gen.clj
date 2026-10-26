(ns gen.device.pjrt-test
  "XLA's PJRT C API by FFM (gen.device.pjrt): hand-written StableHLO
   compiled and run on a device, the arrays checked on the host.

   Needs a PJRT plugin, -J-Dgen.xla.plugin=<xla_cuda_plugin.so>; skipped
   with a note otherwise. Run on fenrir's RTX PRO 6000 with jax-cuda13-plugin
   0.7.2's plugin (PJRT API 0.76)."
  (:require [clojure.test :refer [deftest is testing use-fixtures]]
            [gen.device.pjrt :as pjrt])
  (:import [clojure.lang ExceptionInfo]))

(def ^:private ^:dynamic *client* nil)

(use-fixtures :once
  (fn [f]
    (if (System/getProperty "gen.xla.plugin")
      (with-open [c (pjrt/client)]
        (binding [*client* c] (f)))
      (do (println "Skipping the PJRT tests: no plugin (-J-Dgen.xla.plugin=<xla_cuda_plugin.so>)")
          (f)))))

(defmacro ^:private on-device [& body]
  `(when *client* ~@body))

(defn- run
  "Compiles mlir, runs it on inputs ([data shape] or [data shape dtype]) and
   returns its outputs read back."
  [mlir & inputs]
  (with-open [exe (pjrt/compile *client* mlir)]
    (let [args (mapv #(apply pjrt/->device *client* %) inputs)]
      (try
        (let [outs (pjrt/execute exe args)]
          (try (mapv pjrt/->host outs)
               (finally (run! pjrt/close outs))))
        (finally (run! pjrt/close args))))))

(deftest a-client-has-a-device
  (on-device
    (is (= 0 (first (:version (pjrt/plugin)))))
    (is (string? (:platform *client*)))
    (is (pos? (:devices *client*)))
    (is (string? (:device-kind *client*)))))

(deftest f32-add
  (on-device
    (let [[out :as outs]
          (run "module @m {
                  func.func public @main(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = stablehlo.add %a, %b : tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }"
               [(float-array [1 2 3 4]) [4]]
               [(float-array [10 20 30 0.5]) [4]])]
      (is (= 1 (count outs)))
      (is (= [:f32 [4]] [(:dtype out) (:shape out)]))
      (is (= [11.0 22.0 33.0 4.5] (map double (:data out)))))))

(deftest a-while-loop
  (testing "the sum of 0 … n-1, n a scalar argument, with a counter"
    (on-device
      (let [[out] (run "module @m {
                          func.func public @main(%n: tensor<i32>) -> tensor<i32> {
                            %c0 = stablehlo.constant dense<0> : tensor<i32>
                            %c1 = stablehlo.constant dense<1> : tensor<i32>
                            %r:2 = stablehlo.while(%i = %c0, %s = %c0) : tensor<i32>, tensor<i32>
                             cond {
                              %p = stablehlo.compare LT, %i, %n : (tensor<i32>, tensor<i32>) -> tensor<i1>
                              stablehlo.return %p : tensor<i1>
                            } do {
                              %s2 = stablehlo.add %s, %i : tensor<i32>
                              %i2 = stablehlo.add %i, %c1 : tensor<i32>
                              stablehlo.return %i2, %s2 : tensor<i32>, tensor<i32>
                            }
                            return %r#1 : tensor<i32>
                          }
                        }"
                       [(int-array [100]) []])]
        (is (= [:i32 []] [(:dtype out) (:shape out)]))
        (is (= [4950] (vec (:data out))))))))

(defn- u32 ^long [^long x] (bit-and x 0xFFFFFFFF))

(deftest uint32-bits
  (testing "Threefry's operations on ui32: shifts, or, xor, and an add that wraps"
    (on-device
      (let [a [0 1 0xDEADBEEF 0xFFFFFFFF]
            b [0x12345678 0xFFFFFFFF 0x0F0F0F0F 0x80000000]
            [rot mix] (run "module @m {
                              func.func public @main(%a: tensor<4xui32>, %b: tensor<4xui32>)
                                  -> (tensor<4xui32>, tensor<4xui32>) {
                                %c13 = stablehlo.constant dense<13> : tensor<4xui32>
                                %c19 = stablehlo.constant dense<19> : tensor<4xui32>
                                %0 = stablehlo.shift_left %a, %c13 : tensor<4xui32>
                                %1 = stablehlo.shift_right_logical %a, %c19 : tensor<4xui32>
                                %2 = stablehlo.or %0, %1 : tensor<4xui32>
                                %3 = stablehlo.xor %2, %b : tensor<4xui32>
                                %4 = stablehlo.add %3, %a : tensor<4xui32>
                                return %2, %4 : tensor<4xui32>, tensor<4xui32>
                              }
                            }"
                           [(int-array (map unchecked-int a)) [4] :u32]
                           [(int-array (map unchecked-int b)) [4] :u32])
            rotl (fn [^long x] (u32 (bit-or (bit-shift-left x 13) (unsigned-bit-shift-right x 19))))
            bits (fn [out] (mapv #(Integer/toUnsignedLong %) (:data out)))]
        (is (= [:u32 [4]] [(:dtype rot) (:shape rot)]))
        (is (= (mapv rotl a) (bits rot)))
        (is (= (mapv (fn [^long x ^long y] (u32 (+ (bit-xor (long (rotl x)) y) x))) a b)
               (bits mix)))))))

(deftest a-reduction
  (testing "f64 rows summed with stablehlo.reduce's add region"
    (on-device
      (let [[out] (run "module @m {
                          func.func public @main(%a: tensor<2x3xf64>) -> tensor<2xf64> {
                            %z = stablehlo.constant dense<0.0> : tensor<f64>
                            %0 = stablehlo.reduce(%a init: %z) across dimensions = [1]
                                   : (tensor<2x3xf64>, tensor<f64>) -> tensor<2xf64>
                              reducer(%x: tensor<f64>, %y: tensor<f64>) {
                                %s = stablehlo.add %x, %y : tensor<f64>
                                stablehlo.return %s : tensor<f64>
                              }
                            return %0 : tensor<2xf64>
                          }
                        }"
                       [(double-array [1 2 3 0.1 0.2 0]) [2 3]])]
        (is (= [:f64 [2]] [(:dtype out) (:shape out)]))
        (is (= [6.0 (+ 0.1 0.2)] (vec (:data out))) "0.30000000000000004: an f64 sum")))))

(deftest predicates
  (testing "a predicate in (select) and out (compare)"
    (on-device
      (let [[sel gt] (run "module @m {
                             func.func public @main(%p: tensor<4xi1>, %x: tensor<4xf32>, %y: tensor<4xf32>)
                                 -> (tensor<4xf32>, tensor<4xi1>) {
                               %0 = stablehlo.select %p, %x, %y : tensor<4xi1>, tensor<4xf32>
                               %1 = stablehlo.compare GT, %x, %y : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
                               return %0, %1 : tensor<4xf32>, tensor<4xi1>
                             }
                           }"
                          [(boolean-array [true false true false]) [4]]
                          [(float-array [1 2 3 4]) [4]]
                          [(float-array [4 3 2 1]) [4]])]
        (is (= [1.0 3.0 3.0 1.0] (map double (:data sel))))
        (is (= [:bool [4]] [(:dtype gt) (:shape gt)]))
        (is (= [false false true true] (vec (:data gt))))))))

(deftest errors-are-loud
  (on-device
    (testing "malformed MLIR: XLA's message, as ex-info"
      (let [e (try (pjrt/compile *client* "module @m { func.func public @main( }") nil
                   (catch ExceptionInfo e e))]
        (is (some? e))
        (is (= :PJRT_Client_Compile (:pjrt/fn (ex-data e))))
        (is (keyword? (:pjrt/code (ex-data e))))
        (is (re-find #"(?i)expected|parse|error" (:pjrt/message (ex-data e))))))
    (testing "ill-typed StableHLO"
      (is (thrown? ExceptionInfo
                   (pjrt/compile *client* "module @m {
                                             func.func public @main(%a: tensor<4xf32>, %b: tensor<4xi32>) -> tensor<4xf32> {
                                               %0 = stablehlo.add %a, %b : tensor<4xf32>
                                               return %0 : tensor<4xf32>
                                             }
                                           }"))))
    (testing "the wrong number of arguments, at execute"
      (is (thrown? ExceptionInfo
                   (run "module @m {
                           func.func public @main(%a: tensor<2xf32>, %b: tensor<2xf32>) -> tensor<2xf32> {
                             %0 = stablehlo.add %a, %b : tensor<2xf32>
                             return %0 : tensor<2xf32>
                           }
                         }"
                        [(float-array [1 2]) [2]]))))
    (testing "host data that does not fit its shape or dtype"
      (is (thrown-with-msg? ExceptionInfo #"needs 6 elements"
                            (pjrt/->device *client* (float-array 4) [2 3])))
      (is (thrown-with-msg? ExceptionInfo #"takes a float\[\]"
                            (pjrt/->device *client* (double-array 4) [4] :f32))))
    (testing "a closed buffer"
      (let [b (pjrt/->device *client* (float-array [1 2]) [2])]
        (pjrt/close b)
        (pjrt/close b)
        (is (thrown-with-msg? ExceptionInfo #"closed" (pjrt/->host b)))))
    (testing "the client still works after them"
      (is (= [3.0] (map double (:data (first (run "module @m {
                                                     func.func public @main(%a: tensor<1xf32>) -> tensor<1xf32> {
                                                       %0 = stablehlo.add %a, %a : tensor<1xf32>
                                                       return %0 : tensor<1xf32>
                                                     }
                                                   }"
                                                  [(float-array [1.5]) [1]])))))))))
