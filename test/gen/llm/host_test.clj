(ns gen.llm.host-test
  "The LLM layer's host work (gen.llm.host), which the llama.cpp engine's
   rows and token SMC's particles share the code of (the post-M5 review's
   I4-04: each had a copy): the default size, the pools, and the ordered
   map, whose results and errors are the calling thread's; exp, and the
   rows of logits normalized (genclj-9y6y).

   Mutations checked (2026-10-22, genclj-egmg), with the tests that caught
   them: the results in the order the calls end, not their inputs'
   (in-order-is-mapv-on-a-pool); an error wrapped in its
   ExecutionException (a-call-s-error-is-raised-as-itself); no bindings on
   the pool (a-call-sees-the-calling-thread-s-bindings).

   Mutations checked (2026-11-07, genclj-9y6y), with the tests that caught
   them: exp's table without its tails (exp-is-within-half-an-ulp,
   planted in place: 1.0 ulp against the band's 0.52); a sum a unit
   instead of a block, so that the sums depend on the threads
   (the-sums-are-the-same-for-any-number-of-threads, by an edit of
   log-normalizers: 29 of its checks fail; the floats alone did not show
   it); the rows' log-normalizers left unchecked (a-row-not-finite-is-loud,
   by an edit: 6 failures, 4 errors); the row's largest logit taken as
   its log-normalizer (rows-are-log-softmax-in-double, planted in
   place)."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.llm.branching-contract :as bc]
            [gen.llm.host :as host]
            [gen.test-support :as ts])
  (:import [java.math BigDecimal MathContext]
           [java.util Arrays Random]
           [java.util.concurrent ExecutorService ThreadPoolExecutor]))

(deftest the-default-size-is-a-positive-integer-or-loud
  (let [threads #'host/threads]
    (is (= 4 (threads nil 10)) "half the processors")
    (is (= 4 (threads nil 32)) "up to 4")
    (is (= 1 (threads nil 1)))
    (is (= 3 (threads "3" 10)))
    (is (= 1 (threads "1" 10)) "the off switch")
    (doseq [bad ["four" "4 " "" "0" "-2" "2.5"]]
      (is (= :invalid-option (bc/reason #(threads bad 10))) (pr-str bad))))
  (is (pos? (host/default-threads))))

(deftest a-pool-has-daemon-threads-named-for-it
  (with-open [^ExecutorService pool (host/pool "gen.llm.host-test-" 2)]
    (let [threads (host/in-order pool (fn [_] (Thread/currentThread)) (range 8))]
      (is (every? #(.isDaemon ^Thread %) threads))
      (is (every? #(.startsWith (.getName ^Thread %) "gen.llm.host-test-") threads))
      (is (= 2 (.getMaximumPoolSize ^ThreadPoolExecutor pool)))))
  (doseq [n [0 -1 2.0 nil]]
    (is (= :invalid-args (bc/reason #(host/pool "p" n))) (pr-str n))))

(deftest in-order-is-mapv-on-a-pool
  ;; later inputs finish first, the results are still in the inputs' order
  (with-open [^ExecutorService pool (host/pool "gen.llm.host-test-" 4)]
    (let [f (fn [i] (Thread/sleep (long (- 40 (* 5 i)))) (* i i))]
      (is (= (mapv f (range 8)) (host/in-order pool f (range 8))))
      (is (pos? (.getCompletedTaskCount ^ThreadPoolExecutor pool)) "the pool ran them")))
  (let [caller (Thread/currentThread)]
    (is (= [caller caller] (host/in-order false (fn [_] (Thread/currentThread)) [1 2])) "no pool: the calling thread")
    (with-open [^ExecutorService pool (host/pool "gen.llm.host-test-" 2)]
      (is (= [caller] (host/in-order pool (fn [_] (Thread/currentThread)) [1])) "one call: the calling thread"))))

(deftest a-call-s-error-is-raised-as-itself
  (with-open [^ExecutorService pool (host/pool "gen.llm.host-test-" 2)]
    (is (= :planted (bc/reason #(host/in-order pool (fn [i] (if (= i 3) (throw (ex-info "planted" {:reason :planted})) i))
                                               (range 6)))))))

(def ^:dynamic *bound* :root)

(deftest a-call-sees-the-calling-thread-s-bindings
  (with-open [^ExecutorService pool (host/pool "gen.llm.host-test-" 2)]
    (binding [*bound* :caller]
      (is (= [:caller :caller :caller] (host/in-order pool (fn [_] *bound*) (range 3)))))))

;; ---------------------------------------------------------------------------
;; exp
;; ---------------------------------------------------------------------------

(defn- exact-exp
  "e^x to about 45 digits: e^(x/2^12) by its Taylor series, squared 12
   times."
  ^BigDecimal [^double x]
  (let [mc (MathContext. 60)
        y (.divide (BigDecimal. x) (BigDecimal. 4096) mc)
        e (loop [i 1 term BigDecimal/ONE sum BigDecimal/ONE]
            (if (neg? (.compareTo (.abs term) (BigDecimal. "1e-62")))
              sum
              (let [term (.divide (.multiply term y mc) (BigDecimal/valueOf i) mc)]
                (recur (inc i) term (.add sum term mc)))))]
    (nth (iterate #(.multiply ^BigDecimal % ^BigDecimal % mc) e) 12)))

(defn- ulps
  "|got - exact| in ulps of exact rounded to double."
  ^double [^double got ^BigDecimal exact]
  (/ (.doubleValue (.abs (.subtract (BigDecimal. got) exact))) (Math/ulp (.doubleValue exact))))

(defn- worst-ulps
  "The largest error of host/exp, in ulps, over n points of [-708, 0]
   (from a fixed seed, most of them in [-60, 0], where logits less their
   largest fall) and the ends."
  ^double [^long n]
  (let [r (Random. 9)
        xs (concat [0.0 -1e-300 -1e-9 -0.5 -1.0 -708.0 -707.9]
                   (repeatedly n #(if (< (.nextDouble r) 0.8) (* -60.0 (.nextDouble r)) (* -708.0 (.nextDouble r)))))]
    (reduce max (map #(ulps (host/exp %) (exact-exp %)) xs))))

(deftest exp-is-within-half-an-ulp
  ;; Tang's method with the table's tails is within 0.51 ulp of e^x
  ;; (measured 0.503 here; fdlibm's, Math/exp's on the M4, is within 1):
  ;; the band 0.52. Power:
  ;; the table without its tails is 1.0 ulp off at worst over these points
  (is (<= (worst-ulps 6000) 0.52))
  (is (< 0.9 (with-redefs-fn {#'host/exp-tail (double-array 128)} #(worst-ulps 6000))) "the tails dropped")
  (testing "outside [-708, 0], Math/exp's value"
    (doseq [x [-708.5 -745.0 -746.0 Double/NEGATIVE_INFINITY 1e-300 0.5 700.0 710.0 Double/POSITIVE_INFINITY]]
      (is (= (Math/exp x) (host/exp x)) (str x)))
    (is (Double/isNaN (host/exp Double/NaN)))
    (is (= 1.0 (host/exp 0.0) (host/exp -0.0)))))

;; ---------------------------------------------------------------------------
;; Rows of logits normalized
;; ---------------------------------------------------------------------------

(defn- rows
  "b rows of n logits N(0, scale²) from r, as one float array."
  ^floats [^Random r b n scale]
  (float-array (repeatedly (* (long b) (long n)) #(* (double scale) (.nextGaussian r)))))

(defn- exact-normalizer
  "m + log Σ e^(x - m) over row i of xs, the terms StrictMath's, summed
   exactly."
  ^double [^floats xs i n]
  (let [off (* (long i) (long n))
        m (double (reduce max (map #(aget xs (+ off (long %))) (range n))))]
    (+ m (Math/log (.doubleValue ^BigDecimal (reduce (fn [^BigDecimal s j]
                                                       (.add s (BigDecimal. (StrictMath/exp (- (double (aget xs (+ off (long j)))) m)))))
                                                     BigDecimal/ZERO (range n)))))))

(defn- float-ulps
  "How many floats apart a and b are (finite, of one sign)."
  ^long [a b]
  (Math/abs (- (Float/floatToIntBits a) (Float/floatToIntBits b))))

(deftest rows-are-log-softmax-in-double
  ;; each float is x - z rounded, z the row's log-normalizer: within a
  ;; float's ulp of x - z for the exact z (the floats move with z only at
  ;; a rounding's boundary), across block ends and Qwen3's and Qwen3.5's
  ;; vocabularies. Power: z taken as the row's largest logit is off by
  ;; log Σ e^(x - m), many ulps
  (let [r (Random. 5)
        check (fn [xs n]
                (let [out (host/log-softmax-rows nil xs n "t")]
                  (reduce max 0 (for [i (range (count out)) :let [z (exact-normalizer xs i n) ^floats o (out i)]
                                      j (range n)]
                                  (float-ulps (aget o j) (float (- (double (aget ^floats xs (+ (* (long i) (long n)) (long j)))) z)))))))]
    (doseq [n [1 2 7 128 4095 4096 4097 10000 151936 248320] scale [1.0 8.0 30.0]
            :let [b (if (< n 10000) 3 1)]]
      (is (<= (check (rows r b n scale) n) 1) (pr-str [n scale])))
    (is (< 1000 (with-redefs-fn {#'host/log-normalizer (fn ^double [^doubles st ^long g0 ^long g1] (#'host/blocks-max st g0 g1))}
                  #(check (rows r 2 10000 1.0) 10000)))
        "the largest logit as z")))

(defn- normalizers
  "The rows' log-normalizers on pool, as gen.llm.host computes them."
  ^"[D" [pool ^floats xs ^long n]
  (let [w (#'host/workers pool)]
    (#'host/log-normalizers pool w xs (#'host/units (quot (alength xs) n) n w))))

(deftest the-sums-are-the-same-for-any-number-of-threads
  ;; the blocks' sums are combined in their order, whatever the threads
  ;; and the units they take: the log-normalizers bit for bit (the floats
  ;; then too), on no pool and pools of 1 to 8 threads, from one row to
  ;; more rows than units. (The floats alone would not show a sum taken
  ;; in another order: a change of z in its last bits moves a float
  ;; only at a rounding's boundary.)
  (let [pools (mapv #(host/pool "gen.llm.host-test-" %) [1 2 3 5 8])
        r (Random. 3)]
    (try
      (doseq [b [1 2 3 16 41] n [5 4097 30000 151936] scale [1.0 30.0]
              :when (or (< n 100000) (< b 4))
              :let [xs (rows r b n scale)
                    z (normalizers nil xs n)
                    alone (host/log-softmax-rows nil xs n "t")]]
        (doseq [p pools]
          (is (Arrays/equals z (normalizers p xs n)) (pr-str [b n scale (.getMaximumPoolSize ^ThreadPoolExecutor p)]))
          (is (every? true? (map #(Arrays/equals ^floats %1 ^floats %2) alone (host/log-softmax-rows p xs n "t")))
              (pr-str [b n scale (.getMaximumPoolSize ^ThreadPoolExecutor p)]))))
      (finally (run! #(.shutdown ^ExecutorService %) pools)))))

(deftest a-row-not-finite-is-loud
  ;; a -Inf logit is a probability 0 (a block of them adds nothing); a NaN
  ;; or +Inf logit, or a row of -Inf alone, is loud whatever the threads
  (with-open [^ExecutorService pool (host/pool "gen.llm.host-test-" 3)]
    (let [r (Random. 1)]
      (doseq [p [nil pool]]
        (let [xs (rows r 2 10000 8.0)
              _ (aset xs 17 Float/NEGATIVE_INFINITY)
              _ (Arrays/fill xs 4096 8192 Float/NEGATIVE_INFINITY)
              [a] (host/log-softmax-rows p xs 10000 "t")]
          (is (= Float/NEGATIVE_INFINITY (aget ^floats a 17) (aget ^floats a 5000)))
          (is (< (Math/abs (- 1.0 (double (reduce + (map #(Math/exp (aget ^floats a %)) (range 10000)))))) 1e-5)))
        (doseq [bad [Float/NaN Float/POSITIVE_INFINITY]
                :let [xs (rows r 3 10000 8.0)]]
          (aset xs 25000 (float bad))
          (let [data (ts/thrown-data (host/log-softmax-rows p xs 10000 "advance"))]
            (is (= :decode-failed (:reason data)))
            (is (not (Double/isFinite (:log-normalizer data))))))
        (is (= :decode-failed (:reason (ts/thrown-data (host/log-softmax-rows p (float-array (repeat 64 Float/NEGATIVE_INFINITY)) 32 "t"))))
            "every logit -Inf")
        (is (= [] (host/log-softmax-rows p (float-array 0) 7 "t")) "no rows")))))
