(ns gen.llm.host
  "The LLM layer's host work: pools of daemon platform threads, an ordered
   map over one, and rows of logits normalized on one. The engines
   normalize a decode's rows on a pool of their own (gen.llm.llama,
   gen.llm.array-engine), token SMC draws its particles' tokens on
   another (gen.llm.smc). Each keeps a pool of its own: one pool shared by
   both could deadlock, a task waiting on tasks queued behind it.

     (pool name n)              → a fixed pool of n daemon threads, name-0, name-1, …
     (default-threads)          → the default size: the system property
                                  gen.llm.host-threads, or half the processors up to 4
     (in-order pool f xs)       → (mapv f xs), the calls on pool
     (log-softmax-rows pool xs n what)
                                → the rows of xs (n logits each) as log-probabilities,
                                  normalized in double on pool and the calling thread
     (exp x)                    → e^x, within 0.51 ulp on [-708, 0]"
  {:api :experimental}
  (:import [java.math BigDecimal MathContext]
           [java.util Collection]
           [java.util.concurrent Callable ExecutionException ExecutorService Executors Future ThreadPoolExecutor]
           [java.util.concurrent.atomic AtomicLong]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

(defn pool
  "A fixed pool of n daemon platform threads, named name-0, name-1, …;
   shut it down when done, or let the JVM's exit end it."
  ^ExecutorService [^String name n]
  (when-not (pos-int? n)
    (throw (ex-info (str "host/pool: a pool has a positive number of threads, not " (pr-str n))
                    {:reason :invalid-args :n n})))
  (Executors/newFixedThreadPool (int n) (-> (Thread/ofPlatform) (.name name 0) (.daemon true) .factory)))

(defn- threads
  "The default size from the property's value (a string, nil when it is
   not set): a positive integer, loud otherwise; without it, half of
   processors, up to 4."
  ^long [property ^long processors]
  (if (nil? property)
    (max 1 (min 4 (quot processors 2)))
    (let [n (parse-long property)]
      (when-not (and n (pos? (long n)))
        (throw (ex-info (str "gen.llm.host-threads is a positive integer, not " (pr-str property))
                        {:reason :invalid-option :host-threads property})))
      (long n))))

(def ^:private default
  (delay (threads (System/getProperty "gen.llm.host-threads") (.availableProcessors (Runtime/getRuntime)))))

(defn default-threads
  "The default size of a host pool, read once: the system property
   gen.llm.host-threads (a positive integer; anything else is loud) or,
   without it, half the available processors, up to 4."
  ^long []
  @default)

(defn- rethrown
  "The value of fut, what its call threw rethrown as itself."
  [^Future fut]
  (try (.get fut)
       (catch ExecutionException x (throw (.getCause x)))))

(defn in-order
  "(f x) for each x of xs, in order, as a vector: on pool when there is
   one (an ExecutorService; false or nil: none) and xs has several, each
   call under the calling thread's bindings; else on the calling thread.
   What a call threw is rethrown as itself, once every call has ended."
  [pool f xs]
  (if (and pool (next xs))
    (let [f (bound-fn* f)]
      (mapv rethrown (.invokeAll ^ExecutorService pool ^Collection (mapv (fn [x] (fn [] (f x))) xs))))
    (mapv f xs)))

;; ---------------------------------------------------------------------------
;; e^x
;; ---------------------------------------------------------------------------

;; A row's normalization takes an exponential an entry, 151,936 of them for
;; Qwen3's vocabulary: Math/exp's 4.2 ns (fdlibm's, StrictMath's, under the
;; Graal JIT on an M4) were most of the host's time a step (genclj-9y6y).
;; exp is Tang's table-driven method, with each entry's tail kept (as ARM's
;; optimized-routines do): x = (k/128) ln 2 + r with |r| <= ln 2/256, e^x =
;; 2^(k/128) e^r, 2^(j/128) from a table of 128 doubles and their tails,
;; e^r - 1 by its Taylor polynomial of degree 5 (truncated at r^6/720 <
;; 6e-19). Within 0.51 ulp of e^x on [-708, 0] (gen.llm.host-test: 0.503
;; at worst; fdlibm's within 1), in about 1.4 ns on the M4.

(def ^:private ^BigDecimal ln2
  "ln 2 to 60 digits."
  (BigDecimal. "0.693147180559945309417232121458176568075500134360255254120680"))

(def ^:private two-to-the-j-128
  "2^(j/128) for j from 0 below 128, to 45 digits: powers of 2^(1/128),
   e^((ln 2)/128) by its Taylor series."
  (let [mc (MathContext. 45)
        x (.divide ln2 (BigDecimal. 128) mc)
        root (loop [i 1 term BigDecimal/ONE sum BigDecimal/ONE]
               (if (neg? (.compareTo (.abs term) (BigDecimal. "1e-50")))
                 sum
                 (let [term (.divide (.multiply term x mc) (BigDecimal/valueOf i) mc)]
                   (recur (inc i) term (.add sum term mc)))))]
    (vec (take 128 (iterate #(.multiply ^BigDecimal % root mc) BigDecimal/ONE)))))

(def ^:private ^"[D" exp-hi
  "2^(j/128) rounded to double."
  (double-array (map #(.doubleValue ^BigDecimal %) two-to-the-j-128)))

(def ^:private ^"[D" exp-tail
  "2^(j/128)'s relative rounding error in exp-hi: 2^(j/128) = hi (1 + tail)."
  (double-array (map (fn [^BigDecimal t]
                       (let [hi (BigDecimal. (.doubleValue t))]
                         (.doubleValue (.divide (.subtract t hi) hi (MathContext. 20)))))
                     two-to-the-j-128)))

(def ^:private ^:const exp-scale
  "128/ln 2, rounded."
  (.doubleValue (.divide (BigDecimal. 128) ln2 (MathContext. 40))))

(def ^:private ^:const exp-step-hi
  "(ln 2)/128 rounded to double."
  (/ (.doubleValue ln2) 128.0))

(def ^:private ^:const exp-step-lo
  "(ln 2)/128 less exp-step-hi."
  (/ (.doubleValue (.subtract ln2 (BigDecimal. (.doubleValue ln2)))) 128.0))

(def ^:private ^:const exp-shift
  "1.5 × 2^52: y + exp-shift rounds y to an integer, held in its low bits."
  6755399441055744.0)

(def ^:private ^:const exp-shift-bits (Double/doubleToRawLongBits 6755399441055744.0))

(defmacro ^:private exp-in-range
  "e^d for a double d in [-708, 0], hi and tail locals holding exp-hi and
   exp-tail: 2^(j/128) 2^e (1 + tail + q(r)), for d = k (ln 2)/128 + r and
   k = 128 e + j."
  [hi tail d]
  `(let [d# ~d
         kd# (+ (* d# exp-scale) exp-shift)
         k# (- (Double/doubleToRawLongBits kd#) exp-shift-bits)
         kd# (- kd# exp-shift)
         r# (Math/fma (- kd#) exp-step-lo (Math/fma (- kd#) exp-step-hi d#))
         j# (bit-and k# 127)
         ;; 2^(j/128) 2^e: e added to the exponent of 2^(j/128), in [1, 2)
         s# (Double/longBitsToDouble (+ (Double/doubleToRawLongBits (aget ~hi j#))
                                        (bit-shift-left (bit-shift-right k# 7) 52)))
         r2# (* r# r#)
         q# (Math/fma r2# (Math/fma r2# (Math/fma r# 0.008333333333333333 0.041666666666666664)
                                    (Math/fma r# 0.16666666666666666 0.5))
                      (+ r# (aget ~tail j#)))]
     (Math/fma s# q# s#)))

(defn exp
  "e^x: within 0.51 ulp for x in [-708, 0] (a log-probability, or a logit
   less the largest), at about a third of Math/exp's cost (Tang's method,
   a table of 128 entries); Math/exp's value elsewhere (0 at -Inf, NaN for
   NaN)."
  ^double [^double x]
  (if (and (<= -708.0 x) (<= x 0.0))
    (let [hi exp-hi tail exp-tail] (exp-in-range hi tail x))
    (Math/exp x)))

;; ---------------------------------------------------------------------------
;; Rows of logits normalized
;; ---------------------------------------------------------------------------

;; A row's log-normalizer is defined by blocks of `block` entries, whatever
;; the threads: each block's largest logit m_b and its sum s_b of
;; e^(x - m_b), in order; then z = m + log Σ_b s_b e^(m_b - m), the blocks
;; in order, m the row's largest. The rows' blocks are spread over the
;; pool's threads and the calling thread in units (a row's run of blocks)
;; that each takes as it is free, so the floats are the same for any
;; number of threads, bit for bit.

(def ^:private ^:const block
  "Entries in a block of a row (16 KiB of floats, in a core's L1 cache)."
  4096)

;; Floats are converted four at a time, the four held together, and a
;; block is widened to doubles before its max and sum read it. The Graal
;; JIT on x86 (fenrir's) compiles a float's conversion to double, or
;; back, into an instruction that waits on its register's last value
;; (cvtss2sd's false dependency) and vectorizes no Clojure loop (a long
;; index), so a loop converting one value at a time runs at the latency of
;; everything between two conversions: 9 ns an entry for an exp over
;; floats against 2 over doubles, 2.3 ns for x - z rounded against 0.43
;; four at a time (genclj-9y6y). C2 and the Graal JIT on the M4 run either
;; form alike.

(defn- widen!
  "The k floats of xs from `from`, as doubles, into d."
  [^floats xs ^long from ^long k ^doubles d]
  (loop [j 0]
    (cond (<= (+ j 4) k)
          (let [i (+ from j)
                a (double (aget xs i)) b (double (aget xs (+ i 1)))
                c (double (aget xs (+ i 2))) e (double (aget xs (+ i 3)))]
            (aset d j a) (aset d (+ j 1) b) (aset d (+ j 2) c) (aset d (+ j 3) e)
            (recur (+ j 4)))
          (< j k) (do (aset d j (double (aget xs (+ from j)))) (recur (inc j))))))

(defn- shifted!
  "out[o + j] = xs[from + j] - z rounded to float, for j below k."
  [^floats xs from ^floats out o k z]
  (let [from (long from) o (long o) k (long k) z (double z)]
    (loop [j 0]
      (cond (<= (+ j 4) k)
            (let [i (+ from j)
                  p (+ o j)
                  a (float (- (double (aget xs i)) z)) b (float (- (double (aget xs (+ i 1))) z))
                  c (float (- (double (aget xs (+ i 2))) z)) e (float (- (double (aget xs (+ i 3))) z))]
              (aset out p a) (aset out (+ p 1) b) (aset out (+ p 2) c) (aset out (+ p 3) e)
              (recur (+ j 4)))
            (< j k) (do (aset out (+ o j) (float (- (double (aget xs (+ from j))) z))) (recur (inc j)))))))

(defn- block-max
  "The largest of the k doubles of d, NaN when one is."
  ^double [^doubles d ^long k]
  (loop [j 0 m Double/NEGATIVE_INFINITY]
    (if (< j k) (recur (inc j) (Math/max m (aget d j))) m)))

(defn- block-sum
  "Σ e^(x - m) over the k doubles x of d, in order, for m their largest
   (finite)."
  ^double [^doubles d ^long k ^double m]
  (let [hi exp-hi tail exp-tail]
    (loop [j 0 s 0.0]
      (if (< j k)
        (let [x (- (aget d j) m)]
          (recur (inc j) (+ s (if (< x -708.0) (Math/exp x) (exp-in-range hi tail x)))))
        s))))

(defn- blocks-max
  "The largest of the blocks' largest logits from g0 below g1 (st holds a
   block g's largest logit at 2g, its sum at 2g + 1)."
  ^double [^doubles st ^long g0 ^long g1]
  (loop [g g0 m Double/NEGATIVE_INFINITY]
    (if (< g g1) (recur (inc g) (Math/max m (aget st (* 2 g)))) m)))

(defn- blocks-sum
  "Σ s_b e^(m_b - m) over the blocks from g0 below g1, in order, for m
   their largest (finite); a block of -Inf logits adds nothing."
  ^double [^doubles st ^long g0 ^long g1 ^double m]
  (loop [g g0 s 0.0]
    (if (< g g1)
      (let [mb (aget st (* 2 g))]
        (recur (inc g) (if (== mb Double/NEGATIVE_INFINITY) s (+ s (* (aget st (inc (* 2 g))) (exp (- mb m)))))))
      s)))

(defn- log-normalizer
  "A row's log-normalizer from its blocks' statistics, g0 below g1 of st,
   in double: its largest logit when that is not finite (a NaN or +Inf
   logit, or every one -Inf)."
  ^double [^doubles st ^long g0 ^long g1]
  (let [m (blocks-max st g0 g1)]
    (if (Double/isFinite m) (+ m (Math/log (blocks-sum st g0 g1 m))) m)))

(defn- check-normalizer
  "z, loud when it is not finite (a logit NaN or +Inf, or every one -Inf)."
  ^double [^double z what]
  (when-not (Double/isFinite z)
    (throw (ex-info (str what ": the model's logits are not finite") {:reason :decode-failed :log-normalizer z})))
  z)

(defn- workers
  "The threads that normalize on pool: its own and the calling thread."
  ^long [pool]
  (if (instance? ThreadPoolExecutor pool) (inc (.getMaximumPoolSize ^ThreadPoolExecutor pool)) 1))

(defn- in-units!
  "(f u) for each unit u from 0 below k, on pool's threads (at most
   workers - 1 of them) and the calling thread, each taking the next unit
   when it is free. What a call threw is rethrown as itself, once every
   thread is done."
  [pool ^long workers ^long k f]
  (let [next (AtomicLong.)
        work (fn [] (loop [] (let [u (.getAndIncrement next)] (when (< u k) (f u) (recur)))))
        futs (mapv (fn [_] (.submit ^ExecutorService pool ^Callable (fn [] (work) nil)))
                   (range (min (dec workers) (dec k))))]
    (try (work)
         (finally (run! #(try (.get ^Future %) (catch Throwable _)) futs)))
    (run! rethrown futs)))

(defn- units
  "How rows of n entries are cut for w threads: {:rows :n, :blocks a
   row's, :parts a row's units (runs of its blocks): at least four units
   a thread, a row whole when there are rows enough}."
  [^long rows ^long n ^long w]
  (let [nb (quot (+ n (dec block)) block)
        r (max 1 rows)]
    {:rows rows :n n :blocks nb :parts (min nb (quot (+ (* 4 w) (dec r)) r))}))

(defn- unit
  "[row, first block, end block] of unit u."
  [{:keys [^long blocks ^long parts]} ^long u]
  (let [q (rem u parts)]
    [(quot u parts) (quot (* q blocks) parts) (quot (* (inc q) blocks) parts)]))

(defn- log-normalizers
  "Each row's log-normalizer (log-normalizer's), a double array: the
   blocks' statistics taken by units on pool and the calling thread (w of
   them), then each row's combined on the calling thread."
  ^"[D" [pool ^long w ^floats xs {:keys [^long rows ^long n ^long blocks ^long parts] :as us}]
  (let [st (double-array (* 2 rows blocks))
        zs (double-array rows)]
    (in-units! pool w (* rows parts)
               (fn [u]
                 (let [[r b0 b1] (unit us u)
                       d (double-array block)]
                   (loop [i (long b0)]
                     (when (< i (long b1))
                       (let [k (- (min n (* (inc i) block)) (* i block))
                             g (+ (* (long r) blocks) i)]
                         (widen! xs (+ (* (long r) n) (* i block)) k d)
                         (let [m (block-max d k)]
                           (aset st (* 2 g) m)
                           (when (Double/isFinite m) (aset st (inc (* 2 g)) (block-sum d k m))))
                         (recur (inc i))))))))
    (dotimes [r rows]
      (aset zs r (log-normalizer st (* r blocks) (* (inc r) blocks))))
    zs))

(defn log-softmax-rows
  "The rows of xs (a float array of rows of n logits, row after row) as
   log-probabilities: a vector of new float arrays, row i's entries x - z_i
   rounded to float, z_i its log-normalizer in double (by blocks of 4096
   entries, the same for any number of threads). The work is spread over
   pool (an ExecutorService from pool, or nil for none) and the calling
   thread. Loud (:decode-failed, what naming the operation) when a row
   holds NaN or +Inf, or only -Inf; a -Inf logit is a probability 0."
  [pool ^floats xs n what]
  (let [n (long n)
        w (workers pool)
        {:keys [^long rows ^long parts] :as us} (units (quot (alength xs) n) n w)
        ^doubles zs (log-normalizers pool w xs us)
        outs (vec (repeatedly rows #(float-array n)))]
    (dotimes [r rows] (check-normalizer (aget zs r) what))
    ;; each entry less its row's log-normalizer, rounded
    (in-units! pool w (* rows parts)
               (fn [u]
                 (let [[r b0 b1] (unit us u)
                       j0 (* (long b0) block)]
                   (shifted! xs (+ (* (long r) n) j0) (outs r) j0 (- (min n (* (long b1) block)) j0) (aget zs r)))))
    outs))
