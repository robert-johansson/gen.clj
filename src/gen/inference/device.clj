(ns gen.inference.device
  "kernel/run's and importance sampling's :device option (experimental:
   outside the 1.x promise; DESIGN §9, genclj-fpnf).

   A run with {:device :jvm} (or :metal) stages the run once from gen.clj's
   own code (gen.array.stage): kernel/run's chain, the kernel's step under
   array/scan from the start trace, or importance sampling's particle, a
   constrained generate; compiles it with the device lowerings (gen.device,
   the extension ext/device); runs it; and rebuilds the result kernel/run
   or importance-sampling returns, with :device added. The model's
   arguments are constants of the program; the trace's choices, score and
   return value, or the observations, are its inputs.

   Programs are cached. A kernel of kernel/mala or kernel/hmc is keyed by
   its kind and parameters (another kernel by its identity), the model by
   its identity, its arguments one by one (host scalars by value, a double
   by its bits; arrays and the rest by identity) and the chain by its
   length: a run alike reuses the program, a run on new data arrays or of
   another length compiles anew.

   The run is the handler's (decision 5): in float64 on the JVM backend bit
   for bit, else to float32 rounding. Programs stage at the current
   backend's float (:float overrides it). Per-particle keys: importance
   sampling's particle i runs on (rng/split key n)[i], the unbatched run's
   draws (decision 3).

   A run that does not stage or lower (a branch on a draw, an op a
   lowering lacks, ext/device absent, a kernel reporting no single accept)
   runs on the handler instead, its result carrying :device-fallback
   {:device :reason}: never another result."
  {:api :experimental}
  (:require [gen.array :as array]
            [gen.array.dtype :as dt]
            [gen.array.stage :as st]
            [gen.array.tree :as tree]
            [gen.batched :as batched]
            [gen.choicemap :as cm]
            [gen.gfi :as gfi]
            [gen.inference.kernel :as kernel]
            [gen.random :as rng]
            [gen.trace :as trace]))

(set! *warn-on-reflection* true)

;; ---------------------------------------------------------------------------
;; Compiled programs, cached by the identity of what they close over
;; ---------------------------------------------------------------------------

(deftype ^:private Id [o]
  Object
  (equals [_ x] (and (instance? Id x) (identical? o (.-o ^Id x))))
  (hashCode [_] (System/identityHashCode o)))

(defonce ^:private cache (atom {}))

(def ^:private cache-limit 64)

(defn- cached
  "The value at k in the program cache, made by (make) when absent (the
   cache emptied past cache-limit entries)."
  [k make]
  (or (get @cache k)
      (let [v (make)]
        (swap! cache (fn [c] (assoc (if (>= (count c) cache-limit) {} c) k v)))
        v)))

(defn- arg-key
  "An argument's part of a program's key: a host scalar by value (a double
   by its bits, so 0.0 and -0.0 make two programs), anything else (an
   array, a collection, a model) by identity."
  [a]
  (cond (double? a) [Double (Double/doubleToRawLongBits a)]
        (or (number? a) (keyword? a) (string? a) (boolean? a) (nil? a)) [(class a) a]
        :else (Id. a)))

(defn- args-key [args] (mapv arg-key args))

(defn ^:no-doc clear-cache!
  "Forget every compiled program."
  []
  (reset! cache {}))

(defn- compile-program [device program]
  (let [f (try (requiring-resolve 'gen.device/compile-program)
               (catch Exception e
                 (throw (ex-info "the device lowerings (ext/device) are not on the classpath" {:reason :no-lowering} e))))]
    (f device program)))

(defn- float-dtype [opts] (or (:float opts) (:float (array/defaults array/*backend*))))

(defn- spec-of
  "The staging spec of a value: its shape and dtype, a float as fl, an int
   as int32."
  [fl x]
  (let [d (array/dtype x)]
    [(vec (array/shape x)) (case (dt/kind d) :float fl :int (if (= :uint32 d) :uint32 :int32) d)]))

(defmacro ^:private with-fallback
  "body's value with :device device, or, when it throws, (handler)'s with
   :device-fallback {:device :reason}."
  [device handler & body]
  `(try (assoc (do ~@body) :device ~device)
        (catch Exception e#
          (assoc (~handler) :device-fallback {:device ~device :reason (or (ex-message e#) (str (class e#)))}))))

;; ---------------------------------------------------------------------------
;; kernel/run
;; ---------------------------------------------------------------------------

(defn- constant?
  "Does the chain start by holding a start leaf as a constant? A host
   integer or boolean (a count of the arguments, a discrete choice, a
   flag), which host code may read: (repeat n slope). Arrays and floats are
   carried."
  [x]
  (not (or (array/backend-array? x) (float? x))))

(defn- same-host? [a b] (and (not (array/backend-array? b)) (= (class a) (class b)) (= a b)))

(defn- merged
  "Every leaf of a trace: constants (position to host value) at their
   positions, the carried leaves in order at the others."
  [n constants carried]
  (loop [i 0 cs (seq carried) out (transient [])]
    (cond (= i n) (persistent! out)
          (contains? constants i) (recur (inc i) cs (conj! out (constants i)))
          :else (recur (inc i) (next cs) (conj! out (first cs))))))

(defn- carried [constants leaves]
  (into [] (keep-indexed (fn [i v] (when-not (contains? constants i) v))) leaves))

(defn- aliases
  "For the carried leaves of the chains' starts (one vector per chain),
   each position's representative: the first position holding the same
   array object in every chain (a trace holding one array twice: a model
   whose return value is its observation), else the position itself."
  [starts]
  (let [first-start (first starts)
        same-object? (fn [i j] (every? #(let [x (nth % i)] (and (array/backend-array? x) (identical? x (nth % j)))) starts))]
    (mapv (fn [j] (or (first (filter #(and (identical? (nth first-start %) (nth first-start j)) (same-object? % j)) (range j))) j))
          (range (count first-start)))))

(defn- chain-program
  "kernel's chain of steps from a trace of skeleton sk (on args) whose
   leaves are constants (position to host value) and carried ones, the
   carried positions sharing a representative (rep, aliases') one value:
   {:program :unchanged :reps}, a program of a key and the
   representatives' values (specs) whose outputs are each step's {:c the
   representatives a step changes} (and :acc, :div, :ok when the kernel
   reports them), unchanged the representatives (their indices in reps)
   a step passes on as it got them (the observations, say). Throws
   ::carry, the constants' positions a step computes, and ::alias, the
   positions a step separates from their representative's value."
  [kernel sk args n constants rep specs steps fl]
  (let [f (or (:device-f kernel) (if (map? kernel) (:f kernel) kernel))
        reps (vec (distinct rep))
        at (zipmap reps (range))
        unchanged (volatile! #{})
        program
        (binding [st/*float* fl]
          (st/stage
           (fn [k cs]
             (let [args (st/constants args)
                   [_ ys] (array/scan
                           (fn [c t]
                             (let [full (mapv #(c (at (rep %))) (range (count rep)))
                                   res (kernel/step-result (f (trace/from-leaves sk (merged n constants full) args) (st/key-at k t)))
                                   [sk' out] (trace/as-leaves (:trace res))
                                   out-c (carried constants out)
                                   computed (into #{} (keep (fn [[i _]] (when (array/backend-array? (out i)) i))) constants)
                                   separated (into #{} (keep (fn [j] (let [r (rep j)] (when (and (not= r j) (not (identical? (out-c j) (out-c r)))) j))))
                                                   (range (count rep)))]
                               (when (contains? res :accepted)
                                 (throw (ex-info "the kernel reports several accept decisions a step (a combinator's)" {})))
                               (when-not (= sk sk')
                                 (throw (ex-info (trace/structure-change sk sk') {})))
                               (when (seq computed)
                                 (throw (ex-info "a step computes a leaf held as a constant" {::carry computed})))
                               (when (seq separated)
                                 (throw (ex-info "a step gives an aliased leaf a value of its own" {::alias separated})))
                               (when-not (every? (fn [[i v]] (same-host? v (out i))) constants)
                                 (throw (ex-info "the kernel changes a host integer or boolean of the trace" {})))
                               (let [out-r (mapv out-c reps)]
                                 ;; a representative the step returns as it got it: the staged value itself
                                 (vreset! unchanged (into #{} (keep-indexed (fn [i [a b]] (when (identical? a b) i))) (map vector c out-r)))
                                 [out-r
                                  (cond-> {:c (into [] (keep-indexed (fn [i v] (when-not (@unchanged i) v))) out-r)}
                                    (contains? res :accepted?) (assoc :acc (:accepted? res))
                                    (contains? res :diverged?) (assoc :div (:diverged? res))
                                    (contains? res :start-ok) (assoc :ok (:start-ok res)))])))
                           cs nil steps)]
               ys))
           (st/key-spec) specs))]
    {:program program :unchanged @unchanged :reps reps}))

(defn- chain
  "chain-program's {:program :unchanged :reps} with its constants and rep:
   the start's host integers and booleans held as constants, but for those
   a step computes (a running count), carried instead; carried leaves that
   are one array in every start (starts: one vector of carried leaves per
   chain, or nil for the first's alone) one value, but for those a step
   separates; staged again until neither happens."
  ([kernel sk args leaves steps fl] (chain kernel sk args leaves steps fl nil))
  ([kernel sk args leaves steps fl starts]
   (loop [constants (into {} (keep-indexed (fn [i v] (when (constant? v) [i v]))) leaves)
          apart #{}]
     (let [cs (or (some->> starts (mapv #(carried constants %))) [(carried constants leaves)])
           rep (mapv #(if (contains? apart %2) %2 %1) (aliases cs) (range))
           r (try (chain-program kernel sk args (count leaves) constants rep
                                 (mapv #(spec-of fl ((first cs) %)) (distinct rep)) steps fl)
                  (catch clojure.lang.ExceptionInfo e
                    (let [causes (take-while some? (iterate ex-cause e))
                          carry (some #(::carry (ex-data %)) causes)
                          alias (some #(::alias (ex-data %)) causes)]
                      (cond carry {::carry carry}
                            alias {::alias alias}
                            :else (throw e)))))]
       (cond (::carry r) (recur (apply dissoc constants (::carry r)) #{})
             (::alias r) (recur constants (into apart (::alias r)))
             :else (assoc r :constants constants :rep rep))))))

(defn- host-element
  "Element i of a flat Java array as a host number or boolean: a float as a
   double, a boolean as Boolean/TRUE or Boolean/FALSE (reflect.Array's
   are new Booleans, and Clojure's if takes any Boolean but FALSE as true)."
  [data ^long i]
  (let [v (java.lang.reflect.Array/get data (int i))]
    (cond (instance? Float v) (.doubleValue ^Float v)
          (instance? Boolean v) (Boolean/valueOf (.booleanValue ^Boolean v))
          :else v)))

(defn- element-reader
  "A function of i giving element i of a flat Java array as a host number
   or boolean (a float as a double, an int as a long)."
  [data]
  (condp instance? data
    double/1 (let [^doubles a data] (fn [^long i] (aget a i)))
    float/1 (let [^floats a data] (fn [^long i] (double (aget a i))))
    long/1 (let [^longs a data] (fn [^long i] (aget a i)))
    int/1 (let [^ints a data] (fn [^long i] (long (aget a i))))
    boolean/1 (let [^booleans a data] (fn [^long i] (aget a i)))
    (fn [^long i] (host-element data i))))

(defn- data-of
  "A host output's data: a flat Java array or nested vectors (:metal's read
   from the GPU when first asked for, a delay)."
  [o]
  (force (:data o)))

(defn- row-reader
  "A function of i giving row i of a host output of the lowerings ({:data
   :shape :dtype}, gen.device's: its rows along the leading axis, :data a
   flat Java array or nested vectors) as an array of the current backend,
   made from that row alone: a 0-d one from its value."
  [{:keys [shape dtype] :as o}]
  (let [data (data-of o)
        row (vec (rest shape))
        opts {:dtype dtype}]
    (if (vector? data)
      (fn [^long i] (array/array (nth data i) opts))
      (let [m (long (reduce * 1 row))
            elem (element-reader data)]
        (if (empty? row)
          (fn [^long i] (array/array (elem i) opts))
          (fn [^long i] (array/reshape (array/array (mapv #(elem (+ (* i m) (long %))) (range m)) opts) row)))))))

(defn- host-doubles
  "A host output's elements as a vector of doubles."
  [o]
  (let [data (data-of o)]
   (if (vector? data)
    (mapv double data)
    (let [elem (element-reader data)]
      (mapv #(double (elem %)) (range (java.lang.reflect.Array/getLength data)))))))

(defn- host-shaped
  "Flat host values as nested vectors of shape (a host value for [])."
  [vs shape]
  (if (empty? shape)
    (first vs)
    (reduce (fn [v d] (mapv vec (partition d v))) (vec vs) (reverse (rest shape)))))

(defn- host-row-reader
  "A function of i giving row i of a host output as host values: a number
   or boolean for a 0-d row, nested vectors otherwise (as kernel/keeping
   reads an array on the host)."
  [o]
  (let [data (data-of o)
        row (vec (rest (:shape o)))
        m (long (reduce * 1 row))]
    (if (vector? data)
      (fn [^long i] (nth data i))
      (let [elem (element-reader data)]
        (if (empty? row)
          (fn [^long i] (elem i))
          (fn [^long i] (host-shaped (mapv #(elem (+ (* i m) (long %))) (range m)) row)))))))

(defn- count-true
  "How many elements of a host output, from index from (inclusive) to to
   (exclusive), are true."
  ^long [o ^long from ^long to]
  (let [data (data-of o)
        elem (if (vector? data) #(nth data %) (element-reader data))]
    (loop [i from c 0]
      (if (< i to) (recur (inc i) (if (true? (elem i)) (inc c) c)) c))))

(defn- step-carried
  "The carried leaves after a step: the start's where the step passes a
   leaf on unchanged (indices), the step's rows, in order, at the others."
  [start unchanged rows]
  (loop [j 0 rs (seq rows) out (transient [])]
    (cond (= j (count start)) (persistent! out)
          (contains? unchanged j) (recur (inc j) rs (conj! out (start j)))
          :else (recur (inc j) (next rs) (conj! out (first rs))))))

(defn- choice-sources
  "Where each address of keep-choices is read from in a chain program's
   results: {address [:constant v] | [:start i] | [:row r]}, a constant of
   the program, the chain's start's representative i (one no step
   changes), or the r-th :c output. An address's leaf is found by a probe:
   the skeleton rebuilt with each leaf its index. Loud for an address with
   no choice."
  [{:keys [sk args n constants unchanged rep reps]} keep-choices]
  (let [probe (trace/from-leaves sk (vec (range n)) args)
        carried-at (into {} (map-indexed (fn [c j] [j c])) (remove #(contains? constants %) (range n)))
        at (zipmap reps (range))
        changed (vec (remove #(contains? unchanged %) (range (count reps))))
        row-of (zipmap changed (range))]
    (into {} (for [a keep-choices
                   :let [j (try (cm/get-choice (:choices probe) a) (catch Exception _ nil))]]
               (do (when-not (integer? j)
                     (throw (ex-info (str "run-chains :keep-choices: no choice at " (pr-str a)) {:reason :no-choice :address a})))
                   (let [i (when-not (contains? constants j) (at (rep (carried-at j))))]
                     [a (cond (nil? i) [:constant (constants j)]
                              (contains? unchanged i) [:start i]
                              :else [:row (row-of i)])]))))))

(defn- chain-result
  "kernel/run's result for one chain from a program's host outputs: outs
   (the chain program's out-tree, each output rows of [runs·steps …]),
   readers (row-reader of each :c output), the chain's run offset and its
   start (the representatives' values). With :sources (choice-sources')
   its :draws are the kept choices read from the outputs as host values,
   no trace made. Loud when a step's start was not finite (the handler
   throws there)."
  [{:keys [sk args n constants unchanged rep reps steps burn thin keep sources host-readers]} outs readers offset start]
  (let [steps (long steps)
        base (* (long offset) steps)
        _ (when-let [ok (:ok outs)]
            (when (< (count-true ok base (+ base steps)) steps)
              (throw (ex-info "a step's start was not finite" {}))))
        kept (filterv #(let [i' (inc (long %))] (and (> i' (long burn)) (zero? (mod (- i' (long burn)) (long thin))))) (range steps))
        at (zipmap reps (range))
        expand (fn [vs] (mapv #(vs (at (rep %))) (range (count rep))))
        trace-at (fn [i] (trace/from-leaves sk (merged n constants (expand (step-carried start unchanged (mapv #(% (+ base (long i))) readers)))) args))
        [read finish] (if (some? keep) (kernel/keeping keep) [identity nil])
        value (fn [[kind x] i] (case kind
                                 :constant x
                                 :start (kernel/control-value (start x))
                                 :row ((host-readers x) (+ base (long i)))))]
    (assoc (cond sources {:draws (into {} (map (fn [[a src]] [a (mapv #(value src %) kept)])) sources)}
                 finish {:draws (finish (mapv (comp read trace-at) kept))}
                 :else {:traces (mapv trace-at kept)})
           :acceptance-rate (when (and (:acc outs) (pos? steps)) (/ (double (count-true (:acc outs) base (+ base steps))) steps))
           :divergences (if (:div outs) (count-true (:div outs) base (+ base steps)) 0))))

(defn- kernel-key
  "A kernel's part of a program's key: its kind and parameters
   (:device-key), else its identity."
  [kernel]
  (or (and (map? kernel) (:device-key kernel)) (Id. kernel)))

(defn run-kernel
  "kernel/run's :device path: kernel's chain from trace as a compiled
   program, its result kernel/run's (handler, a function of no arguments,
   runs it on the handler for the fallback). opts are run's, :key given.
   The result is made on the host from the program's outputs: a kept
   step's trace from its rows alone, the leaves a step does not change
   the start's, a leaf the trace holds twice (one array) read once."
  [kernel trace {:keys [samples burn thin key device] :or {burn 0 thin 1} :as opts} handler]
  (with-fallback device handler
    (let [steps (+ burn (* samples thin))
          {:keys [gen-fn args]} trace
          [sk leaves] (trace/as-leaves trace)
          fl (float-dtype opts)
          ;; a host integer or boolean may be a constant of the program: its value keys it
          k [:kernel device fl (kernel-key kernel) (Id. gen-fn) (args-key args) sk
             (mapv #(if (constant? %) [(class %) %] (spec-of fl %)) leaves) steps
             (aliases [(carried (into {} (keep-indexed (fn [i v] (when (constant? v) [i v]))) leaves) leaves)])]
          {:keys [program unchanged constants rep reps run]}
          (cached k (fn [] (let [c (chain kernel sk args leaves steps fl)]
                             (assoc c :run (compile-program device (:program c))))))
          start (mapv (carried constants leaves) reps)
          outs (tree/refill (:out-tree program) ((:run-one run) (rng/key-data key) start))]
      (chain-result (assoc opts :sk sk :args args :n (count leaves) :constants constants :unchanged unchanged
                           :rep rep :reps reps :steps steps :burn burn :thin thin)
                    outs (mapv row-reader (:c outs)) 0 start))))

;; ---------------------------------------------------------------------------
;; many chains
;; ---------------------------------------------------------------------------

(defn- runs-as-rows
  "A host output of n runs, [n steps …], as rows [n·steps …]: run r's step
   s at r·steps + s."
  [{:keys [data shape] :as o}]
  (let [[a b & more] shape]
    (assoc o :shape (into [(* (long a) (long b))] more)
             :data (delay (let [d (force data)] (if (vector? d) (into [] cat d) d))))))

(defn- gathered
  "One kernel/run result per chain as run-chains' result: :traces or :draws,
   :acceptance-rates and :divergences, one entry per chain."
  [results]
  (cond-> {:acceptance-rates (mapv :acceptance-rate results)
           :divergences (mapv :divergences results)}
    (contains? (first results) :draws) (assoc :draws (mapv :draws results))
    (contains? (first results) :traces) (assoc :traces (mapv :traces results))))

(defn- device-chains
  "run-chains' device path: every chain one run of the chain program
   compiled once, each chain's start its own (per-run) inputs, what every
   chain holds as one array one shared input."
  [kernel traces {:keys [samples burn thin key device] :or {burn 0 thin 1} :as opts}]
  (let [n (count traces)
        steps (+ burn (* samples thin))
        {:keys [gen-fn args]} (first traces)
        fl (float-dtype opts)
        walked (mapv trace/as-leaves traces)
        [sk leaves] (first walked)
        spec (fn [ls] (mapv #(if (constant? %) [(class %) %] (spec-of fl %)) ls))
        _ (when-not (every? (fn [t] (and (identical? gen-fn (:gen-fn t)) (= (args-key args) (args-key (:args t))))) traces)
            (throw (ex-info "the chains' traces are of different generative functions or arguments" {})))
        _ (when-not (every? (fn [[sk' ls]] (and (= sk sk') (= (spec leaves) (spec ls)))) walked)
            (throw (ex-info "the chains' traces differ in structure, shapes or host values" {})))
        constants0 (into {} (keep-indexed (fn [i v] (when (constant? v) [i v]))) leaves)
        all-starts (mapv (fn [[_ ls]] ls) walked)
        k0 [:chain-program (kernel-key kernel) (Id. gen-fn) (args-key args) sk (spec leaves) steps fl
            (aliases (mapv #(carried constants0 %) all-starts))]
        {:keys [constants rep reps] :as c0} (cached k0 #(chain kernel sk args leaves steps fl all-starts))
        ;; each chain's representatives' values
        starts (mapv (fn [ls] (mapv (carried constants ls) reps)) all-starts)
        ;; a representative the same value in every chain (the observations, one
        ;; object) is one input all chains share; the others each chain's own
        same? (fn [a b] (or (identical? a b) (and (array/host? a) (= a b))))
        per-run (into (sorted-set) (remove (fn [i] (let [v0 (nth (first starts) i)] (every? #(same? v0 (nth % i)) starts))))
                      (range (count reps)))
        k (conj k0 per-run device)
        {:keys [program unchanged run]}
        (cached k (fn [] (assoc c0 :run (compile-program device (assoc (:program c0) :per-run per-run)))))
        inputs (mapv (fn [i] (let [vs (mapv #(nth % i) starts)]
                               (if (contains? per-run i)
                                 ;; every chain's, stacked
                                 (array/array (mapv array/->clj vs) {:dtype (second (spec-of fl (first vs)))})
                                 (first vs))))
                     (range (count reps)))
        outs (tree/refill (:out-tree program) (mapv runs-as-rows ((:run-many run) n (rng/key-data key) inputs)))
        settings (assoc opts :sk sk :args args :n (count leaves) :constants constants :unchanged unchanged
                        :rep rep :reps reps :steps steps :burn burn :thin thin)
        sources (when-let [ks (:keep-choices opts)] (choice-sources settings ks))
        ;; with :keep-choices no trace is made: only the outputs they read
        readers (if sources [] (mapv row-reader (:c outs)))
        ;; host readers only of the outputs a kept choice reads
        settings (assoc settings :sources sources
                        :host-readers (into {} (for [[_ [kind r]] sources :when (= :row kind)] [r (host-row-reader ((:c outs) r))])))]
    (gathered (mapv #(chain-result settings outs readers % (starts %)) (range n)))))

(defn- batched-chains
  "run-chains' fallback for a kernel with a batched form bf: every chain a
   particle of one batched run (the traces' choices its per-particle
   constraints), stepped by bf on the keys (rng/split key steps); its
   draws the batched run's, not the device's chains (the same
   distribution; DESIGN §9, owner, 2026-11-04)."
  [bf traces {:keys [samples burn thin key keep keep-choices] :or {burn 0 thin 1}}]
  (let [n (count traces)
        {:keys [gen-fn args]} (first traces)
        ps0 (:particles (batched/generate gen-fn args cm/EMPTY n (rng/key 0) {:per-particle (mapv :choices traces)}))
        steps (+ burn (* samples thin))
        kept? (fn [i] (let [i' (inc (long i))] (and (> i' (long burn)) (zero? (mod (- i' (long burn)) (long thin))))))
        [read finish] (when keep (kernel/keeping keep))
        record (fn [ps] (cond keep-choices (let [cols (mapv #(array/->clj (batched/choice ps %)) keep-choices)]
                                              (mapv (fn [i] (zipmap keep-choices (map #(nth % i) cols))) (range n)))
                              keep (mapv read (batched/traces ps))
                              :else (batched/traces ps)))
        count-in (fn [counts flags] (if flags (mapv #(if %2 (inc (long %1)) %1) counts flags) counts))
        [_ acc div kept]
        (reduce (fn [[ps acc div kept] [i k]]
                  (let [r (kernel/particles-result (bf ps k))]
                    [(:particles r) (count-in acc (:accepted r)) (count-in div (:diverged r))
                     (if (kept? i) (conj kept (record (:particles r))) kept)]))
                [ps0 (vec (repeat n 0)) (vec (repeat n 0)) []]
                (map vector (range steps) (rng/split key steps)))
        per-chain (fn [c] (mapv #(nth % c) kept))]
    (cond-> {:acceptance-rates (mapv #(/ (double %) steps) acc) :divergences div}
      keep-choices (assoc :draws (mapv (fn [c] (into {} (map (fn [a] [a (mapv #(get % a) (per-chain c))])) keep-choices)) (range n)))
      keep (assoc :draws (mapv (fn [c] (finish (per-chain c))) (range n)))
      (not (or keep keep-choices)) (assoc :traces (mapv per-chain (range n))))))

(defn run-chains
  "n chains of kernel (gen.inference.kernel's), chain i from (traces i) on
   the key (rng/split key n)[i], each kernel/run's from its trace on that
   key: opts as run's (:samples, :burn, :thin, :keep, :key, the implicit
   source's without one) and :device (:jvm, every chain on a CPU core of
   a pool; :metal, every chain a GPU thread of one kernel). :keep-choices,
   a vector of addresses, keeps those choices as a :keep of
   {address #(% address)} would, but read from the program's outputs with
   no trace made: what many chains of many steps want (a trace a kept step
   is millions of traces there). Returns {:traces (or :draws with :keep or
   :keep-choices) :acceptance-rates :divergences}, one entry per chain, as
   kernel/run's per chain, with :device. Without :device, kernel/run per
   chain. Where the chains do not stage or lower on the device (traces of
   other structures, a kernel or model it cannot run), one batched run of
   the kernel's batched form (its draws the batched run's, the same
   distribution; chain by chain for a kernel with none), with
   :device-fallback {:device :reason :ran (:batched or :chain-by-chain)}.
   The chains' traces share their generative function, arguments and
   structure. Experimental (DESIGN §9, genclj-ke07)."
  [kernel traces {:keys [key device keep keep-choices] :as opts}]
  (let [traces (vec traces)
        key (or key (gfi/next-key))
        opts (assoc opts :key key)
        ;; kernel/run's :keep for the handler: {address #(% address)}
        run-opts (cond-> (dissoc opts :device :keep-choices)
                   keep-choices (assoc :keep (into {} (map (fn [a] [a #(% a)])) keep-choices)))
        on-handler (fn [] (gathered (mapv (fn [t k] (kernel/run kernel t (assoc run-opts :key k)))
                                          traces (rng/split key (count traces)))))]
    (when (empty? traces) (throw (ex-info "run-chains: no traces" {:reason :no-chains})))
    (when (and keep keep-choices)
      (throw (ex-info "run-chains: :keep or :keep-choices, not both" {:reason :keep-and-keep-choices})))
    (when (and keep-choices (not (and (sequential? keep-choices) (seq keep-choices))))
      (throw (ex-info "run-chains: :keep-choices is a vector of addresses" {:reason :invalid-keep-choices :keep-choices keep-choices})))
    (if-not device
      (on-handler)
      (try (assoc (device-chains kernel traces opts) :device device)
           (catch Exception e
             (let [reason (or (ex-message e) (str (class e)))
                   bf (kernel/batched-form kernel)
                   {:keys [gen-fn args]} (first traces)]
               ;; one batched run needs one generative function on the same arguments
               (if (and bf (every? #(and (identical? gen-fn (:gen-fn %)) (= (args-key args) (args-key (:args %)))) traces))
                 (assoc (batched-chains bf traces opts) :device-fallback {:device device :reason reason :ran :batched})
                 (assoc (on-handler) :device-fallback {:device device :reason reason :ran :chain-by-chain}))))))))

;; ---------------------------------------------------------------------------
;; importance sampling
;; ---------------------------------------------------------------------------

(defn importance-sampling
  "importance-sampling's :device path: n particles of model's constrained
   generate as a compiled program, particle i on (rng/split key n)[i], the
   result importance-sampling's unbatched one (summary, a function of the
   log weights, gives its weights' part; handler runs it on the handler for
   the fallback)."
  [model args observations n {:keys [key device proposal] :as opts} summary handler]
  (with-fallback device handler
    (when proposal (throw (ex-info "a :proposal under :device, not yet" {})))
    (let [fl (float-dtype opts)
          [osk ovals] (trace/as-leaves observations)
          ospecs (mapv #(spec-of fl %) ovals)
          k [:importance device fl (Id. model) (args-key args) osk ospecs]
          {:keys [program run sk observed]}
          (cached k (fn []
                      (let [info (volatile! nil)
                            observed (volatile! {})
                            p (binding [st/*float* fl]
                                (st/stage
                                 (fn [kk os]
                                   (let [{:keys [trace weight]} (gfi/generate model (st/constants args) (trace/from-leaves osk os nil) kk)
                                         [sk leaves] (trace/as-leaves trace)
                                         ;; a leaf that is an observation as given: every
                                         ;; particle holds it, so it is not an output
                                         at (into {} (keep-indexed (fn [i l] (when-let [j (first (keep-indexed (fn [j o] (when (identical? l o) j)) os))] [i j]))) leaves)]
                                     (vreset! info sk)
                                     (vreset! observed at)
                                     {:w weight :c (into [] (keep-indexed (fn [i l] (when-not (contains? at i) l))) leaves)}))
                                 (st/key-spec) ospecs))]
                        {:sk @info :observed @observed :program p :run (compile-program device p)})))
          outs (tree/refill (:out-tree program) ((:run-many run) n (rng/key-data key) ovals))
          readers (mapv row-reader (:c outs))
          n-leaves (+ (count readers) (count observed))
          ;; an observation as given, as the handler's trace holds it
          obs (into {} (map (fn [[i j]] [i (ovals j)])) observed)
          leaves-at (fn [i] (loop [p 0 rs (seq readers) out (transient [])]
                              (cond (= p n-leaves) (persistent! out)
                                    (contains? obs p) (recur (inc p) rs (conj! out (obs p)))
                                    :else (recur (inc p) (next rs) (conj! out ((first rs) i))))))]
      (assoc (summary (host-doubles (:w outs)))
             :traces (mapv (fn [i] (trace/from-leaves sk (leaves-at i) args)) (range n))))))
