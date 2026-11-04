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

(defn- row "x's element i along its leading axis." [x i] (array/take x i 0))

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

(defn- chain-program
  "kernel's chain of steps from a trace of skeleton sk (on args) whose
   leaves are constants (position to host value) and carried ones (specs):
   {:program :unchanged}, a program of a key and the carried leaves whose
   outputs are each step's {:c carried leaves} (and :acc, :div, :ok when
   the kernel reports them), and the indices of the carried leaves a step
   passes on as it got them (the observations, say), which :c leaves out:
   every step holds the start's. Throws ::carry, the constants' positions
   a step computes."
  [kernel sk args n constants specs steps fl]
  (let [f (or (:device-f kernel) (if (map? kernel) (:f kernel) kernel))
        unchanged (volatile! #{})
        program
        (binding [st/*float* fl]
          (st/stage
           (fn [k cs]
             (let [args (st/constants args)
                   [_ ys] (array/scan
                           (fn [c t]
                             (let [res (kernel/step-result (f (trace/from-leaves sk (merged n constants c) args) (st/key-at k t)))
                                   [sk' out] (trace/as-leaves (:trace res))
                                   out-c (carried constants out)
                                   computed (into #{} (keep (fn [[i _]] (when (array/backend-array? (out i)) i))) constants)]
                               (when (contains? res :accepted)
                                 (throw (ex-info "the kernel reports several accept decisions a step (a combinator's)" {})))
                               (when-not (= sk sk')
                                 (throw (ex-info (trace/structure-change sk sk') {})))
                               (when (seq computed)
                                 (throw (ex-info "a step computes a leaf held as a constant" {::carry computed})))
                               (when-not (every? (fn [[i v]] (same-host? v (out i))) constants)
                                 (throw (ex-info "the kernel changes a host integer or boolean of the trace" {})))
                               ;; a leaf the step returns as it got it: the staged value itself
                               (vreset! unchanged (into #{} (keep-indexed (fn [j [a b]] (when (identical? a b) j))) (map vector c out-c)))
                               [out-c
                                (cond-> {:c (into [] (keep-indexed (fn [j v] (when-not (@unchanged j) v))) out-c)}
                                  (contains? res :accepted?) (assoc :acc (:accepted? res))
                                  (contains? res :diverged?) (assoc :div (:diverged? res))
                                  (contains? res :start-ok) (assoc :ok (:start-ok res)))]))
                           cs nil steps)]
               ys))
           (st/key-spec) specs))]
    {:program program :unchanged @unchanged}))

(defn- chain
  "chain-program's {:program :unchanged} and its constants: the start's
   host integers and booleans, but for those a step computes (a running
   count), carried instead, the chain staged again."
  [kernel sk args leaves steps fl]
  (loop [constants (into {} (keep-indexed (fn [i v] (when (constant? v) [i v]))) leaves)]
    (let [r (try (chain-program kernel sk args (count leaves) constants
                                (mapv #(spec-of fl %) (carried constants leaves)) steps fl)
                 (catch clojure.lang.ExceptionInfo e
                   (if-let [c (some #(::carry (ex-data %)) (take-while some? (iterate ex-cause e)))]
                     {::carry c}
                     (throw e))))]
      (if-let [c (::carry r)]
        (recur (apply dissoc constants c))
        (assoc r :constants constants)))))

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

(defn- row-reader
  "A function of i giving row i of a host output of the lowerings ({:data
   :shape :dtype}, gen.device's run-one: its rows along the leading axis)
   as an array of the current backend, made from that row alone: a 0-d one
   from its value."
  [{:keys [data shape dtype]}]
  (let [row (vec (rest shape))
        m (long (reduce * 1 row))
        elem (element-reader data)
        opts {:dtype dtype}]
    (if (empty? row)
      (fn [^long i] (array/array (elem i) opts))
      (fn [^long i] (array/reshape (array/array (mapv #(elem (+ (* i m) (long %))) (range m)) opts) row)))))

(defn- host-trues
  "How many elements of a host output are true."
  ^long [{:keys [data]}]
  (let [n (java.lang.reflect.Array/getLength data)]
    (loop [i 0 c 0]
      (if (< i n) (recur (inc i) (if (true? (host-element data i)) (inc c) c)) c))))

(defn- step-carried
  "The carried leaves after a step: the start's where the step passes a
   leaf on unchanged (indices), the step's rows, in order, at the others."
  [start unchanged rows]
  (loop [j 0 rs (seq rows) out (transient [])]
    (cond (= j (count start)) (persistent! out)
          (contains? unchanged j) (recur (inc j) rs (conj! out (start j)))
          :else (recur (inc j) (next rs) (conj! out (first rs))))))

(defn run-kernel
  "kernel/run's :device path: kernel's chain from trace as a compiled
   program, its result kernel/run's (handler, a function of no arguments,
   runs it on the handler for the fallback). opts are run's, :key given.
   The result is made on the host from the program's outputs: a kept
   step's trace from its rows alone, the leaves a step does not change
   the start's."
  [kernel trace {:keys [samples burn thin key keep device] :or {burn 0 thin 1} :as opts} handler]
  (with-fallback device handler
    (let [steps (+ burn (* samples thin))
          {:keys [gen-fn args]} trace
          [sk leaves] (trace/as-leaves trace)
          n (count leaves)
          fl (float-dtype opts)
          ;; a host integer or boolean may be a constant of the program: its value keys it
          k [:kernel device fl (or (and (map? kernel) (:device-key kernel)) (Id. kernel)) (Id. gen-fn) (args-key args) sk
             (mapv #(if (constant? %) [(class %) %] (spec-of fl %)) leaves) steps]
          {:keys [program unchanged constants run]} (cached k (fn [] (let [c (chain kernel sk args leaves steps fl)]
                                                                        (assoc c :run (compile-program device (:program c))))))
          start (carried constants leaves)
          outs (tree/refill (:out-tree program) ((:run-one run) (rng/key-data key) start))
          _ (when-let [ok (:ok outs)]
              (when (< (host-trues ok) (java.lang.reflect.Array/getLength (:data ok)))
                ;; the handler throws there (a log density not finite at the start)
                (throw (ex-info "a step's start was not finite" {}))))
          kept (filterv #(let [i' (inc %)] (and (> i' burn) (zero? (mod (- i' burn) thin)))) (range steps))
          readers (mapv row-reader (:c outs))
          trace-at (fn [i] (trace/from-leaves sk (merged n constants (step-carried start unchanged (mapv #(% i) readers))) args))
          [read finish] (if (some? keep) (kernel/keeping keep) [identity nil])]
      (assoc (if finish {:draws (finish (mapv (comp read trace-at) kept))} {:traces (mapv trace-at kept)})
             :acceptance-rate (when (and (:acc outs) (pos? steps)) (/ (double (host-trues (:acc outs))) steps))
             :divergences (if (:div outs) (host-trues (:div outs)) 0)))))

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
          {:keys [program run sk]}
          (cached k (fn []
                      (let [info (volatile! nil)
                            p (binding [st/*float* fl]
                                (st/stage
                                 (fn [kk os]
                                   (let [{:keys [trace weight]} (gfi/generate model (st/constants args) (trace/from-leaves osk os nil) kk)
                                         [sk leaves] (trace/as-leaves trace)]
                                     (vreset! info sk)
                                     {:w weight :c leaves}))
                                 (st/key-spec) ospecs))]
                        {:sk @info :program p :run (compile-program device p)})))
          outs (tree/refill (:out-tree program) ((:run-many run) n (rng/key-data key) ovals))
          weights (mapv double (array/->clj (:w outs)))]
      (assoc (summary weights)
             :traces (mapv (fn [i] (trace/from-leaves sk (mapv #(row % i) (:c outs)) args)) (range n))))))
