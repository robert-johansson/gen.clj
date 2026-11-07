(ns gen.array.transform
  "gen.array's function transformations and the particles of batched
   values (docs/array-api.md §8, §8b): the bodies of gen.array's scan,
   compile, checkpoint, batched?, batch-shape, particle-backend, particle, particles
   and vmap, whose docstrings are there. Written in gen.array's ops, this
   namespace requires gen.array, which loads it at the first call of one."
  {:api :internal}
  (:refer-clojure :exclude [+ - * / < > <= >= == compile])
  (:require [clojure.core :as c]
            [gen.array :as array :refer [< ==]]
            [gen.array.batch :as batch]
            [gen.array.protocols :as p]
            [gen.array.shape :as sh]
            [gen.array.tree :as tree]))

(set! *warn-on-reflection* true)

;; ---------------------------------------------------------------------------
;; Loops (experimental, outside the 1.x promise: DESIGN §9, genclj-yuau)
;; ---------------------------------------------------------------------------

(defn- step-x
  "xs' slice i along its leading axis (a tree), or the step index i."
  [xs i]
  (if (nil? xs)
    (array/array i {:dtype :int32})
    (array/tree-map #(array/take % i 0) xs)))

(defn- check-scan-carry
  "carry' (a step's) keeps carry's structure, shapes and dtypes: a host
   number's dtype is weak, so only its kind must stay."
  [carry carry']
  (let [refuse (fn [what] (throw (ex-info (str "scan: the step changed the carry's " what)
                                          {:reason :invalid-carry :changed what})))]
    (when-not (= (tree/skeleton carry) (tree/skeleton carry'))
      (refuse "structure"))
    (doseq [[c c'] (map vector (array/tree-leaves carry) (array/tree-leaves carry'))]
      (when-not (and (array/array? c') (= (array/shape c) (array/shape c')))
        (refuse (str "shape, from " (array/shape c) " to " (when (array/array? c') (array/shape c')))))
      (when-not (if (or (array/host? c) (array/host? c'))
                  (= (array/dtype-kind (array/dtype c)) (array/dtype-kind (array/dtype c')))
                  (= (array/dtype c) (array/dtype c')))
        (refuse (str "dtype, from " (array/dtype c) " to " (array/dtype c')))))))

(defn scan
  "gen.array/scan's body."
  [f carry xs n]
  (let [n (long n)
        b (array/backend-of :scan (concat (filter array/backend-array? (array/tree-leaves carry))
                                          (when xs (filter array/backend-array? (array/tree-leaves xs)))))]
    (when xs
      (doseq [l (array/tree-leaves xs)]
        (when-not (and (array/array? l) (seq (array/shape l)) (c/== n (long (first (array/shape l)))))
          (throw (ex-info (str "scan: every leaf of xs has a leading axis of n = " n ", not "
                               (if (array/array? l) (str "shape " (array/shape l)) (pr-str l)))
                          {:reason :invalid-args :n n :shape (when (array/array? l) (array/shape l))})))))
    (if (satisfies? p/ILooping b)
      (p/-scan-loop b f carry xs n)
      (loop [i 0 c carry ys []]
        (if (< i n)
          (let [[c' y] (f c (step-x xs i))]
            (check-scan-carry carry c')
            (recur (inc i) c' (conj ys y)))
          (let [y0 (if (seq ys)
                     (first ys)
                     ;; no step ran: f once on zeros of x's shapes, for the ys'
                     (second (f carry (if (nil? xs)
                                        (step-x nil 0)
                                        (array/tree-map #(array/zeros (vec (rest (array/shape %))) {:dtype (array/dtype %)}) xs)))))
                sk (tree/skeleton y0)
                columns (apply map vector (map array/tree-leaves (if (seq ys) ys [y0])))]
            [c (tree/refill sk (vec (if (seq ys)
                                      (map #(array/stack (vec %) 0) columns)
                                      (map #(array/zeros (into [0] (array/shape %)) {:dtype (array/dtype %)}) (array/tree-leaves y0)))))]))))))

;; ---------------------------------------------------------------------------
;; Compilation (docs/array-api.md §8)
;; ---------------------------------------------------------------------------

(defn- check-donate
  "compile's options with :donate (when given) checked and made a set of
   argument positions."
  [opts]
  (let [{:keys [donate]} opts]
    (when-not (or (nil? donate)
                  (and (coll? donate) (not (map? donate)) (every? #(and (integer? %) (c/<= 0 (long %))) donate)))
      (throw (ex-info (str "compile: :donate is a collection of argument positions (integers from 0), not " (pr-str donate))
                      {:reason :invalid-args :donate donate})))
    (cond-> opts donate (assoc :donate (set (map long donate))))))

(defn- donated-leaves
  "The positions, among args' leaves (argument by argument, as a trace
   takes them), of the leaves of the arguments at the positions donate."
  [args donate]
  (let [counts (mapv #(count (array/tree-leaves %)) args)
        starts (vec (reductions c/+ 0 counts))]
    (doseq [i donate]
      (when-not (c/< (long i) (count args))
        (throw (ex-info (str "compile: :donate names argument " i ", but the call has " (count args))
                        {:reason :invalid-args :donate donate :arguments (count args)}))))
    (into #{} (mapcat (fn [i] (range (starts i) (c/+ (long (starts i)) (long (counts i)))))) donate)))

(defn compile
  "gen.array/compile's body."
  [f opts]
  (let [opts (check-donate opts)
        compiled (atom {})]
    (fn [& args]
      (let [args (vec args)
            leaves (vec (array/tree-leaves args))]
        (doseq [l leaves]
          (when-not (array/array? l)
            (throw (ex-info (str "compile: the arguments must be arrays (or trees of them), not "
                                 (pr-str l)
                                 (when (instance? gen.random.threefry.Key l)
                                   ": a key's draws would be replayed by every call; draw outside, or close over the key"))
                            {:value l}))))
        (let [b (array/backend-of :compile leaves)
              donated (when (:donate opts) (donated-leaves args (:donate opts)))]
          (if-not (instance? gen.array.protocols.ICompiling b)
            ;; compile is f itself here (the JVM backend, the tape, the
            ;; batch tracer): nothing is donated
            (apply f args)
            (let [skel (tree/skeleton args)
                  entry (or (get @compiled [b skel])
                            (let [out-skel (atom nil)
                                  g (p/-compile b (fn [ins]
                                                    (let [out (apply f (tree/refill skel ins))]
                                                      (reset! out-skel (tree/skeleton out))
                                                      (vec (array/tree-leaves out))))
                                                (cond-> (dissoc opts :donate) donated (assoc :donate donated)))
                                  e {:g g :out out-skel :in skel}]
                              (swap! compiled assoc [b skel] e)
                              e))
                  ins (array/with-backend b (mapv #(if (array/host? %) (array/array % {:dtype (array/weak-dtype % b)}) %)
                                                  (tree/leaves-in (:in entry) args)))
                  ;; apply first: the first call traces f, which records the output's structure
                  outs ((:g entry) ins)]
              (tree/refill @(:out entry) outs))))))))

;; ---------------------------------------------------------------------------
;; Checkpoints (experimental, outside the 1.x promise: DESIGN §9)
;; ---------------------------------------------------------------------------

(defn- checkpoint-input?
  "Is the leaf x one of a checkpointed call's inputs: a float array of a
   backend (one being differentiated, or that may be)? Host numbers, int
   and bool arrays and every other leaf are constants of the call, which f
   receives as they are."
  [x]
  (and (array/backend-array? x) (= :float (array/dtype-kind (array/dtype x)))))

(defn checkpoint
  "gen.array/checkpoint's body: (apply f args), its float arrays the
   inputs of one node whose gradient recomputes f."
  [f args]
  (let [args (vec args)
        leaves (vec (tree/leaves args))
        at (vec (keep-indexed (fn [i x] (when (checkpoint-input? x) i)) leaves))
        call (fn [ins]
               (let [out (apply f (tree/refill args (reduce (fn [ls [i x]] (assoc ls i x)) leaves (map vector at ins))))]
                 (when-not (and (array/array? out) (= :float (array/dtype-kind (array/dtype out))))
                   (throw (ex-info (str "checkpoint: f returns one float array, not "
                                        (if (array/array? out) (str "an array of " (array/dtype out)) (pr-str out)))
                                   {:reason :invalid-result :op :checkpoint})))
                 out))
        ins (mapv leaves at)
        b (when (seq ins) (array/backend-of :checkpoint ins))]
    (if (instance? gen.array.protocols.ICheckpoint b)
      (p/-checkpoint b call ins)
      (call ins))))

;; ---------------------------------------------------------------------------
;; The particles of a batched value (docs/array-api.md §8b)
;; ---------------------------------------------------------------------------

(defn- batch-of
  "x as a batched value (gen.array.batch), seen through the tape (its
   primal, when x is being differentiated), or nil."
  [x]
  (let [v (if (array/tracked? x) (p/-primal x) x)]
    (when (instance? gen.array.protocols.IBatchedValue v) v)))

(defn batched?
  "gen.array/batched?'s body."
  [x]
  (some? (batch-of x)))

(defn batch-shape
  "gen.array/batch-shape's body."
  [x]
  (when-let [v (batch-of x)]
    (let [s (array/shape (p/-all-particles v))]
      (subvec s 0 (c/- (count s) (count (array/shape v)))))))

(defn particle-backend
  "gen.array/particle-backend's body."
  [x]
  (when-let [v (batch-of x)]
    (p/-inner (p/-backend v))))

(defn- batch-to-read
  "x's batched value for op to read its particles, or nil when x is not
   batched; one being differentiated is refused, since its particles'
   values, arrays apart from the run, would carry no gradient."
  [op x]
  (let [v (batch-of x)]
    (when (and v (array/tracked? x))
      (throw (ex-info (str (name op) ": the particles of a value being differentiated would carry no gradient;"
                           " read those of (stop-gradient x) to drop it on purpose")
                      {:op op :shape (array/shape x) :dtype (array/dtype x)})))
    v))

(defn- particle-rows
  "A batched value's particles as an array of the backend it wraps, one
   particle per row: its batch axes flattened, row-major."
  [v]
  (let [all (p/-all-particles v)
        s (array/shape all)
        k (c/- (count s) (count (array/shape v)))]
    (if (== 1 k) all (array/reshape all (into [(sh/size (subvec s 0 k))] (subvec s k))))))

(defn- check-particle
  "Throws unless i is a particle of a batch of n (any non-negative integer
   when n is nil: a shared value's)."
  [op i n]
  (when-not (and (integer? i) (c/<= 0 i) (or (nil? n) (c/< i n)))
    (throw (ex-info (str (name op) ": " (pr-str i) " is not a particle"
                         (when n (str " of a batch of " n)) " (a host integer from 0)")
                    {:op op :index i :particles n}))))

(defn particle
  "gen.array/particle's body."
  [x i]
  (if-let [v (batch-to-read :particle x)]
    (let [rows (particle-rows v)]
      (check-particle :particle i (first (array/shape rows)))
      (array/take rows (long i) 0))
    x))

(defn particles
  "gen.array/particles' body."
  [x idx]
  (let [idx (vec idx)]
    (if-let [v (batch-to-read :particles x)]
      (let [rows (particle-rows v)
            n (first (array/shape rows))
            inner (p/-inner (p/-backend v))]
        (doseq [i idx] (check-particle :particles i n))
        (array/take rows (array/with-backend inner (array/array (mapv long idx) {:dtype (:int (array/defaults inner))})) 0))
      (do (when-not (array/array? x) (array/not-an-array :particles x))
          (doseq [i idx] (check-particle :particles i nil))
          (array/broadcast-to x (into [(count idx)] (array/shape x)))))))

;; ---------------------------------------------------------------------------
;; A function batched over a leading axis (docs/array-api.md §8b)
;; ---------------------------------------------------------------------------

(defn vmap
  "gen.array/vmap's body."
  [f opts]
  (when-not (ifn? f)
    (throw (ex-info (str "vmap: f must be a function, not " (pr-str f)) {:op :vmap :value f})))
  (when-not (or (nil? opts) (map? opts))
    (throw (ex-info (str "vmap: the options are a map, {:in-axes …}, not " (pr-str opts)) {:op :vmap :value opts})))
  (let [{:keys [in-axes]} opts]
    (when-let [unknown (seq (remove #{:in-axes} (keys opts)))]
      (throw (ex-info (str "vmap: unknown options " (vec unknown) "; the option is :in-axes")
                      {:op :vmap :unknown (vec unknown)})))
    (when-not (or (nil? in-axes)
                  (and (vector? in-axes) (every? #(or (nil? %) (and (integer? %) (zero? (long %)))) in-axes)))
      (throw (ex-info (str "vmap: :in-axes is a vector of 0 (batched along the leading axis) or nil (shared) per"
                           " argument, not " (pr-str in-axes) "; move another axis to the front with array/transpose")
                      {:op :vmap :in-axes in-axes})))
    (fn [& args]
      (let [args (vec args)]
        (when (and in-axes (not= (count in-axes) (count args)))
          (throw (ex-info (str "vmap: :in-axes " (pr-str in-axes) " names " (count in-axes) " arguments; the call gives "
                               (count args))
                          {:op :vmap :in-axes in-axes :arguments (count args)})))
        (batch/call-batched f (or in-axes (vec (repeat (count args) 0))) args)))))
