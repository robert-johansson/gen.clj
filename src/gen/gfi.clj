(ns gen.gfi
  "The Generative Function Interface, shaped like Gen.jl's.

   Every operation is a pure function of its arguments and an explicit PRNG
   key: the same key gives the same result. The public functions take the key
   as an optional last argument. Omitting it draws a key from the implicit key
   source, the single impure point in gen.clj: a root key folded with a call
   counter. The root seed is random for each JVM (so separate processes are
   independent); implicit-seed reads it back, and seed! fixes it, so any run can
   be reproduced.

     (simulate gf args)                          → trace
     (generate gf args constraints)              → {:trace :weight}
     (update trace constraints)                  → {:trace :weight :retdiff :discard}
     (update trace args argdiffs constraints)
     (regenerate trace selection)                → {:trace :weight :retdiff}
     (regenerate trace args argdiffs selection)
     (assess gf args choices)                    → {:weight :retval}
     (propose gf args)                           → {:choices :weight :retval}
     (project trace selection)                   → weight

   update and regenerate without arguments keep the trace's own arguments,
   with no-change argdiffs; with arguments they take one argdiff per
   argument. Weights follow Gen.jl and the Cusumano-Towner thesis
   (docs/research/genjl-parity.md §3).

   A generative function may own randomness that is not among its choices
   (encapsulated randomness, thesis §4.5): its score is then an unbiased
   estimate of its density, and assess, which draws that randomness, is
   random too.

   A trace is called on an address or path for its choice there, like
   Gen.jl's trace[addr]: (tr :slope), (tr [:y 0]); :choices, :retval, :score
   and :args are its fields.

   update and regenerate of a trace by a generative function that does not
   compute as the trace's maker (same-gen-fn?), such as a trace given
   another :gen-fn, keep nothing of it but its choices (revise)."
  (:refer-clojure :exclude [update -])
  (:require [gen.array :as array :refer [-]]
            [gen.array.batch :as batch]
            [gen.choicemap :as cm]
            [gen.diff :as diff]
            [gen.random :as rng]
            [gen.selection :as sel]
            [gen.trace :as trace]))

(set! *warn-on-reflection* true)

(defprotocol IGenerativeFunction
  (-simulate [gf args key])
  (-generate [gf args constraints key])
  (-update [gf trace args argdiffs constraints key])
  (-regenerate [gf trace args argdiffs selection key])
  (-assess [gf args choices key])
  (-propose [gf args key])
  (-project [gf trace selection]))

(def ^:dynamic *exact-density*
  "The name of the operation in effect that needs exact densities (a
   gradient, an exact enumeration, a replayed proposal), or nil. Under it,
   encapsulated randomness is a loud error: the score would be an estimate
   at one draw of it, which the operation would silently fix."
  nil)

(defmacro with-exact-density
  "body, in which encapsulated randomness is an error naming op."
  [op & body]
  `(binding [*exact-density* ~op] ~@body))

(defn check-exact-density!
  "Throws when an operation that needs exact densities is in effect: what
   (an untraced draw, an estimated density) makes the score an estimate."
  [what]
  (when-let [op *exact-density*]
    (throw (ex-info (str op " needs exact densities, but " what " makes the score an estimate"
                         " (encapsulated randomness, thesis §4.5)")
                    {:op op}))))

(defn generative-function?
  "Does x implement the GFI? Inline implementations answer by an instance
   check; satisfies? (slow when the answer is no) only covers extensions."
  [x]
  (or (instance? gen.gfi.IGenerativeFunction x)
      (and (some? x) (not (keyword? x)) (not (vector? x)) (not (number? x))
           (satisfies? IGenerativeFunction x))))

;; ---------------------------------------------------------------------------
;; Unvisited constraints
;; ---------------------------------------------------------------------------

(defn- unvisited-error [op paths detail cause]
  (ex-info (str (name op) ": the constraint at " (first paths) " was not visited" (when detail (str " (" detail ")")))
           {:op op :unvisited true :path (first paths) :paths (vec paths) :detail detail}
           cause))

(defn unvisited
  "The error for constraints at paths (relative to the generative function
   whose run left them unvisited) that a run did not visit, flagged for the
   calls above and for revise: ex-data :unvisited, :paths and :path (the
   first). detail, if given, follows the message in parentheses."
  ([op paths] (unvisited op paths nil))
  ([op paths detail] (unvisited-error op paths detail nil)))

(defn under
  "e, raised inside a call at path at (the call's address in its caller),
   with its unvisited paths under at; any other error is e itself."
  [e at]
  (let [{:keys [op paths detail] :as data} (ex-data e)]
    (if (and (:unvisited data) (seq at))
      (unvisited-error op (mapv #(into (vec at) %) paths) detail e)
      e)))

;; ---------------------------------------------------------------------------
;; What a generative function computes, and traces another one made
;; ---------------------------------------------------------------------------

(defprotocol IRevisable
  "How a generative function stands to traces it may not have made. Every
   generative function has the defaults (extended to Object): a record
   computes as another of its class whose fields are the same values
   (same-value?), and revises a trace by generating from its choices
   (generate-again)."
  (-same-gen-fn? [gf other budget]
    "Does gf compute as other, a generative function of gf's class? budget
     as same-value?'s.")
  (-revise [gf op args opts project-old]
    "op (:update or :regenerate, opts as run-op's) of (:prev opts), a trace
     gf may not have made: revise's result. (project-old prev selection) is
     prev's projection under the context prev ran in."))

(def ^:private same-limit
  "How many values one comparison looks at before it calls the rest
   different (which is sound: a different gen-fn only runs again)."
  256)

(def ^:private ^java.lang.ClassValue closure-fields
  "The instance fields of a fn class: the values it closes over."
  (proxy [java.lang.ClassValue] []
    (computeValue [^Class c]
      (vec (for [^java.lang.reflect.Field f (.getDeclaredFields c)
                 :when (not (java.lang.reflect.Modifier/isStatic (.getModifiers f)))]
             (doto f (.setAccessible true)))))))

(defn- atomic? [x]
  (or (nil? x) (keyword? x) (symbol? x) (string? x) (array/host? x) (char? x) (class? x) (var? x)))

(declare same-value?)

(defn- same-seq?
  "Are seqs xs and ys of one length and the same values pairwise, in order?"
  [xs ys budget]
  (loop [xs (seq xs) ys (seq ys)]
    (cond (nil? xs) (nil? ys)
          (nil? ys) false
          (same-value? (first xs) (first ys) budget) (recur (next xs) (next ys))
          :else false)))

(defn- same-entries?
  "Are maps a and b of one length and the same keys and values pairwise, in
   iteration order? An entry is no value of its own: an identical key or
   value costs one identical? test and no budget (review 7, R7-2)."
  [a b budget]
  (let [^java.util.Iterator xs (.iterator ^Iterable a)
        ^java.util.Iterator ys (.iterator ^Iterable b)]
    (loop []
      (if (.hasNext xs)
        (and (.hasNext ys)
             (let [^java.util.Map$Entry x (.next xs) ^java.util.Map$Entry y (.next ys)]
               (and (same-value? (.getKey x) (.getKey y) budget) (same-value? (.getValue x) (.getValue y) budget)
                    (recur))))
        (not (.hasNext ys))))))

(defn same-value?
  "Are a and b the same value, judged without reading an array (a host read
   inside a run) and as code holding them could tell (review 6, R6-1)? They
   must be of one class. Then: atomic values (numbers, booleans, keywords,
   strings, nil) by .equals, which tells -0.0 from 0.0 and takes NaN for
   NaN; generative functions by their kind's rule (same-gen-fn?); fns by
   their closed-over values (a fn built again from the same expression and
   locals); maps, sets, vectors and lists by their metadata and their
   elements pairwise in iteration order (an array-map's order, a sorted
   collection's and its comparator, a vector against a list all count), a
   map's entries by key and value, never by hashing. An array is the same
   only as itself, and so is anything else. A comparison looks at up to 256
   values before it calls the rest different, which is sound: a value
   called different only makes what holds it run again."
  ([a b] (or (identical? a b) (same-value? a b (long-array 1 same-limit))))
  ([a b ^longs budget]
   (cond
     (identical? a b) true
     (neg? (aset budget 0 (dec (aget budget 0)))) false
     (not (identical? (class a) (class b))) false
     (atomic? a) (.equals ^Object a b)
     (array/array? a) false
     (instance? gen.gfi.IGenerativeFunction a) (-same-gen-fn? a b budget)
     (fn? a) (every? (fn [^java.lang.reflect.Field f] (same-value? (.get f a) (.get f b) budget))
                     (.get closure-fields (class a)))
     (or (map? a) (set? a) (and (sequential? a) (counted? a)))
     (and (== (count a) (count b)) (same-value? (meta a) (meta b) budget)
          ;; the comparator decides where conj puts a value: (sorted-set-by
          ;; < 5) is not (sorted-set-by > 5) (review 7, R7-6)
          (or (not (instance? clojure.lang.Sorted a))
              (same-value? (.comparator ^clojure.lang.Sorted a) (.comparator ^clojure.lang.Sorted b) budget))
          (if (map? a) (same-entries? a b budget) (same-seq? a b budget)))
     :else false)))

(defn same-gen-fn?
  "Do generative functions a and b compute the same: identical, or of one
   class and the same by their kind's rule (IRevisable; a record's fields
   the same values, same-value?)? A combinator or a wrapper built again
   around the same generative functions and values is then the same, and
   one around another value is not."
  [a b]
  (or (identical? a b)
      (and (identical? (class a) (class b)) (-same-gen-fn? a b (long-array 1 same-limit)))))

(defn- without
  "choices without anything at path p."
  [choices p]
  (case (count p)
    0 cm/EMPTY
    1 (cm/remove-submap choices (first p))
    (let [sub (without (cm/get-submap choices [(first p)]) (subvec p 1))]
      (if (cm/empty? sub) (cm/remove-submap choices (first p)) (cm/set-submap choices (first p) sub)))))

(def ^:dynamic ^:no-doc *revising*
  "While generate-again's generate runs, {:constraints :prefix}: the
   revise's constraints, with any a run adds (constraining), and the path
   of the call running now (a call extends it, gen.dynamic/run-call). Each
   call's generate in it drops the old choices its callee leaves unvisited
   (generate-dropping), never a constraint (review 6, R6-3), so a structure
   change costs every call one more run of its own, not the whole revise
   one per call. nil otherwise. Public operations run without it
   (outside-revise): a caller's own constraints stay checked."
  nil)

(defmacro ^:no-doc outside-revise
  "body, a public operation: out of reach of a revise in progress."
  [& body]
  `(if *revising* (binding [*revising* nil] ~@body) (do ~@body)))

(defn ^:no-doc constraining
  "(f), a run of the call running now that takes choices (at its own
   addresses) as constraints too, such as a proposal's: in a revise they
   are constraints of the revise, never dropped, so one the run leaves is
   the unvisited error, as outside a revise (review 7, R7-1)."
  [choices f]
  (let [{:keys [constraints prefix]} *revising*]
    (if (and *revising* (not (cm/empty? choices)))
      (binding [*revising* (assoc *revising* :constraints
                                  (cm/override constraints (if (seq prefix) (cm/set-choice cm/EMPTY prefix choices) choices)))]
        (f))
      (f))))

(defn ^:no-doc generate-dropping
  "(generate sigma)'s result and the sigma it ran with, in a revise
   (*revising*) at the call at path prefix: sigma less the paths of every
   unvisited error that sigma holds as old choices (not under the revise's
   constraints), each error's paths dropped at once and generate run again.
   Any other error is raised. Each round drops paths the sigma of that
   round holds, so it ends: a path raised again once dropped is raised."
  [generate sigma prefix]
  (let [constraints (:constraints *revising*)]
    (loop [sigma sigma]
      (let [old? (fn [p] (and (not (cm/empty? (cm/get-submap sigma p)))
                              (cm/empty? (cm/get-submap constraints (into prefix p)))))
            r (try (generate sigma)
                   (catch clojure.lang.ExceptionInfo e
                     (let [{flagged :unvisited paths :paths} (ex-data e)]
                       (if (and flagged (seq paths) (every? old? paths)) {::drop paths} (throw e)))))]
        (if-let [paths (::drop r)]
          (recur (reduce without sigma paths))
          [r sigma])))))

(defn ^:no-doc rerun-without-drops
  "In a revise: (attempt sigma), whose calls drop the old choices they
   leave, run again with sigma less every old choice its trace lacks, until
   it lacks none: [result sigma]. For code that reads its constraints
   before its model runs (a proposal override's q), which then sees the
   final ones (review 6, R6-4)."
  [attempt sigma]
  (loop [sigma sigma]
    (let [r (attempt sigma)
          new (:choices (:trace r))
          dropped (remove #(cm/has-value? (cm/get-submap new %)) (cm/addresses sigma))]
      (if (seq dropped) (recur (reduce without sigma dropped)) [r sigma]))))

(defn- survivors
  "The paths of old (choices) that the new choices still hold, less those
   under constraints: the old values a revise kept."
  [old new constraints]
  (filter #(and (cm/has-value? (cm/get-submap new %)) (cm/empty? (cm/get-submap constraints %)))
          (if (cm/has-value? old) [[]] (cm/addresses old))))

(defn generate-again
  "op (:update or :regenerate, opts as run-op's) of prev, (:prev opts), as
   generate from prev's choices: update's (prev's choices overridden by
   the constraints) or regenerate's (the unselected ones). An old choice
   the new run leaves unvisited is dropped where it is left, at the call
   (*revising*, generate-dropping), and generate runs that call again; a
   constraint so left is an unvisited error, as in the update's own run,
   and is never dropped.
   update's weight is generate's less prev's score, its discard every old
   choice that does not survive unchanged; regenerate's weight is
   generate's less prev's projection on the old choices it kept (Eq. 4.1,
   by (project-old prev selection)): a dropped choice cancels against the
   reverse move, which draws it afresh. The retdiff is unknown-change.
   IRevisable's default."
  [gf op args {:keys [prev constraints selection key]} project-old]
  (let [constraints (or constraints cm/EMPTY)
        old (:choices prev)
        sigma (if (= op :update) (cm/override old constraints) (sel/get-selected old (sel/complement selection)))
        [{t :trace w :weight}] (binding [*revising* {:constraints constraints :prefix []}]
                                 (generate-dropping #(-generate gf args % key) sigma []))
        new (:choices t)]
    (when-let [missed (seq (remove #(cm/has-value? (cm/get-submap new %)) (cm/addresses constraints)))]
      (throw (unvisited op missed)))
    (if (= op :update)
      (let [kept (set (survivors old new constraints))]
        {:trace t :weight (- w (:score prev)) :retdiff diff/unknown-change
         :discard (cond (empty? kept) old
                        (cm/has-value? old) cm/EMPTY
                        :else (reduce (fn [d p] (if (kept p) d (cm/set-choice d p (cm/get-choice old p))))
                                      cm/EMPTY (cm/addresses old)))})
      (let [unselected (if (cm/has-value? sigma) [[]] (cm/addresses sigma))
            kept (survivors sigma new cm/EMPTY)]
        {:trace t :retdiff diff/unknown-change
         :weight (- w (project-old prev (if (== (count kept) (count unselected))
                                          (sel/complement selection)
                                          (sel/from-paths kept))))}))))

(extend-protocol IRevisable
  Object
  (-same-gen-fn? [gf other budget] (and (record? gf) (same-entries? gf other budget)))
  (-revise [gf op args opts project-old] (generate-again gf op args opts project-old)))

(defn project-by-maker
  "tr's projection on selection by its maker (which a trace given another
   gen-fn is not projected by)."
  [tr selection]
  (-project (trace/maker tr) tr selection))

(defn revise
  "op (:update or :regenerate, opts as run-op's) of (:prev opts), a trace
   gf did not make or that ran in another context (a call's changed store):
   nothing of it is kept but its choices, whatever kind gf is (no argument
   shows such a change, so no kind's own skipping could see it). By gf's
   IRevisable: by default generate-again, with prev projected by its maker
   (project-by-maker) or by (project-old prev selection)."
  ([gf op args opts] (revise gf op args opts project-by-maker))
  ([gf op args opts project-old] (-revise gf op args opts project-old)))

(defn run-op
  "Run a GFI op on gf from a map of its inputs (:key :constraints :prev
   :argdiffs :selection). With a non-empty :batch-shape the op runs batched:
   its scalar code under the batch tracer (gen.array.batch), which gives
   every generative function a batched form with the same meaning. Returns
   the op's result map ({:trace} for simulate). An update or regenerate of
   a trace whose maker gf does not compute as (same-gen-fn?) revises it.
   The one dispatch point for generative functions that call others
   (splices, combinators), and for update and regenerate."
  [gf op args {:keys [key constraints prev argdiffs selection batch-shape] :as opts}]
  (if (seq batch-shape)
    (batch/batched batch-shape (run-op gf op args (dissoc opts :batch-shape)))
    (case op
      :simulate {:trace (-simulate gf args key)}
      :generate (-generate gf args constraints key)
      :assess (-assess gf args constraints key)
      :update (if (same-gen-fn? gf (trace/maker prev))
                (-update gf prev args argdiffs constraints key)
                (revise gf op args opts))
      :regenerate (if (same-gen-fn? gf (trace/maker prev))
                    (-regenerate gf prev args argdiffs selection key)
                    (revise gf op args opts)))))

;; ---------------------------------------------------------------------------
;; The implicit key source
;; ---------------------------------------------------------------------------

(defonce ^:private implicit
  (atom (let [seed (.nextLong (java.security.SecureRandom.))]
          {:seed seed :key (rng/key seed)})))

(defonce ^:private counter (java.util.concurrent.atomic.AtomicLong.))

(defn implicit-seed
  "The seed of the implicit key source: random for each JVM unless seed! set
   it. Record it to reproduce a run."
  []
  (:seed @implicit))

(defn seed!
  "Reset the implicit key source to seed: later calls without a key repeat
   exactly."
  [seed]
  (reset! implicit {:seed (long seed) :key (rng/key seed)})
  (.set ^java.util.concurrent.atomic.AtomicLong counter 0)
  nil)

(def ^:private backward-salt 0x6261636b)

(defn backward-key
  "A key for a move's backward assess, folded from the move's key k apart
   from any key a generative function derives from k (sites split it, Map
   and Unfold fold it with small element indices), so encapsulated
   randomness there never repeats the model's."
  [k]
  (rng/fold-in k backward-salt))

(defn next-key
  "A fresh key from the implicit key source (deterministic given the seed and
   the order of calls)."
  []
  (rng/fold-in (:key @implicit) (.getAndIncrement ^java.util.concurrent.atomic.AtomicLong counter)))

;; ---------------------------------------------------------------------------
;; Operations
;; ---------------------------------------------------------------------------

(defn check-args
  "The arguments as a vector: they must be a sequential collection (or nil,
   no arguments)."
  [op args]
  (if (or (nil? args) (sequential? args))
    (vec args)
    (throw (ex-info (str (name op) ": the arguments must be a vector of arguments, got " (pr-str args))
                    {:op op :args args}))))

(defn check-choice-map
  "x, a choice map (nil is the empty one). A plain Clojure map is an error:
   it would be read as no choices at all."
  [op what x]
  (cond (nil? x) cm/EMPTY
        (cm/choicemap? x) x
        :else (throw (ex-info (str (name op) ": " what " must be a choice map, not a " (.getName (class x))
                                   (when (map? x) ": use (gen.choicemap/from-map m)"))
                              {:op op :value-type (class x)}))))

(defn check-selection
  "x, a selection."
  [op x]
  (if (sel/selection? x)
    x
    (throw (ex-info (str (name op) ": the selection must be a selection, such as (gen.selection/select :a), not "
                         (pr-str x))
                    {:op op :value x}))))

(defn eval-trace!
  "Force the evaluation of tr's arrays (choices, return value and score) on
   a lazy backend, and return tr. Inference loops call it once per
   iteration (docs/array-api.md §9), so that a lazy backend's graph stays
   one iteration deep. With also, a tree of other arrays (a weight, say) is
   evaluated with it. A no-op on eager backends."
  ([tr] (eval-trace! tr nil))
  ([tr also]
   ;; one evaluation for the trace and also (e.g. a weight)
   (array/eval! [(:choices tr) (:retval tr) (:score tr) also])
   tr))

(defn simulate
  "Sample a trace of gf on args."
  ([gf args] (simulate gf args (next-key)))
  ([gf args key] (-simulate gf (check-args :simulate args) key)))

(defn generate
  "A trace of gf on args consistent with constraints (a choice map), and the
   log weight log p(τ; x) - log q(υ; x, σ), where υ are the choices the
   internal proposal sampled and σ the constraints."
  ([gf args constraints] (generate gf args constraints (next-key)))
  ([gf args constraints key]
   (outside-revise
    (-generate gf (check-args :generate args) (check-choice-map :generate "the constraints" constraints) key))))

(defn check-argdiffs
  "One argdiff per argument: a missing one would silently mean no-change."
  [op args argdiffs]
  (when-not (= (count args) (count argdiffs))
    (throw (ex-info (str (name op) ": " (count args) " arguments but " (count argdiffs) " argdiffs")
                    {:args (count args) :argdiffs (count argdiffs)}))))

(defn update
  "The trace with new arguments and constrained choices. Returns the new trace,
   the log weight, the retdiff, and the discarded choices (overwritten or no
   longer visited), which make the update reversible. The trace's :gen-fn
   runs the update: given another gen-fn than its maker, it revises it."
  ([trace constraints] (update trace constraints (next-key)))
  ([trace constraints key]
   (let [args (:args trace)]
     (update trace args (diff/no-changes (count args)) constraints key)))
  ([trace args argdiffs constraints] (update trace args argdiffs constraints (next-key)))
  ([trace args argdiffs constraints key]
   (check-argdiffs :update (check-args :update args) argdiffs)
   (outside-revise
    (run-op (:gen-fn trace) :update (vec args)
            {:prev trace :argdiffs (vec argdiffs) :constraints (check-choice-map :update "the constraints" constraints)
             :key key}))))

(defn regenerate
  "The trace with new arguments and the selected choices resampled from the
   internal proposal. Returns the new trace, the log weight and the retdiff.
   As update, by the trace's :gen-fn."
  ([trace selection] (regenerate trace selection (next-key)))
  ([trace selection key]
   (let [args (:args trace)]
     (regenerate trace args (diff/no-changes (count args)) selection key)))
  ([trace args argdiffs selection] (regenerate trace args argdiffs selection (next-key)))
  ([trace args argdiffs selection key]
   (check-argdiffs :regenerate (check-args :regenerate args) argdiffs)
   (outside-revise
    (run-op (:gen-fn trace) :regenerate (vec args)
            {:prev trace :argdiffs (vec argdiffs) :selection (check-selection :regenerate selection) :key key}))))

(defn assess
  "The log density of fully specified choices, and the return value. The
   key draws the encapsulated randomness of a generative function that has
   some; with the same key, assess's weight is generate's on the same
   choices."
  ([gf args choices] (assess gf args choices (next-key)))
  ([gf args choices key]
   (-assess gf (check-args :assess args) (check-choice-map :assess "the choices" choices) key)))

(defn propose
  "Sample choices from gf, with their log density and the return value."
  ([gf args] (propose gf args (next-key)))
  ([gf args key] (-propose gf (check-args :propose args) key)))

(defn project
  "The log density of the selected choices of a trace."
  [trace selection]
  (-project (:gen-fn trace) trace (check-selection :project selection)))
