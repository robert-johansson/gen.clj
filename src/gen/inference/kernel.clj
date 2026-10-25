(ns gen.inference.kernel
  "MCMC kernels and their combinators.

   A kernel is a function (kernel trace key) → {:trace trace' :accepted? …}:
   Gen.jl's (trace, metadata), as a map, :accepted? whether it took its
   move. Every kernel of the library returns it: MH's and the gradient
   kernels' with :log-alpha, NUTS's beside its statistics, and a
   combinator's (did any of its moves). A function that returns a bare
   trace is not a kernel, and running one is a loud error. A Kernel value
   can also carry its reversal, the kernel that undoes it in distribution
   (Gen.jl's reversal). MH-family kernels are reversible, so they are their
   own reversals. Combinators of kernels with reversals have reversals too:
   the reversal of a chain is the chain of the reversals, in reverse order.

     (mh selection), (mh proposal args), (involutive-mh proposal args h)
     (hmc selection opts), (mala selection tau opts), (nuts selection opts),
     (elliptical-slice addr mu cov)
                           gen.inference.mh's and mcmc's moves as kernels,
                           each its own reversal
     (map-optimize selection opts)
                           a step of gradient ascent, with no reversal

     (chain k1 k2 k3)      k1, then k2, then k3
     (repeat n k)          k, n times
     (cycle n [k1 k2])     k1, k2, k1, … for n steps in all
     (mix [k1 k2] [w1 w2]) one of the kernels, chosen with probability ∝ w
     (checked k observations)  k, then check that it left the observations alone
     (kern [args] stmt …)  Gen.jl's @kern: statements, loops and branches over
                           the arguments, as one chain with its reversal
                           (with no arguments, (kern [] stmt …) is the
                           kernel itself)
     (kern ^:trace [tr args] stmt …)
                           the same, reading the trace in its loops,
                           branches, lets and kernels' arguments, with the
                           thesis's invariant checked after each

   A combinator's result reports the accept decisions of the moves it
   made, in order, as :accepted (a vector of booleans) and :accepted? (any
   of them) when they all report one, and how many of its proposals
   diverged (gen.inference.mcmc's :diverged?) as :divergences, when any
   did.

   A kernel may also carry a batched form (batched-form), which moves
   every particle of a gen.batched collection at once, as its -particles
   move does (mh/mh-particles, mcmc/hmc-particles, mala-particles,
   map-optimize-particles): the mh, hmc, mala and map-optimize kernels
   have one, and chain, repeat, cycle and kern (without ^:trace) have one
   when each of their kernels does. A batched particle filter's
   rejuvenation stays batched with such a kernel
   (gen.inference.particle-filter/rejuvenate).

   (run kernel trace opts) runs a chain and collects samples: its traces,
   or with :keep a function of each, read on the host."
  {:api :stable}
  (:refer-clojure :exclude [repeat cycle reversible?])
  (:require [gen.array :as array]
            [gen.choicemap :as cm]
            [gen.gfi :as gfi]
            [gen.inference.mcmc :as mcmc]
            [gen.inference.mh :as mh]
            [gen.random :as rng]))

(set! *warn-on-reflection* true)

(defrecord Kernel [f reverse-f particles reverse-particles]
  clojure.lang.IFn
  (invoke [_ trace] (f trace (gfi/next-key)))
  (invoke [_ trace key] (f trace key))
  (applyTo [this args] (clojure.lang.AFn/applyToHelper this args)))

(defn kernel
  "A kernel from (fn [trace key] → {:trace …}), with its reversal when
   given. Options: :particles and :reverse-particles, the batched forms of
   the kernel and of its reversal (batched-form)."
  ([f] (->Kernel f nil nil nil))
  ([f reverse-f] (->Kernel f reverse-f nil nil))
  ([f reverse-f opts]
   (when-let [unknown (seq (remove #{:particles :reverse-particles} (keys opts)))]
     (throw (ex-info (str "kernel: unknown options " (vec unknown) "; the options are :particles and :reverse-particles")
                     {:unknown (vec unknown)})))
   (->Kernel f reverse-f (:particles opts) (:reverse-particles opts))))

(defn symmetric
  "A kernel that is its own reversal; with particles, its batched form,
   which is its reversal's too."
  ([f] (->Kernel f f nil nil))
  ([f particles] (->Kernel f f particles particles)))

(defn reversible? [k] (and (instance? Kernel k) (some? (:reverse-f k))))

(defn reversal
  "The declared reversal of a kernel (with the reversal's batched form,
   when the kernel declares one)."
  [k]
  (if (reversible? k)
    (->Kernel (:reverse-f k) (:f k) (:reverse-particles k) (:particles k))
    (throw (ex-info "The kernel has no declared reversal" {:kernel k}))))

(defn batched-form
  "k's batched form, or nil: (fn [ps key] → {:particles ps' …}), the
   kernel applied to every particle of a gen.batched collection at once,
   n independent chains, each particle's move distributed as k's on its
   trace. gen.inference.particle-filter/rejuvenate moves a batched filter
   by it, and one particle at a time when k has none (recording
   :fallback). The library's kernels have one where a -particles move
   exists: mh (of a selection, or of a proposal: mh/mh-particles), hmc
   and mala (mcmc/hmc-particles and mala-particles, which drop
   :compile?: a batched step runs under the batch tracer) and
   map-optimize; and chain, repeat, cycle and kern (without ^:trace) when
   every kernel they combine has one. Its result holds the -particles
   moves' keys: {:particles}, :accepted (n booleans: did any of its moves
   take each particle) when every move reports them, and :fallback, why a
   move ran one particle at a time, when one did. involutive-mh, nuts,
   elliptical-slice, mix, checked and (kern ^:trace …) have none."
  [k]
  (when (instance? Kernel k) (:particles k)))

(defn- step-fn
  "The (fn [trace key]) of a kernel or a plain function."
  [k]
  (if (instance? Kernel k) (:f k) k))

(defn ^:no-doc step-result
  "r, a kernel's result, when it is one ({:trace …}); a loud error
   otherwise, such as for a bare trace."
  [r]
  (if (and (map? r) (contains? r :trace))
    r
    (throw (ex-info (str "A kernel returns {:trace …}, not " (if (nil? r) "nil" (.getName (class r)))
                         ": a function f of a trace and a key that returns the trace alone is the kernel"
                         " (fn [t key] {:trace (f t key)})")
                    {:result-type (some-> r class .getName)}))))

(defn ^:no-doc particles-result
  "r, a batched form's result, when it is one ({:particles …}); a loud
   error otherwise."
  [r]
  (if (and (map? r) (contains? r :particles))
    r
    (throw (ex-info (str "A kernel's batched form returns {:particles …}, not "
                         (if (nil? r) "nil" (.getName (class r))))
                    {:result-type (some-> r class .getName)}))))

(defn- with-reversals
  "A kernel built by combine from the step fns of ks, reversible when every
   k is: its reversal is combine over the reversals in reverse-order. With
   combine-particles, its batched form combines the ks' batched forms, when
   every k has one (and the reversal's, the reversals')."
  [combine combine-particles ks reverse-order]
  (let [rs (when (every? reversible? ks) (mapv reversal (reverse-order ks)))
        batched (fn [ks] (when (and combine-particles (every? batched-form ks))
                           (combine-particles (mapv batched-form ks))))]
    (->Kernel (combine (mapv step-fn ks))
              (when rs (combine (mapv step-fn rs)))
              (batched ks)
              (when rs (batched rs)))))

(defn- accepted-moves
  "The accept decisions a kernel result reports: one for an MH step, a
   combinator's sequence, or nil when it reports none."
  [r]
  (cond (contains? r :accepted) (:accepted r)
        (contains? r :accepted?) [(:accepted? r)]))

(defn- divergences
  "How many proposals a kernel result reports diverged: an HMC or MALA
   move's :diverged?, or a combinator's :divergences."
  ^long [r]
  (cond (contains? r :divergences) (long (:divergences r))
        (:diverged? r) 1
        :else 0))

(defn- combined
  "A combinator's result: the trace t, and the moves' accept decisions as
   :accepted and :accepted? (any of them) when every move reported one."
  [t moves diverged]
  (cond-> {:trace t}
    moves (assoc :accepted? (boolean (some true? moves)) :accepted moves)
    (pos? diverged) (assoc :divergences diverged)))

(defn- sequence-fn [fs]
  (fn [trace key]
    (let [[t moves diverged] (reduce (fn [[t moves diverged] [f k]]
                                       (let [r (step-result (f t k))]
                                         [(:trace r) (when moves (some->> (accepted-moves r) (into moves)))
                                          (+ diverged (divergences r))]))
                                     [trace [] 0]
                                     (map vector fs (rng/split key (count fs))))]
      (combined t moves diverged))))

(defn- particles-sequence-fn
  "The batched forms pfs applied in sequence, the i-th with the key's i-th
   split, as sequence-fn applies its moves: {:particles}, :accepted (did
   any move take each particle) when every move reports it, and the first
   :fallback."
  [pfs]
  (fn [ps key]
    (let [[ps accepted m] (reduce (fn [[ps accepted m] [pf k]]
                                    (let [r (particles-result (pf ps k))]
                                      [(:particles r)
                                       (when (and accepted (:accepted r)) (mapv #(or %1 %2) accepted (:accepted r)))
                                       (or m (:fallback r))]))
                                  [ps (vec (clojure.core/repeat (:n ps) false)) nil]
                                  (map vector pfs (rng/split key (count pfs))))]
      (cond-> {:particles ps}
        accepted (assoc :accepted accepted)
        m (assoc :fallback m)))))

(defn chain
  "The kernels applied in sequence; batched when each of them is
   (batched-form)."
  [& kernels]
  (with-reversals sequence-fn particles-sequence-fn (vec kernels) reverse))

(declare kern-form)

(defn- statements
  "A body's statements as one kernel: the lone statement's, or their chain."
  [forms]
  (if (= 1 (count forms)) (kern-form (first forms)) `(chain ~@(map kern-form forms))))

(defn checked-kernel
  "k, when it is a kernel (a function); a loud error naming form otherwise."
  [k form]
  (if (ifn? k)
    k
    (throw (ex-info (str "kern: " (pr-str form) " gave " (pr-str k) ", not a kernel") {:form form}))))

(defn- kern-form
  "A kern statement as a kernel expression: a loop is the chain of its
   iterations, a branch the chain of its taken side (or none), a let binds
   over its statements; any other form is a kernel."
  [form]
  (let [head (when (and (seq? form) (symbol? (first form))) (name (first form)))]
    (case head
      ("doseq" "for") (let [[_ bindings & body] form] `(apply chain (for ~bindings ~(statements body))))
      "when" (let [[_ c & body] form] `(if ~c ~(statements body) (chain)))
      "if" (let [[_ c a b] form] `(if ~c ~(kern-form a) ~(if (some? b) (kern-form b) `(chain))))
      "let" (let [[_ bindings & body] form] `(let ~bindings ~(statements body)))
      `(checked-kernel ~form '~form))))

(defn ^:no-doc control-value
  "A value on the host (what a (kern ^:trace …) body's control flow read,
   or what run's :keep keeps): an array read at the inference loop's
   boundary, in collections too."
  [v]
  (cond (array/array? v) (array/->clj v)
        (sequential? v) (mapv control-value v)
        (and (map? v) (not (record? v))) (into {} (map (fn [[k x]] [k (control-value x)])) v)
        :else v))

(def ^:private ^java.lang.ClassValue closure-fields
  "The instance fields of a fn class: the values it closes over."
  (proxy [java.lang.ClassValue] []
    (computeValue [^Class c]
      (vec (for [^java.lang.reflect.Field f (.getDeclaredFields c)
                 :when (not (java.lang.reflect.Modifier/isStatic (.getModifiers f)))]
             (doto f (.setAccessible true)))))))

(defn- control-difference
  "nil when control values a and b are the same, else the first pair of
   their parts that differ. Host data compares by =, with no budget (a loop
   may run over a large collection), and what a construct built from it by
   what that holds: an array by its host value, a generative function by
   its kind's rule (gfi/same-gen-fn?), a fn by its class and closed-over
   values, a record by its class and fields. So a kernel built again from
   the same values is the same kernel. Closures that hold each other (a
   letfn's) compare co-inductively: a pair of fns met again on the way
   down is taken as the same."
  ([a b] (control-difference a b #{}))
  ([a b seen]
   (let [pairwise (fn [as bs seen] (some identity (map #(control-difference %1 %2 seen) as bs)))]
     (cond (identical? a b) nil
           (array/array? a) (when-not (and (array/array? b) (= (array/->clj a) (array/->clj b))) [a b])
           (instance? gen.gfi.IGenerativeFunction a) (when-not (gfi/same-gen-fn? a b) [a b])
           ;; fns are equal by identity, so the pair is too
           (fn? a) (cond (contains? seen [a b]) nil
                         (identical? (class a) (class b))
                         (let [fields (.get closure-fields (class a))]
                           (pairwise (map #(.get ^java.lang.reflect.Field % a) fields)
                                     (map #(.get ^java.lang.reflect.Field % b) fields)
                                     (conj seen [a b])))
                         :else [a b])
           (record? a) (if (identical? (class a) (class b)) (control-difference (into {} a) (into {} b) seen) [a b])
           (map? a) (if (and (map? b) (= (count a) (count b)) (every? #(contains? b %) (keys a)))
                      (some (fn [[k v]] (control-difference v (get b k) seen)) a)
                      [a b])
           (sequential? a) (if (and (sequential? b) (= (count a) (count b))) (pairwise a b seen) [a b])
           :else (when-not (= a b) [a b])))))

(defn- noting
  "[the control values (reads note) passed to note, on the host, and what it
   returned]."
  [reads]
  (let [seen (volatile! [])
        x (reads (fn [v] (vswap! seen conj v) v))]
    [(mapv control-value @seen) x]))

(defn- construct-head
  "The head of a loop, branch or let form, as a name; nil for a kernel
   statement."
  [form]
  (when (and (seq? form) (symbol? (first form)))
    (#{"doseq" "for" "when" "if" "let"} (name (first form)))))

(defn- construct-name [form] (str "(" (first form) " " (pr-str (second form)) " …)"))

;; The check is per construct, as in the thesis (Prop. 3.4.1, pp. 99-101) and
;; Gen.jl. A loop, branch or let is a dependent mixture: its control value
;; v(t) picks the kernel K_v of its body (a loop's: its iterations in turn),
;; whose reversal R_v has p(t) K_v(t, t') = p(t') R_v(t', t). The forward run
;; checks that K_v keeps v(t) = v, the reversal's that R_v keeps v(t') = v,
;; so [v(t) = v] = [v(t') = v] wherever K_v(t, t') > 0, and term by term
;; p(t) [v(t) = v] K_v(t, t') = p(t') [v(t') = v] R_v(t', t). Summed over v:
;; the reversal is the construct over the R_v, reading v from its own input,
;; which is what backward runs. A chain's reversal is its statements'
;; reversals in reverse order, which asks nothing across statements: a later
;; statement may change what an earlier construct read. A kernel statement
;; whose arguments read the trace is a dependent mixture too, its control
;; value the kernel it builds, so it is checked as a let is (the post-M5
;; review's P3-02; the thesis and Gen.jl leave it unchecked). (The check
;; sees the runs that happen, as Gen.jl's does, not every run the
;; proposition quantifies over.)
(defn ^:no-doc trace-kernel
  "The kernel of a (kern ^:trace …) body. forward and backward are (fn [t
   apply! construct]) running the statements (backward: reversed, loops
   backwards) against the volatile t. apply! applies a kernel to @t, the
   i-th applied with the key folded with i. (construct form reads run) runs
   a loop, branch, let or kernel statement: (reads note) reads its control
   values from @t (a loop's collections and conditions, a condition, a
   let's values, a statement's kernel), passing each to note, and returns
   what run takes; run runs the body. Then reads reads again, and values
   that differ (control-difference) are a loud error."
  [forward backward]
  (letfn [(run [body reversal? trace key]
            (let [t (volatile! trace)
                  n (volatile! 0)
                  moves (volatile! [])
                  diverged (volatile! 0)
                  apply! (fn [k]
                           (let [r (step-result ((step-fn (if reversal? (reversal k) k)) @t (rng/fold-in key @n)))]
                             (vswap! n inc)
                             (vswap! moves #(when % (some->> (accepted-moves r) (into %))))
                             (vswap! diverged + (divergences r))
                             (vreset! t (:trace r))))
                  construct (fn [form reads run-body]
                              (let [[read x] (noting reads)]
                                (run-body x)
                                (let [[again _] (noting reads)]
                                  (when-let [[was is] (control-difference read again)]
                                    (throw (ex-info (str "kern: "
                                                         (if (construct-head form)
                                                           (str "the body of " (construct-name form) " changed what it"
                                                                " reads (a loop's collection, a condition or a let's value)")
                                                           (str "the statement " (pr-str form) " changed what its"
                                                                " arguments read from the trace"))
                                                         ", so the reversal would not undo the kernel: read "
                                                         (pr-str was) ", then " (pr-str is))
                                                    {:form form :read read :again again}))))))]
              (body t apply! construct)
              (combined @t @moves @diverged)))]
    (kernel (fn [trace key] (run forward false trace key)) (fn [trace key] (run backward true trace key)))))

(declare traced-statements)

(defn- traced-form
  "A (kern ^:trace …) statement as code against the volatile t, with the
   trace parameter tr bound to @t wherever an expression is evaluated;
   apply! and construct as in trace-kernel. reverse? gives the reversal's
   code: statements reversed, loops backwards, and apply! applies
   reversals."
  [form tr t apply! construct reverse?]
  (let [head (when (and (seq? form) (symbol? (first form))) (name (first form)))
        at (fn [expr] `(let [~tr (deref ~t)] ~expr))
        note (gensym "note")
        noted (fn [expr] `(~note ~(at expr)))
        body (fn [forms] (traced-statements forms tr t apply! construct reverse?))
        checked (fn [reads run] `(~construct '~form (fn [~note] ~reads) ~run))]
    (case head
      ("doseq" "for")
      ;; the iterations as thunks, the collections read once at the loop
      (let [[_ bindings & stmts] form
            bindings' (vec (mapcat (fn [[b e]]
                                     (case b
                                       (:when :while) [b `(control-value ~(noted e))]
                                       :let [b (vec (mapcat (fn [[x v]] [x (noted v)]) (partition 2 e)))]
                                       [b (noted `(vec ~e))]))
                                   (partition 2 bindings)))
            iterations (gensym "iterations")]
        (checked `(vec (for ~bindings' (fn [] ~(body stmts))))
                 `(fn [~iterations] (run! #(%) ~(if reverse? `(rseq ~iterations) iterations)))))
      "when" (let [[_ c & stmts] form c' (gensym "c")]
               (checked `(control-value ~(noted c)) `(fn [~c'] (when ~c' ~(body stmts)))))
      "if" (let [[_ c a b] form c' (gensym "c")]
             (checked `(control-value ~(noted c)) `(fn [~c'] (if ~c' ~(body [a]) ~(when (some? b) (body [b]))))))
      "let" (let [[_ bindings & stmts] form
                  pairs (partition 2 bindings)
                  vs (vec (repeatedly (count pairs) #(gensym "v")))]
              ;; each value through a symbol of its own, so a binding may destructure
              (checked `(let [~@(mapcat (fn [v [x e]] [v (noted e) x v]) vs pairs)] ~vs)
                       `(fn [~vs] (let [~@(mapcat (fn [v [x _]] [x v]) vs pairs)] ~(body stmts)))))
      ;; a statement that reads the trace, as a let of its kernel
      (if (some #{tr} (tree-seq coll? seq form))
        (let [k (gensym "k")]
          (checked (noted form) `(fn [~k] (~apply! (checked-kernel ~k '~form)))))
        `(~apply! (checked-kernel ~(at form) '~form))))))

(defn- traced-statements [forms tr t apply! construct reverse?]
  `(do ~@(map #(traced-form % tr t apply! construct reverse?) (if reverse? (reverse forms) forms)) nil))

(defmacro kern
  "Gen.jl's @kern: (kern [args] statement …) is a function of args returning
   one kernel, the statements' kernels in order; with no arguments,
   (kern [] statement …) is that kernel. A statement is a kernel, or
   doseq/for, if/when or let over statements; their conditions and ranges
   are computed from the arguments when the kernel is made, never from the
   trace, so the result is a chain and its reversal is automatic: every
   statement reversed, in reverse order (loops run backwards).

     (kern [n] (mh (sel/select :mu)) (doseq [i (range n)] (mh (sel/select [:z i]))))

   (kern ^:trace [tr args] statement …) reads the trace, as Gen.jl's @kern
   does: tr is the trace as it is when each statement is evaluated (after
   the kernels before it), in a kernel statement's arguments and in the
   loops, conditions and lets. Kernels apply in order, the i-th with the
   key folded with i; the reversal runs the statements reversed, loops
   backwards, each kernel's reversal (loud for a kernel without one).
   After the body of each loop, branch and let, what the construct read
   (its collections, its condition, its values) is read again from the
   trace, and a body that changed it is a loud error, in the reversal too:
   the thesis's invariant (pp. 100-101), which Gen.jl checks with
   check=true and gen.clj always. A kernel statement whose arguments read
   tr is checked the same way, as a let of its kernel: built again from
   the trace it left, the kernel must be the same (fns compare by their
   closed-over values, so a kernel built again from the same values is
   the same). A later statement may change what an earlier one read, as
   the thesis's p. 102 kernel does. Conditions are Clojure's truthiness of
   host values (an array is read on the host). With tr alone,
   (kern ^:trace [tr] statement …) is the kernel itself.

     (kern ^:trace [tr]
       (doseq [i (range (inc (array/item (tr :degree))))] (mh (sel/select [:coeff i])))
       (mh (sel/select :degree))
       (doseq [i (range (inc (array/item (tr :degree))))] (mh (sel/select [:coeff i]))))"
  [params & body]
  (if (:trace (meta params))
    (let [[tr & args] params
          t (gensym "t") apply! (gensym "apply") construct (gensym "construct")]
      (let [k `(trace-kernel (fn [~t ~apply! ~construct] ~(traced-statements body tr t apply! construct false))
                             (fn [~t ~apply! ~construct] ~(traced-statements body tr t apply! construct true)))]
        (if (seq args) `(fn [~@args] ~k) k)))
    (if (seq params) `(fn ~params ~(statements body)) (statements body))))

(defn repeat
  "The kernel applied n times."
  [n kernel]
  (apply chain (clojure.core/repeat n kernel)))

(defn cycle
  "The kernels applied in turn, n steps in all (one round by default)."
  ([kernels] (cycle (count kernels) kernels))
  ([n kernels] (apply chain (take n (clojure.core/cycle kernels)))))

(defn mix
  "One kernel chosen at random with probabilities proportional to weights.
   Its reversal mixes the reversals with the same weights."
  [kernels weights]
  (when-not (= (count kernels) (count weights))
    (throw (ex-info "mix needs one weight per kernel" {:kernels (count kernels) :weights (count weights)})))
  (when-not (and (every? #(and (number? %) (>= % 0)) weights) (some pos? weights))
    (throw (ex-info "mix needs non-negative weights, not all zero" {:weights weights})))
  (let [logits (array/log (array/array (mapv double weights)))]
    (with-reversals (fn [fs]
                      (fn [trace key]
                        (let [[k-choose k-step] (rng/split key)]
                          ((fs (array/item (rng/categorical k-choose logits))) trace k-step))))
                    nil
                    (vec kernels)
                    identity)))

(defn- check-observations! [trace observations]
  (doseq [addr (cm/addresses observations)]
    (let [observed (cm/get-choice observations addr)
          actual (cm/get-choice (:choices trace) addr nil)]
      (when-not (and (some? actual) (array/equal? observed actual))
        (throw (ex-info (str "The kernel changed the observation at " addr)
                        {:path addr :observed observed :actual actual}))))))

(defn checked
  "The kernel, checking after each application that the trace still holds
   the observations (Gen.jl's check=true, observations=…)."
  [kernel observations]
  (let [check (fn [f] (fn [trace key]
                        (let [r (step-result (f trace key))]
                          (check-observations! (:trace r) observations)
                          r)))]
    (->Kernel (check (step-fn kernel))
              (when (reversible? kernel) (check (:reverse-f kernel)))
              nil nil)))

;; ---------------------------------------------------------------------------
;; MH kernels
;; ---------------------------------------------------------------------------

(defn mh
  "The MH kernel of a selection, or of a proposal and its arguments;
   batched, mh/mh-particles."
  ([selection] (symmetric (fn [trace key] (mh/mh trace selection key))
                          (fn [ps key] (mh/mh-particles ps selection key))))
  ([proposal proposal-args] (symmetric (fn [trace key] (mh/mh trace proposal proposal-args key))
                                       (fn [ps key] (mh/mh-particles ps proposal proposal-args key)))))

(defn involutive-mh
  "The involutive MCMC kernel of a proposal, its arguments and an
   involution."
  [proposal proposal-args involution]
  (symmetric (fn [trace key] (mh/involutive-mh trace proposal proposal-args involution key))))

;; ---------------------------------------------------------------------------
;; Gradient kernels (gen.inference.mcmc)
;; ---------------------------------------------------------------------------

(defn- kernel-opts
  "A gradient kernel's options, an options map or nil, without :key: the
   kernel takes a key at each application."
  [what opts]
  (when-not (or (nil? opts) (map? opts))
    (throw (ex-info (str what " takes an options map, not " (pr-str opts)) {:value opts})))
  (when (contains? opts :key)
    (throw (ex-info (str what ": a kernel takes its key at each application, not in its options")
                    {:key (:key opts)})))
  opts)

(defn hmc
  "The HMC kernel of a selection, with mcmc/hmc's options (:L, :eps,
   :inv-metric, :jitter, :compile?). Reversible: its own reversal."
  ([selection] (hmc selection nil))
  ([selection opts]
   (let [opts (kernel-opts "hmc" opts)
         batched-opts (dissoc opts :compile?)]
     (symmetric (fn [trace key] (mcmc/hmc trace selection opts key))
                (fn [ps key] (mcmc/hmc-particles ps selection batched-opts key))))))

(defn mala
  "The MALA kernel of a selection and a step size tau, with mcmc/mala's
   options (:inv-metric, :compile?). Reversible: its own reversal."
  ([selection tau] (mala selection tau nil))
  ([selection tau opts]
   (let [opts (kernel-opts "mala" opts)
         batched-opts (dissoc opts :compile?)]
     (symmetric (fn [trace key] (mcmc/mala trace selection tau (assoc opts :key key)))
                (fn [ps key] (mcmc/mala-particles ps selection tau (assoc batched-opts :key key)))))))

(defn nuts
  "The NUTS kernel of a selection: one transition of mcmc/nuts, with its
   options (:eps, :max-depth, :inv-metric, :jitter, :compile?), such as an
   adapted state's: (nuts s (select-keys adapted [:eps :inv-metric])),
   adapted from gen.inference.nuts/adapt. Reversible (the transition keeps
   detailed balance): its own reversal."
  ([selection] (nuts selection nil))
  ([selection opts]
   (let [opts (kernel-opts "nuts" opts)]
     (symmetric (fn [trace key] (mcmc/nuts trace selection opts key))))))

(defn elliptical-slice
  "The elliptical slice sampling kernel of the choice at addr, whose prior
   is N(mu, cov) (mcmc/elliptical-slice). Reversible: its own reversal."
  [addr mu cov]
  (symmetric (fn [trace key] (mcmc/elliptical-slice trace addr mu cov key))))

(defn map-optimize
  "A step of mcmc/map-optimize on a selection, with its options, as a
   kernel (its key unused). It climbs toward a mode and leaves no
   distribution unchanged, so it declares no reversal."
  ([selection] (map-optimize selection nil))
  ([selection opts]
   (let [opts (kernel-opts "map-optimize" opts)]
     (kernel (fn [trace _key] (mcmc/map-optimize trace selection opts))
             nil
             {:particles (fn [ps _key] (mcmc/map-optimize-particles ps selection opts))}))))

;; ---------------------------------------------------------------------------
;; Running a chain
;; ---------------------------------------------------------------------------

(defn ^:no-doc keeping
  "run's and gen.inference.nuts/sample's :keep, checked: [read finish],
   read giving what a kept trace keeps, (keep trace) with its arrays read
   on the host (control-value), and finish the vector of those as :draws.
   For a map of functions, read gives a map and finish a map of vectors."
  [keep]
  (cond (fn? keep) [#(control-value (keep %)) identity]
        (and (map? keep) (seq keep) (every? fn? (vals keep)))
        [(fn [t] (update-vals keep #(control-value (% t))))
         (fn [kept] (into {} (map (fn [k] [k (mapv #(get % k) kept)])) (keys keep)))]
        :else (throw (ex-info (str ":keep is a function of a trace, or a map of them, not " (pr-str keep))
                              {:keep keep}))))

(defn run
  "Apply the kernel repeatedly from trace and collect samples. opts:
   :samples (required), :burn (steps discarded first, 0), :thin (steps per
   sample, 1), :key, :keep. Returns {:traces [...] :acceptance-rate r
   :divergences n}: the rate over every move of every step (a combinator's
   steps make several), or nil when the kernel does not report its accept
   decisions, and how many proposals diverged (HMC's, MALA's and NUTS's
   :diverged?, as Stan counts them). With :keep, a function of a trace,
   the result holds :draws in place of :traces: (keep t) for each kept
   trace t, its arrays read on the host, so that the chain holds its
   current trace alone (a long chain on MLX, say). (run k t {:samples
   1000 :keep #(% :x)}) gives :x's draws as host numbers, which
   gen.inference.diagnostics reads. :keep may be a map of such functions,
   and :draws is then a map of their vectors: {:keep {:x #(% :x) :y #(%
   :y)}}."
  [kernel trace {:keys [samples burn thin key keep] :or {burn 0 thin 1}}]
  (when-not (and (nat-int? samples) (nat-int? burn) (pos-int? thin))
    (throw (ex-info "run needs :samples ≥ 0, :burn ≥ 0 and :thin ≥ 1" {:samples samples :burn burn :thin thin})))
  (let [f (step-fn kernel)
        [read finish] (if (some? keep) (keeping keep) [identity nil])
        steps (+ burn (* samples thin))
        step-keys (if (pos? steps) (rng/split (or key (gfi/next-key)) steps) [])]
    (loop [i 0 t trace kept (transient []) accepted 0 moves 0 reported? true diverged 0]
      (if (< i steps)
        (let [r (step-result (f t (step-keys i)))
              i' (inc i)
              ;; one iteration's graph at a time on a lazy backend
              t' (gfi/eval-trace! (:trace r))
              keep? (and (> i' burn) (zero? (mod (- i' burn) thin)))
              ms (accepted-moves r)]
          (recur i' t' (if keep? (conj! kept (read t')) kept)
                 (+ accepted (count (filter true? ms)))
                 (+ moves (count ms))
                 (and reported? (some? ms))
                 (+ diverged (divergences r))))
        (assoc (if finish {:draws (finish (persistent! kept))} {:traces (persistent! kept)})
               :acceptance-rate (when (and reported? (pos? moves)) (/ (double accepted) moves))
               :divergences diverged)))))
