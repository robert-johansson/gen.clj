(ns gen.inference.kernel
  "MCMC kernels and their combinators.

   A kernel is a function (kernel trace key) → {:trace trace' ...}: Gen.jl's
   (trace, metadata), as a map. The MH kernels add :accepted? and :log-alpha.
   A Kernel value can also carry its reversal, the kernel that undoes it in
   distribution (Gen.jl's reversal). MH-family kernels are reversible, so
   they are their own reversals. Combinators of kernels with reversals have
   reversals too: the reversal of a chain is the chain of the reversals, in
   reverse order.

     (chain k1 k2 k3)      k1, then k2, then k3
     (repeat n k)          k, n times
     (cycle n [k1 k2])     k1, k2, k1, … for n steps in all
     (mix [k1 k2] [w1 w2]) one of the kernels, chosen with probability ∝ w
     (checked k observations)  k, then check that it left the observations alone
     (kern [args] stmt …)  Gen.jl's @kern: statements, loops and branches over
                           the arguments, as one chain with its reversal
     (kern ^:trace [tr args] stmt …)
                           the same, reading the trace in its loops,
                           branches and lets, with the thesis's invariant
                           checked after each of their bodies

   A combinator's result reports the accept decisions of the MH moves it
   made, in order, as :accepted (a vector of booleans) when they all report
   one.

   (run kernel trace opts) runs a chain and collects samples."
  (:refer-clojure :exclude [repeat cycle reversible?])
  (:require [gen.array :as array]
            [gen.choicemap :as cm]
            [gen.gfi :as gfi]
            [gen.inference.mh :as mh]
            [gen.random :as rng]))

(set! *warn-on-reflection* true)

(defrecord Kernel [f reverse-f]
  clojure.lang.IFn
  (invoke [_ trace] (f trace (gfi/next-key)))
  (invoke [_ trace key] (f trace key))
  (applyTo [this args] (clojure.lang.AFn/applyToHelper this args)))

(defn kernel
  "A kernel from (fn [trace key] → {:trace …}), with its reversal when given."
  ([f] (->Kernel f nil))
  ([f reverse-f] (->Kernel f reverse-f)))

(defn symmetric
  "A kernel that is its own reversal."
  [f]
  (->Kernel f f))

(defn reversible? [k] (and (instance? Kernel k) (some? (:reverse-f k))))

(defn reversal
  "The declared reversal of a kernel."
  [k]
  (if (reversible? k)
    (->Kernel (:reverse-f k) (:f k))
    (throw (ex-info "The kernel has no declared reversal" {:kernel k}))))

(defn- step-fn
  "The (fn [trace key]) of a kernel or a plain function."
  [k]
  (if (instance? Kernel k) (:f k) k))

(defn- with-reversals
  "A kernel built by combine from the step fns of ks, reversible when every
   k is: its reversal is combine over the reversals in reverse-order."
  [combine ks reverse-order]
  (->Kernel (combine (mapv step-fn ks))
            (when (every? reversible? ks)
              (combine (mapv (comp step-fn reversal) (reverse-order ks))))))

(defn- accepted-moves
  "The accept decisions a kernel result reports: one for an MH step, a
   combinator's sequence, or nil when it reports none."
  [r]
  (cond (contains? r :accepted) (:accepted r)
        (contains? r :accepted?) [(:accepted? r)]))

(defn- sequence-fn [fs]
  (fn [trace key]
    (let [[t moves] (reduce (fn [[t moves] [f k]]
                              (let [r (f t k)]
                                [(:trace r) (when moves (some->> (accepted-moves r) (into moves)))]))
                            [trace []]
                            (map vector fs (rng/split key (count fs))))]
      (cond-> {:trace t} moves (assoc :accepted moves)))))

(defn chain
  "The kernels applied in sequence."
  [& kernels]
  (with-reversals sequence-fn (vec kernels) reverse))

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
  "A value a (kern ^:trace …) body's control flow read, on the host: an
   array read at the inference loop's boundary, in collections too."
  [v]
  (cond (array/array? v) (array/->clj v)
        (and (sequential? v) (not (record? v))) (mapv control-value v)
        (and (map? v) (not (record? v))) (into {} (map (fn [[k x]] [k (control-value x)])) v)
        :else v))

(defn- noting
  "[the control values (reads note) passed to note, on the host, and what it
   returned]."
  [reads]
  (let [seen (volatile! [])
        x (reads (fn [v] (vswap! seen conj v) v))]
    [(mapv control-value @seen) x]))

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
;; statement may change what an earlier construct read. (The check sees the
;; runs that happen, as Gen.jl's does, not every run the proposition
;; quantifies over.)
(defn ^:no-doc trace-kernel
  "The kernel of a (kern ^:trace …) body. forward and backward are (fn [t
   apply! construct]) running the statements (backward: reversed, loops
   backwards) against the volatile t. apply! applies a kernel to @t, the
   i-th applied with the key folded with i. (construct form reads run) runs
   a loop, branch or let: (reads note) reads its control values from @t (a
   loop's collections and conditions, a condition, a let's values), passing
   each to note, and returns what run takes; run runs the body. Then reads
   reads again, and values that differ are a loud error."
  [forward backward]
  (letfn [(run [body reversal? trace key]
            (let [t (volatile! trace)
                  n (volatile! 0)
                  moves (volatile! [])
                  apply! (fn [k]
                           (let [r ((step-fn (if reversal? (reversal k) k)) @t (rng/fold-in key @n))]
                             (vswap! n inc)
                             (vswap! moves #(when % (some->> (accepted-moves r) (into %))))
                             (vreset! t (:trace r))))
                  construct (fn [form reads run-body]
                              (let [[read x] (noting reads)]
                                (run-body x)
                                (let [[again _] (noting reads)]
                                  (when-not (= read again)
                                    (throw (ex-info (str "kern: the body of " (construct-name form) " changed what it"
                                                         " reads (a loop's collection, a condition or a let's value),"
                                                         " so the reversal would not undo the kernel: read "
                                                         (pr-str read) ", then " (pr-str again))
                                                    {:form form :read read :again again}))))))]
              (body t apply! construct)
              (cond-> {:trace @t} @moves (assoc :accepted @moves))))]
    (->Kernel (fn [trace key] (run forward false trace key)) (fn [trace key] (run backward true trace key)))))

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
      `(~apply! (checked-kernel ~(at form) '~form)))))

(defn- traced-statements [forms tr t apply! construct reverse?]
  `(do ~@(map #(traced-form % tr t apply! construct reverse?) (if reverse? (reverse forms) forms)) nil))

(defmacro kern
  "Gen.jl's @kern: (kern [args] statement …) is a function of args returning
   one kernel, the statements' kernels in order. A statement is a kernel, or
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
   check=true and gen.clj always. A later statement may change what an
   earlier one read, as the thesis's p. 102 kernel does. Conditions are
   Clojure's truthiness of host values (an array is read on the host).

     (kern ^:trace [tr]
       (doseq [i (range (inc (array/item (tr :degree))))] (mh (sel/select [:coeff i])))
       (mh (sel/select :degree))
       (doseq [i (range (inc (array/item (tr :degree))))] (mh (sel/select [:coeff i]))))"
  [params & body]
  (if (:trace (meta params))
    (let [[tr & args] params
          t (gensym "t") apply! (gensym "apply") construct (gensym "construct")]
      `(fn [~@args]
         (trace-kernel (fn [~t ~apply! ~construct] ~(traced-statements body tr t apply! construct false))
                       (fn [~t ~apply! ~construct] ~(traced-statements body tr t apply! construct true)))))
    `(fn ~params ~(statements body))))

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
                    (vec kernels)
                    identity)))

(defn- check-observations! [trace observations]
  (doseq [addr (cm/addresses observations)]
    (let [observed (cm/get-choice observations addr)
          actual (cm/get-choice (:choices trace) addr)]
      (when-not (and (some? actual) (array/equal? observed actual))
        (throw (ex-info (str "The kernel changed the observation at " addr)
                        {:path addr :observed observed :actual actual}))))))

(defn checked
  "The kernel, checking after each application that the trace still holds
   the observations (Gen.jl's check=true, observations=…)."
  [kernel observations]
  (let [check (fn [f] (fn [trace key]
                        (let [r (f trace key)]
                          (check-observations! (:trace r) observations)
                          r)))]
    (->Kernel (check (step-fn kernel))
              (when (reversible? kernel) (check (:reverse-f kernel))))))

;; ---------------------------------------------------------------------------
;; MH kernels
;; ---------------------------------------------------------------------------

(defn mh
  "The MH kernel of a selection, or of a proposal and its arguments."
  ([selection] (symmetric (fn [trace key] (mh/mh trace selection key))))
  ([proposal proposal-args] (symmetric (fn [trace key] (mh/mh trace proposal proposal-args key)))))

(defn involutive-mh
  "The involutive MCMC kernel of a proposal, its arguments and an
   involution."
  [proposal proposal-args involution]
  (symmetric (fn [trace key] (mh/involutive-mh trace proposal proposal-args involution key))))

;; ---------------------------------------------------------------------------
;; Running a chain
;; ---------------------------------------------------------------------------

(defn run
  "Apply the kernel repeatedly from trace and collect samples. opts:
   :samples (required), :burn (steps discarded first, 0), :thin (steps per
   sample, 1), :key. Returns {:traces [...] :acceptance-rate r}: the rate
   over every MH move of every step (a combinator's steps make several), or
   nil when the kernel does not report its accept decisions."
  [kernel trace {:keys [samples burn thin key] :or {burn 0 thin 1}}]
  (when-not (and (nat-int? samples) (nat-int? burn) (pos-int? thin))
    (throw (ex-info "run needs :samples ≥ 0, :burn ≥ 0 and :thin ≥ 1" {:samples samples :burn burn :thin thin})))
  (let [f (step-fn kernel)
        steps (+ burn (* samples thin))
        keys (if (pos? steps) (rng/split (or key (gfi/next-key)) steps) [])]
    (loop [i 0 t trace traces (transient []) accepted 0 moves 0 reported? true]
      (if (< i steps)
        (let [r (f t (keys i))
              i' (inc i)
              ;; one iteration's graph at a time on a lazy backend
              t' (gfi/eval-trace! (:trace r))
              keep? (and (> i' burn) (zero? (mod (- i' burn) thin)))
              ms (accepted-moves r)]
          (recur i' t' (if keep? (conj! traces t') traces)
                 (+ accepted (count (filter true? ms)))
                 (+ moves (count ms))
                 (and reported? (some? ms))))
        {:traces (persistent! traces)
         :acceptance-rate (when (and reported? (pos? moves)) (/ (double accepted) moves))}))))
