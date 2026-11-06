(ns gen.combinators.chain
  "Unfold and Scan (gen.combinators): a chain of kernel steps, step t's
   choices at address t (gen.combinators.elements), each step's carry the
   one the step before it passed on (Unfold's state, Scan's carry). The two
   differ only in what a step takes and gives, their chain (unfold-chain,
   scan-chain): one walk runs either's steps afresh (generate, and an
   update's or a regenerate's new steps), one revises the steps a trace
   keeps, and one assesses."
  {:api :internal}
  (:refer-clojure :exclude [+])
  (:require [gen.array :as array :refer [+]]
            [gen.choicemap :as cm]
            [gen.combinators :as-alias comb]
            [gen.combinators.calls :as calls :refer [element-key stopped?]]
            [gen.combinators.elements :as el]
            [gen.diff :as diff]
            [gen.gfi :as gfi]))

(set! *warn-on-reflection* true)

;; A chain is a map:
;;   :kernel           the kernel
;;   :n                the number of steps
;;   :init             the first step's carry
;;   :args             (fn [t carry]): step t's kernel arguments
;;   :out              (fn [t retval stopped?]): [carry y] of step t from
;;                     the kernel's return value: the carry it passes on
;;                     (never read after a step that stopped the run) and
;;                     its output
;;   :carry            (fn [acc t]): the carry a kept step t passes on
;; and for a revise:
;;   :dinit            the first carry's argdiff
;;   :argdiffs         (fn [t cdiff]): step t's argdiffs, cdiff its carry's
;;   :diffs            (fn [rd prev out]): [carry-diff y-diff] of a revisited
;;                     step from the kernel's retdiff rd, its previous return
;;                     value prev and its [carry y] out
;;   :params-changed?  may the parameters have changed? A change in their
;;                     number is a change (DESIGN.md §9, G3)
;;   :inputs           the steps whose input other than the carry may have
;;                     changed: a sorted set, or :all

(defn- params-changed? [args dparams tr]
  (boolean (and tr (or (not= (count args) (count (:args tr))) (some diff/changed? dparams)))))

(defn- check-steps
  "The number of steps as a host integer (a one-element array is read; a
   number that differs across a batch's particles raises the batched-branch
   error, so batched inference falls back)."
  [n]
  (let [n (if (array/backend-array? n) (array/item n) n)]
    (when-not (nat-int? n)
      (throw (ex-info "Unfold's first argument, the number of steps, must be a non-negative integer" {:n n})))
    n))

(defn- unfold-chain
  "Unfold's chain on args (n init & params), and argdiffs and the previous
   trace tr for a revise: step t calls (kernel t state & params), whose
   return value is the next state and the step's output."
  [gf [n init & params :as args] argdiffs tr]
  (let [n (check-steps n)
        [_ dinit & dparams] argdiffs]
    {:kernel (:kernel gf) :n n :init init
     :args (fn [t state] (into [t state] params))
     :out (fn [_ state _] [state state])
     :carry (fn [acc t] (nth (:retval acc) t))
     :dinit dinit
     :argdiffs (fn [_ sdiff] (into [diff/no-change sdiff] dparams))
     :diffs (fn [rd _ _] [rd rd])
     :params-changed? (params-changed? args dparams tr)
     :inputs (sorted-set)}))

(defn- scan-inputs
  "Scan's inputs, its second argument, as a vector of one per step."
  [args]
  (when (< (count args) 2)
    (throw (ex-info (str "Scan takes (init xs & params), the initial carry and the inputs; given " (count args)
                         " arguments")
                    {:args (count args)})))
  (let [xs (second args)]
    (when-not (sequential? xs)
      (throw (ex-info (str "Scan's second argument is its inputs, a vector of one per step, not " (pr-str xs))
                      {:xs xs})))
    (vec xs)))

(defn- step-output
  "Step t's kernel return value, checked to be [carry y]."
  [t out]
  (when-not (and (vector? out) (== 2 (count out)))
    (throw (ex-info (str "Scan's kernel returns [carry y], not " (pr-str out) " (step " t ")") {:step t :retval out})))
  out)

(defn- changed-steps
  "The steps whose input may have changed, by the inputs' argdiff d: a
   sorted set, or :all."
  [d]
  (cond (diff/no-change? d) (sorted-set)
        (diff/vector-diff? d) (into (sorted-set) (keys (:updated d)))
        :else :all))

(defn- part-diff
  "The diff of part i (0 the carry, 1 the output) of a step's [carry y],
   prev before and now after, from the kernel's retdiff rd: the part's own
   (from a vector-diff), or no-change when it is the same value
   (gfi/same-value?), so that a changed output alone leaves the next step
   be."
  [rd prev now i]
  (let [d (el/element-diff rd i)]
    (if (and (diff/changed? d) (gfi/same-value? (nth prev i) (nth now i))) diff/no-change d)))

(defn- scan-chain
  "Scan's chain on args (init xs & params), and argdiffs and the previous
   trace tr for a revise: step t calls (kernel carry (nth xs t) & params),
   which returns [carry' y]; a step that stopped the run has neither."
  [gf [init _ & params :as args] argdiffs tr]
  (let [xs (scan-inputs args)
        [dinit dxs & dparams] argdiffs]
    {:kernel (:kernel gf) :n (count xs) :init init
     :args (fn [t carry] (into [carry (xs t)] params))
     :out (fn [t retval stopped] (if stopped [nil nil] (step-output t retval)))
     :carry (fn [acc t] (first (:retval ((:subs acc) t))))
     :dinit dinit
     :argdiffs (fn [t cdiff] (into [cdiff (el/element-diff dxs t)] dparams))
     :diffs (fn [rd prev out] [(part-diff rd prev out 0) (part-diff rd prev out 1)])
     :params-changed? (params-changed? args dparams tr)
     :inputs (when tr (changed-steps dxs))}))

;; ---------------------------------------------------------------------------
;; The walks
;; ---------------------------------------------------------------------------

(defn- fresh-steps
  "{:carry :acc :stop}: acc with the chain's steps from t on run afresh
   after carry, generated under the constraints cs, or simulated when cs
   is nil, until one stops the run (stopped?): stop is its index (the
   carry nil), else nil."
  [{:keys [kernel n args out]} t carry acc cs key]
  (loop [t t carry carry acc acc]
    (if (< t n)
      (let [r (calls/fresh-at kernel (args t carry) (when cs (cm/get-submap cs t cm/EMPTY)) (element-key key t) [t])
            stop? (stopped? (:trace r))
            [c y] (out t (:retval (:trace r)) stop?)
            acc (el/put-run acc t nil r y diff/no-change)]
        (if stop? {:acc acc :stop t} (recur (inc t) c acc)))
      {:carry carry :acc acc})))

(defn- among? [steps t] (or (= :all steps) (contains? steps t)))

(defn- next-among
  "The first of steps (a sorted set, or :all) from t on, else end."
  [steps t end]
  (if (= :all steps) t (or (first (subseq steps >= t)) end)))

(defn- last-stopped
  "The index of the last of subs, the steps of an old trace, when its run
   stopped there, else nil: a later update that keeps it as it is stops
   there too."
  [subs]
  (let [n (count subs)]
    (when (and (pos? n) (stopped? (subs (dec n)))) (dec n))))

(defn- revise-steps
  "{:carry :cdiff :acc :stop}: the steps below kept of the trace whose
   steps are old-subs revised in order into acc (update, op :update with
   the constraints cs, or regenerate, op :regenerate with the selection
   cs), the carry and its diff after them. A step is revisited when its
   carry, its input, the parameters, or what reaches it (targets) may have
   changed, and the walk skips straight to the next such step otherwise.
   stop: the step at which the run stopped (stopped?), else nil."
  [{:keys [kernel init dinit args argdiffs out diffs carry params-changed? inputs]}
   old-subs acc kept targets op cs key]
  (let [touched (if (or (= :all targets) (= :all inputs)) :all (into targets inputs))
        old-stop (last-stopped old-subs)]
    (loop [t 0 c init cdiff dinit acc acc]
      (cond
        (>= t kept) {:carry c :cdiff cdiff :acc acc}

        (or params-changed? (diff/changed? cdiff) (among? touched t))
        (let [old (old-subs t)
              r (calls/revise-at op kernel old (args t c) (argdiffs t cdiff) (calls/reaching op cs t)
                                 (element-key key t) [t])
              stop? (stopped? (:trace r))
              [c' y :as o] (out t (:retval (:trace r)) stop?)]
          (if stop?
            {:cdiff diff/unknown-change :acc (el/put-run acc t old r y diff/unknown-change) :stop t}
            (let [[cd yd] (diffs (:retdiff r) (:retval old) o)]
              (recur (inc t) c' cd (el/put-run acc t old r y yd)))))

        :else
        (let [t' (long (min (next-among touched t kept) kept))]
          ;; a kept step that stopped the old run stops this one
          (if (and old-stop (<= t old-stop) (< old-stop t'))
            {:cdiff diff/unknown-change :acc acc :stop old-stop}
            (recur t' (carry acc (dec t')) diff/no-change acc)))))))

(defn- revise-chain
  "{:carry :cdiff :acc :stop}: update or regenerate (op, cs as
   revise-steps') of tr, a trace of the chain, whose outputs are given:
   its steps revised (revise-steps), then the run extended to the chain's
   steps or truncated. stop: the step at which the run stopped
   (stopped?), else nil."
  [{:keys [n] :as chain} tr outputs op cs key]
  (let [old-subs (::comb/subtraces tr)
        kept (min n (count old-subs))
        {:keys [carry cdiff acc stop]} (revise-steps chain old-subs (el/trace-acc tr outputs) kept (el/targets op cs n)
                                                     op cs key)]
    (if stop
      {:cdiff cdiff :acc (el/drop-elements acc (inc stop) op) :stop stop}
      (assoc (fresh-steps chain kept carry (el/drop-elements acc kept op) (when (= op :update) cs) key) :cdiff cdiff))))

(defn- assess-steps
  "{:weight :carry :ys}: the chain's steps assessed in order under
   choices, until one stops the run (stopped?): the weights' sum, the last
   carry (nil after a stop) and the steps' outputs."
  [{:keys [kernel n init args out]} choices key]
  ;; weight starts boxed: it accumulates arrays on array backends
  (loop [t 0 carry init weight (Double/valueOf 0.0) ys []]
    (if (< t n)
      (let [r (calls/assess-at kernel (args t carry) (cm/get-submap choices t cm/EMPTY) (element-key key t) [t])
            weight (+ weight (:weight r))
            stop? (stopped? (:weight r))
            [c y] (out t (:retval r) stop?)]
        (if stop?
          {:weight weight :ys (conj ys y)}
          (recur (inc t) c weight (conj ys y))))
      {:weight weight :carry carry :ys ys})))

;; ---------------------------------------------------------------------------
;; Unfold
;; ---------------------------------------------------------------------------

(defn unfold-generate
  "generate (constraints, or nil to simulate) of an Unfold on args."
  [gf args constraints key]
  (let [{:keys [n init] :as chain} (unfold-chain gf args nil nil)]
    (when constraints (el/indices :generate constraints n))
    (let [{:keys [acc]} (fresh-steps chain 0 init el/empty-acc constraints key)]
      {:trace (el/elements-trace gf (vec args) acc) :weight (:weight acc)})))

(defn unfold-revise
  "update or regenerate of an Unfold trace: revisits a step when its state
   input, the parameters, its constraints or its selection may have
   changed; then extends or truncates."
  [gf tr args argdiffs op cs key]
  (let [{:keys [n] :as chain} (unfold-chain gf args argdiffs tr)
        prev-n (count (::comb/subtraces tr))
        {:keys [acc stop]} (revise-chain chain tr (:retval tr) op cs key)]
    (el/incremental-result gf (vec args) acc op n prev-n stop)))

(defn unfold-assess
  "assess of an Unfold on args: its steps', until one stops the run."
  [gf args choices key]
  (let [{:keys [n] :as chain} (unfold-chain gf args nil nil)]
    (el/indices :assess choices n)
    (let [{:keys [weight ys]} (assess-steps chain choices key)]
      {:weight weight :retval ys})))

;; ---------------------------------------------------------------------------
;; Scan
;; ---------------------------------------------------------------------------

(defn- scan-trace [gf args carry acc]
  (el/elements-trace gf args (assoc acc :retval [carry (:retval acc)])))

(defn- scan-retdiff
  "The diff of [carry ys] from its parts'."
  [carry-diff ys-diff]
  (if (and (diff/no-change? carry-diff) (diff/no-change? ys-diff))
    diff/no-change
    (diff/vector-diff 2 2 (cond-> {}
                            (diff/changed? carry-diff) (assoc 0 carry-diff)
                            (diff/changed? ys-diff) (assoc 1 ys-diff)))))

(defn scan-generate
  "generate (constraints, or nil to simulate) of a Scan on args."
  [gf args constraints key]
  (let [{:keys [n init] :as chain} (scan-chain gf args nil nil)]
    (when constraints (el/indices :generate constraints n))
    (let [{:keys [carry acc]} (fresh-steps chain 0 init el/empty-acc constraints key)]
      {:trace (scan-trace gf (vec args) carry acc) :weight (:weight acc)})))

(defn scan-revise
  "update or regenerate of a Scan trace, as Unfold's: revisits a step when
   its carry, its input, the parameters, its constraints or its selection
   may have changed; then extends or truncates."
  [gf tr args argdiffs op cs key]
  (let [{:keys [n] :as chain} (scan-chain gf args argdiffs tr)
        prev-n (count (::comb/subtraces tr))
        {:keys [carry cdiff acc stop]} (revise-chain chain tr (second (:retval tr)) op cs key)
        ;; the last carry is another step's when the length changed
        carry-diff (cond (== n prev-n) cdiff
                         (gfi/same-value? (first (:retval tr)) carry) diff/no-change
                         :else diff/unknown-change)]
    (cond-> {:trace (scan-trace gf (vec args) carry acc) :weight (:weight acc)
             :retdiff (if stop diff/unknown-change (scan-retdiff carry-diff (el/retdiff acc n prev-n)))}
      (= op :update) (assoc :discard (:discard acc)))))

(defn scan-assess
  "assess of a Scan on args: its steps', until one stops the run."
  [gf args choices key]
  (let [{:keys [n] :as chain} (scan-chain gf args nil nil)]
    (el/indices :assess choices n)
    (let [{:keys [weight carry ys]} (assess-steps chain choices key)]
      {:weight weight :retval [carry ys]})))
