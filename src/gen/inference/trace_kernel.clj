(ns gen.inference.trace-kernel
  "GenTraceKernelDSL's trace kernels: stochastic maps between traces that mix
   sampling with deterministic code, as proposals for MH and as the steps of
   SMCP3.

   A trace kernel is a generative function (gen) whose first argument is a
   model trace and whose return value is [update backward]: the choice map
   of new values for the model's trace, and the choices of the reverse kernel
   that undo the move. Its own choices are the auxiliary randomness; it
   reads the model's choices by calling the trace, and may call other
   generative functions (whose return values it gets, where
   GenTraceKernelDSL hands over their traces):

     (def split-merge
       (gen [tr]
         (if (tr :two?)
           [(cm/choicemap :two? false :mu (/ (+ (tr :mu1) (tr :mu2)) 2.0))
            (cm/choicemap :u (/ (- (tr :mu2) (tr :mu1)) 2.0))]
           (let [u (trace :u (d/normal 0 1))]
             [(cm/choicemap :two? true :mu1 (- (tr :mu) u) :mu2 (+ (tr :mu) u))
              (cm/choicemap)]))))

   (mh trace kernel args key) is an MH move with the kernel as its own
   reverse (GenTraceKernelDSL's metropolis_hastings), and (mh-kernel kernel
   args) the same as a gen.inference.kernel kernel. (smcp3 {…}) is an SMCP3
   step with a forward kernel k and a backward kernel l
   (GenTraceKernelDSL's run_smcp3_step), a translator for
   gen.inference.particle-filter/move-reweight.

   Weights include log |det J|, by automatic differentiation: the inputs are
   the kernel's continuous choices and the trace's continuous choices that
   the update rewrites or discards (less observations SMCP3's
   :new-observations replace) or that an output depends on, the outputs
   every continuous value of update and backward, and a continuous choice
   of the trace that the update keeps and no output rewrites, but that an
   output depends on, is an unchanged output. The kernel runs on the trace
   the substituted choices make (generate, every choice constrained), so
   what it reads from a call's value, the return value or the score is
   differentiated too. The model may hold encapsulated randomness (a
   pseudo-marginal move), unless the kernel's outputs depend on it: then
   they differ at the trace's own choices, a loud error. GenTraceKernelDSL also drops outputs that depend on
   no input and kernel choices no output depends on, which hides moves that
   are not bijections (a constant output, randomness thrown away, a value
   overwritten); here they make the matrix non-square or singular, a loud
   error."
  (:refer-clojure :exclude [+ -])
  (:require [gen.array :as array :refer [+ -]]
            [gen.array.linalg :as la]
            [gen.choicemap :as cm]
            [gen.diff :as diff]
            [gen.gfi :as gfi]
            [gen.gradients :as grad]
            [gen.inference.kernel :as k]
            [gen.inference.mh :as mh]
            [gen.inference.transform :as tf]
            [gen.random :as rng]))

(set! *warn-on-reflection* true)

(defn- continuous-addresses [choices]
  (filterv #(grad/continuous? (cm/get-choice choices %)) (cm/addresses choices)))

(defn- columns
  "[space path] of each flattened element of the values at addrs."
  [space choices addrs]
  (vec (mapcat (fn [p] (repeat (array/size (cm/get-choice choices p)) [space p])) addrs)))

(defn log-abs-det
  "log |det J| of kernel's move from trace (see the namespace doc): kc are
   the kernel's choices on (trace, args…), [update backward] its return
   value, and retained? says which of the trace's addresses the move leaves
   alone: the update keeps them, or replaces them as data (SMCP3's new
   observations)."
  [kernel trace args kc [update backward] retained?]
  (let [mc (:choices trace)
        ins (into (mapv #(vector :model %) (continuous-addresses mc)) (map #(vector :kernel %)) (continuous-addresses kc))
        outs (into (mapv #(vector :update %) (continuous-addresses update)) (map #(vector :backward %))
                   (continuous-addresses backward))
        in-values (mapv (fn [[sp p]] (cm/get-choice (if (= sp :model) mc kc) p)) ins)
        out-values (fn [[u b]] (mapv (fn [[sp p]] (cm/get-choice (if (= sp :update) u b) p)) outs))
        m (reduce + 0 (map array/size (out-values [update backward])))
        rows (if (and (seq ins) (seq outs))
               (let [[x unflatten] (grad/flattened in-values)
                     substituted (fn [x] (reduce (fn [[mc kc] [[sp p] v]]
                                                   (if (= sp :model) [(cm/set-choice mc p v) kc] [mc (cm/set-choice kc p v)]))
                                                 [mc kc] (map vector ins (unflatten x))))
                     ;; the kernel on the trace the substituted choices make,
                     ;; so its reads of a call's value, the return value and
                     ;; the score are tracked too. Only the kernel needs exact
                     ;; densities: the model may hold encapsulated randomness
                     ;; (a pseudo-marginal move), drawn here at key 0, which
                     ;; the check below makes loud wherever the kernel's
                     ;; outputs depend on it
                     kernel-outputs (fn [x]
                                      (let [[mc' kc'] (substituted x)
                                            trace' (:trace (gfi/generate (:gen-fn trace) (:args trace) mc' (rng/key 0)))]
                                        (:retval (gfi/with-exact-density "A trace kernel's Jacobian"
                                                   (gfi/assess kernel (into [trace'] args) kc' (rng/key 0))))))
                     continuous-outs (fn [ub] (reduce (fn [c [[sp p] v]] (cm/set-choice c (into [sp] p) v))
                                                      cm/EMPTY (map vector outs (out-values ub))))]
                 (when-not (tf/same-choices? (continuous-outs (kernel-outputs x)) (continuous-outs [update backward]))
                   (throw (ex-info (str "The trace kernel gives other outputs on the trace its choices make than on the"
                                        " trace itself: they depend on the model's encapsulated randomness, which its"
                                        " Jacobian cannot follow")
                                   {:outputs (mapv vec (map rest outs))})))
                 (array/->clj (grad/jacobian #(array/concatenate (mapv array/flatten (out-values (kernel-outputs %)))) x m)))
               (vec (repeat m [])))
        cols (into (columns :model mc (continuous-addresses mc)) (columns :kernel kc (continuous-addresses kc)))
        ;; every output and every kernel choice counts: a constant output,
        ;; or randomness no output keeps, is not a bijection (the thesis
        ;; leaves out only what is copied); so does a trace choice the
        ;; update rewrites or discards, whose old value would be lost
        ;; otherwise. Trace choices the move leaves alone and no output
        ;; depends on are left out
        rewritten (set (continuous-addresses update))
        consumed? (fn [p] (or (rewritten p) (not (retained? p))))
        used (filterv (fn [c] (let [[sp p] (cols c)] (or (= :kernel sp) (consumed? p) (some #(not (zero? (% c))) rows))))
                      (range (count cols)))
        kept (filterv (fn [c] (let [[sp p] (cols c)] (and (= sp :model) (not (consumed? p))))) used)
        square (into (mapv (fn [row] (mapv row used)) rows)
                     (map (fn [c] (mapv #(if (= % c) 1.0 0.0) used))) kept)]
    (when-not (= (count square) (count used))
      (throw (ex-info (str "The trace kernel's continuous part maps " (count used) " dimensions to " (count square)
                           "; a bijection keeps them equal")
                      {:inputs (mapv cols used)})))
    (if (empty? used)
      0.0
      (let [[sign logabs] (la/slogdet (array/array square))]
        (when (zero? (double (array/item sign)))
          (throw (ex-info "The trace kernel's Jacobian is singular here: its move is not a bijection" {})))
        logabs))))

(defn- kept-by [discard] (fn [p] (not (cm/has-value? discard p))))

(defn mh
  "An MH move with trace kernel k (on (trace, args…)) as its own reverse:
   {:trace :accepted? :log-alpha}. The update may sample nothing: every new
   choice comes from the kernel's update, a loud error otherwise."
  ([trace kernel args] (mh trace kernel args (gfi/next-key)))
  ([trace kernel args key]
   (let [[k-propose k-update k-accept] (rng/split key 3)
         {kc :choices fwd :weight [update backward :as ret] :retval} (gfi/propose kernel (into [trace] args) k-propose)
         {t' :trace w :weight discard :discard} (gfi/update trace update k-update)]
     ;; a move outside the support (gfi/impossible?: the update stopped at a
     ;; choice there) is rejected, unweighed: t' holds the choices up to it
     (if (gfi/impossible? w)
       (mh/accept trace t' ##-Inf k-accept)
       (let [_ (when-let [extra (seq (tf/unconstrained-additions (:choices trace) update discard (:choices t')))]
                 (throw (ex-info (str "trace-kernel/mh: the kernel's update leaves new choices unwritten: " (vec extra))
                                 {:addresses (vec extra)})))
             {bwd :weight} (gfi/assess kernel (into [t'] args) backward (gfi/backward-key k-update))]
         (mh/accept trace t' (+ (- w fwd) bwd (log-abs-det kernel trace args kc ret (kept-by discard))) k-accept))))))

(defn mh-kernel
  "The MH move of trace kernel k as a gen.inference.kernel kernel."
  [kernel args]
  (k/symmetric (fn [trace key] (mh trace kernel args key))))

(defn smcp3
  "An SMCP3 step (GenTraceKernelDSL's run_smcp3_step) as a translator,
   (fn [trace key] → {:trace :weight}). The forward kernel k on (trace,
   k-args…) proposes [update backward]; the trace is updated to p-new-args
   (unknown changes unless :p-argdiffs) with update and :new-observations;
   the backward kernel l on (trace', l-args…) assesses backward and returns
   [its update, its backward choices]. The weight is the update's
   + log l - log k + log |det J|. :check? throws unless l's backward choices
   are k's and l's update restores the trace (with :prev-observations). A
   step outside the support (the update stopped at a choice there,
   gfi/impossible?) weighs -Inf, l unrun on trace', which holds the choices
   up to that one."
  [{:keys [p-new-args p-argdiffs new-observations k k-args l l-args check? prev-observations]}]
  (fn [trace key]
    (let [[k-propose k-update k-check] (rng/split key 3)
          {kc :choices fwd :weight [update backward :as ret] :retval} (gfi/propose k (into [trace] k-args) k-propose)
          [args argdiffs] (tf/new-arguments trace p-new-args p-argdiffs)
          {t' :trace w :weight discard :discard} (gfi/update trace args argdiffs (cm/merge update new-observations) k-update)]
      (if (gfi/impossible? w)
        {:trace t' :weight ##-Inf}
        (let [{bwd :weight [l-update l-backward] :retval} (gfi/assess l (into [t'] l-args) backward (gfi/backward-key k-update))]
          (when check?
            (let [{t'' :trace} (gfi/update t' (:args trace) (vec (repeat (count (:args trace)) diff/unknown-change))
                                           (cm/merge l-update prev-observations) k-check)]
              (when-not (tf/same-choices? kc l-backward)
                (throw (ex-info "smcp3: the backward kernel does not give back the forward kernel's choices"
                                {:forward (cm/to-map kc) :given-back (cm/to-map l-backward)})))
              (when-not (tf/same-choices? (:choices trace) (:choices t''))
                (throw (ex-info "smcp3: the backward kernel's update does not restore the trace"
                                {:trace (cm/to-map (:choices trace)) :restored (cm/to-map (:choices t''))})))))
          ;; an observation the new ones replace is data, not part of the move
          {:trace t' :weight (+ w (- bwd fwd) (log-abs-det k trace k-args kc ret #(or ((kept-by discard) %)
                                                                                       (cm/has-value? new-observations %))))})))))
