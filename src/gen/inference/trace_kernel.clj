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
   the continuous choices of the model's trace and of the kernel, the
   outputs the continuous values of update and backward. Inputs that no
   output depends on, and outputs that depend on no input, are left out (as
   GenTraceKernelDSL); a continuous choice of the trace that the update
   keeps and no output rewrites, but that an output depends on, is an
   unchanged output (GenTraceKernelDSL would find the matrix not square)."
  (:refer-clojure :exclude [+ - *])
  (:require [gen.array :as array :refer [+ - *]]
            [gen.array.dtype :as dt]
            [gen.array.linalg :as la]
            [gen.choicemap :as cm]
            [gen.diff :as diff]
            [gen.gfi :as gfi]
            [gen.inference.kernel :as k]
            [gen.inference.mh :as mh]
            [gen.inference.transform :as tf]
            [gen.random :as rng]))

(set! *warn-on-reflection* true)

(defn- continuous? [v]
  (and (array/array? v) (not (boolean? v)) (= :float (dt/kind (array/dtype v)))))

(defn- continuous-addresses [choices]
  (filterv #(continuous? (cm/get-choice choices %)) (cm/addresses choices)))

(defn- size-of [v] (if (array/host? v) 1 (array/size v)))

(defn- columns
  "[space path] of each flattened element of the values at addrs."
  [space choices addrs]
  (vec (mapcat (fn [p] (repeat (size-of (cm/get-choice choices p)) [space p])) addrs)))

(defn log-abs-det
  "log |det J| of kernel's move from trace (see the namespace doc): kc are
   the kernel's choices on (trace, args…), [update backward] its return
   value, and retained? says which of the trace's addresses the update
   keeps."
  [kernel trace args kc [update backward] retained?]
  (let [mc (:choices trace)
        ins (into (mapv #(vector :model %) (continuous-addresses mc)) (map #(vector :kernel %)) (continuous-addresses kc))
        outs (into (mapv #(vector :update %) (continuous-addresses update)) (map #(vector :backward %))
                   (continuous-addresses backward))]
    (if (or (empty? ins) (empty? outs))
      0.0
      (let [source (fn [sp] (if (= sp :model) mc kc))
            in-values (mapv (fn [[sp p]] (cm/get-choice (source sp) p)) ins)
            shapes (mapv #(if (array/host? %) [] (array/shape %)) in-values)
            offsets (vec (reductions clojure.core/+ 0 (map size-of in-values)))
            substituted (fn [x]
                          (reduce (fn [[mc kc] [i [sp p]]]
                                    (let [v (array/reshape (array/slice x 0 (offsets i) (offsets (inc i))) (shapes i))]
                                      (if (= sp :model) [(cm/set-choice mc p v) kc] [mc (cm/set-choice kc p v)])))
                                  [mc kc] (map-indexed vector ins)))
            outputs (fn [x]
                      (let [[mc' kc'] (substituted x)
                            [u b] (:retval (gfi/assess kernel (into [(assoc trace :choices mc')] args) kc'))]
                        (array/concatenate (mapv (fn [[sp p]] (array/flatten (cm/get-choice (if (= sp :update) u b) p)))
                                                 outs))))
            x (array/concatenate (mapv array/flatten in-values))
            m (reduce clojure.core/+ 0 (map (fn [[sp p]] (size-of (cm/get-choice (if (= sp :update) update backward) p))) outs))
            eye (array/eye m)
            jacobian (mapv (fn [j] (vec (array/->clj ((array/grad #(array/sum (* (outputs %) (array/take eye j)))) x))))
                           (range m))
            cols (into (columns :model mc (continuous-addresses mc)) (columns :kernel kc (continuous-addresses kc)))
            rows (filterv #(some (complement zero?) %) jacobian)
            used (filterv (fn [c] (some #(not (zero? (% c))) rows)) (range (count cols)))
            rewritten (set (continuous-addresses update))
            kept (filterv (fn [c] (let [[sp p] (cols c)] (and (= sp :model) (not (rewritten p)) (retained? p)))) used)
            square (into (mapv (fn [row] (mapv row used)) rows)
                         (map (fn [c] (mapv #(if (= % c) 1.0 0.0) used))) kept)]
        (when-not (= (count square) (count used))
          (throw (ex-info (str "The trace kernel's continuous part maps " (count used) " dimensions to " (count square)
                               "; a bijection keeps them equal")
                          {:inputs (mapv cols used)})))
        (let [[sign logabs] (la/slogdet (array/array square))]
          (when (zero? (double (array/item sign)))
            (throw (ex-info "The trace kernel's Jacobian is singular here: its move is not a bijection" {})))
          logabs)))))

(defn- unconstrained-additions [old-choices update discard new-choices]
  (remove (fn [p] (or (cm/has-value? (cm/get-submap update p))
                      (and (cm/has-value? (cm/get-submap old-choices p))
                           (not (cm/has-value? (cm/get-submap discard p))))))
          (cm/addresses new-choices)))

(defn- kept-by [discard] (fn [p] (not (cm/has-value? (cm/get-submap discard p)))))

(defn mh
  "An MH move with trace kernel k (on (trace, args…)) as its own reverse:
   {:trace :accepted? :log-alpha}. The update may sample nothing: every new
   choice comes from the kernel's update, a loud error otherwise."
  ([trace kernel args] (mh trace kernel args (gfi/next-key)))
  ([trace kernel args key]
   (let [[k-propose k-update k-accept] (rng/split key 3)
         {kc :choices fwd :weight [update backward :as ret] :retval} (gfi/propose kernel (into [trace] args) k-propose)
         {t' :trace w :weight discard :discard} (gfi/update trace update k-update)
         _ (when-let [extra (seq (unconstrained-additions (:choices trace) update discard (:choices t')))]
             (throw (ex-info (str "trace-kernel/mh: the kernel's update leaves new choices unwritten: " (vec extra))
                             {:addresses (vec extra)})))
         {bwd :weight} (gfi/assess kernel (into [t'] args) backward)]
     (mh/accept trace t' (+ (- w fwd) bwd (log-abs-det kernel trace args kc ret (kept-by discard))) k-accept))))

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
   are k's and l's update restores the trace (with :prev-observations)."
  [{:keys [p-new-args p-argdiffs new-observations k k-args l l-args check? prev-observations]}]
  (fn [trace key]
    (let [[k-propose k-update] (rng/split key)
          {kc :choices fwd :weight [update backward :as ret] :retval} (gfi/propose k (into [trace] k-args) k-propose)
          args (if p-new-args (vec p-new-args) (:args trace))
          argdiffs (if p-argdiffs (vec p-argdiffs) (vec (repeat (count args) diff/unknown-change)))
          {t' :trace w :weight discard :discard}
          (gfi/update trace args argdiffs (cm/merge update (or new-observations cm/EMPTY)) k-update)
          {bwd :weight [l-update l-backward] :retval} (gfi/assess l (into [t'] l-args) backward)]
      (when check?
        (let [{t'' :trace} (gfi/update t' (:args trace) (vec (repeat (count (:args trace)) diff/unknown-change))
                                       (cm/merge l-update (or prev-observations cm/EMPTY)) k-update)]
          (when-not (tf/same-choices? kc l-backward)
            (throw (ex-info "smcp3: the backward kernel does not give back the forward kernel's choices"
                            {:forward (cm/to-map kc) :given-back (cm/to-map l-backward)})))
          (when-not (tf/same-choices? (:choices trace) (:choices t''))
            (throw (ex-info "smcp3: the backward kernel's update does not restore the trace"
                            {:trace (cm/to-map (:choices trace)) :restored (cm/to-map (:choices t''))})))))
      {:trace t' :weight (+ w (- bwd fwd) (log-abs-det k trace k-args kc ret (kept-by discard)))})))
