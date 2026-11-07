(ns gen.gated-delta-bench
  "What gen.array/gated-delta's gradient costs (genclj-8h0l): training
   differentiates the rule's composite, since MLX's Metal kernel (mlx-lm's)
   has no VJP and is declined under gradients:

     clojure -J-Dgen.backend=mlx [-J-Dgen.mlx.grad=native] -M:bench -m gen.gated-delta-bench [<Qwen3.5 GGUF>]

   One GPU JVM at a time. Sections (gen.bench.sections, default
   op,model), at the lengths gen.bench.lengths (256,512,1024):
   - op: the rule alone at Qwen3.5-0.8B's GatedDeltaNet sizes (1 row, 16
     heads, Dk = Dv = 128, float32): the forward by :auto (the Metal
     kernel where there is one), the forward by the composite (:chunked,
     chunks of 64), and the value and gradient (with respect to q, k, v, g
     and beta) of Σ y·r through the composite;
   - model (with a GGUF): the model's nll of one row of the length, its
     output projection in vocabulary chunks of 32768: the forward
     (log-probs, no gradient), and the value and gradient with respect to
     a LoRA adapter (rank 8, lora/init's default targets), checkpointed;
     then the same with the rule replaced by a stand-in of no cost (y =
     β·v, the state as it was), whose difference is the rule's share of a
     training step (its forward, recompute and backward in every
     GatedDeltaNet layer).
   Times are the median of gen.bench.runs (5) evaluations, after two
   untimed ones; ms."
  (:require [clojure.string :as str]
            [gen.array :as array]
            [gen.llm.lora :as lora]
            [gen.llm.models.qwen35 :as qwen35]
            [gen.random :as rng]))

(set! *warn-on-reflection* true)

(defn- prop [k default] (System/getProperty (str "gen.bench." k) (str default)))

(def ^:private sections (set (map keyword (str/split (prop "sections" "op,model") #","))))
(def ^:private lengths (mapv parse-long (str/split (prop "lengths" "256,512,1024") #",")))
(def ^:private runs (parse-long (prop "runs" 5)))

(defn- ms
  "The median time of (array/eval! (f)) over runs, after two untimed
   evaluations, in ms."
  ^double [f]
  (dotimes [_ 2] (array/eval! (f)))
  (let [ts (sort (for [_ (range runs)]
                   (let [t0 (System/nanoTime)]
                     (array/eval! (f))
                     (/ (- (System/nanoTime) t0) 1e6))))]
    (nth ts (quot (count ts) 2))))

(defn- operands
  "The rule's operands for t tokens: q and k l2-normed [1 t 16 128], v,
   log decays g <= 0 and β in (0, 1) [1 t 16], a zero state."
  [t]
  (let [ks (rng/split (rng/key 1) 6)
        unit (fn [x] (array// x (array/sqrt (array/+ 1e-6 (array/sum (array/* x x) {:axis -1 :keep-dims? true})))))
        [q k v g beta r] (map (fn [key shape] (rng/normal key shape)) ks
                              [[1 t 16 128] [1 t 16 128] [1 t 16 128] [1 t 16] [1 t 16] [1 t 16 128]])]
    {:xs [(unit q) (unit k) v (array/negate (array/softplus g)) (array/sigmoid beta) (array/zeros [1 16 128 128])]
     :r r}))

(defn- op-row [t]
  (let [{:keys [xs r]} (operands t)
        [q k v g beta state] xs
        _ (array/eval! (conj xs r))
        loss (fn [q k v g beta] (array/sum (array/* r (first (array/gated-delta q k v g beta state)))))
        vg (array/value-and-grad loss {:argnums [0 1 2 3 4]})]
    {:auto (ms #(array/gated-delta q k v g beta state))
     :chunked (ms #(array/gated-delta q k v g beta state {:method :chunked}))
     :grad (ms #(vg q k v g beta))}))

(defn- stand-in
  "gated-delta replaced by y = β·v, the state as it was: no cost."
  [_q _k v _g beta state & _]
  [(array/* v (array/expand-dims beta -1)) state])

(defn- model-row [m ad t]
  (let [tokens [(mapv #(mod (+ 1000 (* 7919 (long %))) (long (get-in m [:config :vocab]))) (range t))]
        opts {:chunk 32768}
        loss (fn [params] (qwen35/nll (lora/adapt m (assoc ad :params params)) tokens (assoc opts :checkpoint? true)))
        vg (array/value-and-grad loss)
        step #(vg (:params ad))]
    {:forward (ms #(qwen35/log-probs m tokens opts))
     :step (ms step)
     :step-without-rule (with-redefs [array/gated-delta stand-in] (ms step))}))

(defn -main [& [path]]
  (println "gated-delta's gradient: gen.backend" (System/getProperty "gen.backend" "jvm")
           "gen.mlx.device" (System/getProperty "gen.mlx.device" "default")
           "gen.mlx.grad" (System/getProperty "gen.mlx.grad" "tape"))
  (when (:op sections)
    (println "\nthe rule alone, 1 x T x 16 heads x 128, float32: ms")
    (println (format "%6s %10s %10s %10s %8s" "T" ":auto" ":chunked" "grad" "grad/auto"))
    (doseq [t lengths
            :let [{:keys [auto chunked grad]} (op-row t)]]
      (println (format "%6d %10.2f %10.2f %10.2f %8.1f" t auto chunked grad (/ grad auto)))))
  (when (and path (:model sections))
    (let [m (qwen35/load path)
          ad (lora/init m nil (rng/key 1))
          n (count (filter #{:delta} (get-in m [:config :kinds])))]
      (array/eval! (:weights m))
      (println (format "\nthe model, %s (%d GatedDeltaNet layers), 1 row, LoRA rank 8 on %s, checkpointed: ms"
                       (last (str/split path #"/")) n (pr-str (vec (distinct (mapcat keys (vals (:params ad))))))))
      (println (format "%6s %10s %10s %14s %12s" "T" "forward" "step" "step w/o rule" "rule's share"))
      (doseq [t lengths
              :let [{:keys [forward step step-without-rule]} (model-row m ad t)]]
        (println (format "%6d %10.1f %10.1f %14.1f %11.0f%%" t forward step step-without-rule
                         (* 100.0 (/ (- step step-without-rule) step))))))))
