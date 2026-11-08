(ns gen.llm.models.against-llama
  "A forward over gen.array (gen.llm.models.qwen3, or qwen35 for Qwen3.5
   and Qwen3.6) against a reference on the same GGUF: the llama.cpp engine
   (gen.llm.llama), for the tests (gen.llm.models.qwen3-llama-test,
   qwen35-llama-test) and their bands' bench
   (bench/gen/qwen3_llama_noise.clj); or the same forward on another
   backend (a model loaded there: MLX's CPU, for
   gen.llm.models.cuda-reference-test and its bench,
   bench/gen/cuda_reference_noise.clj). The next-token log-probabilities
   of one prompt, by both, compared as the Branching contract compares two
   computations (gen.llm.branching-contract's max-diff: max |Δ log p| over
   the reference's 64 most likely tokens), with the argmax and the top 5
   (as sets; a swap across the boundary between tokens the reference holds
   within a tie of each other is noise). The forward's log-probabilities
   are its logits' log-softmax in float32 (a half-float model's logits
   widened first, as the engine normalizes them on the host).

   Each prompt is read three ways by the forward:
   - :whole, a prefill of the prompt;
   - :chunk, a prefill of its first half and a step of the rest;
   - :step i, the prompt's cache stepped i times by greedy tokens (the
     reference's argmax), against the reference advanced so.
   The reference's readings are taken first (reference), so that a planted
   error (below) changes only the forward under test, even where the
   reference is the same forward. On a mixture of experts they can hold
   the reference's experts (a model's, or llama.cpp's from an engine
   recording them through its eval callback), which compare-readings'
   :forced forces on the forward: the router's near-ties, where a float's
   difference picks other experts, then move nothing (forced-band).

   And the planted convention errors (mutations), as with-redefs-fn's
   redefinitions of gen.array's operations, each self-consistent:
   - :kv-tiled: query head h reads KV head h mod KV, where Qwen3 (and
     gen.array/sdpa) groups them, h / (H/KV);
   - :rope-traditional: RoPE's consecutive pairs for NEOX's halves;
   - :qk-norms-skipped: q and k not normalized per head;
   and Qwen3.5's own (qwen35-mutations, with those), of its layers:
   - :conv-oldest: the conv state keeping its oldest inputs, not its
     newest (only a cached read sees it);
   - :conv-reversed: the conv kernel applied newest input first;
   - :query-gate-swapped: attn_q's query and output gate swapped;
   on a model of fewer key heads than value heads (value-heads-mutations):
   - :value-heads-grouped: value head h reading key head h / (value-heads
     / key-heads), as Hugging Face's checkpoints order them, where the
     converter tiles them (h mod key-heads);
   and on a mixture of experts (moe-mutations):
   - :experts-gate-up-swapped: the routed experts' gate and up stacks
     swapped (silu of the up projection, times the gate's);
   - :router-unnormalized: the top k experts weighted by their router
     probabilities, not renormalized to sum 1;
   - :shared-expert-ungated: the shared expert's output not gated by
     sigmoid(ffn_gate_inp_shexp · x)."
  (:require [gen.array :as array]
            [gen.llm.branching :as br]
            [gen.llm.branching-contract :refer [max-diff top-tokens]]
            [gen.array.gguf :as gguf]
            [gen.llm.models.common :as common]
            [gen.llm.models.qwen3 :as qwen3]
            [gen.llm.models.qwen35 :as qwen35]))

(defn- qwen35? [m] (contains? #{"qwen35" "qwen35moe"} (get-in m [:config :arch])))

(defn- prefill [m & args] (apply (if (qwen35? m) qwen35/prefill qwen3/prefill) m args))
(defn- step [m & args] (apply (if (qwen35? m) qwen35/step qwen3/step) m args))

(defn architecture
  "The architecture of the GGUF file at path (its header's)."
  [path]
  (get-in (gguf/header path) [:metadata "general.architecture"]))

(defn load-model
  "The model of the GGUF file at path, by its architecture: qwen3/load or
   qwen35/load, with their opts (:dtype)."
  ([path] (load-model path nil))
  ([path opts]
   (if (#{"qwen35" "qwen35moe"} (architecture path)) (qwen35/load path opts) (qwen3/load path opts))))

(defn k-quants?
  "Does the GGUF file at path hold K-quant tensors?"
  [path]
  (boolean (some #{:q4_k :q5_k :q6_k} (map :type (:tensors (gguf/header path))))))

(defn band
  "The band of the forward against llama.cpp on the GGUF file at path
   (compare-prompt's |Δ log p|, and its tie), per platform as llama.cpp's
   own bands are (DESIGN §9): twice the largest |Δ log p| that
   bench/gen/qwen3_llama_noise.clj measured, rounded up, by the model's
   family and quantization. qwen3-llama-test's and qwen35-llama-test's
   docstrings give the measurements.
   - On Metal: Qwen3 0.075 (Q8_0), 0.47 (K-quants); Qwen3.5 0.035 (Q8_0)
     and 11.7 (K-quants: Qwen3.6-35B-A3B's Q4_K experts, the MacBook M2
     Max, 2026-11-08, the general and the digit prompts; a few readings'
     routing near-ties make it, and llama.cpp's own spread on Metal
     reaches 7.1).
   - Elsewhere, against llama.cpp's CUDA and x86 CPU builds, which
     quantize the activations to 8 bits for their dot products (fenrir,
     2026-11-08, the general and the digit prompts): Qwen3 2.4 (Q8_0;
     K-quants unmeasured), Qwen3.5 7.1 (Q8_0) and 12.3 (K-quants:
     Qwen3.6-35B-A3B's Q4_K experts, where llama.cpp's own spread on CUDA
     reaches 6.1)."
  [path]
  (let [qwen35? (#{"qwen35" "qwen35moe"} (architecture path))
        k? (k-quants? path)]
    (if (.startsWith (System/getProperty "os.name") "Mac")
      (cond (and qwen35? k?) 11.7 qwen35? 0.035 k? 0.47 :else 0.075)
      (cond (and qwen35? k?) 12.3 qwen35? 7.1 :else 2.4))))

(defn forced-band
  "The band of the forward against llama.cpp with llama.cpp's experts
   forced on it (compare-readings' :forced, the readings of an engine
   recording them: reference's :experts) on the mixture of experts at
   path, per platform as band's: twice the largest |Δ log p| that
   bench/gen/qwen3_llama_noise.clj measured forced, rounded up; band's
   where unmeasured (every other model). qwen35-llama-test's docstring
   gives the measurements.
   - On Metal: 0.16 on K-quants (Qwen3.6-35B-A3B's Q4_K experts, the
     MacBook M2 Max, 2026-11-08, the general and the digit prompts: 6.4e-3
     and 8.0e-2; the same readings unforced 0.97 and 5.97).
   - Elsewhere, against llama.cpp's CUDA build (which quantizes the
     activations to 8 bits): 3.9 on K-quants (fenrir, 2026-11-08, the
     general and the digit prompts: 0.45 and 1.91; unforced 2.19 and
     6.24), as Qwen3.5's dense 7.1 is llama.cpp's.
   - Every other model: band's (unmeasured)."
  [path]
  (if (and (= "qwen35moe" (architecture path)) (k-quants? path))
    (if (.startsWith (System/getProperty "os.name") "Mac") 0.16 3.9)
    (band path)))

(def ^:private sdpa array/sdpa)
(def ^:private rope array/rope)
(def ^:private rms-norm array/rms-norm)

(def mutations
  "Planted convention errors: name to with-redefs-fn's map."
  {:kv-tiled {#'array/sdpa (fn sdpa-tiled
                             ([q k v] (sdpa-tiled q k v nil))
                             ([q k v opts]
                              (let [rep (quot (long (second (array/shape q))) (long (second (array/shape k))))
                                    tile (fn [x] (array/concatenate (vec (repeat rep x)) 1))]
                                (sdpa q (tile k) (tile v) opts))))}
   :rope-traditional {#'array/rope (fn rope-traditional
                                     ([x dims] (rope-traditional x dims nil))
                                     ([x dims opts] (rope x dims (assoc opts :traditional? true))))}
   ;; every per-head norm (4-D, and Qwen3.5's GatedDeltaNet q and k
   ;; normed as one 5-D array since genclj-yesw), the hidden state's
   ;; (3-D) kept
   :qk-norms-skipped {#'array/rms-norm (fn [x w eps] (if (<= 4 (array/ndim x)) x (rms-norm x w eps)))}})

(def ^:private causal-conv @#'qwen35/causal-conv)

(def qwen35-mutations
  "Planted convention errors of Qwen3.5 (and qwen3's, which its
   attention layers share): name to with-redefs-fn's map."
  (merge mutations
         {:conv-oldest {#'qwen35/conv-state (fn [padded t]
                                              (array/slice padded 1 0 (- (long (second (array/shape padded))) (long t))))}
          :conv-reversed {#'qwen35/causal-conv (fn [padded kernel t] (causal-conv padded (array/slice kernel 1 nil nil -1) t))}
          :query-gate-swapped {#'qwen35/query-and-gate (fn [qg] [(array/slice qg 3 1 2) (array/slice qg 3 0 1)])}}))

(def value-heads-mutations
  "The planted error of a GatedDeltaNet of fewer key heads than value
   heads: name to with-redefs-fn's map."
  {:value-heads-grouped {#'qwen35/key-heads-tiled
                         (fn [y hv]
                           (let [[b t hk d] (array/shape y)]
                             (array/reshape (array/broadcast-to (array/expand-dims y 3) [b t hk (quot hv hk) d])
                                            [b t hv d])))}})

(def ^:private moe @#'qwen35/moe)
(def ^:private top-k @#'qwen35/top-k)

(def moe-mutations
  "The planted errors of Qwen3.5's mixture of experts: name to
   with-redefs-fn's map."
  {:experts-gate-up-swapped {#'qwen35/moe (fn [cfg w x] (moe cfg (assoc w :gate (:up w) :up (:gate w)) x))}
   ;; top-k through its var, which a forced read (with-experts) redefines
   :router-unnormalized {#'qwen35/routing (fn [probs k] (#'qwen35/top-k probs k))}
   ;; the routed experts' part (no :shared), plus the shared expert's SwiGLU
   :shared-expert-ungated {#'qwen35/moe (fn [cfg w x]
                                          (let [s (array/shape x)
                                                shared (common/swiglu (:shared w) (array/reshape x [-1 (peek s)]))]
                                            (array/+ (moe cfg (dissoc w :shared) x) (array/reshape shared s))))}})

(defn mutations-of
  "The planted errors that apply to model m: qwen3's mutations, or
   qwen35-mutations with value-heads-mutations where m has fewer key heads
   than value heads and moe-mutations where it has experts."
  [m]
  (if (qwen35? m)
    (let [{:keys [delta-key-heads delta-value-heads experts]} (:config m)]
      (cond-> qwen35-mutations
        (not= delta-key-heads delta-value-heads) (merge value-heads-mutations)
        experts (merge moe-mutations)))
    mutations))

(defn- logprobs
  "The log-probabilities of logits' last row (logits [1 T vocab]), in
   float32, a float[]."
  ^floats [logits]
  (let [t (long (second (array/shape logits)))
        last-row (array/slice logits 1 (dec t) t)]
    (array/->floats (array/log-softmax (if (= :float32 (array/dtype last-row)) last-row (array/astype last-row :float32))))))

(defn- same-top?
  "Are lp's k most likely tokens ref's, but for a swap across the k-th
   place between tokens ref holds within tie of its k-th (a tie inside
   the noise)?"
  [^floats ref ^floats lp k tie]
  (let [r (top-tokens ref k)
        l (top-tokens lp k)
        kth (aget ref (int (peek r)))
        tied? (fn [t] (<= (Math/abs (- (aget ref (int t)) kth)) (double tie)))]
    (and (every? tied? (remove (set r) l))
         (every? tied? (remove (set l) r)))))

(defn- verdict
  "{:max-diff :argmax? :top5?} of lp against the reference ref: the
   argmax and the top 5 (as sets) the same, but for swaps across the
   boundary between tokens ref holds within tie of each other."
  [^floats ref ^floats lp tie]
  {:max-diff (max-diff ref lp 64)
   :argmax? (same-top? ref lp 1 tie)
   :top5? (same-top? ref lp 5 tie)})

(defn- model-backend
  "The backend model m's weights live on."
  [m]
  (array/backend (first (filter array/backend-array? (array/tree-leaves (:weights m))))))

(defn- recorded
  "[(f)'s result, the experts each of its mixture-of-experts layers' top k
   picked, by position: [position [layer [k indices]]], host vectors
   (empty without such layers)]. f is a forward of a qwen35 model."
  [f]
  (let [calls (atom [])
        out (with-redefs-fn {#'qwen35/top-k (fn [p k] (let [r (top-k p k)] (swap! calls conj (first r)) r))} f)
        layers (mapv array/->clj @calls)]
    [out (if (seq layers) (apply mapv vector layers) [])]))

(defn- with-experts
  "(f), a forward of a qwen35 model at the positions positions (one a row
   of its tokens), each of its mixture-of-experts layers' top k the
   experts' (position → layer → indices, reference's :experts), weighted
   by the model's own router; (f) when experts is nil. A layer's indices
   nil at a position (llama.cpp's last layer computes only the rows whose
   logits a decode reads) leave the model's own there: that row feeds no
   logits read. A position whose experts are not one per layer of the
   forward, or not k a layer, is a loud error."
  [experts positions f]
  (if-not experts
    (f)
    (let [layer (atom -1)
          out (with-redefs-fn
                {#'qwen35/top-k (fn [p k]
                                  (let [l (swap! layer inc)
                                        rows (mapv #(get-in experts [% l]) positions)
                                        k (long k)]
                                    (when-let [bad (first (remove #(or (nil? %) (== k (count %))) rows))]
                                      (throw (ex-info (str "forced experts " bad " at layer " l " are not " k)
                                                      {:layer l :experts bad :k k})))
                                    (if (every? some? rows)
                                      (let [idx (array/array rows {:dtype :int32})]
                                        [idx (array/take-along-axis p idx -1)])
                                      (let [[own] (top-k p k)
                                            forced? (array/array (mapv some? rows) {:dtype :bool})
                                            idx (array/where (array/expand-dims forced? -1)
                                                             (array/array (mapv #(or % (vec (repeat k 0))) rows) {:dtype :int32})
                                                             own)]
                                        [idx (array/take-along-axis p idx -1)]))))}
                f)
          layers (inc (long @layer))]
      (when-let [p (first (remove #(== layers (count (get experts %))) positions))]
        (throw (ex-info (str "the forced experts at position " p " are of " (count (get experts p))
                             " layers, the forward's " layers)
                        {:position p :layers layers :forced (count (get experts p))})))
      out)))

(defn- llama-recording?
  "Does the Branching engine e record llama.cpp's experts (gen.llm.llama's
   :record of ffn_moe_topk)?"
  [e]
  (contains? (:record (br/info e)) "ffn_moe_topk"))

(defn- llama-experts
  "The experts llama.cpp's engine e recorded since the last call
   (gen.llm.llama's recorded and experts), the decodes of one chain of
   branches: [position [layer [k indices], nil where its layer computed
   no row]]."
  [e]
  (let [by-slot ((requiring-resolve 'gen.llm.llama/experts) ((requiring-resolve 'gen.llm.llama/recorded) e))
        by-pos (apply merge (vals by-slot))]
    (when (seq by-pos)
      (when-not (= (sort (keys by-pos)) (range (count by-pos)))
        (throw (ex-info "llama.cpp's recorded experts are not of positions 0 to n" {:positions (sort (keys by-pos))})))
      (mapv by-pos (range (count by-pos))))))

(defn reference
  "The readings of the reference ref on the prompt's tokens (a vector of
   at least 2) and steps greedy tokens after it, each the reference's
   argmax: {:tokens :root lp :steps [[t lp] …]}, each lp a float[] of the
   next token's log-probabilities. ref is a Branching engine (llama.cpp's)
   or a model (qwen3/load's, qwen35/load's) on any backend, read on its
   own (the backend its weights live on). A mixture of experts' readings
   hold :experts too, the experts its layers' top k picked at each
   position (the prompt's, then the steps' tokens'): [position [layer [k
   indices]]], which compare-readings can force on the forward under test
   (genclj-fxag). So do llama.cpp's, from an engine recording its experts
   (gen.llm.llama's :record #{\"ffn_moe_topk\"}, through its eval
   callback): its last layer's nil but at the positions read (the
   prompt's last, the steps')."
  [ref tokens steps]
  (if (map? ref)
    (array/with-backend (model-backend ref)
      (let [moe? (some? (get-in ref [:config :experts]))
            run (fn [f] (if moe? (recorded f) [(f) nil]))
            [[lg cache] ex] (run #(prefill ref [tokens] {:logits :last}))]
        (loop [i 1 c cache lp (logprobs lg) out (cond-> {:tokens tokens :root lp :steps []} moe? (assoc :experts ex))]
          (if (> i steps)
            out
            (let [t (first (top-tokens lp 1))
                  [[lg c'] ex] (run #(step ref c [[t]]))
                  lp' (logprobs lg)]
              (recur (inc i) c' lp' (cond-> (update out :steps conj [t lp'])
                                      moe? (update :experts into ex))))))))
    (let [record? (llama-recording? ref)
          _ (when record? (llama-experts ref))
          root (br/open ref tokens)]
      (try
        (let [lp (br/logprobs root)]
          (loop [i 1 b (br/fork root) lp lp out {:tokens tokens :root lp :steps []}]
            (if (> i steps)
              (do (br/dispose! b)
                  (cond-> out record? (assoc :experts (llama-experts ref))))
              (let [t (first (top-tokens lp 1))
                    [b'] (br/advance [b] [t])
                    lp' (br/logprobs b')]
                (recur (inc i) b' lp' (update out :steps conj [t lp']))))))
        (finally (br/dispose! root))))))

(defn compare-readings
  "The forward of model m against a reference's readings (reference's):
   a vector of {:what :max-diff :argmax? :top5?}, :what :whole, :chunk or
   [:step i] (i from 1, one a reading's step); tie as verdict's (0:
   exactly the same tokens). Readings with :forced (reference's :experts)
   force those experts on m's mixture-of-experts layers (with-experts):
   the forward with the reference's routing, so that a near-tie of the
   router's top k (which half floats flip) moves nothing (genclj-fxag)."
  [m {:keys [tokens root steps forced]} tie]
  (array/with-backend (model-backend m)
    (let [n (count tokens)
          h (quot n 2)
          [whole cache] (with-experts forced (range n) #(prefill m [tokens] {:logits :last}))
          [_ half] (with-experts forced (range h) #(prefill m [(subvec tokens 0 h)]))
          [chunk] (with-experts forced (range h n) #(step m half [(subvec tokens h)] {:logits :last}))]
      (loop [i 1 c cache sts (seq steps)
             out [(assoc (verdict root (logprobs whole) tie) :what :whole)
                  (assoc (verdict root (logprobs chunk) tie) :what :chunk)]]
        (if-let [[t lp] (first sts)]
          (let [[lg c'] (with-experts forced [(+ n i -1)] #(step m c [[t]]))]
            (recur (inc i) c' (next sts) (conj out (assoc (verdict lp (logprobs lg) tie) :what [:step i]))))
          out)))))

(defn compare-prompt
  "The forward of model m against the reference e (a Branching engine,
   llama.cpp's, or a model on another backend: reference's) on the
   prompt's tokens (a vector of at least 2): compare-readings of e's
   readings and steps greedy steps, :what :whole, :chunk or [:step i] (i
   from 1 to steps); tie as verdict's (0: exactly the same tokens)."
  [e m tokens steps tie]
  (compare-readings m (reference e tokens steps) tie))

(defn mutated
  "The worst :max-diff of the forward under mutation (a key of
   (mutations-of m)) on the prompt's tokens, against the reference e (as
   compare-prompt's), or against readings (reference's, taken once for
   several mutations): its :whole and :chunk readings, and one step."
  ([e m tokens mutation] (mutated m (reference e tokens 1) mutation))
  ([m readings mutation]
   (let [readings (update readings :steps #(vec (take 1 %)))]
     (with-redefs-fn ((mutations-of m) mutation)
       #(reduce max (map :max-diff (compare-readings m readings 0)))))))
