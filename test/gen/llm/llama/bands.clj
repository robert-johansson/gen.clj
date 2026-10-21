(ns gen.llm.llama.bands
  "The bands of the llama.cpp tests' noise checks, by device (DESIGN §9,
   llama.cpp's band by device; the measurements are DESIGN §6's): twice the
   largest |Δ log p| (over the reference's 64 most likely tokens) that
   bench/gen/llama_noise.clj and bench/gen/llama_long_noise.clj measured
   between two computations of one distribution there.
   - Metal, any weights: 0.1, and 0.6 for branches past a sliding window
     (M2 Max and Mac mini M4, 2026-10-20 and 21; Q8_0: max 5.2e-2 and 0.28).
   - CUDA with float weights: 0.26, and 0.23 for long branches (RTX PRO
     6000, sm_120, Qwen3-0.6B f16, 2026-10-21: max 0.13, and 0.11 past 512
     tokens; no float model with a sliding window measured there).
   - CUDA with quantized weights: no band. There llama.cpp's results vary
     with batch composition by about 0.2 nats a token (up to about 1) and
     differ from f32 by as much, so the noise checks are skipped with a
     note, never given a band wide enough to hide that.
   - The x86 CPU build is batch-invariant (0 measured): Metal's bands."
  (:require [clojure.string :as str]))

(defn device
  "The backend holding the model's weights, from the engine's footprint
   (llama.cpp's buffer names): :metal, :cuda or :cpu."
  [info]
  (let [names (keep (fn [{:keys [kind buffer]}] (when (= kind :model) (str buffer)))
                    (get-in info [:footprint :buffers]))]
    (cond (some #(str/starts-with? % "CUDA") names) :cuda
          (some #(str/starts-with? % "Metal") names) :metal
          :else :cpu)))

(defn quantized?
  "Are the model's weights quantized? llama.cpp's description of a model
   ends with its file type: Q8_0, Q4_K_M, F16, BF16, F32, …"
  [info]
  (not (#{"F16" "BF16" "F32"} (str/upper-case (peek (str/split (str (get-in info [:model :desc])) #" "))))))

(defn bands
  "The noise checks' bands for the engine whose info is given: {:band
   :long-band}, or {:skip note} where a band could only hide a measured
   difference."
  [info]
  (let [d (device info)]
    (if (and (= d :cuda) (quantized? info))
      {:skip (str "Skipping llama.cpp's noise checks: on CUDA, quantized weights ("
                  (get-in info [:model :desc]) ") vary with batch composition by about 0.2 nats a"
                  " token (DESIGN §6); run them with float weights there")}
      (case d
        :cuda {:band 0.26 :long-band 0.23}
        {:band 0.1 :long-band 0.6}))))
