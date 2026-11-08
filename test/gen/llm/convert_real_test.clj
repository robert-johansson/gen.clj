(ns gen.llm.convert-real-test
  "gen.llm.convert/convert! on real models against llama.cpp's quantizer:
   each converted file's tensors are the reference's, type and bytes, and
   so is its metadata. The pairs are given as
   -J-Dgen.convert.pairs=<source>=<reference>@<recipe>,… (a recipe a key
   of gen.llm.convert/file-types, e.g. q4_k_m), each reference written by
   llama.cpp's quantizer from its source (test/fixtures/gguf/generate.py
   --quantize); skipped without. Host work only: it needs no backend.

   Measured (2026-11-08, the M4, llama.cpp v0.9.0's quantizer as
   native/build-llama.sh builds it): ggml-org's Qwen3-0.6B F16 as Q8_0
   (311 tensors, 2.2 s) and as Q4_K_M (27 s), and Qwen3.5-0.8B BF16 as
   Q4_K_M (335 tensors, 26 s) and Q8_0 (1.9 s; some of its blocks so
   small that a scale's reciprocal overflows, whose codes are C's
   conversions of infinities), each file byte for byte llama.cpp's, its
   size too. llama.cpp reads them as its own: the same log-probabilities,
   exactly, on three prompts (Metal, 2026-11-08).
   Planted: make_qkx2_quants' first error sum fused, which clang does not
   (its loop vectorized), changes one tensor of Qwen3-0.6B's 169 Q4_K
   ones (blk.21.attn_v) and no byte of the fixtures
   (gen.array.gguf-quants-test): a tie that comes about once in 10^8
   values."
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [gen.array.gguf :as gguf]
            [gen.llm.convert :as convert])
  (:import [java.nio.file Files]
           [java.nio.file.attribute FileAttribute]))

(def ^:private pairs
  (when-let [p (System/getProperty "gen.convert.pairs")]
    (for [pair (str/split p #",")
          :let [[_ source reference recipe] (re-find #"^(.+)=(.+)@(.+)$" pair)]]
      (do (when-not source
            (throw (ex-info (str "gen.convert.pairs: " pair " is not <source>=<reference>@<recipe>") {:pair pair})))
          [source reference (keyword recipe)]))))

(defn- file-bytes
  ^bytes [path {:keys [offset bytes]}]
  (let [out (byte-array bytes)]
    (with-open [f (java.io.RandomAccessFile. (str path) "r")]
      (.seek f (long offset))
      (.readFully f out))
    out))

(deftest ^:slow converted-models-are-llama-cpp-s
  (if-not pairs
    (println "Skipping gen.llm.convert-real-test: no -J-Dgen.convert.pairs=<source>=<reference>@<recipe>,…")
    (doseq [[source reference recipe] pairs]
      (testing (str source " as " (name recipe))
        (let [out (str (Files/createTempDirectory "convert-real" (make-array FileAttribute 0)) "/converted.gguf")]
          (try
            (convert/convert! source out {:recipe recipe})
            (let [ours (gguf/header out)
                  theirs (into {} (map (juxt :name identity)) (:tensors (gguf/header reference)))]
              (is (= (count theirs) (count (:tensors ours))))
              (doseq [{:keys [name type] :as info} (:tensors ours)
                      :let [ref (theirs name)]]
                (is (and ref (= (:type ref) type) (java.util.Arrays/equals (file-bytes out info) (file-bytes reference ref)))
                    name))
              (is (= (:metadata (gguf/header reference)) (:metadata ours)) "the metadata")
              (is (= (.length (java.io.File. ^String reference)) (.length (java.io.File. ^String out))) "the size"))
            (finally (Files/deleteIfExists (.toPath (java.io.File. ^String out))))))))))
