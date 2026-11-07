(ns gen.llm.models
  "The model families of gen.llm.models.* by one name: functions that
   dispatch on a model's architecture to its family's, for code over any
   family (training: gen.llm.sft).

     (architecture m)       → m's architecture, a string (\"qwen3\", \"qwen35\", …)
     (log-probs m tokens)   → the family's log-probs: log p [B T-1] of each next token
     (nll m tokens opts)    → the family's nll: -Σ log p, :mask's weights given

   Each is a multimethod on (architecture m), its methods registered by the
   family's own namespace (gen.llm.models.qwen3 registers \"qwen3\"), so a
   model, made by its family's namespace, finds its methods loaded. Their
   options are the families' (:mask, :checkpoint?, :chunk, log-probs'
   :temperature), and their values and gradients too. A model of an architecture no family
   registered is loud (:unsupported).

   A model's architecture is its config's :arch, or else its
   :architecture (Qwen3's): a GGUF's general.architecture.

   Experimental: outside the 1.x promise (DESIGN §9)."
  {:api :experimental})

(set! *warn-on-reflection* true)

(defn architecture
  "The architecture of model m (a family's model {:config :weights}): its
   config's :arch, or else its :architecture."
  [m]
  (let [{:keys [arch architecture]} (:config m)]
    (or arch architecture)))

(defn- dispatch [m & _] (architecture m))

(defn- unsupported [f m]
  (let [arch (architecture m)]
    (throw (ex-info (str "gen.llm.models/" f ": no family of the architecture " (pr-str arch) " is loaded")
                    {:reason :unsupported :fn f :architecture arch}))))

(defmulti log-probs
  "The log-probabilities [B T-1] of each row of tokens at positions 1 …
   T-1, each given the tokens before it, by m's family (qwen3/log-probs).
   Options: the family's (:checkpoint?, :chunk, :temperature)."
  {:arglists '([m tokens] [m tokens opts])}
  dispatch)

(defmethod log-probs :default [m & _] (unsupported 'log-probs m))

(defmulti nll
  "The negative log-likelihood of tokens by m's family (qwen3/nll), -Σ
   log-probs, each position's times :mask's weight where given: a 0-d
   array. Options: the family's (:mask, :checkpoint?, :chunk)."
  {:arglists '([m tokens] [m tokens opts])}
  dispatch)

(defmethod nll :default [m & _] (unsupported 'nll m))
