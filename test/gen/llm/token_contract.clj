(ns gen.llm.token-contract
  "The token generative function (gen.llm.token) and token SMC
   (gen.llm.smc) over an LM, as checks any engine must pass: the GFI laws
   within one LM (exact whatever the engine's float noise: each token is
   scored once, gen.llm.lm's ledger), forced text, token SMC under a
   lipogram (no token whose bytes hold an e: a grammar's mask in
   miniature) by the masked proposal and the bootstrap, and under a
   grammar (a date, gen.llm.grammar) by the masked proposal, and a step of
   SMC after the first as one read. Each check takes a setup and options
   and returns a vector of failures (strings), empty when it passes; one
   that throws fails with its message. Run on llama.cpp
   (gen.llm.llama-token-test, gen.llm.llama-smc-test) and on gen.array's
   engine (gen.llm.array-engine-test).

   The setup: {:engine :lm (an LM over the engine) :completion
   ((token/completion lm))}. Options: :prompt (tokens), :forced (tokens
   generate forces after it), :sea-prompt (tokens: the lipogram's
   prompt), :date-prompt (tokens: the grammar's), :particles (32),
   :min-weight (the forced tokens' least weight, -20),
   :trials (the laws' test.check trials, 10)."
  (:require [clojure.test.check :as tc]
            [gen.gfi :as gfi]
            [gen.laws :as laws]
            [gen.llm.branching :as br]
            [gen.llm.grammar :as grammar]
            [gen.llm.lm :as lm]
            [gen.llm.potential :as potential]
            [gen.llm.smc :as smc]
            [gen.llm.token :as token]
            [gen.random :as rng]))

(set! *warn-on-reflection* true)

(defn run-check
  "The failures of (check setup opts), an exception counted as one."
  [check setup opts]
  (try
    (vec (check setup opts))
    (catch Throwable t
      [(str "threw " (.getName (class t)) ": " (ex-message t) " " (pr-str (ex-data t)))])))

(defn- no-backstop
  "failures, and one more if the engine ran its GC backstop."
  [failures {:keys [engine]}]
  (cond-> failures
    (pos? (long (:collections (br/info engine)))) (conj "the engine ran its GC backstop")))

(defn check-laws
  "The GFI laws (laws/all and laws/conventions) over the completion, within
   one LM: exact, since every token is scored once."
  [{:keys [completion] :as setup} {:keys [prompt trials] :or {trials 10}}]
  (let [cases (laws/cases [[:completion completion [prompt 4] [[prompt 2] [prompt 6]]]])]
    (-> (reduce (fn [fs [k law]]
                  (let [r (tc/quick-check trials (law cases) :seed 20261021)]
                    (cond-> fs (not (:pass? r)) (conj (str k " " (pr-str (select-keys r [:fail :shrunk])))))))
                [] (sort-by key (merge laws/all laws/conventions)))
        (no-backstop setup))))

(defn check-forced
  "generate forces the tokens given, which a model's own continuation
   weighs (a log probability below 0, above :min-weight, -20 by default,
   for a likely one); the
   trace's score is assess's weight, bit for bit."
  [{:keys [lm completion] :as setup} {:keys [prompt forced min-weight] :or {min-weight -20.0}}]
  (let [{tr :trace w :weight} (gfi/generate completion [prompt 12] (token/forced forced) (rng/key 1))
        ^String text (lm/decode lm (:retval tr))]
    (println "completion:" (pr-str text))
    (-> (cond-> []
          (not= forced (subvec (:retval tr) 0 (min (count (:retval tr)) (count forced)))) (conj "the forced tokens")
          (not (< (double min-weight) (double w) 0.0)) (conj (str "the forced tokens' weight " w))
          (not (.startsWith text ^String (lm/decode lm forced))) (conj (str "the text " (pr-str text)))
          (not= (:score tr) (:weight (gfi/assess completion [prompt 12] (:choices tr))))
          (conj "score once: assess's weight is not the trace's score"))
        (no-backstop setup))))

(defn lipogram
  "The potential that forbids every token whose bytes hold an e: 0 or
   -Inf, a constraint. The end is allowed."
  [m]
  (let [n (long (:n-vocab (lm/vocab m)))
        banned (boolean-array (map (fn [^bytes b] (boolean (some #(== 0x65 (long %)) b))) (lm/byte-vocab m)))
        allowed? (fn [ctx] (not-any? #(aget banned (int %)) ctx))
        row (let [w (float-array (inc n))]
              (dotimes [t n] (when (aget banned t) (aset w t (unchecked-float ##-Inf))))
              w)]
    (potential/potential n {:prefix #(if (allowed? %) 0.0 ##-Inf)
                            :complete #(if (allowed? %) 0.0 ##-Inf)
                            :logw-next (fn [_] (aclone row))})))

(defn- text-without-e?
  "Does tokens's text, the end tokens aside, hold no e?"
  [m tokens]
  (not (.contains ^String (lm/decode m (vec (remove (:eog (lm/vocab m)) tokens))) "e")))

(defn check-lipogram
  "Token SMC under the lipogram, by the masked proposal (every particle
   alive) and the bootstrap: every particle's text holds no e; the
   estimate is a finite log probability; the particles' log p are their
   traces' scores, which assess gives bit for bit."
  [{:keys [lm completion] :as setup} {:keys [sea-prompt particles] :or {particles 32}}]
  (let [args [sea-prompt 12]
        pot (lipogram lm)]
    (-> (reduce (fn [fs proposal]
                  (let [s (smc/run completion args particles {:potential pot :proposal proposal :key (rng/key 1)})
                        live (remove :dead? (:particles s))
                        trs (remove nil? (smc/traces s))
                        what (fn [msg] (str (name proposal) ": " msg))]
                    (when (seq live)
                      (println (str "SMC (" (name proposal) "): " (pr-str (lm/decode lm (:tokens (first live)))))))
                    (cond-> fs
                      (empty? live) (conj (what "no particle survives"))
                      (and (= :masked proposal) (not= particles (count live))) (conj (what "a masked particle died"))
                      (not (every? #(text-without-e? lm (:tokens %)) live)) (conj (what "a particle's text holds an e"))
                      (not (Double/isFinite (smc/log-ml-estimate s))) (conj (what "the estimate is not finite"))
                      (not (<= (smc/log-ml-estimate s) 0.0)) (conj (what "the estimate is above 0"))
                      (not= (mapv :log-p live) (mapv :score trs)) (conj (what "the particles' log p are not their traces' scores"))
                      (not (every? #(= (:score %) (:weight (gfi/assess completion args (:choices %)))) trs))
                      (conj (what "score once: assess's weight is not a trace's score")))))
                [] [:masked :lm])
        (no-backstop setup))))

(def date #" ?[0-9]{4}-[0-9]{2}-[0-9]{2}")

(defn check-grammar
  "Token SMC under a grammar (a date), by the masked proposal: every
   particle stays alive and ends, its text a string of the language; the
   estimate (the probability that the completion is a date) is a finite
   log probability; the particles' log p are their traces' scores."
  [{:keys [lm completion] :as setup} {:keys [date-prompt particles] :or {particles 32}}]
  (let [g (grammar/grammar lm (str date))
        args [date-prompt 16]
        s (smc/run completion args particles {:potential g :proposal :masked :key (rng/key 3)})
        live (remove :dead? (:particles s))
        ends (:eog (lm/vocab lm))
        texts (mapv #(lm/decode lm (vec (remove ends (:tokens %)))) live)
        trs (smc/traces s)]
    (println "SMC under a date grammar:" (pr-str (frequencies texts)) (format "log Z %.3f" (smc/log-ml-estimate s)))
    (-> (cond-> []
          (not= particles (count live)) (conj "a masked particle died")
          (not (every? #(contains? ends (peek (:tokens %))) live)) (conj "a particle did not end")
          (not (every? #(re-matches date %) texts)) (conj (str "a text is not a date: " (pr-str (frequencies texts))))
          (not (Double/isFinite (smc/log-ml-estimate s))) (conj "the estimate is not finite")
          (not (<= (smc/log-ml-estimate s) 0.0)) (conj "the estimate is above 0")
          (not= (mapv :log-p live) (mapv :score trs)) (conj "the particles' log p are not their traces' scores")
          (not (every? #(= (:score %) (:weight (gfi/assess completion args (:choices %)))) trs))
          (conj "score once: assess's weight is not a trace's score"))
        (no-backstop setup))))

(defn check-one-read
  "A step of token SMC after the first is one LM read (one batched
   advance)."
  [{:keys [lm completion]} {:keys [sea-prompt particles] :or {particles 32}}]
  (let [reads #(:reads (lm/info lm))
        s1 (smc/step (smc/initialize completion [sea-prompt 8] particles {:potential (lipogram lm) :proposal :masked
                                                                          :key (rng/key 2)}))
        per-step (loop [s s1 out []]
                   (if (or (smc/done? s) (= 6 (count out)))
                     out
                     (let [r0 (reads)
                           s (smc/step (smc/maybe-resample s {}))]
                       (recur s (conj out (- (long (reads)) (long r0)))))))]
    (cond-> []
      (empty? per-step) (conj "no step after the first")
      (not (every? #(= 1 %) per-step)) (conj (str "reads a step: " per-step)))))

(def checks
  "Every check, by name."
  {:laws check-laws
   :forced check-forced
   :lipogram check-lipogram
   :grammar check-grammar
   :one-read check-one-read})
