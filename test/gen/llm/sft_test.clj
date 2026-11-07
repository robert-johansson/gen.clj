(ns gen.llm.sft-test
  "Supervised fine-tuning (gen.llm.sft, experimental) of LoRA adapters on
   the tiny random Qwen3 of test/fixtures/gguf (vocabulary 128, hidden 64,
   3 blocks), F32, on the backend under test and its gradients; the
   tokenizer a byte tokenizer of the chat template's pieces (ChatML's
   <|im_start|> and <|im_end|> one token each, the think tags, newline
   and printable ASCII; BOS added), the examples questions answered in a
   JSON format:
   1. the tokens trained on are the assistant turns' content and end
      token, every turn of :messages, only the completion of :prompt; a
      token crossing a trained span's edge, and tokens that are not the
      text's bytes, are loud;
   2. batches pad, truncate and mask the targets;
   3. the loss is the mean NLL per trained token, against the forward's
      log-softmax read by hand;
   4. two batches accumulated step as one batch of both;
   5. a run saved at step 2, loaded and trained on to step 4 is the run
      trained to 4 without a stop, bit for bit (its checkpoints' files
      identical), through dropout, warmup and AdamW's moments;
   6. a loss that is not finite is loud;
   7. checkpoints: their files, the latest, an existing one refused;
   8. each epoch takes every example once, in an order of its own;
   9. training lowers the held-out NLL;
   10. misuse is loud.
   Planted bugs (2026-11-07, the JVM backend, with-redefs-fn), each failing
   its test, in place: the targets' mask a position early (3, 9); each
   batch's loss divided by the batches' number before the mean (4);
   AdamW's moments, or the schedule's step count, not restored from the
   checkpoint (5); the loss ascended (9)."
  (:require [clojure.java.io :as io]
            [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [gen.array.gguf :as gguf]
            [gen.learning :as learn]
            [gen.llm.byte-tokenizer :as bt]
            [gen.llm.chat :as chat]
            [gen.llm.lora :as lora]
            [gen.llm.models.qwen3 :as qwen3]
            [gen.llm.sft :as sft]
            [gen.random :as rng]
            [gen.test-support :as ts])
  (:import [java.nio.file Files]
           [java.nio.file.attribute FileAttribute]))

(def ^:private model (delay (qwen3/load "test/fixtures/gguf/qwen3-tiny-f32.gguf")))

(def ^:private pieces
  "The chat template's pieces, then printable ASCII: tokens 0 to 99, BOS 100,
   EOS 101."
  (concat ["<|im_start|>" "<|im_end|>" "<think>" "</think>" "\n"] (map #(str (char %)) (range 32 127))))

(def ^:private tokenizer (bt/tokenizer pieces))

(defn- qa
  "Question i, answered in the JSON format trained."
  [i]
  {:prompt (str "What is " i " plus " (inc i) "?") :completion (str "{\"answer\": " (+ i i 1) "}")})

(def ^:private examples (delay (mapv #(sft/example tokenizer (qa %)) (range 12))))

(def ^:private held-out (delay (mapv #(sft/example tokenizer (qa %)) (range 40 44))))

(defn- adapter
  "An adapter of the tiny model, rank 4 on q, v and down of every layer,
   its B drawn so that it changes the model, but for opts."
  ([] (adapter nil))
  ([opts]
   (let [ad (lora/init @model (merge {:rank 4 :alpha 8 :targets [:q :v :down]} opts) (rng/key 7))]
     (update ad :params
             (fn [params]
               (into {} (for [[i ts] params]
                          [i (into {} (for [[t {:keys [a b]}] ts]
                                        [t {:a a :b (array/* 0.1 (rng/normal (rng/fold-in (rng/key 3) (+ (* 10 i) (count (name t))))
                                                                             (array/shape b)))}]))])))))))

(defn- trained-text
  "The text of an example's tokens trained on."
  [{:keys [tokens mask]}]
  ((:decode tokenizer) (keep-indexed (fn [i t] (when (mask i) t)) tokens)))

(defn- temp-dir []
  (str (Files/createTempDirectory "sft-test" (make-array FileAttribute 0))))

(defn- delete-tree [path]
  (doseq [f (reverse (file-seq (io/file path)))] (io/delete-file f true)))

(def ^:private band
  "|Δ| between two computations of one value, relative: float64's
   round-off on the JVM backend, float32's on MLX."
  (ts/tol 1e-12))

;; ---------------------------------------------------------------------------
;; 1. The tokens trained on
;; ---------------------------------------------------------------------------

(deftest the-assistant-s-tokens-are-trained-on
  (let [q (qa 1)
        ex (sft/example tokenizer q)]
    (is (= "{\"answer\": 3}<|im_end|>" (trained-text ex)))
    (is (= (str "<s>" (chat/render [{:role :user :content (:prompt q)} {:role :assistant :content (:completion q)}]
                                   {:generation-prompt? false}))
           ((:decode tokenizer) (:tokens ex)))
        "the whole conversation, BOS first")
    (is (= false (first (:mask ex))) "BOS is not trained on")
    (is (= "{\"answer\": 3}" (trained-text (sft/example tokenizer q {:end-token? false}))) "without the end token"))
  (testing "every assistant turn of :messages, only the completion of :prompt"
    (let [msgs [{:role :system :content "Be brief."} {:role :user :content "Hi"} {:role :assistant :content "Hello" :reasoning "greet"}
                {:role :user :content "Bye"} {:role :assistant :content "Ciao"}]]
      (is (= "Hello<|im_end|>Ciao<|im_end|>" (trained-text (sft/example tokenizer {:messages msgs}))))
      (is (= "Ciao<|im_end|>" (trained-text (sft/example tokenizer {:prompt (pop msgs) :completion "Ciao"}))))))
  (testing "a think block is context"
    (let [ex (sft/example tokenizer {:messages [{:role :user :content "Hi"} {:role :assistant :content "Hello" :reasoning "greet"}]}
                          {:chat {:thinking? true}})]
      (is (= "Hello<|im_end|>" (trained-text ex)))
      (is (re-find #"<think>\ngreet\n</think>" ((:decode tokenizer) (:tokens ex))))))
  (testing "tokens that cross a trained span's edge, or are not the text's bytes, are loud"
    (is (= :straddling-token (:reason (ts/thrown-data (sft/example (bt/tokenizer (cons "\n\n{" pieces)) (qa 1))))))
    (is (= :misaligned-tokens (:reason (ts/thrown-data (sft/example (update tokenizer :encode (fn [enc] (fn [t o] (enc (str " " t) o))))
                                                                    (qa 1))))))))

;; ---------------------------------------------------------------------------
;; 2. Batches
;; ---------------------------------------------------------------------------

(deftest batches-pad-truncate-and-mask
  (let [[a b] [(@examples 0) (sft/example tokenizer {:prompt "What is 1000 plus 2000?" :completion "{\"answer\": 3000}"})]
        t (max (count (:tokens a)) (count (:tokens b)))
        bt (sft/batch [a b] {:pad 7})
        trained (fn [ex] (count (filter true? (rest (:mask ex)))))]
    (is (not= (count (:tokens a)) (count (:tokens b))))
    (is (= [2 t] (array/shape (:tokens bt))))
    (is (= [2 (dec t)] (array/shape (:mask bt))))
    (is (= (+ (trained a) (trained b)) (:count bt)))
    (is (= (mapv #(into (:tokens %) (repeat (- t (count (:tokens %))) 7)) [a b]) (ts/host (:tokens bt))) "right-padded")
    (is (= (mapv (fn [ex] (into (mapv #(if % 1.0 0.0) (rest (:mask ex))) (repeat (- t (count (:tokens ex))) 0.0))) [a b])
           (ts/host (:mask bt)))
        "the targets: token t + 1 trained on at position t")
    (testing "truncation keeps the start"
      (let [n (- (count (:tokens a)) 3)
            cut (sft/batch [a] {:max-length n})
            short (sft/example tokenizer {:messages [{:role :assistant :content "Hi"}]})]
        (is (= [1 n] (array/shape (:tokens cut))))
        (is (= (- (trained a) 2) (:count cut)) "the end token and the brace before it cut, and an untrained newline")
        (is (= {:reason :nothing-trained :example 1 :max-length 20}
               (select-keys (ts/thrown-data (sft/batch [short a] {:max-length 20})) [:reason :example :max-length]))
            "an example cut to nothing to train on is loud, by its index")
        (is (= 2 (:count (sft/batch [short] {:max-length 20}))) "Hi within the first 20, its end token past them")))))

;; ---------------------------------------------------------------------------
;; 3. The loss, by hand
;; ---------------------------------------------------------------------------

(defn- nll-by-hand
  "Σ -log p of every trained token of exs, from m's forward of each alone
   (no padding, no mask), its log-softmax read on the host."
  [m exs]
  (reduce + (for [{:keys [tokens mask]} exs
                  :let [lp (ts/host (array/log-softmax (qwen3/forward m [tokens])))]
                  i (range 1 (count tokens))
                  :when (mask i)]
              (- (double (get-in lp [0 (dec i) (tokens i)]))))))

(defn- loss-is-by-hand?
  "Is a step's loss on exs, and their evaluate, the mean NLL per trained
   token by hand?"
  [exs]
  (let [ad (adapter)
        m (lora/adapt @model ad)
        n (count (filter true? (mapcat (comp rest :mask) exs)))
        want (/ (nll-by-hand m exs) n)
        [_ info] (sft/step @model (sft/trainer ad (learn/sgd {:lr 0.0}) (rng/key 1)) [(sft/batch (take 2 exs)) (sft/batch (drop 2 exs))])
        {:keys [nll tokens]} (sft/evaluate m exs {:batch-size 3})]
    (and (= n (:tokens info) tokens)
         (ts/close? (:loss info) want {:rtol band :atol 0.0})
         (ts/close? nll want {:rtol band :atol 0.0}))))

(deftest the-loss-is-the-mean-nll-of-the-trained-tokens
  (let [exs (subvec @examples 0 5)]
    (is (loss-is-by-hand? exs))
    (is (not (with-redefs-fn {#'sft/targets (fn [mask] (mapv #(if % 1.0 0.0) (butlast mask)))} #(loss-is-by-hand? exs)))
        "planted: the targets a position early")))

;; ---------------------------------------------------------------------------
;; 4. Accumulation
;; ---------------------------------------------------------------------------

(defn- accumulation-matches?
  "Do two batches of two accumulated step as one batch of the four (SGD,
   so the step is the mean gradient's): the same loss, tokens and
   parameters after?"
  []
  (let [tr (sft/trainer (adapter) (learn/sgd {:lr 0.5}) (rng/key 3))
        exs (subvec @examples 2 6)
        [t1 i1] (sft/step @model tr [(sft/batch (subvec exs 0 2)) (sft/batch (subvec exs 2 4))])
        [t2 i2] (sft/step @model tr (sft/batch exs))
        flat #(array/concatenate (mapv (fn [x] (array/reshape x [-1])) (array/tree-leaves (get-in % [:adapter :params]))))
        moved (array/max (array/abs (array/- (flat t2) (flat tr))))]
    (and (= (:tokens i1) (:tokens i2))
         (ts/close? (:loss i1) (:loss i2) {:rtol band :atol 0.0})
         (ts/close? (:grad-norm i1) (:grad-norm i2) {:rtol (ts/tol 1e-10) :atol 0.0})
         (< 1e-4 (ts/number moved))
         (ts/close? (flat t1) (flat t2) {:rtol 0.0 :atol (* (ts/tol 1e-10) (ts/number moved))}))))

(deftest accumulated-batches-step-as-one
  (is (accumulation-matches?))
  (let [accumulated @#'sft/accumulated]
    (is (not (with-redefs-fn {#'sft/accumulated (fn [m adapter batches keys opts]
                                                  (let [k (double (count batches))
                                                        [l g n] (accumulated m adapter batches keys opts)]
                                                    [(array// l k) (array/tree-map #(array// % k) g) n]))}
               accumulation-matches?))
        "planted: the sums divided by the batches' number, then by the tokens")))

;; ---------------------------------------------------------------------------
;; 5. Resuming from a checkpoint
;; ---------------------------------------------------------------------------

(defn- file-bytes [path] (Files/readAllBytes (.toPath (io/file path))))

(defn- resumes-exactly?
  "Is a run of 4 steps, saved at step 2 (dir a), the run from that
   checkpoint to 4 (dir b): the same steps, and the same files at 4?"
  []
  (let [opt (learn/warmup-cosine (learn/clip-norm (learn/adamw {:lr 1e-2}) {:max-norm 1.0}) {:warmup 1 :steps 4})
        tr (sft/trainer (adapter {:dropout 0.2}) opt (rng/key 5))
        opts {:steps 4 :batch-size 2 :accumulate 2}
        a (temp-dir)
        b (temp-dir)]
    (try
      (let [whole (sft/train @model tr @examples (assoc opts :save-dir a :save-every 2))
            resumed (sft/train @model (sft/load-checkpoint (str a "/step-00000002")) @examples (assoc opts :save-dir b))
            at-4 (fn [dir f] (file-bytes (str dir "/step-00000004/" f)))]
        (and (= 4 (count (:history whole)))
             (= (drop 2 (:history whole)) (:history resumed))
             (java.util.Arrays/equals ^bytes (at-4 a "trainer.edn") ^bytes (at-4 b "trainer.edn"))
             (java.util.Arrays/equals ^bytes (at-4 a "adapter.gguf") ^bytes (at-4 b "adapter.gguf"))))
      (finally (delete-tree a) (delete-tree b)))))

(deftest a-checkpoint-resumes-exactly
  (is (resumes-exactly?))
  (let [load sft/load-checkpoint]
    (is (not (with-redefs-fn {#'sft/load-checkpoint
                              (fn [path] (update-in (load path) [:optimizer-state :state]
                                                    #(assoc % :m (array/tree-map array/zeros-like (:m %))
                                                            :v (array/tree-map array/zeros-like (:v %)))))}
               resumes-exactly?))
        "planted: AdamW's moments not restored")
    (is (not (with-redefs-fn {#'sft/load-checkpoint (fn [path] (assoc-in (load path) [:optimizer-state :t] 0))}
               resumes-exactly?))
        "planted: the schedule's step count not restored")))

;; ---------------------------------------------------------------------------
;; 6. A loss that is not finite
;; ---------------------------------------------------------------------------

(deftest a-loss-that-is-not-finite-is-loud
  (let [ad (assoc-in (adapter) [:params 1 :v :b] (array/full [32 4] ##NaN))
        data (ts/thrown-data (sft/step @model (sft/trainer ad (learn/adamw {}) (rng/key 1)) (sft/batch (subvec @examples 0 2))))]
    (is (= {:reason :non-finite :step 0} (select-keys data [:reason :step])))
    (is (Double/isNaN (:loss data)))))

;; ---------------------------------------------------------------------------
;; 7. Checkpoints
;; ---------------------------------------------------------------------------

(deftest checkpoints-are-written-whole
  (let [dir (temp-dir)
        [t1] (sft/step @model (sft/trainer (adapter) (learn/adamw {:lr 1e-2}) (rng/key 2)) (sft/batch (subvec @examples 0 2)))
        f32 #(ts/host (array/astype % :float32))]
    (try
      (let [path (sft/save-checkpoint t1 dir)]
        (is (= (str dir "/step-00000001") path))
        (is (= (ts/host t1) (ts/host (sft/load-checkpoint path))) "the trainer, exactly")
        (is (= "adapter" (get-in (gguf/header (str path "/adapter.gguf")) [:metadata "general.type"])))
        (is (= (array/tree-map f32 (get-in t1 [:adapter :params])) (array/tree-map f32 (:params (lora/load-gguf (str path "/adapter.gguf")))))
            "llama.cpp's adapter, in float32")
        (is (= :exists (:reason (ts/thrown-data (sft/save-checkpoint t1 dir)))))
        (.mkdirs (io/file dir ".step-00000009.tmp-1"))
        (spit (io/file dir ".step-00000009.tmp-1" "trainer.edn") "{}")
        (.mkdirs (io/file dir "step-00000007"))
        (is (= path (sft/latest-checkpoint dir)) "a hidden or incomplete one is not a checkpoint")
        (let [[t2] (sft/step @model t1 (sft/batch (subvec @examples 2 4)))]
          (is (= (str dir "/step-00000002") (sft/save-checkpoint t2 dir) (sft/latest-checkpoint dir))))
        (is (nil? (sft/latest-checkpoint (str dir "/step-00000007"))))
        (is (= :no-checkpoint (:reason (ts/thrown-data (sft/load-checkpoint (str dir "/step-00000007")))))))
      (finally (delete-tree dir)))))

;; ---------------------------------------------------------------------------
;; 8. The order of the examples
;; ---------------------------------------------------------------------------

(deftest each-epoch-takes-every-example-once
  (let [order @#'sft/order
        k (rng/key 9)
        epoch (fn [e] (vec (mapcat #(order k 12 % 4) (range (* 3 e) (* 3 (inc e))))))]
    (is (= (range 12) (sort (epoch 0))))
    (is (= (range 12) (sort (epoch 1))))
    (is (not= (epoch 0) (epoch 1)) "each epoch in an order of its own")
    (is (= (order k 12 2 5) (into (subvec (epoch 0) 10) (subvec (epoch 1) 0 3))) "a step across epochs")
    (is (= (epoch 0) (vec (mapcat #(order k 12 % 4) (range 3)))) "a function of the key")))

;; ---------------------------------------------------------------------------
;; 9. Training lowers the held-out NLL
;; ---------------------------------------------------------------------------

(defn- held-out-nlls
  "The held-out NLL before and after 9 steps from a new adapter (B zero:
   the base model's first)."
  []
  (let [ad (lora/init @model {:rank 4 :targets (vec (keys lora/targets))} (rng/key 1))
        opt (learn/warmup-cosine (learn/clip-norm (learn/adamw {:lr 2e-2 :weight-decay 0.0}) {:max-norm 1.0}) {:warmup 1 :steps 9})
        {:keys [evaluations history]} (sft/train @model (sft/trainer ad opt (rng/key 2)) @examples
                                                 {:steps 9 :batch-size 4 :held-out @held-out})]
    [(mapv :nll evaluations) (mapv :loss history)]))

(deftest training-lowers-the-held-out-nll
  ;; measured 2026-11-07 (the JVM backend): 5.93 → 0.90 nats a trained
  ;; token held out; the training loss 6.13 → 0.53 over the 9 steps
  (let [[[before after] losses] (held-out-nlls)]
    (is (< after (* 0.6 before)) (pr-str [before after losses]))
    (is (< (last losses) (* 0.6 (first losses))) (pr-str losses)))
  (let [summed-loss @#'sft/summed-loss
        [[before after]] (with-redefs-fn {#'sft/summed-loss (fn [& args]
                                                              (let [[l g] (apply summed-loss args)]
                                                                [l (array/tree-map array/negate g)]))}
                           held-out-nlls)]
    (is (not (< after (* 0.6 before))) "planted: the loss ascended")))

;; ---------------------------------------------------------------------------
;; 10. Misuse
;; ---------------------------------------------------------------------------

(deftest misuse-is-loud
  (let [reason (fn [f] (:reason (ts/thrown-data (f))))
        ex (@examples 0)
        tr (sft/trainer (adapter) (learn/sgd {:lr 0.1}) (rng/key 1))]
    (testing "examples"
      (is (= :nothing-trained (reason #(sft/example tokenizer {:messages [{:role :user :content "Hi"}]}))))
      (is (= :nothing-trained (reason #(sft/example tokenizer {:prompt "Hi" :completion ""} {:end-token? false}))))
      (is (= :invalid-args (reason #(sft/example tokenizer {:text "Hi"}))))
      (is (= :invalid-args (reason #(sft/example tokenizer {:prompt "Hi" :completion {:role :assistant :content "x"}}))))
      (is (= :invalid-args (reason #(sft/example {:n-vocab 3} (qa 1)))))
      (is (= :invalid-args (reason #(sft/example tokenizer (qa 1) {:chat {:generation-prompt? true}}))))
      (is (= :unknown-options (reason #(sft/example tokenizer (qa 1) {:end? false})))))
    (testing "batches"
      (is (= :invalid-args (reason #(sft/batch []))))
      (is (= :invalid-args (reason #(sft/batch [ex] {:max-length 1}))))
      (is (= :invalid-args (reason #(sft/batch [ex] {:pad -1}))))
      (is (= :invalid-args (reason #(sft/batch [(assoc ex :mask [true])]))))
      (is (= :unknown-options (reason #(sft/batch [ex] {:length 20})))))
    (testing "the trainer and training"
      (is (= :invalid-args (reason #(sft/trainer (adapter) (learn/sgd {:lr 0.1}) 7))) "a key, not a seed")
      (is (= :invalid-args (reason #(sft/trainer {:params {}} (learn/sgd {:lr 0.1}) (rng/key 1)))))
      (is (= :invalid-args (reason #(sft/train @model (assoc tr :step 3) @examples {:steps 2}))))
      (is (= :invalid-args (reason #(sft/train @model tr @examples {:steps 1 :batch-size 0}))))
      (is (= :unknown-options (reason #(sft/train @model tr @examples {:steps 1 :epochs 2}))))
      (is (= :unknown-options (reason #(sft/step @model tr (sft/batch [ex]) {:mask nil}))))
      (is (= :unsupported (reason #(sft/nll (assoc-in @model [:config :architecture] "llama") (sft/batch [ex]))))))))
