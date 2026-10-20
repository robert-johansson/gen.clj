(ns gen.llm.toy-engine
  "A pure-JVM Slots engine (gen.llm.slots) for the Branching tests. Its
   'language model' gives each token sequence a fixed pseudo-random
   next-token distribution, a function of the whole sequence computed the
   same way however it is batched, so the Branching contract holds exactly
   (band 0). It checks llama.cpp's invariants loudly (a slot's contents end
   where a run or step starts; a copy's destination is empty; a step's slots
   are distinct; at most n-batch tokens a decode), so the bookkeeping of
   gen.llm.slots is tested on every call.

   (engine {:n-vocab :max-branches :n-ctx :n-batch :max-cells :on-close :bug})
   :max-cells bounds the tokens held in all slots together, as llama.cpp's
   KV cells (but copies count again: the toy shares nothing): a run or step
   past it throws {:reason :kv-full :exhausted? true}, changing nothing.
   :on-close is called when the engine closes.
   :bug plants one mutation of a Slots operation, for the power tests:
     :copy-perturbs  a copy changes the last token of its destination
     :step-rotates   a step returns its rows rotated by one
     :step-ignores   a step reads token 0, not the token given
     :run-shifts     a run's scores read the target after the next token
     :run-stale      a run returns the distribution before its last token
     :copy-fails     every second copy throws (copies must not, by the
                     Slots contract: for the tests of what the engine does
                     if one does)"
  (:require [gen.llm.slots :as slots]))

(set! *warn-on-reflection* true)

(defn- mix
  "SplitMix64's finalizer."
  ^long [^long z]
  (let [z (unchecked-multiply (bit-xor z (unsigned-bit-shift-right z 30)) -4658895280553007687)
        z (unchecked-multiply (bit-xor z (unsigned-bit-shift-right z 27)) -7723592293110705685)]
    (bit-xor z (unsigned-bit-shift-right z 31))))

(defn- sequence-hash ^long [tokens]
  (reduce (fn [^long h t] (mix (bit-xor h (unchecked-multiply (inc (long t)) -7046029254386353131))))
          0x5eed tokens))

(defn toy-logprobs
  "The toy model's next-token log-probabilities after tokens: logits in
   [-4, 4), normalized in double, as floats."
  ^floats [^long n-vocab tokens]
  (let [h (sequence-hash tokens)
        logits (double-array n-vocab)]
    (dotimes [j n-vocab]
      (aset logits j (- (* 8.0 (/ (double (unsigned-bit-shift-right (mix (bit-xor h (* 0x9e3779b9 (inc j)))) 11))
                                  9007199254740992.0))
                        4.0)))
    (let [m (areduce logits j m Double/NEGATIVE_INFINITY (Math/max m (aget logits j)))
          z (+ m (Math/log (areduce logits j s 0.0 (+ s (Math/exp (- (aget logits j) m))))))
          out (float-array n-vocab)]
      (dotimes [j n-vocab] (aset out j (float (- (aget logits j) z))))
      out)))

(defn- fail [msg data]
  (throw (ex-info (str "toy engine: " msg) data)))

(defn- check-cells [^objects contents max-cells more]
  (when max-cells
    (let [held (areduce contents i n 0 (+ n (count (aget contents i))))]
      (when (< (long max-cells) (+ held (long more)))
        (throw (ex-info "toy engine: no room for the tokens" {:reason :kv-full :exhausted? true}))))))

(deftype ToySlots [^long n-vocab ^long n-slots ^long n-ctx ^long n-batch ^objects contents bug max-cells on-close
                   copies]
  slots/Slots
  (slots-info [_] {:n-vocab n-vocab :n-slots n-slots :n-ctx n-ctx :n-batch n-batch :engine :toy})

  (slots-run [_ slot pos tokens targets]
    (let [have (aget contents slot)]
      (when-not (= pos (count have))
        (fail "a run starts away from the end of its slot" {:slot slot :pos pos :have (count have)}))
      (when (and targets (not= (count targets) (dec (count tokens))))
        (fail "a run's targets are one fewer than its tokens" {:tokens (count tokens) :targets (count targets)}))
      (check-cells contents max-cells (count tokens))
      (let [all (into have tokens)
            scores (when targets
                     (double-array
                      (for [i (range (count targets))
                            :let [after (subvec all 0 (+ pos i 1 (if (= bug :run-shifts) 1 0)))]]
                        (let [^floats lp (toy-logprobs n-vocab after)]
                          (aget lp (int (targets i)))))))]
        (aset contents slot all)
        {:logprobs (toy-logprobs n-vocab (if (= bug :run-stale) (pop all) all))
         :scores (some-> scores vec)})))

  (slots-step [_ slots positions tokens]
    (when-not (apply distinct? slots) (fail "a step's slots repeat" {:slots slots}))
    (when (< n-batch (count slots)) (fail "a step over n-batch" {:n (count slots)}))
    (check-cells contents max-cells (count slots))
    (let [rows (mapv (fn [slot pos t]
                       (let [have (aget contents slot)]
                         (when-not (= pos (count have))
                           (fail "a step away from the end of its slot" {:slot slot :pos pos :have (count have)}))
                         (let [all (conj have (if (= bug :step-ignores) 0 t))]
                           (aset contents slot all)
                           (toy-logprobs n-vocab all))))
                     slots positions tokens)]
      (if (= bug :step-rotates) (into (subvec rows 1) [(first rows)]) rows)))

  (slots-copy [_ src dst]
    (when (seq (aget contents dst)) (fail "a copy into a slot that is not empty" {:src src :dst dst}))
    (when (and (= bug :copy-fails) (odd? (vswap! copies inc)))
      (fail "a copy failed (planted)" {:src src :dst dst}))
    (let [have (aget contents src)]
      (aset contents dst (if (and (= bug :copy-perturbs) (seq have))
                           (conj (pop have) (mod (inc (long (peek have))) n-vocab))
                           have))))

  (slots-clear [_ slot] (aset contents slot []))

  (slots-close [_] (when on-close (on-close))))

(defn engine
  [{:keys [n-vocab max-branches n-ctx n-batch bug max-cells on-close]
    :or {n-vocab 97 max-branches 8 n-ctx 256 n-batch 64}}]
  (let [n-slots (inc max-branches)]
    (slots/engine #(ToySlots. n-vocab n-slots n-ctx n-batch (object-array (repeat n-slots [])) bug max-cells on-close
                              (volatile! 0))
                  "toy")))
