(ns gen.llm.cell-engine
  "A pure-JVM Slots engine (gen.llm.slots) whose KV cells behave as
   llama.cpp's unified cache, for the LM protocol's tests: a read takes a
   new cell for each token, the lowest free first, from :max-cells; a copy
   shares its source's cells; a cell is freed when no slot holds it; a read
   that finds too few free cells throws {:reason :kv-full :exhausted? true}
   and changes nothing. (gen.llm.toy-engine's cells count copies again.)
   It checks llama.cpp's invariants loudly, as the toy engine does.

   Its distributions are the toy model's (gen.llm.toy-engine/toy-logprobs),
   exact by default: a function of the tokens alone. With :noise, each is
   perturbed by up to :noise nats, by a function of the read's shape (its
   rows, the row's place among them) and the cell the row's token landed
   in, as llama.cpp's kernels and cache layout perturb theirs: the same
   read from the same state gives the same result, but two ways to compute
   a context give two.

   (engine {:n-vocab :max-branches :n-ctx :n-batch :max-cells :noise :fail-reads})
   :fail-reads, a set of read numbers (from 0) that throw a decode failure
   (restoring the slots), plants failures for the tests of what the LM does
   when a read fails.
   (counts engine) → {:cells (held now) :reads :kv-full :failed}"
  (:require [gen.llm.slots :as slots]
            [gen.llm.toy-engine :as toy]))

(set! *warn-on-reflection* true)

(defn- mix
  "SplitMix64's finalizer."
  ^long [^long z]
  (let [z (unchecked-multiply (bit-xor z (unsigned-bit-shift-right z 30)) -4658895280553007687)
        z (unchecked-multiply (bit-xor z (unsigned-bit-shift-right z 27)) -7723592293110705685)]
    (bit-xor z (unsigned-bit-shift-right z 31))))

(defn- unit
  "A double in [-1, 1) from a hash."
  ^double [^long h]
  (- (* 2.0 (/ (double (unsigned-bit-shift-right (mix h) 11)) 9007199254740992.0)) 1.0))

(defn- perturbed
  "lp, perturbed by up to noise nats as a function of key and renormalized
   in double."
  ^floats [^floats lp ^double noise ^long key]
  (let [n (alength lp)
        x (double-array n)]
    (dotimes [j n]
      (aset x j (+ (aget lp j) (* noise (unit (bit-xor key (unchecked-multiply (inc j) -7046029254386353131)))))))
    (let [m (areduce x j m Double/NEGATIVE_INFINITY (Math/max m (aget x j)))
          z (+ m (Math/log (areduce x j s 0.0 (+ s (Math/exp (- (aget x j) m))))))
          out (float-array n)]
      (dotimes [j n] (aset out j (float (- (aget x j) z))))
      out)))

(defn- row
  "The distribution after tokens, read as row i of a read of rows rows,
   the last token in cell cell."
  ^floats [n-vocab noise tokens rows i cell]
  (let [lp (toy/toy-logprobs n-vocab tokens)]
    (if noise
      (perturbed lp noise (mix (+ (* 1000003 (long rows)) (* 1009 (long i)) (long cell))))
      lp)))

(defn- fail [msg data]
  (throw (ex-info (str "cell engine: " msg) data)))

(declare take-cells)

(deftype CellSlots [^long n-vocab ^long n-slots ^long n-ctx ^long n-batch ^long max-cells noise fail-reads
                    ^objects tokens ^objects cells ^ints holders free counts]
  slots/Slots
  (slots-info [_] {:n-vocab n-vocab :n-slots n-slots :n-ctx n-ctx :n-batch n-batch :engine :cells})

  (slots-run [this slot pos ts targets]
    (let [have (aget tokens slot)]
      (when-not (= pos (count have))
        (fail "a run starts away from the end of its slot" {:slot slot :pos pos :have (count have)}))
      (when (and targets (not= (count targets) (dec (count ts))))
        (fail "a run's targets are one fewer than its tokens" {:tokens (count ts) :targets (count targets)}))
      (let [ids (take-cells this (count ts))
            all (into have ts)
            n (count ts)
            at (fn [^long k] (row n-vocab noise (subvec all 0 (+ (long pos) k 1)) n k (ids k)))]
        (aset tokens slot all)
        (aset cells slot (into (aget cells slot) ids))
        {:logprobs (at (dec n))
         :scores (when targets (mapv (fn [k] (double (aget ^floats (at k) (int (targets k))))) (range (count targets))))})))

  (slots-step [this ss positions ts]
    (when-not (apply distinct? ss) (fail "a step's slots repeat" {:slots ss}))
    (when (< n-batch (count ss)) (fail "a step over n-batch" {:n (count ss)}))
    (doseq [[slot pos] (map vector ss positions)]
      (when-not (= pos (count (aget tokens slot)))
        (fail "a step away from the end of its slot" {:slot slot :pos pos})))
    (let [ids (take-cells this (count ss))
          m (count ss)]
      (mapv (fn [i slot t]
              (let [all (conj (aget tokens slot) t)]
                (aset tokens slot all)
                (aset cells slot (conj (aget cells slot) (ids i)))
                (row n-vocab noise all m i (ids i))))
            (range m) ss ts)))

  (slots-copy [_ src dst]
    (when (seq (aget tokens dst)) (fail "a copy into a slot that is not empty" {:src src :dst dst}))
    (aset tokens dst (aget tokens src))
    (aset cells dst (aget cells src))
    (doseq [id (aget cells src)] (aset holders (int id) (inc (aget holders (int id))))))

  (slots-clear [_ slot]
    (doseq [id (aget cells slot)]
      (let [h (dec (aget holders (int id)))]
        (aset holders (int id) h)
        (when (zero? h) (vswap! free conj id))))
    (aset tokens slot [])
    (aset cells slot []))

  (slots-close [_])

  Object
  (toString [_] "#<CellSlots>"))

(defn- take-cells
  "n free cells, the lowest first, each held once: a vector, or a loud
   :kv-full that changes nothing. Read number r of the engine; a planted
   failure throws instead."
  [^CellSlots s ^long n]
  (let [counts (.-counts s)
        r (:reads @counts)]
    (vswap! counts update :reads inc)
    (when (contains? (.-fail-reads s) r)
      (vswap! counts update :failed inc)
      (throw (ex-info "cell engine: a planted decode failure" {:reason :decode-failed :read r})))
    (when (< (count @(.-free s)) n)
      (vswap! counts update :kv-full inc)
      (throw (ex-info "cell engine: no room for the tokens" {:reason :kv-full :exhausted? true})))
    (let [ids (vec (take n @(.-free s)))]
      (vswap! (.-free s) #(reduce disj % ids))
      (doseq [id ids] (aset ^ints (.-holders s) (int id) 1))
      ids)))

(defn engine
  [{:keys [n-vocab max-branches n-ctx n-batch max-cells noise fail-reads]
    :or {n-vocab 97 max-branches 8 n-ctx 256 max-cells 1024 fail-reads #{}}}]
  (let [n-slots (inc max-branches)
        ;; as llama.cpp's engine: a decode holds a token for every slot
        n-batch (max (long (or n-batch 64)) n-slots)]
    (slots/engine #(CellSlots. n-vocab n-slots n-ctx n-batch max-cells noise fail-reads
                               (object-array (repeat n-slots [])) (object-array (repeat n-slots []))
                               (int-array max-cells) (volatile! (into (sorted-set) (range max-cells)))
                               (volatile! {:reads 0 :kv-full 0 :failed 0}))
                  "cells")))

(defn counts
  "{:cells held now, :reads (runs and steps), :kv-full (reads refused for
   want of cells), :failed (planted failures)}."
  [e]
  (slots/call e (fn [^CellSlots s]
                  (assoc @(.-counts s) :cells (- (.-max-cells s) (count @(.-free s)))))))
