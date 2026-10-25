(ns gen.llm.llama.recurrent
  "Where llama.cpp keeps a recurrent or hybrid model's states, replayed:
   the bookkeeping of llama_memory_recurrent (src/llama-memory-recurrent.cpp
   at llama.cpp 620d222, which v0.7.0 builds: seq_cp, seq_rm, find_slot),
   fed every memory operation of the engine, so that gen.llm.llama can
   split a step whose sequences would land far apart. Internal to
   gen.llm.llama. Pinned to that commit: gen.llama.LlamaRuntime refuses
   any libllama but v0.7.0, and a change of version needs this file read
   against the new llama-memory-recurrent.cpp.

   Why: a decode gathers the states of every cell from the lowest to the
   highest its sequences land on (n_rs of them, build_rs in
   llama-graph.cpp), and writes back the ones between that it does not
   change. A step whose new states land on cells 2 and 63 copies the 60
   others. On Qwen3.5-0.8B (Metal, Mac mini M4) a state is 19 MB, and each
   one in the span costs 3.5 ms, against 2.1 ms for each sequence read and
   about 10 ms for the decode itself (docs/research/llm-branching.md,
   Recurrent states and split steps). An engine whose live branches are
   all in its steps keeps them on adjacent cells; one that holds others
   between them, as the LM protocol's cache does, pays for every one in
   each step's span.

   Where a state lives changes no result, only the cost: a replay that
   drifts from llama.cpp costs time, never correctness.

   A replay is a map:
     :size   cells (the context's n_seq_max)
     :head   where the search for an empty cell starts
     :used   cells holding a state
     :cells  each {:pos :tail :seq}, as llama.cpp's mem_cell: :seq the
             sequences whose state the cell holds, :pos the state's
             position, and :tail, kept in cell i, the cell holding
             sequence i's state (-1 for none)
   and after a decode :n-rs (the cells it gathered) and :taken (the cell
   each of its sequences took, before the gather moved them)."
  {:api :experimental})

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

(def ^:private no-cell {:pos -1 :tail -1 :seq #{}})

(defn empty-cells
  "The replay of a context with size sequences, before any decode."
  [size]
  {:size size :head 0 :used 0 :cells (vec (repeat size no-cell))})

(defn- empty-cell? [st i] (empty? (get-in st [:cells i :seq])))

(defn- free-cell
  "Cell i, now holding no sequence, emptied."
  [st i]
  (-> st (assoc-in [:cells i :pos] -1) (update :used dec)))

(defn copy
  "llama_memory_seq_cp(src, dst) of the whole sequence: dst shares src's
   cell."
  [st src dst]
  (let [old (long (get-in st [:cells dst :tail]))
        st (if (neg? old)
             st
             (let [st (-> st
                          (update-in [:cells old :seq] disj dst)
                          (assoc-in [:cells dst :tail] -1))]
               (if (empty-cell? st old) (free-cell st old) st)))
        cell (long (get-in st [:cells src :tail]))]
    (if (neg? cell)
      st
      (-> st
          (update-in [:cells cell :seq] conj dst)
          (assoc-in [:cells dst :tail] cell)))))

(defn clear
  "llama_memory_seq_rm(s, -1, -1): s leaves its cell, and the cell is
   empty when no other sequence shares it. The search starts again at the
   lowest cell this empties, if that is below the head."
  [st s]
  (let [cell (long (get-in st [:cells s :tail]))]
    (if (or (neg? cell) (neg? (long (get-in st [:cells cell :pos]))))
      st
      (let [st (-> st
                   (assoc-in [:cells s :tail] -1)
                   (update-in [:cells cell :seq] disj s))]
        (if (empty-cell? st cell)
          (cond-> (free-cell st cell)
            (< cell (long (:head st))) (assoc :head cell))
          st)))))

(defn- next-empty
  "The first empty cell from cell i on, wrapping round (i itself if none
   is)."
  [st ^long i]
  (let [size (long (:size st))]
    (loop [c i k 0]
      (let [c (if (>= c size) (- c size) c)]
        (if (or (empty-cell? st c) (>= k size)) c (recur (inc c) (inc k)))))))

(defn- swap-cells
  "Cells a and b exchange their states, and every tail follows."
  [st ^long a ^long b]
  (let [ca (get-in st [:cells a]) cb (get-in st [:cells b])]
    (-> st
        (assoc-in [:cells a] (assoc cb :tail (:tail ca)))
        (assoc-in [:cells b] (assoc ca :tail (:tail cb)))
        (update :cells (fn [cells]
                         (mapv (fn [c] (let [t (long (:tail c))]
                                         (cond (== t b) (assoc c :tail a)
                                               (== t a) (assoc c :tail b)
                                               :else c)))
                               cells))))))

(defn decode
  "find_slot for a decode of one token for each of seqs (distinct), at
   positions, in batch order: a sequence that shares its cell (a fork, or a
   branch others were forked from) takes the next empty cell, searched from
   the head; then the decode's sequences move to adjacent cells from the
   lowest one (the gather), and the head is that cell."
  [st seqs positions]
  (let [n (count seqs)
        st (if (> (long (:head st)) (+ (long (:used st)) (* 2 n))) (assoc st :head 0) st)
        [st lo hi] (loop [st st s 0 free (next-empty st (:head st)) lo (dec (long (:size st))) hi 0]
                     (if (== s n)
                       [st lo hi]
                       (let [q (seqs s)
                             cell (long (get-in st [:cells q :tail]))
                             own? (and (<= 0 cell) (== 1 (count (get-in st [:cells cell :seq]))))
                             [st free] (if own?
                                         [st free]
                                         (let [st (if (neg? cell)
                                                    st
                                                    (-> st
                                                        (assoc-in [:cells free :pos] (get-in st [:cells cell :pos]))
                                                        (update-in [:cells cell :seq] disj q)
                                                        (update-in [:cells free :seq] conj q)))]
                                           [(assoc-in st [:cells q :tail] free)
                                            (if (< (inc s) n) (next-empty st (inc (long free))) free)]))
                             t (long (get-in st [:cells q :tail]))]
                         (recur st (inc s) free (min lo t) (max hi t)))))
        taken (mapv #(get-in st [:cells % :tail]) seqs)
        st (reduce (fn [st s]
                     (let [dst (+ (long lo) (long s)) src (long (get-in st [:cells (seqs s) :tail]))]
                       (if (== dst src) st (swap-cells st dst src))))
                   st (range n))
        st (reduce (fn [st s]
                     (let [cell (+ (long lo) (long s)) q (seqs s)]
                       (-> st
                           (update-in [:cells cell] assoc :pos (positions s) :seq #{q})
                           (assoc-in [:cells q :tail] cell))))
                   st (range n))]
    (assoc st
           :head lo
           :used (count (remove #(empty? (:seq %)) (:cells st)))
           :n-rs (inc (- (long hi) (long lo)))
           :taken taken)))

;; ---------------------------------------------------------------------------
;; Splitting a step
;; ---------------------------------------------------------------------------

(defn- extra
  "The states a decode gathers but does not read: its span less its
   sequences."
  ^long [st]
  (- (long (:n-rs st)) (count (:taken st))))

(defn- cost
  "The cost of decoding groups (of indices into seqs) in turn from st, in
   states copied: fixed for each decode, and the extra states it gathers
   (the sequences' own cost is the same however they are grouped)."
  [st seqs positions groups fixed]
  (loop [st st [g & gs] groups total 0.0]
    (if g
      (let [st (decode st (mapv seqs g) (mapv positions g))]
        (recur st gs (+ total (double fixed) (extra st))))
      total)))

(defn- runs
  "The cells, in order, cut into runs of adjacent ones, with the cut that
   minimizes the fixed cost of each run plus its span."
  [cells ^double fixed]
  (let [cs (vec (sort cells))
        n (count cs)
        cuts (reduce (fn [[best from] j]
                       (let [[b i] (reduce (fn [[b _ :as m] i]
                                             (let [c (+ (double (best i)) fixed
                                                        (inc (- (long (cs (dec (long j)))) (long (cs i)))))]
                                               (if (< c (double b)) [c i] m)))
                                           [Double/POSITIVE_INFINITY 0] (range j))]
                         [(conj best b) (conj from i)]))
                     [[0.0] [0]] (range 1 (inc n)))
        from (second cuts)]
    (loop [j n out ()]
      (if (zero? j) (vec out) (recur (long (from j)) (cons (subvec cs (from j) j) out))))))

(defn plan
  "How to decode a step of one token for each of seqs at positions: groups
   of indices into seqs, decoded in turn. One group, the whole step, unless
   the replay finds splitting it along runs of the cells its sequences take
   cheaper by fixed, a decode's own cost in states copied.

   A sequence that keeps its cell (it owns it, or is the last of the
   step's sequences sharing it) must be decoded after the others sharing
   it, which leave for empty cells; those take them in the order of the
   search, so the runs that hold empty cells go in that order, and each
   takes the leavers due soonest."
  [st seqs positions ^double fixed]
  (let [n (count seqs)
        whole [(vec (range n))]
        full (decode st seqs positions)]
    (if (zero? (extra full))
      whole
      (let [taken (:taken full)
            before (mapv #(get-in st [:cells % :tail]) seqs)
            stays? (fn [i] (= (taken i) (before i)))
            leavers (vec (remove stays? (range n)))
            ;; the empty cells, in the order the search found them
            found (into {} (map-indexed (fn [k i] [(taken i) k])) leavers)
            rs (runs taken fixed)
            run-of (into {} (for [[r cs] (map-indexed vector rs) c cs] [c r]))
            order (vec (sort-by (fn [r] [(reduce min Long/MAX_VALUE (keep found (rs r))) r]) (range (count rs))))
            at (into {} (map-indexed (fn [p r] [r p])) order)
            stayer (into {} (for [i (range n) :when (stays? i)] [(taken i) i]))
            ;; a leaver must go no later than the sequence keeping its old cell
            due (fn [i] (if-let [j (stayer (before i))] (long (at (run-of (taken j)))) Long/MAX_VALUE))
            [groups left ok?]
            (reduce (fn [[groups pool ok?] p]
                      (let [k (count (filter found (rs (order p))))
                            [go stay] (split-at k (sort-by due pool))
                            keepers (filter #(and (stays? %) (== (long p) (long (at (run-of (taken %)))))) (range n))]
                        [(conj groups (vec (concat go keepers)))
                         stay
                         (and ok? (== k (count go)) (every? #(>= (long (due %)) (long p)) go) (every? #(> (long (due %)) (long p)) stay))]))
                    [[] leavers true] (range (count order)))]
        ;; whole costs fixed and full's extra states, decoded already
        (if (and ok? (empty? left) (< (double (cost st seqs positions groups fixed)) (+ fixed (extra full))))
          groups
          whole)))))
