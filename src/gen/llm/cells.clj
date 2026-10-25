(ns gen.llm.cells
  "The KV cells an LM's branches hold, counted as llama.cpp's unified cache
   counts them, so that gen.llm.cache can keep them under a bound: a read
   of n tokens takes n new cells, a fork shares its branch's cells, and a
   cell is freed when no branch holds it.

   A ledger records the cells as segments, one for each read. A branch's
   state is a segment: the one its read made, or its parent's for a fork.
   A segment is held by the branches whose state it is and by the segments
   read after it, so its cells are freed when no live branch descends from
   it.

     (allocate ledger parent n) → [ledger id]  a read of n cells after segment parent (nil for none)
     (share ledger id)                        one more branch with state id (a fork)
     (release ledger id)                      one branch fewer (consumed or disposed)
     (freed ledger id)                        the cells that release would free
     (used ledger)                            the cells held"
  {:api :experimental})

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

;; {:used cells held, :next the next id, :segments {id {:parent id-or-nil
;; :size cells :refs holders}}}

(def empty-ledger
  "The ledger of no cells."
  {:used 0 :next 0 :segments {}})

(defn used
  "The cells the ledger's branches hold."
  ^long [ledger]
  (:used ledger))

(defn- segment [ledger id]
  (or (get-in ledger [:segments id])
      (throw (ex-info (str "cells: no segment " id) {:reason :no-segment :id id}))))

(defn allocate
  "The ledger after a read of n cells by a branch with state parent (nil
   for a new branch), and the id of the reading branch's successor's state:
   [ledger id]. The reading branch, when it is consumed, is released
   separately."
  [ledger parent ^long n]
  (let [id (:next ledger)]
    (when parent (segment ledger parent))
    [(cond-> (-> ledger
                 (update :next inc)
                 (update :used + n)
                 (assoc-in [:segments id] {:parent parent :size n :refs 1}))
       parent (update-in [:segments parent :refs] inc))
     id]))

(defn share
  "The ledger with one more branch with state id."
  [ledger id]
  (segment ledger id)
  (update-in ledger [:segments id :refs] inc))

(defn release
  "The ledger with one branch with state id fewer: segments no branch
   descends from any more are freed."
  [ledger id]
  (loop [ledger ledger id id]
    (let [{:keys [refs size parent]} (segment ledger id)]
      (if (< 1 (long refs))
        (assoc-in ledger [:segments id :refs] (dec (long refs)))
        (let [ledger (-> ledger
                         (update :segments dissoc id)
                         (update :used - size))]
          (if parent (recur ledger parent) ledger))))))

(defn freed
  "The cells (release ledger id) frees."
  ^long [ledger id]
  (loop [id id total 0]
    (let [{:keys [refs size parent]} (segment ledger id)]
      (cond
        (< 1 (long refs)) total
        parent (recur parent (+ total (long size)))
        :else (+ total (long size))))))
