(ns gen.llm.paged
  "A paged KV cache's books, as values (genclj-1daf): a block allocator
   and sequences' block tables, for gen.array's paged operations
   (paged-write, copy-blocks, paged-attention) and the array engine's
   paged mode (gen.llm.array-engine).

   An allocator holds a pool's blocks: {:size n :free a sorted set of
   free block ids :refs {id holders}}. A block is free or held by one or
   more sequences: a fork shares its parent's blocks (each one's holders
   counted), so forking a sequence copies nothing. A sequence is
   {:blocks [ids …] :len n}: its positions 0..n-1 in its blocks, position
   p at offset p mod BS of block p/BS.

   Before a sequence is written at positions len..len+n-1, prepare makes
   them its own: a last block it shares and has filled in part is copied
   (copy-on-write: the copy's block ids are returned, for
   gen.array/copy-blocks, before the write), and new blocks are taken for
   the rest. Blocks it shares and has filled are never written again, so
   forks share them for good.

   Everything here is a value on the host: the allocator and the tables
   change only by these functions' results. Experimental: outside the 1.x
   promise (DESIGN §9)."
  {:api :experimental})

(set! *warn-on-reflection* true)

(defn- invalid [message data]
  (ex-info (str "paged: " message) (merge {:reason :invalid-args} data)))

;; ---------------------------------------------------------------------------
;; The allocator
;; ---------------------------------------------------------------------------

(defn allocator
  "An allocator of a pool of n blocks, the first reserved ones never given
   out (a caller's scratch blocks)."
  ([n] (allocator n 0))
  ([n reserved]
   (when-not (and (nat-int? reserved) (pos-int? n) (< (long reserved) (long n)))
     (throw (invalid (str "a pool of " (pr-str n) " blocks cannot reserve " (pr-str reserved)) {:size n :reserved reserved})))
   {:size (long n) :free (into (sorted-set) (range reserved n)) :refs {}}))

(defn free-count
  "The free blocks of allocator a."
  ^long [a]
  (count (:free a)))

(defn holders
  "The sequences holding block id (0 when it is free)."
  ^long [a id]
  (long (get (:refs a) id 0)))

(defn allocate
  "[a' ids]: k free blocks of a taken, each held once. Loud when fewer are
   free (:reason :blocks-exhausted, with :free and :wanted), a' unmade."
  [a k]
  (let [k (long k)]
    (when (< (free-count a) k)
      (throw (ex-info (str "paged: " k " blocks wanted, " (free-count a) " free")
                      {:reason :blocks-exhausted :free (free-count a) :wanted k})))
    (let [ids (vec (take k (:free a)))]
      [(-> a
           (update :free #(apply disj % ids))
           (update :refs #(reduce (fn [m id] (assoc m id 1)) % ids)))
       ids])))

(defn retain
  "a with each block of ids held once more (a fork's)."
  [a ids]
  (update a :refs #(reduce (fn [m id]
                             (when-not (contains? m id)
                               (throw (invalid (str "block " id " is free: nothing to share") {:block id})))
                             (update m id inc))
                           % ids)))

(defn release
  "a with each block of ids held once less, freed when no sequence holds
   it."
  [a ids]
  (reduce (fn [a id]
            (let [n (holders a id)]
              (cond (zero? n) (throw (invalid (str "block " id " is free: nothing to release") {:block id}))
                    (= 1 n) (-> a (update :refs dissoc id) (update :free conj id))
                    :else (update-in a [:refs id] dec))))
          a ids))

(defn grow
  "a over a pool of n blocks (at least its size): the new ones free."
  [a n]
  (when-not (and (integer? n) (<= (long (:size a)) (long n)))
    (throw (invalid (str "grow: a pool grows to at least its " (:size a) " blocks, not " (pr-str n)) {:size (:size a) :to n})))
  (-> a (assoc :size (long n)) (update :free into (range (:size a) n))))

;; ---------------------------------------------------------------------------
;; Sequences
;; ---------------------------------------------------------------------------

(def empty-sequence
  "A sequence of no positions."
  {:blocks [] :len 0})

(defn fork
  "[a' s']: s shared, its blocks held once more: the fork's table is
   s's."
  [a s]
  [(retain a (:blocks s)) s])

(defn dispose
  "a with sequence s's blocks released."
  [a s]
  (release a (:blocks s)))

(defn slot
  "The slot of position p of sequence s, blocks of bs."
  ^long [s ^long bs ^long p]
  (+ (* (long (nth (:blocks s) (quot p bs))) bs) (rem p bs)))

(defn blocks-for
  "The blocks n positions take, blocks of bs."
  ^long [^long n ^long bs]
  (quot (+ n (dec bs)) bs))

(defn prepare
  "[a' s' slots copies]: sequence s (blocks of bs) made ready for n
   positions more, written next at the slots (a vector of n): s' holds
   len+n positions, every block they fall in held by s' alone. A last
   block s shares and has filled in part is copied: copies is a vector of
   [from to] block pairs, to copy (gen.array/copy-blocks) before the
   write. Loud when a's free blocks do not suffice (:blocks-exhausted), a
   unmade."
  [a s ^long n ^long bs]
  (let [{:keys [blocks len]} s
        len (long len)
        need (- (blocks-for (+ len n) bs) (count blocks))
        partial? (pos? (rem len bs))
        last-id (peek blocks)
        cow? (and partial? (pos? n) (< 1 (holders a last-id)))
        [a ids] (allocate a (+ need (if cow? 1 0)))
        [a blocks copies] (if cow?
                            [(release a [last-id]) (conj (pop blocks) (first ids)) [[last-id (first ids)]]]
                            [a blocks []])
        s' {:blocks (into blocks (if cow? (rest ids) ids)) :len (+ len n)}]
    [a s' (mapv #(slot s' bs %) (range len (+ len n))) copies]))

(defn table-row
  "s's blocks as a block table's row of width (a vector), padded with
   pad."
  [s width pad]
  (let [bs (:blocks s)]
    (when (< (long width) (count bs))
      (throw (invalid (str "a table of " width " blocks cannot hold a sequence of " (count bs)) {:width width :blocks (count bs)})))
    (into bs (repeat (- (long width) (count bs)) pad))))
