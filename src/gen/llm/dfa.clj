(ns gen.llm.dfa
  "Deterministic finite automata over bytes, the languages of
   grammar-constrained decoding (gen.llm.grammar). gen.llm.regex builds them
   from regular expressions; (dfa spec) from a transition map.

     (dfa {:start :accept :transitions})  → a DFA ({state {byte state}})
     (start d), (size d)                  → its start state, its number of states
     (step d state byte)                  → the state after one byte, or -1
     (walk d state bytes)                 → the state after a byte array, or -1
     (accepting? d state)                 → is the language's string complete there?
     (matches? d bytes)                   → is a byte array in the language?
     (->data d)                           → the transition map

   A DFA is trimmed, minimal and canonical: every state is reached from the
   start and reaches an accepting state, so a byte with no live successor
   leads to -1 (dead) and a walk that is not -1 has read a prefix of the
   language; no two states have the same language; and states are numbered
   in breadth-first order from the start, bytes in order. So two DFAs of the
   same language are equal (=), and the empty language's has no states, its
   start -1. table gives the transitions, for the hot loops of
   gen.llm.grammar: state s's successor on byte b at s * 256 + b."
  (:import (java.nio IntBuffer)
           (java.util Arrays)))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

(deftype Dfa [^long start ^ints table ^booleans accept]
  Object
  (equals [this o]
    (or (identical? this o)
        (and (instance? Dfa o)
             (== start (.-start ^Dfa o))
             (Arrays/equals table ^ints (.-table ^Dfa o))
             (Arrays/equals accept ^booleans (.-accept ^Dfa o)))))
  (hashCode [_] (unchecked-int (+ (* 31 (Arrays/hashCode table)) (Arrays/hashCode accept))))
  (toString [_]
    (str "#<DFA " (alength accept) " states, "
         (areduce accept s n 0 (if (aget accept s) (inc n) n)) " accepting>")))

(defmethod print-method Dfa [d ^java.io.Writer w]
  (.write w (str d)))

(defn dfa? [x] (instance? Dfa x))

(defn start
  "The start state (-1 for the empty language)."
  ^long [^Dfa d]
  (.-start d))

(defn size
  "The number of states (0 for the empty language)."
  ^long [^Dfa d]
  (alength ^booleans (.-accept d)))

(defn table
  "The transitions: state s's successor on byte b at (+ (* s 256) b), -1
   for dead. Read it, never write it."
  ^ints [^Dfa d]
  (.-table d))

(defn accepting?
  "Is state s accepting (the bytes read so far a string of the language)?
   False for -1."
  [^Dfa d ^long s]
  (and (<= 0 s) (aget ^booleans (.-accept d) s)))

(defn step
  "The state after byte b (an integer 0 to 255, or a byte) from state s, or
   -1 (dead). -1 stays -1."
  ^long [^Dfa d ^long s ^long b]
  (if (neg? s) -1 (aget ^ints (.-table d) (+ (* s 256) (bit-and b 0xff)))))

(defn walk
  "The state after the bytes of bs (a byte array) from state s, or -1."
  ^long [^Dfa d ^long s ^bytes bs]
  (let [^ints t (.-table d)
        n (alength bs)]
    (loop [s s i 0]
      (if (or (neg? s) (== i n))
        s
        (recur (aget t (+ (* s 256) (bit-and (long (aget bs i)) 0xff))) (inc i))))))

(defn matches?
  "Is the byte array bs a string of d's language?"
  [^Dfa d ^bytes bs]
  (accepting? d (walk d (start d) bs)))

;; ---------------------------------------------------------------------------
;; Construction: minimal and canonical
;; ---------------------------------------------------------------------------

(defn- byte-classes
  "The bytes grouped by their columns of table (n states): bytes in one
   class lead every state to the same successor. Vectors of bytes, in
   order, ordered by their first byte."
  [^ints table ^long n]
  (let [column (fn [^long b]
                 (let [col (int-array n)]
                   (dotimes [s n] (aset col s (aget table (+ (* s 256) b))))
                   (IntBuffer/wrap col)))]
    (sort-by first (vals (group-by column (range 256))))))

(defn- partition-states
  "The coarsest partition of the states 0 … n - 1 of a complete automaton
   (delta holds state s's successor on class c at s * k + c) that separates
   accepting states from the others and is stable under every class, by
   Hopcroft's algorithm: each state's block, an int array. States share a
   block exactly when their languages are equal."
  ^ints [^ints delta ^long n ^long k ^booleans accept]
  (let [;; the states s with delta(s, c) = t: inv[off[c n + t] … off[c n + t + 1])
        off (int-array (inc (* k n)))
        inv (int-array (* k n))
        _ (dotimes [s n]
            (dotimes [c k]
              (let [i (inc (+ (* c n) (aget delta (+ (* s k) c))))] (aset off i (inc (aget off i))))))
        _ (dotimes [i (* k n)] (aset off (inc i) (+ (aget off (inc i)) (aget off i))))
        fill (aclone off)
        _ (dotimes [s n]
            (dotimes [c k]
              (let [i (+ (* c n) (aget delta (+ (* s k) c)))]
                (aset inv (aget fill i) (int s))
                (aset fill i (inc (aget fill i))))))
        ;; block b's states are elems[lo[b] … hi[b]), the first marked[b] of
        ;; them marked by the current splitter
        elems (int-array (sort-by #(if (aget accept (long %)) 0 1) (range n)))
        where (int-array n)
        block (int-array n)
        lo (int-array n)
        hi (int-array n)
        marked (int-array n)
        waiting (boolean-array n)
        n-accept (count (filter #(aget accept (long %)) (range n)))
        blocks (if (or (zero? n-accept) (== n-accept n)) 1 2)]
    (dotimes [i n]
      (let [s (aget elems i)]
        (aset where s (int i))
        (aset block s (int (if (and (== blocks 2) (<= n-accept i)) 1 0)))))
    (aset hi 0 (int (if (== blocks 2) n-accept n)))
    (when (== blocks 2) (aset lo 1 (int n-accept)) (aset hi 1 (int n)))
    (dotimes [b blocks] (aset waiting b true))
    (loop [todo (vec (range blocks)) blocks blocks]
      (if-let [a (peek todo)]
        (let [splitter (Arrays/copyOfRange elems (aget lo (long a)) (aget hi (long a)))
              _ (aset waiting (long a) false)
              [todo blocks]
              (reduce
               (fn [[todo ^long blocks] ^long c]
                 ;; mark the predecessors on class c, moving each to the front of its block
                 (let [touched (reduce
                                (fn [touched t]
                                  (let [i (+ (* c n) (long t))]
                                    (reduce (fn [touched ^long j]
                                              (let [s (aget inv j)
                                                    b (aget block s)
                                                    front (+ (aget lo b) (aget marked b))]
                                                (if (< (aget where s) front)
                                                  touched
                                                  (let [u (aget elems front)]
                                                    (aset elems (aget where s) u)
                                                    (aset where u (aget where s))
                                                    (aset elems front s)
                                                    (aset where s (int front))
                                                    (aset marked b (inc (aget marked b)))
                                                    (if (== 1 (aget marked b)) (conj touched b) touched)))))
                                            touched (range (aget off i) (aget off (inc i))))))
                                [] splitter)]
                   ;; split each touched block into its marked part (a new block) and the rest
                   (reduce (fn [[todo ^long blocks] ^long b]
                             (let [m (aget marked b)]
                               (aset marked b 0)
                               (if (== m (- (aget hi b) (aget lo b)))
                                 [todo blocks]
                                 (let [new blocks]
                                   (aset lo new (aget lo b))
                                   (aset hi new (int (+ (aget lo b) m)))
                                   (aset lo b (aget hi new))
                                   (dotimes [i m] (aset block (aget elems (+ (aget lo new) i)) (int new)))
                                   (let [smaller (if (<= m (- (aget hi b) (aget lo b))) new b)
                                         add (if (aget waiting b) new smaller)]
                                     (aset waiting add true)
                                     [(conj todo add) (inc blocks)])))))
                           [todo blocks] touched)))
               [(pop todo) blocks] (range k))]
          (recur todo (long blocks)))
        block))))

(defn from-table
  "The minimal, canonical DFA of an automaton of n states (any: perhaps
   with unreachable, dead or equivalent states). table holds state s's
   successor on byte b at s * 256 + b (-1 for none), accept the accepting
   states, start the start state (-1 for none)."
  [^long start ^ints table ^booleans accept]
  (let [n (alength accept)]
    (when-not (and (== (alength table) (* n 256)) (< start n) (<= -1 start)
                   (every? #(< -2 (aget table (long %)) n) (range (alength table))))
      (throw (ex-info "dfa: a table of 256 successors (states, or -1) for each state, and a start among the states"
                      {:reason :invalid-dfa :states n :table (alength table) :start start})))
    ;; complete it with a sink (state n), the dead state, then merge states
    ;; of equal languages: the sink's block is every dead state's
    (let [sink n
          classes (byte-classes table n)
          k (count classes)
          delta (int-array (* (inc n) k) (int sink))
          _ (doseq [s (range n), [^long c bytes] (map-indexed vector classes)]
              (let [s' (aget table (+ (* (long s) 256) (long (first bytes))))]
                (aset delta (+ (* (long s) k) c) (int (if (neg? s') sink s')))))
          ^ints block (partition-states delta (inc n) k (boolean-array (concat accept [false])))
          dead (aget block sink)
          rep (reduce (fn [rep ^long s] (if (contains? rep (aget block s)) rep (assoc rep (aget block s) s)))
                      {} (range n))
          succ (fn ^long [^long b ^long byte]
                 (let [s' (aget table (+ (* (long (rep b)) 256) byte))]
                   (if (neg? s') dead (aget block s'))))
          first-block (if (neg? start) dead (aget block start))]
      (if (== first-block dead)
        (Dfa. -1 (int-array 0) (boolean-array 0))
        ;; number the blocks breadth-first from the start's, bytes in order
        (let [[order queue] (loop [order {first-block 0} queue [first-block] i 0]
                              (if (== i (count queue))
                                [order queue]
                                (let [[order queue] (reduce (fn [[order queue] ^long byte]
                                                              (let [b (long (succ (queue i) byte))]
                                                                (if (or (== b dead) (contains? order b))
                                                                  [order queue]
                                                                  [(assoc order b (count order)) (conj queue b)])))
                                                            [order queue] (range 256))]
                                  (recur order queue (inc i)))))
              m (count queue)
              out (int-array (* m 256))
              out-accept (boolean-array m)]
          (doseq [[^long i b] (map-indexed vector queue)]
            (aset out-accept i (aget accept (long (rep b))))
            (dotimes [byte 256]
              (let [b' (long (succ b byte))]
                (aset out (+ (* i 256) byte) (int (if (== b' dead) -1 (order b')))))))
          (Dfa. 0 out out-accept))))))

(defn dfa
  "The DFA of a transition map: {:start s, :accept #{states}, :transitions
   {state {byte state}}}, its states any values, its bytes integers 0 to
   255; a missing transition is dead. Trimmed, minimized and numbered
   canonically (see the namespace's docstring)."
  [{:keys [start accept transitions] :as spec}]
  (when-not (and (map? spec) (set? accept) (map? transitions) (every? map? (vals transitions)))
    (throw (ex-info "dfa: a map {:start s :accept #{states} :transitions {state {byte state}}}"
                    {:reason :invalid-dfa :spec spec})))
  (doseq [[s row] transitions, [b s'] row]
    (when-not (and (integer? b) (<= 0 b 255))
      (throw (ex-info (str "dfa: a transition's byte is an integer 0 to 255, not " (pr-str b))
                      {:reason :invalid-dfa :state s :byte b :to s'}))))
  (let [states (distinct (concat [start] accept (keys transitions) (mapcat vals (vals transitions))))
        index (zipmap states (range))
        n (count states)
        t (int-array (* n 256) -1)]
    (doseq [[s row] transitions, [b s'] row]
      (aset t (+ (* (long (index s)) 256) (long b)) (int (index s'))))
    (from-table (index start) t (boolean-array (map #(contains? accept %) states)))))

(defn ->data
  "d as a transition map ({:start :accept :transitions}, dfa's form)."
  [^Dfa d]
  (let [^ints t (.-table d)]
    {:start (start d)
     :accept (into #{} (filter #(accepting? d %)) (range (size d)))
     :transitions (into {} (map (fn [s] [s (into {} (keep (fn [b] (let [s' (aget t (+ (* (long s) 256) (long b)))]
                                                                   (when-not (neg? s') [b s']))))
                                           (range 256))]))
                        (range (size d)))}))
