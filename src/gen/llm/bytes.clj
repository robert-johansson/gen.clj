(ns gen.llm.bytes
  "A vocabulary's bytes: the token byte trie, which grammar masks walk
   (gen.llm.grammar).

     (byte-trie byte-vocab)       → the trie of the tokens' bytes (a vector of
                                    byte arrays by token, gen.llm.lm/byte-vocab's)
     (token-bytes trie t)         → token t's bytes (do not write them)
     (root trie), (child trie node b), (depth trie node)
     (tokens-at trie node)        → the tokens whose bytes end at node
     (tokens-under trie node)     → those whose bytes begin with node's path

   The trie is a few flat arrays, its nodes numbered in preorder with
   children in byte order, so node i's subtree is the nodes i to (end i) - 1
   and the tokens under it are one contiguous run: a walk over every token
   from a DFA state (a grammar's mask) visits each shared prefix once and
   skips a dead subtree in one jump, allocating nothing. Tokens with no
   bytes are kept apart (bare).")

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

;; Node i (0 the root, the empty path): edge[i] its last byte, depth[i] its
;; path's length, end[i] one past its subtree; the tokens ending at it are
;; tokens[first[i] … first[i + 1]), those under it tokens[first[i] …
;; first[end[i]]).
(deftype ByteTrie [^long n-vocab ^objects vocab ^bytes edge ^ints depth ^ints end ^ints first ^ints tokens
                   ^ints bare ^long max-depth]
  Object
  (toString [_] (str "#<ByteTrie " n-vocab " tokens, " (alength depth) " nodes>")))

(defmethod print-method ByteTrie [t ^java.io.Writer w] (.write w (str t)))

(defn byte-trie
  "The trie of byte-vocab's tokens (a vector of byte arrays, by token)."
  ^ByteTrie [byte-vocab]
  (when-not (and (sequential? byte-vocab) (seq byte-vocab) (every? bytes? byte-vocab))
    (throw (ex-info "byte-trie: a byte vocabulary is a non-empty vector of byte arrays, by token"
                    {:reason :invalid-vocab})))
  (let [vocab (object-array byte-vocab)
        n (alength vocab)
        len (fn [t] (alength ^bytes (aget vocab (long t))))
        order (sort (fn [a b] (let [c (java.util.Arrays/compareUnsigned ^bytes (aget vocab (long a)) ^bytes (aget vocab (long b)))]
                                (if (zero? c) (compare a b) c)))
                    (filter #(pos? (long (len %))) (range n)))
        cap (inc (long (reduce + 0 (map len order))))
        max-depth (long (reduce max 0 (map len order)))
        edge (byte-array cap)
        depth (int-array cap)
        end (int-array cap)
        first (int-array (inc cap))
        tokens (int-array (count order))
        path (int-array (inc max-depth))]
    ;; tokens in byte order are the trie's preorder: each token opens the
    ;; nodes past its common prefix with the one before, and closes the
    ;; nodes of that one deeper than it
    (loop [ts (seq order) nodes 1 held 0 ^bytes prev (byte-array 0)]
      (if-let [t (clojure.core/first ts)]
        (let [^bytes bs (aget vocab (long t))
              m (java.util.Arrays/mismatch prev bs)
              common (if (neg? m) (alength bs) m)]
          (loop [d (alength prev)] (when (< common d) (aset end (aget path d) (int nodes)) (recur (dec d))))
          (dotimes [i (- (alength bs) common)]
            (let [d (+ common i 1)
                  node (+ nodes i)]
              (aset edge node (aget bs (dec d)))
              (aset depth node (int d))
              (aset first node (int held))
              (aset path d (int node))))
          (aset tokens held (int t))
          (recur (next ts) (+ nodes (- (alength bs) common)) (inc held) bs))
        (do (loop [d (alength prev)] (when (pos? d) (aset end (aget path d) (int nodes)) (recur (dec d))))
            (aset end 0 (int nodes))
            (aset first nodes (int held))
            (ByteTrie. n (object-array (map #(aclone ^bytes %) byte-vocab))
                       (java.util.Arrays/copyOf edge (int nodes)) (java.util.Arrays/copyOf depth (int nodes))
                       (java.util.Arrays/copyOf end (int nodes)) (java.util.Arrays/copyOf first (int (inc nodes)))
                       tokens (int-array (filter #(zero? (long (len %))) (range n))) max-depth))))))

(defn n-vocab ^long [^ByteTrie trie] (.-n-vocab trie))

(defn token-bytes
  "Token t's bytes (the trie's own array: read it, never write it)."
  ^bytes [^ByteTrie trie ^long t]
  (aget ^objects (.-vocab trie) t))

(defn root ^long [_trie] 0)

(defn depth ^long [^ByteTrie trie ^long node] (aget ^ints (.-depth trie) node))

(defn child
  "The child of node by byte b, or -1."
  ^long [^ByteTrie trie ^long node ^long b]
  (let [^bytes edge (.-edge trie)
        ^ints end (.-end trie)
        stop (aget end node)
        b (bit-and b 0xff)]
    (loop [c (inc node)]
      (if (>= c stop)
        -1
        (let [e (bit-and (long (aget edge c)) 0xff)]
          (cond (== e b) c
                (< b e) -1
                :else (recur (aget end c))))))))

(defn children
  "node's children, in byte order: [byte node] pairs."
  [^ByteTrie trie ^long node]
  (let [^ints end (.-end trie)
        stop (aget end node)]
    (loop [c (inc node) out []]
      (if (< c stop)
        (recur (aget end c) (conj out [(bit-and (long (aget ^bytes (.-edge trie) c)) 0xff) c]))
        out))))

(defn tokens-at
  "The tokens whose bytes are node's path (a vector)."
  [^ByteTrie trie ^long node]
  (let [^ints first (.-first trie)]
    (mapv #(aget ^ints (.-tokens trie) (long %)) (range (aget first node) (aget first (inc node))))))

(defn tokens-under
  "The tokens whose bytes begin with node's path (a vector), the root's
   all but the bare ones."
  [^ByteTrie trie ^long node]
  (let [^ints first (.-first trie)]
    (mapv #(aget ^ints (.-tokens trie) (long %)) (range (aget first node) (aget first (aget ^ints (.-end trie) node))))))

(defn bare
  "The tokens with no bytes (a vector)."
  [^ByteTrie trie]
  (vec (.-bare trie)))

;; ---------------------------------------------------------------------------
;; The sweep: every token from a DFA state
;; ---------------------------------------------------------------------------

(defn sweep
  "The tokens whose bytes lead from state s0 of a DFA (its transition table,
   gen.llm.dfa/table) to a state ok holds: a bitset over the vocabulary
   (a long array, token t at bit t mod 64 of word t / 64). The states they
   lead to are marked in reached, when it is not nil. A walk over the trie
   that skips each subtree the DFA cannot enter."
  ^longs [^ByteTrie trie ^ints table s0 ^booleans ok ^booleans reached]
  (let [s0 (long s0)
        ^bytes edge (.-edge trie)
        ^ints depth (.-depth trie)
        ^ints end (.-end trie)
        ^ints first (.-first trie)
        ^ints tokens (.-tokens trie)
        n-nodes (alength depth)
        mask (long-array (quot (+ (.-n-vocab trie) 63) 64))
        stack (int-array (inc (.-max-depth trie)))]
    (aset stack 0 (int s0))
    (when (aget ok s0)
      (doseq [t (.-bare trie)]
        (when reached (aset reached s0 true))
        (aset mask (bit-shift-right (long t) 6)
              (bit-or (aget mask (bit-shift-right (long t) 6)) (bit-shift-left 1 (bit-and (long t) 63))))))
    (loop [i 1]
      (when (< i n-nodes)
        (let [d (aget depth i)
              s (aget table (+ (* (aget stack (dec d)) 256) (bit-and (long (aget edge i)) 0xff)))]
          (if (neg? s)
            (recur (aget end i))
            (do (aset stack d s)
                (when (aget ok s)
                  (let [lo (aget first i) hi (aget first (inc i))]
                    (when (and reached (< lo hi)) (aset reached s true))
                    (loop [j lo]
                      (when (< j hi)
                        (let [t (aget tokens j)
                              w (bit-shift-right t 6)]
                          (aset mask w (bit-or (aget mask w) (bit-shift-left 1 (bit-and t 63))))
                          (recur (inc j)))))))
                (recur (inc i)))))))
    mask))
