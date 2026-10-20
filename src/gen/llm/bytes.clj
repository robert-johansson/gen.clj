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
   bytes are kept apart (bare).

   And a text's probability under an LM, marginalizing over the ways
   tokens spell it (as GenMLX's bytes.cljs and genlm-bytes):

     (byte-lm lm opts)                    → lm read as bytes (opts :end-tokens, :beam)
     (prefix-logprob b prompt text)       → log P(the text after prompt begins with text)
     (text-logprob b prompt text)         → log P(the text is text, then ends)
     (next-byte-logprobs b prompt text)   → each next byte's, and the end's"
  (:require [gen.llm.lm :as lm]))

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

;; ---------------------------------------------------------------------------
;; A text's probability, over its tokenizations
;; ---------------------------------------------------------------------------

(defn log-mass
  "log Σ exp(row[t]) over the tokens under node (-Inf for none): a
   contiguous run of the trie's tokens."
  ^double [^ByteTrie trie ^floats row ^long node]
  (let [^ints tokens (.-tokens trie)
        ^ints first (.-first trie)
        lo (aget first node)
        hi (aget first (aget ^ints (.-end trie) node))]
    ;; one pass: the running maximum m, and the sum of exp(x - m)
    (loop [j lo m Double/NEGATIVE_INFINITY s 0.0]
      (if (< j hi)
        (let [x (double (aget row (aget tokens j)))]
          (cond (Double/isInfinite x) (recur (inc j) m s)
                (<= x m) (recur (inc j) m (+ s (Math/exp (- x m))))
                :else (recur (inc j) x (+ (* s (Math/exp (- m x))) 1.0))))
        (if (Double/isInfinite m) m (+ m (Math/log s)))))))

(defn- log-add ^double [^double a ^double b]
  (cond (Double/isInfinite a) (if (pos? a) a b)
        (Double/isInfinite b) (if (pos? b) b a)
        :else (let [m (Math/max a b)] (+ m (Math/log (+ (Math/exp (- a m)) (Math/exp (- b m))))))))

(defrecord ByteLM [lm trie end-tokens beam])

(defmethod print-method ByteLM [b ^java.io.Writer w]
  (.write w (str "#<ByteLM " (:lm b) (when (:beam b) (str ", beam " (:beam b))) ">")))

(defn byte-lm
  "lm's text as bytes, marginalizing over tokenizations: for
   prefix-logprob, text-logprob and next-byte-logprobs. opts: :end-tokens,
   the tokens that end a text (default the LM's :eog; gen.llm.token's
   completion's :end-tokens); :beam, at most this many tokenizations kept at each
   token boundary, the most probable (default nil: every one, exact).
   Tokens with no bytes are never in a text (they would let a text have
   infinitely many tokenizations); with a beam, probabilities are lower
   bounds."
  ([lm] (byte-lm lm {}))
  ([lm {:keys [end-tokens beam] :as opts}]
   (let [{:keys [n-vocab eog]} (lm/vocab lm)
         end-tokens (if (contains? opts :end-tokens) end-tokens eog)
         byte-vocab (lm/byte-vocab lm)]
     (when-not (and (set? end-tokens) (every? #(and (integer? %) (< -1 (long %) (long n-vocab))) end-tokens))
       (throw (ex-info (str "byte-lm: :end-tokens is a set of tokens, not " (pr-str end-tokens))
                       {:reason :invalid-end-tokens :end-tokens end-tokens})))
     (when-not (or (nil? beam) (pos-int? beam))
       (throw (ex-info (str "byte-lm: :beam is a positive integer or nil, not " (pr-str beam)) {:reason :invalid-option :beam beam})))
     ;; end tokens are kept out of the trie: they end a text, and hold none of it
     (->ByteLM lm (byte-trie (vec (map-indexed (fn [t bs] (if (end-tokens t) (byte-array 0) bs)) byte-vocab)))
               (into #{} (map long) end-tokens) beam))))

(defn- text-bytes ^bytes [what text]
  (cond (string? text) (.getBytes ^String text java.nio.charset.StandardCharsets/UTF_8)
        (bytes? text) text
        :else (throw (ex-info (str what ": a text is a string or a byte array, not " (pr-str text))
                              {:reason :invalid-text :text text}))))

(defn- walk-text
  "Along b from byte j, from the root: the tokens that end within b, as [k
   t] with k the boundary after t, and the node at b's end (-1 when b's
   bytes from j leave the trie)."
  [^ByteTrie trie ^bytes b ^long j]
  (let [n (alength b)]
    (loop [node 0 k j splits []]
      (if (== k n)
        [splits node]
        (let [c (child trie node (aget b k))]
          (if (neg? c)
            [splits -1]
            (recur c (inc k) (into splits (map (fn [t] [(inc k) t])) (tokens-at trie c)))))))))

(defn- log-sum ^double [^floats row ts]
  (reduce (fn [^double acc t] (log-add acc (double (aget row (int t))))) Double/NEGATIVE_INFINITY ts))

(defn- marginals
  "For the bytes b after prompt, over every tokenization (or the beam's):
   {:prefix log P(the text begins with b), :next log P(it begins with b x)
   by byte x (a double array), :end log P(it is b, ended)}. A tokenization
   of a text that begins with b is a run of tokens spelling b's first j
   bytes, j a token boundary inside b, and then a token that begins with
   the rest of b; the runs are walked boundary by boundary, each
   boundary's contexts read in one batch."
  [{:keys [lm trie end-tokens beam]} prompt ^bytes b]
  (let [n (alength b)
        next (double-array 256 Double/NEGATIVE_INFINITY)]
    (loop [j 0 at {0 [[[] 0.0]]} prefix (if (zero? n) 0.0 Double/NEGATIVE_INFINITY) end Double/NEGATIVE_INFINITY]
      (if (< n j)
        {:prefix prefix :next next :end end}
        (let [particles (cond->> (get at j []) beam (sort-by second >) beam (take beam))
              rows (when (seq particles) (lm/batch-next-logprobs lm (mapv #(into prompt (first %)) particles)))
              [splits node] (walk-text trie b j)
              node (long node)
              [at prefix end]
              (reduce (fn [[at ^double prefix ^double end] [[ctx ^double lp] ^floats row]]
                        (when-not (neg? node)
                          (doseq [[x c] (children trie node)]
                            (aset next (long x) (log-add (aget next (long x)) (+ lp (log-mass trie row c))))))
                        [(reduce (fn [at [k t]] (update at k (fnil conj []) [(conj ctx t) (+ lp (aget row (int t)))]))
                                 at splits)
                         (if (and (< j n) (<= 0 node)) (log-add prefix (+ lp (log-mass trie row node))) prefix)
                         (if (== j n) (log-add end (+ lp (log-sum row end-tokens))) end)])
                      [at prefix end] (map vector particles rows))]
          (recur (inc j) (dissoc at j) (double prefix) (double end)))))))

(defn- prompt-vector [what prompt]
  (when-not (and (sequential? prompt) (seq prompt))
    (throw (ex-info (str what ": the prompt is a non-empty vector of tokens, not " (pr-str prompt))
                    {:reason :invalid-token :prompt prompt})))
  (vec prompt))

(defn prefix-logprob
  "log P(the text after prompt begins with text, a string or byte array),
   over its tokenizations: the LM's probability that a completion's
   bytes, before its end token, begin with text's."
  ^double [bm prompt text]
  (:prefix (marginals bm (prompt-vector "prefix-logprob" prompt) (text-bytes "prefix-logprob" text))))

(defn text-logprob
  "log P(the text after prompt is text, then an end token), over its
   tokenizations."
  ^double [bm prompt text]
  (:end (marginals bm (prompt-vector "text-logprob" prompt) (text-bytes "text-logprob" text))))

(defn next-byte-logprobs
  "The next byte's log-probabilities after prompt and text, over their
   tokenizations: a double array of 257 entries, byte x at x and the end
   (the text ends after text) last. They sum to 1 when every token has
   bytes and every beam is wide enough. Loud when text has probability 0
   (:zero-prefix)."
  ^doubles [bm prompt text]
  (let [{:keys [prefix ^doubles next end]} (marginals bm (prompt-vector "next-byte-logprobs" prompt)
                                                      (text-bytes "next-byte-logprobs" text))
        prefix (double prefix)
        out (double-array 257)]
    (when (Double/isInfinite prefix)
      (throw (ex-info "next-byte-logprobs: the text has probability 0" {:reason :zero-prefix :text text})))
    (dotimes [x 256] (aset out x (- (aget next x) prefix)))
    (aset out 256 (- (double end) prefix))
    out))
