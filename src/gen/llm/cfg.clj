(ns gen.llm.cfg
  "Context-free grammars over bytes, recognized by Earley's algorithm byte
   by byte, as potentials: constraints that a regular language cannot
   state (balanced brackets, recursive JSON, code's nesting).

     (cfg {:start :s :rules {:s [[\"(\" :s \")\"] []]}})  → a grammar
     (regex r)                                      → an element: r's language (a regex
                                                      or a gen.llm.regex term)
     (recognizes? g text), (prefix? g text)         → is text a string of g's language,
                                                      a prefix of one?
     (potential vocab g), (potential vocab g opts)  → g read through vocab's tokens
                                                      (an LM, or a byte vocabulary)

   A grammar is a map: :start, a rule's name, and :rules, a map from each
   rule's name (a keyword) to its alternatives, each a vector of elements:
   a keyword (a rule), a string (its UTF-8 bytes), or (regex r). [] is the
   empty string. Rules that derive no string are dropped (an unproductive
   start is :empty-language); a reference to no rule is loud.

   The recognizer is Earley's (with Aycock and Horspool's handling of
   nullable rules), advanced one byte at a time: its Earley set after a
   text's bytes is a state that a next byte extends, as a DFA's state is.
   A regex element is one terminal, a lexeme its minimal DFA scans: an
   item before it carries the DFA's state while the lexeme's bytes go by,
   and moves past it wherever the DFA accepts, so a string's bytes add no
   items. Since every rule left derives some string, a set that a byte
   leaves non-empty is a prefix of some string of the language: prefix? is
   exact.

   The sets are a lazy DFA: each keeps its successor by each byte once it
   is built, and a successor built from the same items over the same chart
   (the sets that predict, which completions read) is the same set. So the
   sets inside a lexeme (a string's body) are a few, met again and again,
   and a byte there costs a table lookup. The sets live for one call: no
   state is kept between calls (rule 3).

   As a potential (gen.llm.potential), a grammar weighs a context 1 (log
   0) as a prefix when its text is a prefix of the language and as a
   complete sequence when the text is in it, 0 (-Inf) otherwise, its end
   tokens leaving the text (as gen.llm.grammar's: :end-tokens, default an
   LM's :eog, none for a byte vocabulary). logw-next is the context's
   mask, in parts: the tokens that stay inside a lexeme the context is in
   are read from masks the potential makes with the token byte trie (one
   per state of each lexeme's DFA, gen.llm.bytes/sweep-split); the tokens
   that may end the lexeme and go on, and those that begin with a byte an
   item waits for, are swept, the set carried down the trie, a successor
   read per node visited. So inside a string a mask costs about what one
   at a value's start does. prefix and complete recognize the context's
   text from the start, so their cost grows with the text's structure (a
   lexeme's bytes cost a lookup each). Unlike a grammar's exact masks,
   liveness is the bytes': a token may lead to a prefix that no sequence
   of the vocabulary's tokens completes, when the vocabulary lacks bytes
   the language needs (one with every byte as a token, as byte-level
   BPE's, never strands). For a large vocabulary token SMC's :awrs
   proposal checks only the tokens it draws, where a mask reads them all."
  {:api :experimental}
  (:require [gen.llm.bytes :as bytes]
            [gen.llm.dfa :as dfa]
            [gen.llm.lm :as lm]
            [gen.llm.potential :as potential]
            [gen.llm.regex :as regex])
  (:import (java.nio.charset StandardCharsets)
           (java.util ArrayList Arrays HashMap HashSet)))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

;; ---------------------------------------------------------------------------
;; Grammars: rules over bytes
;; ---------------------------------------------------------------------------

(defrecord ^:private Regex [dfa])

(defn regex
  "An element of a rule: r's language (a regex, as gen.llm.regex/parse
   reads it, a gen.llm.regex term, or a DFA)."
  [r]
  (->Regex (if (dfa/dfa? r) r (regex/dfa r))))

;; A compiled grammar: nonterminals 0 … n-1 (n-1 the augmented start, its
;; one rule start' → start); terminals 0 … m-1, each a byte (term-byte, 0
;; to 255) or a lexeme (term-byte -1): a DFA's transitions (lex-table),
;; accepting states (lex-accept) and start (lex-start). An Earley item's
;; rule and dot are one number, a position: the rules' dotted positions
;; numbered rule by rule from 0 (start' → • start), a rule of k symbols
;; holding k + 1 (before each symbol, and its end), so moving the dot is
;; adding 1 and no rule's length is bounded. sym is the symbol after each
;; position (a nonterminal k ≥ 0, a terminal j as -(j + 1), or end),
;; lhs-at its rule's nonterminal, starts the first positions of each
;; nonterminal's rules; nullable by nonterminal (a lexeme whose DFA accepts
;; at its start is nullable too).
(def ^:private ^:const end Integer/MIN_VALUE)

(deftype Grammar [^ints sym ^ints lhs-at ^objects starts ^booleans nullable
                  ^ints term-byte ^objects lex-table ^objects lex-accept ^ints lex-start
                  ^long n-rules names]
  Object
  (toString [_] (str "#<CFG " (count names) " rules, " n-rules " productions>")))

(defmethod print-method Grammar [g ^java.io.Writer w] (.write w (str g)))

(defn cfg? [x] (instance? Grammar x))

(defn- utf8 ^bytes [^String s] (.getBytes s StandardCharsets/UTF_8))

(defn- check-rules [{:keys [start rules] :as g}]
  (when-not (and (map? g) (map? rules) (seq rules) (every? keyword? (keys rules)))
    (throw (ex-info "cfg: a grammar is a map of :start and :rules, a non-empty map of rule names (keywords) to alternatives"
                    {:reason :invalid-grammar :grammar g})))
  (when-not (contains? rules start)
    (throw (ex-info (str "cfg: the start " (pr-str start) " is no rule") {:reason :invalid-grammar :start start})))
  (doseq [[k alts] rules]
    (when-not (and (sequential? alts) (every? vector? alts))
      (throw (ex-info (str "cfg: the rule " k "'s alternatives are a sequence of vectors of elements")
                      {:reason :invalid-grammar :rule k})))
    (doseq [alt alts, e alt]
      (cond (keyword? e) (when-not (contains? rules e)
                           (throw (ex-info (str "cfg: the rule " k " refers to " e ", no rule") {:reason :invalid-grammar :rule k :ref e})))
            (string? e) nil
            (instance? Regex e) nil
            :else (throw (ex-info (str "cfg: an element is a rule (keyword), a string or (regex r), not " (pr-str e))
                                  {:reason :invalid-grammar :rule k :element e}))))))

(defn- productive
  "The rules (names) of rules that derive some string (keywords their
   rules, any other symbol a terminal): a fixpoint."
  [rules]
  (loop [known #{}]
    (let [known' (into known (keep (fn [[k alts]]
                                     (when (some (fn [alt] (every? #(or (not (keyword? %)) (known %)) alt)) alts) k)))
                       rules)]
      (if (= known known') known (recur known')))))

(defn cfg
  "The grammar of g, a map of :start and :rules (see the namespace's
   docstring), compiled: rules that derive no string dropped, a regex
   element a lexeme of its DFA. Loud on a malformed grammar
   (:invalid-grammar) and on an empty language (:empty-language)."
  [g]
  (check-rules g)
  (let [;; elements as rule names, [::byte b] and [::lexeme dfa] terminals; a
        ;; regex of the empty language is ::nothing, which no rule derives
        expand (fn [alt]
                 (into [] (mapcat (fn [e]
                                    (cond (keyword? e) [e]
                                          (string? e) (map (fn [b] [::byte (bit-and (long b) 0xff)]) (utf8 e))
                                          (neg? (dfa/start (:dfa e))) [::nothing]
                                          :else [[::lexeme (:dfa e)]])))
                       alt))
        rules (into {} (map (fn [[k alts]] [k (mapv expand alts)])) (:rules g))
        live (productive rules)
        _ (when-not (live (:start g))
            (throw (ex-info "cfg: the language is empty (the start derives no string)" {:reason :empty-language})))
        rules (into {} (keep (fn [[k alts]]
                               (when (live k)
                                 [k (filterv (fn [alt] (every? #(or (not (keyword? %)) (live %)) alt)) alts)])))
                    rules)
        names (vec (sort-by pr-str (keys rules)))
        index (zipmap names (range))
        start' (count names)
        terminals (ArrayList.)
        term-index (HashMap.)
        symbol (fn [x]
                 (if (keyword? x)
                   (long (index x))
                   (let [j (or (.get term-index x) (let [j (.size terminals)] (.add terminals x) (.put term-index x j) j))]
                     (- (inc (long j))))))
        prods (into [[start' [(index (:start g))]]]
                    (for [k names, alt (rules k)] [(index k) (mapv symbol alt)]))
        n (inc start')
        ;; positions: rule r's first is at (offsets r), its end k after it
        offsets (vec (reductions + 0 (map #(inc (count (second %))) prods)))
        sym (int-array (peek offsets))
        lhs-at (int-array (peek offsets))
        _ (doseq [[[l alt] o] (map vector prods offsets)
                  :let [o (long o)]]
            (dotimes [j (inc (count alt))]
              (aset sym (+ o j) (int (if (< j (count alt)) (long (nth alt j)) end)))
              (aset lhs-at (+ o j) (int l))))
        starts (object-array (for [k (range n)]
                               (int-array (keep (fn [[[l] o]] (when (== (long l) (long k)) o)) (map vector prods offsets)))))
        terms (vec terminals)
        lexeme? (fn [[kind]] (= kind ::lexeme))
        term-byte (int-array (map (fn [[_ x :as t]] (if (lexeme? t) -1 (long x))) terms))
        lex-table (object-array (map (fn [[_ d :as t]] (when (lexeme? t) (dfa/table d))) terms))
        lex-accept (object-array (map (fn [[_ d :as t]]
                                        (when (lexeme? t) (boolean-array (map #(dfa/accepting? d %) (range (dfa/size d))))))
                                      terms))
        lex-start (int-array (map (fn [[_ d :as t]] (if (lexeme? t) (dfa/start d) -1)) terms))
        nullable (boolean-array n)
        nullable-symbol? (fn [s]
                           (let [s (long s)]
                             (if (neg? s)
                               (let [t (- (inc s))]
                                 (and (lexeme? (terms t)) (dfa/accepting? (second (terms t)) (aget lex-start t))))
                               (aget nullable s))))]
    ;; nullable: a fixpoint over the productions
    (loop []
      (let [changed (reduce (fn [changed [l alt]]
                              (if (and (not (aget nullable (long l))) (every? nullable-symbol? alt))
                                (do (aset nullable (long l) true) true)
                                changed))
                            false prods)]
        (when changed (recur))))
    (Grammar. sym lhs-at starts nullable term-byte lex-table lex-accept lex-start (count prods) names)))

;; ---------------------------------------------------------------------------
;; Earley sets
;; ---------------------------------------------------------------------------

;; An item: its origin (the high 32 bits) and its position (the low 32), as
;; a long; the item past its next symbol is (inc it). A lexing item is an
;; item before a lexeme with its DFA's state, a pair.
(defn- item ^long [^long origin ^long pos] (bit-or (bit-shift-left origin 32) pos))
(defn- origin-of ^long [^long it] (unsigned-bit-shift-right it 32))
(defn- pos-of ^long [^long it] (bit-and it 0xFFFFFFFF))

;; The chart: the sets that predict (anchors), by origin, each as its items
;; waiting for each nonterminal (a map from the nonterminal to a long
;; array), which a completion reads; and the sets built over it so far,
;; canon, by the items they were built from. A set that predicts nothing
;; is never an origin, so it is no anchor and its successors share its
;; chart.
(deftype Chart [anchors ^HashMap canon])

;; A set: its items before a byte (scan), its lexing items (lex, lex-at),
;; whether the start is complete from 0 in it, the chart its successors
;; build on (its own, past it, when it predicts), and its successors by
;; byte, filled as they are asked for (a set, dead, or nil: not yet).
(deftype EarleySet [^longs scan ^longs lex ^ints lex-at ^boolean accepts ^Chart chart ^objects next])

(def ^:private dead (Object.))

;; A set's key in its chart's canon: the items it is built from, as a long
;; array (the count of seeds, the seeds, the lexing pairs).
(deftype Key [^longs a ^int h]
  Object
  (hashCode [_] h)
  (equals [_ o] (and (instance? Key o) (Arrays/equals a ^longs (.-a ^Key o)))))

(defn- key-of ^Key [^longs a] (Key. a (Arrays/hashCode a)))

(defn- build
  "The Earley set built over chart from seeds (items past a scanned byte
   or an ended lexeme) and lexing pairs (cont, flat: item, state, …): the
   closure by prediction and completion. A set that predicts is an anchor:
   its successors' chart holds its waiting items."
  ^EarleySet [^Grammar g ^Chart chart ^longs seeds ^longs cont]
  (let [^ints sym (.-sym g)
        ^ints lhs-at (.-lhs-at g)
        ^objects starts (.-starts g)
        ^booleans nullable (.-nullable g)
        ^ints term-byte (.-term-byte g)
        ^objects lex-accept (.-lex-accept g)
        ^ints lex-start (.-lex-start g)
        anchors (.-anchors chart)
        i (count anchors)
        seen (HashSet.)
        queue (ArrayList.)
        predicted (HashSet.)
        waiting (HashMap.)
        scan (ArrayList.)
        lexed (HashSet.)
        lex (ArrayList.)
        lex-at (ArrayList.)
        add (fn [^long it] (when (.add seen it) (.add queue it)))
        add-lex (fn [^long it ^long q]
                  (when (.add lexed (key-of (doto (long-array 2) (aset 0 it) (aset 1 q))))
                    (.add lex it)
                    (.add lex-at q)))]
    (dotimes [j (alength seeds)] (add (aget seeds j)))
    (loop [j 0] (when (< j (alength cont)) (add-lex (aget cont j) (aget cont (inc j))) (recur (+ j 2))))
    (loop [k 0]
      (when (< k (.size queue))
        (let [it (long (.get queue k))
              p (pos-of it)
              s (aget sym p)]
          (cond
            ;; complete: advance the items of its origin's set waiting for its lhs
            (== s end) (let [o (origin-of it)]
                         (when-not (== o i)
                           (when-let [^longs ws (.get ^HashMap (nth anchors o) (aget lhs-at p))]
                             (dotimes [j (alength ws)] (add (inc (aget ws j)))))))
            (neg? s) (let [t (- (inc s))]
                       (if (neg? (aget term-byte t))
                         ;; a lexeme: lexing from its start, and past it now when it accepts there
                         (let [q (aget lex-start t)]
                           (add-lex it q)
                           (when (aget ^booleans (aget lex-accept t) q) (add (inc it))))
                         (.add scan it)))
            :else (do (let [^ArrayList l (or (.get waiting s) (let [l (ArrayList.)] (.put waiting s l) l))]
                        (.add l it))
                      (when (.add predicted s)
                        (let [^ints ps (aget starts s)]
                          (dotimes [j (alength ps)] (add (item i (aget ps j))))))
                      ;; Aycock and Horspool: a nullable nonterminal is also skipped
                      (when (aget nullable s) (add (inc it)))))
          (recur (inc k)))))
    (let [chart (if (.isEmpty waiting)
                  chart
                  (let [frozen (HashMap.)]
                    (doseq [[s ^ArrayList l] waiting] (.put frozen s (long-array l)))
                    (Chart. (conj anchors frozen) (HashMap.))))]
      (EarleySet. (long-array scan) (long-array lex) (int-array lex-at) (.contains seen (item 0 1))
                  chart (object-array 256)))))

(defn- start-set
  "The set of the empty text."
  ^EarleySet [^Grammar g]
  (build g (Chart. [] (HashMap.)) (long-array [(item 0 0)]) (long-array 0)))

(def ^:private ^java.util.Comparator pair-order
  (reify java.util.Comparator
    (compare [_ a b]
      (let [c (Long/compare (aget ^longs a 0) (aget ^longs b 0))]
        (if (zero? c) (Long/compare (aget ^longs a 1) (aget ^longs b 1)) c)))))

(defn- sorted-pairs
  "The pairs of xs (flat: a, b, …), sorted and without repeats, flat."
  ^longs [^ArrayList xs]
  (let [n (quot (.size xs) 2)
        ps (object-array n)
        out (ArrayList.)]
    (dotimes [k n]
      (aset ps k (doto (long-array 2) (aset 0 (long (.get xs (* 2 k)))) (aset 1 (long (.get xs (inc (* 2 k))))))))
    (Arrays/sort ps pair-order)
    (dotimes [k n]
      (let [^longs p (aget ps k)]
        (when-not (and (pos? k) (Arrays/equals p ^longs (aget ps (dec k))))
          (.add out (aget p 0))
          (.add out (aget p 1)))))
    (long-array out)))

(defn- successor
  "The set after byte b from s, built or found in its chart's canon, or nil
   when no item takes b (the text is no prefix of the language)."
  [^Grammar g ^EarleySet s ^long b]
  (let [^ints sym (.-sym g)
        ^ints term-byte (.-term-byte g)
        ^objects lex-table (.-lex-table g)
        ^objects lex-accept (.-lex-accept g)
        ^longs scan (.-scan s)
        ^longs lex (.-lex s)
        ^ints lex-at (.-lex-at s)
        seeds (ArrayList.)
        cont (ArrayList.)]
    (dotimes [j (alength scan)]
      (let [it (aget scan j)]
        (when (== b (aget term-byte (- (inc (aget sym (pos-of it))))))
          (.add seeds (inc it)))))
    (dotimes [j (alength lex)]
      (let [it (aget lex j)
            t (- (inc (aget sym (pos-of it))))
            q (aget ^ints (aget lex-table t) (+ (* (aget lex-at j) 256) b))]
        (when-not (neg? q)
          (.add cont it)
          (.add cont q)
          (when (aget ^booleans (aget lex-accept t) q) (.add seeds (inc it))))))
    (when-not (and (.isEmpty seeds) (.isEmpty cont))
      (let [seeds (let [a (long-array seeds)] (Arrays/sort a) (long-array (distinct a)))
            cont (sorted-pairs cont)
            k (key-of (let [a (long-array (+ 1 (alength seeds) (alength cont)))]
                        (aset a 0 (alength seeds))
                        (System/arraycopy seeds 0 a 1 (alength seeds))
                        (System/arraycopy cont 0 a (inc (alength seeds)) (alength cont))
                        a))
            ^Chart chart (.-chart s)
            ^HashMap canon (.-canon chart)]
        (or (.get canon k)
            (let [n (build g chart seeds cont)]
              (.put canon k n)
              n))))))

(defn- step
  "The set after byte b (0-255) from s, or nil: s's successor, built once."
  [^Grammar g ^EarleySet s ^long b]
  (let [^objects next (.-next s)
        n (aget next b)]
    (cond (nil? n) (let [n (successor g s b)] (aset next b (if (nil? n) dead n)) n)
          (identical? n dead) nil
          :else n)))

(defn- set-of
  "The set after bytes (a byte array) from s, or nil when they are no
   prefix."
  [g s ^bytes bs]
  (loop [s s i 0]
    (cond (nil? s) nil
          (== i (alength bs)) s
          :else (recur (step g s (bit-and (long (aget bs i)) 0xff)) (inc i)))))

(defn- accepts? [s] (.-accepts ^EarleySet s))

(defn- check-grammar [what g]
  (when-not (cfg? g)
    (throw (ex-info (str what ": a grammar is made by gen.llm.cfg/cfg, not " (pr-str g)) {:reason :invalid-grammar}))))

(defn recognizes?
  "Is text (a string, or a byte array) a string of g's language?"
  [g text]
  (check-grammar "recognizes?" g)
  (boolean (some-> (set-of g (start-set g) (if (string? text) (utf8 text) text)) accepts?)))

(defn prefix?
  "Is text (a string, or a byte array) a prefix of some string of g's
   language?"
  [g text]
  (check-grammar "prefix?" g)
  (some? (set-of g (start-set g) (if (string? text) (utf8 text) text))))

;; ---------------------------------------------------------------------------
;; The potential
;; ---------------------------------------------------------------------------

(defn- lexeme-splits
  "For each lexeme of g (by terminal, nil for a byte), for each state of its
   DFA, the trie's tokens from it in two parts (gen.llm.bytes/sweep-split):
   those that stay inside the lexeme, and the subtrees of those that may
   end it and go on."
  ^objects [^Grammar g trie]
  (let [^objects lex-table (.-lex-table g)
        ^objects lex-accept (.-lex-accept g)]
    (object-array
     (for [t (range (alength lex-table))]
       (when-let [^ints table (aget lex-table t)]
         (let [^booleans accept (aget lex-accept t)]
           (object-array (for [q (range (alength accept))] (bytes/sweep-split trie table accept q)))))))))

(defn- mask-of
  "The tokens whose bytes lead from set s (non-nil) to a non-empty set: a
   bitset over the vocabulary (gen.llm.bytes/sweep's). A token's first byte
   is taken by an item before a byte, or by a lexing item, or by none. The
   first are swept under that byte's node. For a lexing item, the tokens
   that stay inside its lexeme are taken (its pair keeps the set non-empty),
   read from splits; and the subtrees of those that end it and go on are
   swept. Every other token dies before its lexemes end, so the mask is
   exact, and inside a lexeme (a string's body) a sweep visits only the
   tokens that leave it."
  ^longs [^Grammar g trie ^objects splits ^EarleySet s]
  (let [^ints sym (.-sym g)
        ^ints term-byte (.-term-byte g)
        ^longs lex (.-lex s)
        ^ints lex-at (.-lex-at s)
        ^longs scan (.-scan s)
        advance (fn [s b] (step g s b))
        mask (bytes/bare-mask trie)
        pairs (HashSet.)
        firsts (boolean-array 256)]
    (dotimes [j (alength lex)]
      (let [t (- (inc (aget sym (pos-of (aget lex j)))))
            q (aget lex-at j)]
        (when (.add pairs (item t q))
          (let [[^longs live ^ints roots] (aget ^objects (aget splits t) q)]
            (dotimes [w (alength mask)] (aset mask w (bit-or (aget mask w) (aget live w))))
            (bytes/sweep-roots! mask trie roots s advance)))))
    (dotimes [j (alength scan)]
      (let [b (aget term-byte (- (inc (aget sym (pos-of (aget scan j))))))]
        (when-not (aget firsts b)
          (aset firsts b true)
          (let [c (bytes/child trie 0 b)]
            (when-not (neg? c) (bytes/sweep-into! mask trie c (step g s b) advance))))))
    mask))

(defn potential
  "g read through vocab's tokens (an LM, or a vector of byte arrays by
   token) as a potential: a context weighs 1 (log 0) as a prefix when its
   text (its tokens' bytes) is a prefix of g's language, as a complete
   sequence when it is in it, 0 (-Inf) otherwise. opts: :end-tokens, the
   tokens that end a text and hold none of it (default an LM's :eog, none
   for a byte vocabulary), its end tokens; a context holding one has no
   weight."
  ([vocab g] (potential vocab g {}))
  ([vocab g opts]
   (check-grammar "cfg/potential" g)
   (let [byte-vocab (bytes/vocab-bytes "cfg/potential" vocab)
         n (count byte-vocab)
         ends (lm/end-tokens-option "cfg/potential" opts (if (satisfies? lm/LM vocab) (:eog (lm/vocab vocab)) #{}) n)
         trie (bytes/byte-trie byte-vocab ends)
         splits (lexeme-splits g trie)
         state (fn [context]
                 (loop [s (start-set g) [t & more :as ts] (seq context)]
                   (cond (nil? s) nil
                         (empty? ts) s
                         (contains? ends t) nil
                         :else (recur (set-of g s (byte-vocab t)) more))))
         check-tokens (fn [context]
                        (doseq [t context]
                          (when-not (and (int? t) (< -1 (long t) n))
                            (throw (ex-info (str "cfg/potential: " (pr-str t) " is no token of the vocabulary")
                                            {:reason :invalid-token :token t}))))
                        context)]
     (potential/potential
      n {:prefix (fn [context] (if (state (check-tokens context)) 0.0 Double/NEGATIVE_INFINITY))
         :complete (fn [context] (if (some-> (state (check-tokens context)) accepts?) 0.0 Double/NEGATIVE_INFINITY))
         :logw-next (fn [context]
                      (let [s (or (state (check-tokens context))
                                  (throw (ex-info "logw-next: the context is no prefix of the grammar's language"
                                                  {:reason :zero-prefix :context context})))
                            ^longs mask (mask-of g trie splits s)
                            out (float-array (inc n) Float/NEGATIVE_INFINITY)]
                        (dotimes [w (alength mask)]
                          (loop [bits (aget mask w)]
                            (when-not (zero? bits)
                              (aset out (+ (* w 64) (Long/numberOfTrailingZeros bits)) (float 0.0))
                              (recur (bit-and bits (dec bits))))))
                        (when (accepts? s) (aset out n (float 0.0)))
                        out))
         :end-tokens ends}))))
