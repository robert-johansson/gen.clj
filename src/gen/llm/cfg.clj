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
   nullable rules), advanced one byte at a time: its chart after a text's
   bytes is a state that a next byte extends, as a DFA's state is. A
   regex element becomes right-linear rules from its minimal DFA, one per
   state, so lexemes and structure are one grammar. Since every rule left
   derives some string, a chart that a byte leaves non-empty is a prefix
   of some string of the language: prefix? is exact.

   As a potential (gen.llm.potential), a grammar weighs a context 1 (log
   0) as a prefix when its text is a prefix of the language and as a
   complete sequence when the text is in it, 0 (-Inf) otherwise, its end
   tokens leaving the text (as gen.llm.grammar's: :end-tokens, default an
   LM's :eog, none for a byte vocabulary). logw-next is a sweep of the
   token byte trie with the chart carried down each depth, an Earley set
   per trie node visited; prefix and complete recognize the context's text
   from the start, so their cost grows with the text (no state is kept
   between calls). Unlike a grammar's exact masks, liveness is the
   bytes': a token may lead to a prefix that no sequence of the
   vocabulary's tokens completes, when the vocabulary lacks bytes the
   language needs (one with every byte as a token, as byte-level BPE's,
   never strands). For a large vocabulary token SMC's :awrs proposal
   checks only the tokens it draws, where a mask sweeps the trie."
  {:api :experimental}
  (:require [gen.llm.bytes :as bytes]
            [gen.llm.dfa :as dfa]
            [gen.llm.lm :as lm]
            [gen.llm.potential :as potential]
            [gen.llm.regex :as regex])
  (:import (java.nio.charset StandardCharsets)
           (java.util ArrayList HashMap HashSet)))

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
;; one rule start' → start); terminals byte sets (long[4] bitsets). An
;; Earley item's rule and dot are one number, a position: the rules'
;; dotted positions numbered rule by rule from 0 (start' → • start), a
;; rule of k symbols holding k + 1 (before each symbol, and its end), so
;; moving the dot is adding 1 and no rule's length is bounded. sym is the
;; symbol after each position (a nonterminal k ≥ 0, a terminal j as -(j +
;; 1), or end), lhs-at its rule's nonterminal, starts the first positions
;; of each nonterminal's rules; nullable by nonterminal.
(def ^:private ^:const end Integer/MIN_VALUE)

(deftype Grammar [^ints sym ^ints lhs-at ^objects starts ^booleans nullable ^objects terminals ^long n-rules names]
  Object
  (toString [_] (str "#<CFG " (count names) " rules, " n-rules " productions>")))

(defmethod print-method Grammar [g ^java.io.Writer w] (.write w (str g)))

(defn cfg? [x] (instance? Grammar x))

(defn- utf8 ^bytes [^String s] (.getBytes s StandardCharsets/UTF_8))

(defn- byte-set
  "The bitset (long[4]) of bytes bs."
  ^longs [bs]
  (let [set (long-array 4)]
    (doseq [b bs :let [b (long b)]]
      (aset set (bit-shift-right b 6) (bit-or (aget set (bit-shift-right b 6)) (bit-shift-left 1 (bit-and b 63)))))
    set))

(defn- holds? [^longs set ^long b]
  (not (zero? (bit-and (aget set (bit-shift-right b 6)) (bit-shift-left 1 (bit-and b 63))))))

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
  "The rules (names) of rules that derive some string, nonterminal? telling
   their names from terminals: a fixpoint."
  [rules nonterminal?]
  (loop [known #{}]
    (let [known' (into known (keep (fn [[k alts]]
                                     (when (some (fn [alt] (every? #(or (not (nonterminal? %)) (known %)) alt)) alts) k)))
                       rules)]
      (if (= known known') known (recur known')))))

(defn- regex-rules
  "Right-linear rules of a regex element's DFA, its states named [id s]
   (the start's [id start]): a state's transitions grouped by their
   target, each a byte set, and [] where it accepts. Only the states that
   reach an accepting one."
  [id d]
  (let [n (dfa/size d)
        ^ints table (dfa/table d)
        target (fn ^long [^long s ^long b] (aget table (+ (* s 256) b)))
        live (loop [live (into #{} (filter #(dfa/accepting? d %)) (range n))]
               (let [live' (into live (filter (fn [s] (some #(live (target s %)) (range 256)))) (range n))]
                 (if (= live live') live (recur live'))))]
    (into {}
          (for [s (range n) :when (live s)]
            [[id s]
             (cond-> (vec (for [[t bs] (group-by #(target s %) (filter #(live (target s %)) (range 256)))]
                            [[::bytes (byte-set bs)] [id t]]))
               (dfa/accepting? d s) (conj []))]))))

(defn cfg
  "The grammar of g, a map of :start and :rules (see the namespace's
   docstring), compiled: rules that derive no string dropped, a regex
   element's DFA as right-linear rules. Loud on a malformed grammar
   (:invalid-grammar) and on an empty language (:empty-language)."
  [g]
  (check-rules g)
  (let [regex-ids (atom {})
        ;; elements as nonterminals (rule names), [::bytes set] terminals
        expand (fn [alt]
                 (into [] (mapcat (fn [e]
                                    (cond (keyword? e) [e]
                                          (string? e) (map (fn [b] [::bytes (byte-set [(bit-and (long b) 0xff)])]) (utf8 e))
                                          :else (let [id (or (@regex-ids e)
                                                             (let [id [::regex (count @regex-ids)]] (swap! regex-ids assoc e id) id))]
                                                  [[id (dfa/start (:dfa e))]]))))
                       alt))
        rules (into {} (map (fn [[k alts]] [k (mapv expand alts)])) (:rules g))
        rules (into rules (mapcat (fn [[e id]] (regex-rules id (:dfa e)))) @regex-ids)
        nonterminal? (fn [x] (or (keyword? x) (and (vector? x) (not= ::bytes (first x)))))
        live (productive rules nonterminal?)
        _ (when-not (live (:start g))
            (throw (ex-info "cfg: the language is empty (the start derives no string)" {:reason :empty-language})))
        rules (into {} (keep (fn [[k alts]]
                               (when (live k)
                                 [k (filterv (fn [alt] (every? #(or (not (nonterminal? %)) (live %)) alt)) alts)])))
                    rules)
        names (vec (sort-by pr-str (keys rules)))
        index (zipmap names (range))
        start' (count names)
        terminals (ArrayList.)
        term-index (HashMap.)
        symbol (fn [x]
                 (if (nonterminal? x)
                   (long (index x))
                   (let [k (vec (second x))
                         j (or (.get term-index k) (let [j (.size terminals)] (.add terminals (second x)) (.put term-index k j) j))]
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
        nullable (boolean-array n)]
    ;; nullable: a fixpoint over the productions
    (loop []
      (let [changed (reduce (fn [changed [l alt]]
                              (if (and (not (aget nullable (long l))) (every? #(and (>= (long %) 0) (aget nullable (long %))) alt))
                                (do (aset nullable (long l) true) true)
                                changed))
                            false prods)]
        (when changed (recur))))
    (Grammar. sym lhs-at starts nullable (.toArray terminals) (count prods) names)))

;; ---------------------------------------------------------------------------
;; Earley sets
;; ---------------------------------------------------------------------------

;; An item: its origin (the high 32 bits) and its position (the low 32), as
;; a long; the item past its next symbol is (inc it).
(defn- item ^long [^long origin ^long pos] (bit-or (bit-shift-left origin 32) pos))
(defn- origin-of ^long [^long it] (unsigned-bit-shift-right it 32))
(defn- pos-of ^long [^long it] (bit-and it 0xFFFFFFFF))

;; A set: its items, those waiting for each nonterminal (a map from the
;; nonterminal to a long array), those before a terminal, and whether the
;; start is complete from 0 in it.
(deftype EarleySet [^longs items ^HashMap waiting ^longs scannable ^boolean accepts])

(defn- build
  "Earley set i from its seed items (scanned), the chart before it: the
   closure by prediction and completion."
  ^EarleySet [^Grammar g chart ^long i seeds]
  (let [^ints sym (.-sym g)
        ^ints lhs-at (.-lhs-at g)
        ^objects starts (.-starts g)
        ^booleans nullable (.-nullable g)
        seen (HashSet.)
        queue (ArrayList.)
        predicted (HashSet.)
        waiting (HashMap.)
        scannable (ArrayList.)
        add (fn [^long it] (when (.add seen it) (.add queue it)))]
    (doseq [it seeds] (add (long it)))
    (loop [k 0]
      (when (< k (.size queue))
        (let [it (long (.get queue k))
              p (pos-of it)
              s (aget sym p)]
          (cond
            ;; complete: advance the items of its origin's set waiting for its lhs
            (== s end) (let [o (origin-of it)]
                         (when-not (== o i)
                           (let [^EarleySet at (nth chart o)]
                             (when-let [^longs ws (.get ^HashMap (.-waiting at) (aget lhs-at p))]
                               (dotimes [j (alength ws)] (add (inc (aget ws j))))))))
            (neg? s) (.add scannable it)
            :else (do (let [^ArrayList l (or (.get waiting s) (let [l (ArrayList.)] (.put waiting s l) l))]
                        (.add l it))
                      (when (.add predicted s)
                        (let [^ints ps (aget starts s)]
                          (dotimes [j (alength ps)] (add (item i (aget ps j))))))
                      ;; Aycock and Horspool: a nullable nonterminal is also skipped
                      (when (aget nullable s) (add (inc it)))))
          (recur (inc k)))))
    (let [frozen (HashMap.)]
      (doseq [[s ^ArrayList l] waiting] (.put frozen s (long-array (map long l))))
      (EarleySet. (long-array (map long queue)) frozen (long-array (map long scannable))
                  (.contains seen (item 0 1))))))

(defn- start-chart
  "The chart of the empty text."
  [^Grammar g]
  [(build g [] 0 [(item 0 0)])])

(defn- advance
  "The chart after one more byte b, or nil when no item takes it (the text
   is no prefix of the language)."
  [^Grammar g chart ^long b]
  (let [^EarleySet last-set (peek chart)
        ^longs scannable (.-scannable last-set)
        ^ints sym (.-sym g)
        ^objects terminals (.-terminals g)
        seeds (ArrayList.)]
    (dotimes [j (alength scannable)]
      (let [it (aget scannable j)
            s (aget sym (pos-of it))]
        (when (holds? (aget terminals (- (inc s))) b)
          (.add seeds (inc it)))))
    (when-not (.isEmpty seeds)
      (conj chart (build g chart (count chart) seeds)))))

(defn- chart-of
  "The chart after bytes (a byte array), or nil when they are no prefix."
  [g ^bytes bs]
  (loop [chart (start-chart g) i 0]
    (cond (nil? chart) nil
          (== i (alength bs)) chart
          :else (recur (advance g chart (bit-and (long (aget bs i)) 0xff)) (inc i)))))

(defn- accepts? [chart] (.-accepts ^EarleySet (peek chart)))

(defn- check-grammar [what g]
  (when-not (cfg? g)
    (throw (ex-info (str what ": a grammar is made by gen.llm.cfg/cfg, not " (pr-str g)) {:reason :invalid-grammar}))))

(defn recognizes?
  "Is text (a string, or a byte array) a string of g's language?"
  [g text]
  (check-grammar "recognizes?" g)
  (boolean (some-> (chart-of g (if (string? text) (utf8 text) text)) accepts?)))

(defn prefix?
  "Is text (a string, or a byte array) a prefix of some string of g's
   language?"
  [g text]
  (check-grammar "prefix?" g)
  (some? (chart-of g (if (string? text) (utf8 text) text))))

;; ---------------------------------------------------------------------------
;; The potential
;; ---------------------------------------------------------------------------

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
         chart (fn [context]
                 (loop [chart (start-chart g) [t & more :as ts] (seq context)]
                   (cond (nil? chart) nil
                         (empty? ts) chart
                         (contains? ends t) nil
                         :else (let [^bytes bs (byte-vocab t)]
                                 (recur (loop [c chart i 0]
                                          (if (or (nil? c) (== i (alength bs)))
                                            c
                                            (recur (advance g c (bit-and (long (aget bs i)) 0xff)) (inc i))))
                                        more)))))
         check-tokens (fn [context]
                        (doseq [t context]
                          (when-not (and (int? t) (< -1 (long t) n))
                            (throw (ex-info (str "cfg/potential: " (pr-str t) " is no token of the vocabulary")
                                            {:reason :invalid-token :token t}))))
                        context)]
     (potential/potential
      n {:prefix (fn [context] (if (chart (check-tokens context)) 0.0 Double/NEGATIVE_INFINITY))
         :complete (fn [context] (if (some-> (chart (check-tokens context)) accepts?) 0.0 Double/NEGATIVE_INFINITY))
         :logw-next (fn [context]
                      (let [c (or (chart (check-tokens context))
                                  (throw (ex-info "logw-next: the context is no prefix of the grammar's language"
                                                  {:reason :zero-prefix :context context})))
                            ^longs mask (bytes/sweep-with trie c (fn [c b] (advance g c b)))
                            out (float-array (inc n) Float/NEGATIVE_INFINITY)]
                        (dotimes [w (alength mask)]
                          (loop [bits (aget mask w)]
                            (when-not (zero? bits)
                              (aset out (+ (* w 64) (Long/numberOfTrailingZeros bits)) (float 0.0))
                              (recur (bit-and bits (dec bits))))))
                        (when (accepts? c) (aset out n (float 0.0)))
                        out))
         :end-tokens ends}))))
