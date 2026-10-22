(ns gen.llm.grammar
  "Grammars: a regular language over bytes (gen.llm.regex, gen.llm.dfa)
   read through an LM's vocabulary, so that token sequences can be held to
   it: masks for grammar-constrained decoding and the weights of a potential.

     (grammar vocab language), (grammar vocab language {:end-tokens ts})
                                → a grammar; vocab an LM or a byte vocabulary
                                  (a vector of byte arrays by token), language
                                  a regex, a term (gen.llm.regex) or a DFA;
                                  ts the end tokens, not text (default: an
                                  LM's :eog, none for a byte vocabulary)
     (state g tokens)           → the DFA state after the tokens' bytes (-1: dead)
     (advance g state t)        → the state after token t
     (live? g state)            → can tokens still complete the language from it?
     (accepting? g state)       → are the bytes so far a string of the language?
     (allowed? g state t), (allowed g state)
                                → may token t come next; the tokens that may
     (automaton g), (end-tokens g), (info g)
                                → its DFA; its end tokens; its sizes

   A grammar is a potential (gen.llm.potential): (potential/prefix g
   tokens), complete and logw-next weigh token sequences by it.

   A token sequence's text is its tokens' bytes, concatenated (an LM's
   decode); the grammar holds the text, read byte by byte, so a token may
   end inside a character or span several of the DFA's transitions. End
   tokens are not text: a completion's end token is the end, allowed where
   the text is complete (the end's weight), so a grammar leaves them out,
   and its end tokens must be the completion's: it declares them
   (potential/end-tokens), and potential/check-end-tokens compares them
   wherever it meets a completion. A language that needs an end token's bytes is inexpressible. A
   token is allowed after a context when its bytes lead to a live state:
   one from which some sequence of tokens (not merely of bytes) reaches an
   accepting state. That is exact: every state the vocabulary reaches from
   the start is explored, and its mask computed, when the grammar is made,
   so a constrained sequence never strands where no token completes it.
   An empty language, or one no sequence of the vocabulary's tokens spells,
   is a loud error.

   As a potential (genlm-control's Potential, a BoolFSA read through the
   tokens), a grammar weighs a context 1 (log 0) as a prefix when its state
   is live, as a complete sequence when it accepts, and 0 (log -Inf)
   otherwise; logw-next is the mask, a new float row of n-vocab + 1
   entries, 0 for each allowed token and, last, for the end when the
   context accepts. A context that is no prefix has no next weights
   (:zero-prefix, as gen.llm.potential's dense ones)."
  (:require [gen.llm.bytes :as bytes]
            [gen.llm.dfa :as dfa]
            [gen.llm.lm :as lm]
            [gen.llm.potential :as potential]
            [gen.llm.regex :as regex])
  (:import (gen.llm.bytes ByteTrie)))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

;; masks: by DFA state, the bitset of allowed tokens (gen.llm.bytes/sweep's
;; form) for the live states the vocabulary reaches, nil for the others;
;; live: the same states; ends: the end tokens, by token.
(declare prefix complete logw-next)

(deftype Grammar [dfa ^ByteTrie trie ^objects masks ^booleans live ^booleans ends end-tokens info]
  potential/Potential
  (-prefix [g context] (prefix g context))
  (-complete [g context] (complete g context))
  (-logw-next [g context] (logw-next g context))

  ;; it reads contexts with these end tokens, left out of the text
  potential/EndTokens
  (-end-tokens [_] end-tokens)

  Object
  (toString [_]
    (str "#<Grammar " (:states info) " states (" (:reachable info) " reached by tokens), "
         (:n-vocab info) " tokens>")))

(defmethod print-method Grammar [g ^java.io.Writer w] (.write w (str g)))

(defn grammar? [x] (instance? Grammar x))

(defn- byte-vocab-of [vocab]
  (cond (satisfies? lm/LM vocab) (lm/byte-vocab vocab)
        (and (sequential? vocab) (every? bytes? vocab)) vocab
        :else (throw (ex-info "grammar: the vocabulary is an LM or a vector of byte arrays, by token"
                              {:reason :invalid-vocab :vocab vocab}))))

(defn- dfa-of [language]
  (if (dfa/dfa? language) language (regex/dfa language)))

(defn- explore
  "The states the vocabulary reaches from the start: by state, its mask
   (every token whose bytes do not lead to -1) and the states its tokens
   lead to."
  [^ByteTrie trie d]
  (let [n (dfa/size d)
        table (dfa/table d)
        any (boolean-array n true)]
    (loop [todo [(dfa/start d)] masks {} succ {}]
      (if-let [s (peek todo)]
        (if (contains? masks s)
          (recur (pop todo) masks succ)
          (let [reached (boolean-array n)
                mask (bytes/sweep trie table s any reached)
                succs (filterv #(aget reached (long %)) (range n))]
            (recur (into (pop todo) (remove masks) succs) (assoc masks s mask) (assoc succ s succs))))
        {:masks masks :succ succ}))))

(defn- token-live
  "Of the states the vocabulary reaches, those from which its tokens reach
   an accepting state: backwards from the accepting ones."
  [d succ]
  (let [preds (reduce (fn [preds [s succs]] (reduce #(update %1 %2 conj s) preds succs)) {} succ)]
    (loop [todo (filterv #(dfa/accepting? d %) (keys succ)) live #{}]
      (if-let [s (peek todo)]
        (if (contains? live s)
          (recur (pop todo) live)
          (recur (into (pop todo) (remove live) (preds s)) (conj live s)))
        live))))

(defn- end-tokens-of
  "The end tokens opts gives, or vocab's default: an LM's :eog, none for a
   byte vocabulary; checked."
  [vocab byte-vocab opts]
  (lm/end-tokens-option "grammar" opts (if (satisfies? lm/LM vocab) (:eog (lm/vocab vocab)) #{}) (count byte-vocab)))

(defn grammar
  "The grammar of language (a regex, a regex term or a DFA) over vocab's
   tokens (an LM, or a vector of byte arrays by token, as
   gen.llm.lm/byte-vocab gives). opts: :end-tokens, the tokens that end a
   text (a set; default an LM's :eog, none for a byte vocabulary), left
   out of the text: a completion under the grammar must have these end
   tokens. Every state the tokens reach is explored and its mask made now.
   Loud when the language is empty (:empty-language) or no sequence of the
   text tokens spells a string of it (:inexpressible)."
  ([vocab language] (grammar vocab language {}))
  ([vocab language opts]
   (let [byte-vocab (byte-vocab-of vocab)
         end-tokens (end-tokens-of vocab byte-vocab opts)
         d (dfa-of language)
         _ (when (zero? (dfa/size d))
             (throw (ex-info "grammar: the language is empty" {:reason :empty-language :language language})))
         trie (bytes/byte-trie byte-vocab end-tokens)
         n (dfa/size d)
         t0 (System/nanoTime)
         {:keys [masks succ]} (explore trie d)
         live-states (token-live d succ)
         _ (when-not (contains? live-states (dfa/start d))
             (throw (ex-info (str "grammar: no sequence of the vocabulary's tokens spells a string of the language"
                                  " (its tokens lack bytes the language needs)")
                             {:reason :inexpressible :language language})))
         live (boolean-array n)
         _ (doseq [s live-states] (aset live (long s) true))
         ;; when some reached state is not live, a token into it is not allowed
         exact (if (= (count live-states) (count masks))
                 masks
                 (into {} (map (fn [s] [s (bytes/sweep trie (dfa/table d) s live nil)])) live-states))
         out (object-array n)
         ends (boolean-array (count byte-vocab))]
     (doseq [[s mask] exact] (aset out (long s) mask))
     (doseq [t end-tokens] (aset ends (long t) true))
     (Grammar. d trie out live ends end-tokens
               {:states n
                :reachable (count masks)
                :live (count live-states)
                :n-vocab (bytes/n-vocab trie)
                :mask-ns (- (System/nanoTime) t0)}))))

(defn automaton "g's DFA." [^Grammar g] (.-dfa g))

(defn end-tokens "g's end tokens, which are not text (a set)." [^Grammar g] (.-end-tokens g))

(defn info
  "g's sizes: :states (its DFA's), :reachable (by tokens from the start),
   :live (of those, the states tokens can complete), :n-vocab, and
   :mask-ns, the time its masks took."
  [^Grammar g]
  (.-info g))

(defn advance
  "The state after token t from state s (-1: dead, and for an end token,
   which is no text)."
  ^long [^Grammar g ^long s ^long t]
  (if (or (neg? s) (aget ^booleans (.-ends g) t)) -1 (dfa/walk (.-dfa g) s (bytes/token-bytes (.-trie g) t))))

(defn state
  "The DFA state after the bytes of tokens, from the start (-1: dead)."
  ^long [^Grammar g tokens]
  (reduce (fn [^long s t] (if (neg? s) (reduced -1) (advance g s (long t)))) (dfa/start (.-dfa g)) tokens))

(defn live?
  "Can tokens complete the language from state s?"
  [^Grammar g ^long s]
  (and (<= 0 s) (aget ^booleans (.-live g) s)))

(defn accepting?
  "Are the bytes that led to state s a string of the language?"
  [^Grammar g ^long s]
  (dfa/accepting? (.-dfa g) s))

(defn allowed?
  "May token t come after state s (its bytes lead to a live state)?"
  [^Grammar g ^long s ^long t]
  (if-let [^longs mask (when (live? g s) (aget ^objects (.-masks g) s))]
    (not (zero? (bit-and (aget mask (bit-shift-right t 6)) (bit-shift-left 1 (bit-and t 63)))))
    false))

(defn allowed
  "The tokens that may come after state s, in order (a vector)."
  [^Grammar g ^long s]
  (filterv #(allowed? g s %) (range (:n-vocab (.-info g)))))

;; ---------------------------------------------------------------------------
;; The potential
;; ---------------------------------------------------------------------------

(defn- prefix
  "log 1 when tokens can be continued to a string of the language, else
   -Inf."
  ^double [^Grammar g tokens]
  (if (live? g (state g tokens)) 0.0 Double/NEGATIVE_INFINITY))

(defn- complete
  "log 1 when the bytes of tokens are a string of the language, else -Inf."
  ^double [^Grammar g tokens]
  (if (accepting? g (state g tokens)) 0.0 Double/NEGATIVE_INFINITY))

(defn- logw-next
  "The weights of the next token after tokens: a new float array of
   n-vocab + 1 entries, 0 for an allowed token and -Inf for the others,
   the last 0 when tokens are complete (the end), else -Inf. Loud when
   tokens are no prefix (:zero-prefix)."
  ^floats [^Grammar g tokens]
  (let [s (state g tokens)
        n (long (:n-vocab (.-info g)))
        out (float-array (inc n) Float/NEGATIVE_INFINITY)]
    (when-not (live? g s)
      (throw (ex-info "logw-next: the tokens are no prefix of the grammar's language"
                      {:reason :zero-prefix :tokens tokens})))
    (let [^longs mask (aget ^objects (.-masks g) s)]
      (dotimes [w (alength mask)]
        (loop [bits (aget mask w)]
          (when-not (zero? bits)
            (aset out (+ (* w 64) (Long/numberOfTrailingZeros bits)) (float 0.0))
            (recur (bit-and bits (dec bits)))))))
    (when (accepting? g s) (aset out n (float 0.0)))
    out))
