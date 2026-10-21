(ns gen.llm.regex
  "Regular languages over bytes, as terms, compiled to minimal DFAs
   (gen.llm.dfa) by Brzozowski derivatives: the languages of
   grammar-constrained decoding (gen.llm.grammar).

     (parse \"[0-9]{3}-[0-9]{4}\")        → the term of a regex
     (dfa r), (dfa r {:max-states n})    → the DFA of a term or a regex
     (matches? r text)                   → is text (a string) in the language?

   Terms are values, compared with =, made by:
     nothing, epsilon                   the empty language; the empty string
     (text s)                           the string s
     (chars spec …)                     one character of the specs: chars,
                                        strings of them, [lo hi] ranges (chars
                                        or code points)
     (chars-except spec …), any-char    one character not of the specs; any
     (bytes-of bs), any-byte            one byte of bs (integers 0 to 255); any
     (cat r …), (alt r …)               concatenation, alternation
     (star r), (plus r), (opt r)        repetition
     (repeat r lo hi)                   lo to hi of r (hi nil: no bound)
     (intersect r …), (complement r)    both languages; every byte string not r's

   A character is a Unicode scalar value, read as its UTF-8 bytes, so a DFA
   reads text byte by byte and a token that ends inside a character leaves
   it inside the character. complement is over byte strings, invalid UTF-8
   among them: (intersect (star any-char) (complement r)) is text not r's.

   The regex syntax (parse) is the common one, matching whole strings, and
   agrees with java.util.regex (with UNIX_LINES) on every regex it accepts:
   - a character stands for itself; \\ before a character that is not an
     ASCII letter or digit makes it literal (\\. \\\\ \\( \\[ \\{ …); \\n \\t \\r \\f,
     \\xhh, \\x{h…}, \\uhhhh are characters;
   - . is any character but a newline; [abc], [a-z], [^…] (any character
     but these); \\d \\w \\s are [0-9], [A-Za-z0-9_] and [ \\t\\n\\x0B\\f\\r],
     \\D \\W \\S any character but those, also inside classes;
   - (…) and (?:…) group, | alternates; * + ? {n} {n,} {n,m} repeat (a lazy
     ? after one changes no language);
   - ^ first and $ last are allowed, and change nothing.
   Anything else is a loud error, rather than another dialect's meaning:
   backreferences, lookaround, word boundaries, anchors elsewhere, inline
   flags, possessive quantifiers, \\p classes, nested classes and &&
   (java.util.regex's), and an unescaped ] { } outside a class or [ inside
   one."
  (:refer-clojure :exclude [cat chars complement repeat])
  (:require [clojure.set :as set]
            [gen.llm.dfa :as dfa])
  (:import (java.nio.charset StandardCharsets)))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

;; ---------------------------------------------------------------------------
;; Terms
;; ---------------------------------------------------------------------------

;; :nothing, :epsilon, [:bytes #{b …}], [:cat r r …], [:or #{r r …}],
;; [:and #{r r …}], [:not r], [:star r], [:repeat r lo hi]. The
;; constructors keep them in a normal form (sets for alternatives and
;; conjuncts, flat concatenations, no identities), which is what makes a
;; term's derivatives finitely many (Brzozowski).

(def nothing "The empty language: no string." :nothing)
(def epsilon "The language of the empty string." :epsilon)

(defn- tag [r] (if (keyword? r) r (nth r 0)))

(defn- term? [r]
  (or (#{:nothing :epsilon} r)
      (and (vector? r) (#{:bytes :cat :or :and :not :star :repeat} (nth r 0 nil)))))

(defn- check-term [what r]
  (when-not (term? r)
    (throw (ex-info (str what ": not a regular expression term: " (pr-str r)) {:reason :invalid-term :term r})))
  r)

(defn- nullable?
  "Does r's language hold the empty string?"
  [r]
  (case (tag r)
    :epsilon true
    (:nothing :bytes) false
    :cat (every? nullable? (subvec r 1))
    :or (boolean (some nullable? (nth r 1)))
    :and (every? nullable? (nth r 1))
    :not (not (nullable? (nth r 1)))
    :star true
    :repeat (or (zero? (long (nth r 2))) (nullable? (nth r 1)))))

(defn bytes-of
  "One byte of bs, a collection of integers 0 to 255."
  [bs]
  (when-not (and (coll? bs) (every? #(and (integer? %) (<= 0 % 255)) bs))
    (throw (ex-info (str "bytes-of: bytes are integers 0 to 255, not " (pr-str bs)) {:reason :invalid-term :bytes bs})))
  (if (empty? bs) nothing [:bytes (into #{} (map long) bs)]))

(def any-byte "Any byte." (bytes-of (range 256)))

(def ^:private universe
  "Every byte string."
  [:not :nothing])

(defn cat
  "The concatenation of rs (epsilon for none)."
  [& rs]
  (let [parts (into [] (comp (map #(check-term "cat" %))
                             (mapcat #(if (= :cat (tag %)) (subvec % 1) [%]))
                             (remove #{:epsilon}))
                    rs)]
    (cond (some #{:nothing} parts) nothing
          (empty? parts) epsilon
          (= 1 (count parts)) (parts 0)
          :else (into [:cat] parts))))

(defn alt
  "The union of rs (nothing for none). Single bytes merge into one set."
  [& rs]
  (let [members (into #{} (comp (map #(check-term "alt" %))
                                (mapcat #(if (= :or (tag %)) (nth % 1) [%]))
                                (remove #{:nothing}))
                      rs)
        {singles true others false} (group-by #(= :bytes (tag %)) members)
        members (cond-> (set others) (seq singles) (conj [:bytes (into #{} (mapcat second) singles)]))
        ;; the empty string beside a nullable alternative adds nothing
        members (if (and (contains? members :epsilon) (some nullable? (disj members :epsilon)))
                  (disj members :epsilon)
                  members)]
    (cond (contains? members universe) universe
          (empty? members) nothing
          (= 1 (count members)) (first members)
          :else [:or members])))

(defn intersect
  "The strings in every one of rs (every byte string for none)."
  [& rs]
  (let [members (into #{} (comp (map #(check-term "intersect" %))
                                (mapcat #(if (= :and (tag %)) (nth % 1) [%]))
                                (remove #{universe}))
                      rs)
        {singles true others false} (group-by #(= :bytes (tag %)) members)
        members (cond-> (set others) (seq singles) (conj (bytes-of (apply set/intersection (map second singles)))))]
    (cond (contains? members nothing) nothing
          (empty? members) universe
          (= 1 (count members)) (first members)
          :else [:and members])))

(defn complement
  "Every byte string not in r's language (invalid UTF-8 among them)."
  [r]
  (check-term "complement" r)
  (if (= :not (tag r)) (nth r 1) [:not r]))

(declare star)

(defn- nested-bound
  "The upper bound of (x{0,b}){0,d}: b·d, exactly (nil, no bound, when
   either is), or false when the product would pass Long/MAX_VALUE: the
   nesting is then left as it is, never wrapped. (A DFA of such a term
   tracks every pair of counts, so its construction is slow to reach
   :max-states.)"
  [d b]
  (if (or (nil? d) (nil? b))
    nil
    (try (Math/multiplyExact (long d) (long b))
         (catch ArithmeticException _ false))))

(defn repeat
  "From lo to hi strings of r's language, concatenated (hi nil: no upper
   bound)."
  [r lo hi]
  (check-term "repeat" r)
  (when-not (and (nat-int? lo) (or (nil? hi) (and (nat-int? hi) (<= (long lo) (long hi)))))
    (throw (ex-info (str "repeat: the bounds are 0 <= lo <= hi (hi nil for none), not " lo " and " (pr-str hi))
                    {:reason :invalid-term :lo lo :hi hi})))
  ;; normal forms: with the empty string in r, fewer copies pad with it, so
  ;; the lower bound is 0; then (x | ε){0,d} is x{0,d}, (x*){0,d} is x*,
  ;; (x{0,b}){0,d} is x{0,bd}
  (let [lo (if (nullable? r) 0 (long lo))]
    (cond (and (zero? lo) (= :or (tag r)) (contains? (nth r 1) :epsilon))
          (repeat (apply alt (disj (nth r 1) :epsilon)) 0 hi)
          (= hi 0) epsilon
          (= :epsilon r) epsilon
          (= :nothing r) (if (zero? lo) epsilon nothing)
          (and (== lo 1) (= hi 1)) r
          (and (zero? lo) (nil? hi)) (star r)
          (= :star (tag r)) r
          (and (= :repeat (tag r)) (zero? lo) (zero? (long (nth r 2))) (not (false? (nested-bound hi (nth r 3)))))
          (repeat (nth r 1) 0 (nested-bound hi (nth r 3)))
          :else [:repeat r lo (some-> hi long)])))

(defn star
  "Any number of r's strings."
  [r]
  (check-term "star" r)
  (case (tag r)
    (:nothing :epsilon) epsilon
    :star r
    ;; (x | ε)* and (x{0,b})* or (x{1,b})* are x*
    :or (if (contains? (nth r 1) :epsilon) (star (apply alt (disj (nth r 1) :epsilon))) [:star r])
    :repeat (if (<= (long (nth r 2)) 1) (star (nth r 1)) [:star r])
    [:star r]))
(defn plus "One or more of r's strings." [r] (repeat r 1 nil))
(defn opt "r's strings or the empty string." [r] (alt r epsilon))

;; ---------------------------------------------------------------------------
;; Characters as UTF-8
;; ---------------------------------------------------------------------------

;; A set of characters is a sorted vector of disjoint, non-adjacent [lo hi]
;; ranges of code points.

(def ^:private ^:const max-char 0x10FFFF)

(defn- ranges
  "rs, sorted and merged."
  [rs]
  (reduce (fn [out [lo hi]]
            (let [[plo phi] (peek out)]
              (if (and phi (<= (long lo) (inc (long phi))))
                (conj (pop out) [plo (max (long phi) (long hi))])
                (conj out [lo hi]))))
          [] (sort rs)))

(defn- ranges-complement [rs]
  (let [rs (ranges rs)
        gaps (map (fn [[_ hi] [lo _]] [(inc (long hi)) (dec (long lo))])
                  (cons [-1 -1] rs) (concat rs [[(inc max-char) 0]]))]
    (filterv (fn [[lo hi]] (<= (long lo) (long hi))) gaps)))

(defn- without-surrogates
  "rs less the surrogates (U+D800 to U+DFFF), which are no characters."
  [rs]
  (into [] (comp (mapcat (fn [[lo hi]] [[lo (min (long hi) 0xD7FF)] [(max (long lo) 0xE000) hi]]))
                 (filter (fn [[lo hi]] (<= (long lo) (long hi)))))
        (ranges rs)))

(defn- utf8
  "The UTF-8 bytes of code point c."
  [^long c]
  (cond (< c 0x80) [c]
        (< c 0x800) [(bit-or 0xC0 (bit-shift-right c 6)) (bit-or 0x80 (bit-and c 0x3F))]
        (< c 0x10000) [(bit-or 0xE0 (bit-shift-right c 12)) (bit-or 0x80 (bit-and (bit-shift-right c 6) 0x3F))
                       (bit-or 0x80 (bit-and c 0x3F))]
        :else [(bit-or 0xF0 (bit-shift-right c 18)) (bit-or 0x80 (bit-and (bit-shift-right c 12) 0x3F))
               (bit-or 0x80 (bit-and (bit-shift-right c 6) 0x3F)) (bit-or 0x80 (bit-and c 0x3F))]))

(defn- utf8-sequences
  "The UTF-8 encodings of the characters lo to hi (no surrogates), as
   sequences of byte ranges, each sequence [[lo hi] …] one range per byte
   (the splitting of Rust's regex-syntax Utf8Sequences)."
  [[^long lo ^long hi]]
  (if-let [m (some #(when (and (<= lo (long %)) (< (long %) hi)) %) [0x7F 0x7FF 0xFFFF])]
    ;; the encodings of one length at a time
    (concat (utf8-sequences [lo m]) (utf8-sequences [(inc (long m)) hi]))
    (or (some (fn [^long i]
                (let [m (dec (bit-shift-left 1 (* 6 i)))]
                  (when (not= (bit-and-not lo m) (bit-and-not hi m))
                    (cond (not (zero? (bit-and lo m)))
                          (concat (utf8-sequences [lo (bit-or lo m)]) (utf8-sequences [(inc (bit-or lo m)) hi]))
                          (not= (bit-and hi m) m)
                          (concat (utf8-sequences [lo (dec (bit-and-not hi m))]) (utf8-sequences [(bit-and-not hi m) hi]))))))
              [1 2 3])
        ;; lo and hi differ only where every byte between theirs is allowed
        [(mapv vector (utf8 lo) (utf8 hi))])))

(defn- chars-term
  "One character of the code point ranges rs."
  [rs]
  (apply alt (for [r (without-surrogates rs), sq (utf8-sequences r)]
               (apply cat (map (fn [[lo hi]] (bytes-of (range lo (inc (long hi))))) sq)))))

(defn- check-character
  "c, a code point, unless it is a surrogate: a loud error."
  ^long [what ^long c spec]
  (when (<= 0xD800 c 0xDFFF)
    (throw (ex-info (str what ": U+" (Long/toHexString c) " is a surrogate, not a character") {:reason :invalid-term :spec spec})))
  c)

(defn- code-points
  "The code points of the string s, checked to be characters (a lone
   surrogate is loud)."
  [what ^String s]
  (mapv #(check-character what (long %) s) (.toArray (.codePoints s))))

(defn- spec-ranges
  "The code point ranges of chars's specs."
  [what specs]
  (into [] (mapcat (fn [spec]
                     (let [cp (fn [x] (let [c (cond (char? x) (long (int x))
                                                    (and (integer? x) (<= 0 x max-char)) (long x)
                                                    :else (throw (ex-info (str what ": a character is a char or a code point, not "
                                                                               (pr-str x))
                                                                          {:reason :invalid-term :spec spec})))]
                                        (check-character what c spec)))]
                       (cond (string? spec) (map (fn [c] [c c]) (code-points what spec))
                             (and (vector? spec) (= 2 (count spec)))
                             (let [[lo hi] (map cp spec)]
                               (when (< (long hi) (long lo))
                                 (throw (ex-info (str what ": the range " (pr-str spec) " is out of order")
                                                 {:reason :invalid-term :spec spec})))
                               [[lo hi]])
                             :else (let [c (cp spec)] [[c c]])))))
        specs))

(defn chars
  "One character of the specs: chars, strings of them, [lo hi] ranges of
   chars or code points. (chars [\\a \\z] \\_) is [a-z_]. A surrogate (U+D800 to
   U+DFFF) is no character: a loud error as a spec or a range's end (a
   range across them holds the characters around them)."
  [& specs]
  (chars-term (spec-ranges "chars" specs)))

(defn chars-except
  "One character, any but the specs' (as chars's)."
  [& specs]
  (chars-term (ranges-complement (spec-ranges "chars-except" specs))))

(def any-char "Any character (a Unicode scalar value, as UTF-8)." (chars [0 max-char]))

(defn text
  "The string s, as UTF-8 (a lone surrogate in it, which UTF-8 cannot
   encode, a loud error)."
  [^String s]
  (when-not (string? s)
    (throw (ex-info (str "text: a string, not " (pr-str s)) {:reason :invalid-term :text s})))
  (code-points "text" s)
  (apply cat (map #(bytes-of [(bit-and (long %) 0xff)]) (.getBytes s StandardCharsets/UTF_8))))

;; ---------------------------------------------------------------------------
;; Derivatives
;; ---------------------------------------------------------------------------

(defn- derivative
  "The strings s for which b s is in r's language, as a term. memo (a map
   local to one construction, of [term byte] to derivative) shares the
   derivatives of subterms that recur across states: the tails of
   concatenations, the bodies of repeats."
  [^java.util.Map memo r ^long b]
  (let [k [r b]]
    (or (.get memo k)
        (let [d (case (tag r)
                  (:nothing :epsilon) nothing
                  :bytes (if (contains? (nth r 1) b) epsilon nothing)
                  :cat (let [head (nth r 1)
                             tail (apply cat (subvec r 2))
                             d (cat (derivative memo head b) tail)]
                         (if (nullable? head) (alt d (derivative memo tail b)) d))
                  :or (apply alt (map #(derivative memo % b) (nth r 1)))
                  :and (apply intersect (map #(derivative memo % b) (nth r 1)))
                  :not (complement (derivative memo (nth r 1) b))
                  :star (cat (derivative memo (nth r 1) b) r)
                  :repeat (let [[_ x lo hi] r]
                            (cat (derivative memo x b) (repeat x (max 0 (dec (long lo))) (when hi (dec (long hi)))))))]
          (.put memo k d)
          d))))

(defn- byte-sets [r]
  (case (tag r)
    (:nothing :epsilon) #{}
    :bytes #{(nth r 1)}
    (:cat) (into #{} (mapcat byte-sets) (subvec r 1))
    (:or :and) (into #{} (mapcat byte-sets) (nth r 1))
    (:not :star :repeat) (byte-sets (nth r 1))))

(defn- byte-classes
  "The bytes partitioned so that every byte set of r is a union of parts
   (so are every derivative's): vectors of bytes, ordered by their first."
  [r]
  (->> (byte-sets r)
       (reduce (fn [parts s] (into [] (comp (mapcat (fn [p] [(set/intersection p s) (set/difference p s)]))
                                            (remove empty?))
                                   parts))
               [(set (range 256))])
       (map sort)
       (sort-by first)
       vec))

;; ---------------------------------------------------------------------------
;; The regex syntax
;; ---------------------------------------------------------------------------

;; A parser reads p = {:regex :cps (its code points)} from position i and
;; returns [value i'].

(defn- fail [p i msg]
  (throw (ex-info (str "regex: " msg " (at " i " in " (pr-str (:regex p)) ")")
                  {:reason :invalid-regex :regex (:regex p) :position i})))

(defn- cp-at [p i] (get (:cps p) i))

(defn- ch
  "The char of code point c, nil past U+FFFF: (char c) there keeps c's low
   16 bits (U+1002A would read as *)."
  [c]
  (when (< (long c) 0x10000) (char c)))

(defn- literal
  "Code point c at i, a character (a lone surrogate is loud)."
  [p i c]
  (when (<= 0xD800 (long c) 0xDFFF) (fail p i "a lone surrogate is not a character"))
  c)

(defn- at?
  "Is the character at i ch?"
  [p i ch]
  (= (cp-at p i) (long (int ch))))

(def ^:private class-escapes
  {\d [[48 57]]
   \w [[48 57] [65 90] [95 95] [97 122]]
   \s [[9 13] [32 32]]})

(defn- hex
  "The character of the hex digits from i to j."
  [p ^long i ^long j]
  (let [digits (subvec (:cps p) (min i (count (:cps p))) (min j (count (:cps p))))]
    (when-not (and (pos? (- j i)) (== (count digits) (- j i)) (every? #(<= 0 (Character/digit (int %) 16)) digits))
      (fail p i "a hexadecimal escape needs hex digits"))
    (let [c (long (reduce (fn [^long c d] (min (inc max-char) (+ (* c 16) (Character/digit (int d) 16)))) 0 digits))]
      (when (< max-char c) (fail p i "no such character"))
      (when (<= 0xD800 c 0xDFFF) (fail p i "a surrogate is not a character"))
      c)))

(defn- escape
  "The escape after a backslash at i - 1: [[:char c] i'] or [[:set ranges] i']."
  [p ^long i]
  (let [c (cp-at p i)]
    (when (nil? c) (fail p i "a backslash ends the regex"))
    (let [ch (when (< (long c) 128) (char c))
          class (when ch (class-escapes (Character/toLowerCase (char ch))))]
      (cond class [[:set (if (Character/isUpperCase (char ch)) (ranges-complement class) class)] (inc i)]
            (= ch \n) [[:char 10] (inc i)]
            (= ch \t) [[:char 9] (inc i)]
            (= ch \r) [[:char 13] (inc i)]
            (= ch \f) [[:char 12] (inc i)]
            (= ch \u) [[:char (hex p (inc i) (+ i 5))] (+ i 5)]
            (and (= ch \x) (at? p (inc i) \{))
            (let [close (loop [j (+ i 2)]
                          (cond (nil? (cp-at p j)) (fail p i "an unclosed \\x{")
                                (at? p j \}) j
                                :else (recur (inc j))))]
              [[:char (hex p (+ i 2) close)] (inc (long close))])
            (= ch \x) [[:char (hex p (inc i) (+ i 3))] (+ i 3)]
            ;; an ASCII letter or digit is reserved (as java.util.regex and
            ;; Python have it); any other character escaped is itself
            (and ch (Character/isDigit (char ch))) (fail p i "backreferences and octal escapes are not supported")
            (and ch (Character/isLetter (char ch))) (fail p i (str "\\" ch " is not supported"))
            :else [[:char (literal p i c)] (inc i)]))))

(defn- class-item
  "One member of a class at i: [[:char c] i'] or [[:set ranges] i']."
  [p ^long i]
  (cond (nil? (cp-at p i)) (fail p i "an unclosed [")
        (at? p i \[) (fail p i "a [ in a class (escape it: nested classes are not supported)")
        (and (at? p i \&) (at? p (inc i) \&)) (fail p i "&& in a class (escape an &)")
        (at? p i \\) (escape p (inc i))
        :else [[:char (literal p i (cp-at p i))] (inc i)]))

(defn- char-class
  "The class after a [ at i - 1: [term i']."
  [p ^long i]
  (let [negated? (at? p i \^)
        start (if negated? (inc i) i)]
    (loop [i start rs []]
      (cond (nil? (cp-at p i)) (fail p i "an unclosed [")
            (at? p i \]) (if (== i start)
                           (fail p i "an empty class (escape a ] that is a member)")
                           [(chars-term (cond-> rs negated? ranges-complement)) (inc i)])
            :else
            (let [[[kind x] j] (class-item p i)
                  j (long j)]
              (if (and (at? p j \-) (some? (cp-at p (inc j))) (not (at? p (inc j) \])))
                (let [[[kind' y] k] (class-item p (inc j))]
                  (when (or (= kind :set) (= kind' :set)) (fail p i "a class escape cannot end a range"))
                  (when (< (long y) (long x)) (fail p i "a range out of order"))
                  (recur (long k) (conj rs [x y])))
                (recur j (if (= kind :set) (into rs x) (conj rs [x x])))))))))

(declare alternation)

(defn- group
  "The group after a ( at i - 1: [term i']."
  [p ^long i]
  (let [non-capturing? (and (at? p i \?) (at? p (inc i) \:))]
    (when (and (at? p i \?) (not non-capturing?))
      (fail p i "(? groups other than (?: are not supported"))
    (let [[r j] (alternation p (if non-capturing? (+ i 2) i))]
      (when-not (at? p j \)) (fail p i "an unclosed ("))
      [r (inc (long j))])))

(defn- atom-at
  "The atom at i: [term i'], or nil at the end of an alternative."
  [p ^long i]
  (when-let [c (cp-at p i)]
    (case (ch c)
      (\| \)) nil
      \( (group p (inc i))
      \[ (char-class p (inc i))
      \. [(chars-term (ranges-complement [[10 10]])) (inc i)]
      \\ (let [[[kind x] j] (escape p (inc i))]
           [(chars-term (if (= kind :set) x [[x x]])) j])
      \^ (if (zero? i) [epsilon (inc i)] (fail p i "^ only first (escape it for the character)"))
      \$ (if (== i (dec (count (:cps p)))) [epsilon (inc i)] (fail p i "$ only last (escape it for the character)"))
      (\* \+ \? \{) (fail p i (str "nothing to repeat before " (ch c)))
      (\] \}) (fail p i (str "an unescaped " (ch c) " (escape it)"))
      (let [c (literal p i c)] [(chars-term [[c c]]) (inc i)]))))

(defn- count-at
  "The decimal count at i, or nil: [n i']."
  [p ^long i]
  (let [digit? (fn [j] (when-let [c (cp-at p j)] (<= 48 (long c) 57)))
        j (long (loop [j i] (if (digit? j) (recur (inc j)) j)))]
    [(when (< i j) (reduce (fn [^long n d] (min Integer/MAX_VALUE (+ (* n 10) (- (long d) 48)))) 0 (subvec (:cps p) i j)))
     j]))

(defn- bounds
  "The bounds after a { at i - 1: [[lo hi] i'] (hi nil for none)."
  [p ^long i]
  (let [[lo j] (count-at p i)
        j (long j)]
    (cond (nil? lo) (fail p i "a { not followed by a count (escape it for the character)")
          (at? p j \}) [[lo lo] (inc j)]
          (and (at? p j \,) (at? p (inc j) \})) [[lo nil] (+ j 2)]
          (at? p j \,) (let [[hi k] (count-at p (inc j))]
                         (when-not (and hi (at? p k \})) (fail p i "a malformed {n,m}"))
                         (when (< (long hi) (long lo)) (fail p i "a {n,m} with m < n"))
                         [[lo hi] (inc (long k))])
          :else (fail p i "a malformed {n,m}"))))

(defn- quantified
  "The atom at i and its quantifier, if any: [term i'], or nil."
  [p ^long i]
  (when-let [[r j] (atom-at p i)]
    (let [j (long j)
          q (some-> (cp-at p j) ch)]
      (if-not (#{\* \+ \? \{} q)
        [r j]
        (let [[r k] (case q
                      \* [(star r) (inc j)]
                      \+ [(plus r) (inc j)]
                      \? [(opt r) (inc j)]
                      \{ (let [[[lo hi] k] (bounds p (inc j))] [(repeat r lo hi) k]))
              ;; a lazy ? matches the same language; another quantifier is loud
              k (if (at? p k \?) (inc (long k)) (long k))]
          (cond (at? p k \+) (fail p k "possessive quantifiers change the language and are not supported")
                (#{\* \? \{} (some-> (cp-at p k) ch)) (fail p k "a quantifier after a quantifier")
                :else [r k]))))))

(defn- concatenation [p ^long i]
  (loop [i i rs []]
    (if-let [[r j] (quantified p i)]
      (recur (long j) (conj rs r))
      [(apply cat rs) i])))

(defn- alternation [p ^long i]
  (loop [[r j] (concatenation p i) rs []]
    (if (at? p j \|)
      (recur (concatenation p (inc (long j))) (conj rs r))
      [(apply alt (conj rs r)) j])))

(defn parse
  "The term of a regex (see the namespace's docstring for the syntax)."
  [^String regex]
  (when-not (string? regex)
    (throw (ex-info (str "parse: a regex is a string, not " (pr-str regex)) {:reason :invalid-regex :regex regex})))
  (let [p {:regex regex :cps (mapv long (.toArray (.codePoints regex)))}
        [r i] (alternation p 0)]
    (when (some? (cp-at p i)) (fail p i "an unbalanced )"))
    r))

;; ---------------------------------------------------------------------------
;; DFAs
;; ---------------------------------------------------------------------------

(defn- term-of [what r]
  (if (string? r) (parse r) (check-term what r)))

(defn dfa
  "The minimal DFA (gen.llm.dfa) of r, a term or a regex, by derivatives:
   each state a derivative of r. opts: :max-states (default 20,000), past
   which construction is a loud error. (A DFA can be exponentially larger
   than its regex: nested counted repeats of alternatives that overlap,
   such as ((a|b)*a(a|b)){2,4}, make it track sets of positions.)"
  ([r] (dfa r {}))
  ([r {:keys [max-states] :or {max-states 20000}}]
   (let [r (term-of "dfa" r)
         memo (java.util.HashMap.)
         classes (byte-classes r)
         reps (mapv first classes)
         class-of (reduce (fn [m [i bs]] (reduce #(assoc %1 %2 i) m bs)) {} (map-indexed vector classes))]
     (loop [ids {r 0} terms [r] rows []]
       (let [i (count rows)]
         (if (== i (count terms))
           (let [n (count terms)
                 table (int-array (* n 256))]
             (doseq [s (range n), ^long b (range 256)]
               (aset table (+ (* (long s) 256) b) (int ((rows s) (class-of b)))))
             (dfa/from-table 0 table (boolean-array (map nullable? terms))))
           (let [succ (mapv #(derivative memo (terms i) %) reps)
                 [ids terms] (reduce (fn [[ids terms] d]
                                       (cond (or (= :nothing d) (contains? ids d)) [ids terms]
                                             (<= (long max-states) (count terms))
                                             (throw (ex-info (str "dfa: more than " max-states " states (:max-states)")
                                                             {:reason :too-many-states :max-states max-states}))
                                             :else [(assoc ids d (count terms)) (conj terms d)]))
                                     [ids terms] succ)]
             (recur ids terms (conj rows (mapv #(if (= :nothing %) -1 (ids %)) succ))))))))))

(defn matches?
  "Is the string text in the language of r (a term, a regex or a DFA)?"
  [r ^String text]
  (dfa/matches? (if (dfa/dfa? r) r (dfa r)) (.getBytes text StandardCharsets/UTF_8)))
