(ns gen.llm.regex-test
  "Regular languages (gen.llm.regex) and their DFAs (gen.llm.dfa).

   - The regex syntax against java.util.regex (UNIX_LINES): random regexes
     over characters of one to four UTF-8 bytes, matched against random
     strings, against members drawn from each regex's tree (an oracle of
     its own), and against strings the DFA itself accepts.
   - The term constructors build the same languages as the syntax.
   - Characters: (chars [lo hi]) accepts exactly the UTF-8 encodings of the
     characters lo to hi, checked over every string of one and two bytes and
     random ones of three and four.
   - intersect and complement over byte strings.
   - DFAs are minimal (every two states distinguished by some string) and
     canonical (equal languages, equal DFAs), and (dfa spec) keeps a
     transition map's language.
   - Misuse is loud.

   Mutations checked: see docs/research/llm-grammar.md, Tests."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [gen.llm.branching-contract :as bc]
            [gen.llm.dfa :as dfa]
            [gen.llm.regex :as regex])
  (:import (java.nio.charset StandardCharsets)
           (java.util.regex Pattern)))

(defn- utf8 ^bytes [^String s] (.getBytes s StandardCharsets/UTF_8))

(defn- java-matches? [^String regex ^String s]
  (.matches (.matcher (Pattern/compile regex Pattern/UNIX_LINES) s)))

(defn- string-of [cps] (String. (int-array cps) 0 (count cps)))

;; ---------------------------------------------------------------------------
;; Random regexes, as trees rendered to the syntax and built with the
;; constructors
;; ---------------------------------------------------------------------------

(def ^:private alphabet
  "Characters of one to four bytes, a newline, and metacharacters."
  [97 98 99 10 45 46 34 233 8364 119070])

(def ^:private others
  "Characters in no tree, for strings outside the trees' classes."
  [100 32 9 11 255 0xFFFF 0x10FFFF])

(def ^:private meta-chars (set (map int "\\.[]{}()*+?|^$")))

(def ^:private gen-char (gen/elements alphabet))

(def ^:private gen-range
  (gen/fmap (fn [[a b]] [(min a b) (max a b)]) (gen/tuple gen-char gen-char)))

(def ^:private gen-char-class
  (gen/let [negated? gen/boolean
            items (gen/vector (gen/one-of [(gen/fmap (fn [c] [:char c]) gen-char)
                                           (gen/fmap (fn [r] [:range r]) gen-range)
                                           (gen/fmap (fn [e] [:escape e]) (gen/elements [\d \w \s \D \W \S]))])
                              1 3)]
    [:class negated? items]))

(def ^:private gen-tree
  (gen/recursive-gen
   (fn [inner]
     (gen/one-of [(gen/fmap #(into [:cat] %) (gen/vector inner 2 3))
                  (gen/fmap #(into [:alt] %) (gen/vector inner 2 3))
                  (gen/fmap (fn [t] [:star t]) inner)
                  (gen/fmap (fn [t] [:plus t]) inner)
                  (gen/fmap (fn [t] [:opt t]) inner)
                  (gen/let [t inner lo (gen/choose 0 2) extra (gen/elements [0 1 2 nil])]
                    [:repeat t lo (some->> extra (+ lo))])]))
   (gen/frequency [[6 (gen/fmap (fn [c] [:char c]) gen-char)]
                   [2 gen-char-class]
                   [1 (gen/return [:dot])]
                   [1 (gen/fmap (fn [e] [:escape e]) (gen/elements [\d \w \s \D]))]
                   [1 (gen/return [:empty])]])))

(defn- render-char [c in-class?]
  (cond (= c 10) "\\n"
        (or (meta-chars c) (and in-class? (#{(int \-) (int \&)} c))) (str "\\" (char c))
        :else (string-of [c])))

(defn- render
  "The regex of a tree."
  [[kind & args :as t]]
  (let [group (fn [t] (if (#{:cat :alt :star :plus :opt :repeat} (first t)) (str "(?:" (render t) ")") (render t)))]
    (case kind
      :char (render-char (first args) false)
      :dot "."
      :empty "(?:)"
      :escape (str "\\" (first args))
      :class (let [[negated? items] args]
               (str "[" (when negated? "^")
                    (apply str (for [[k x] items]
                                 (case k
                                   :char (render-char x true)
                                   :range (str (render-char (x 0) true) "-" (render-char (x 1) true))
                                   :escape (str "\\" x))))
                    "]"))
      :cat (apply str (map #(if (= :alt (first %)) (str "(?:" (render %) ")") (render %)) args))
      :alt (clojure.string/join "|" (map render args))
      :star (str (group (first args)) "*")
      :plus (str (group (first args)) "+")
      :opt (str (group (first args)) "?")
      :repeat (let [[t lo hi] args]
                (str (group t) "{" lo (cond (nil? hi) "," (= lo hi) "" :else (str "," hi)) "}")))))

(def ^:private escape-chars
  {\d [[48 57]] \w [[48 57] [65 90] [95 95] [97 122]] \s [[9 13] [32 32]]})

(defn- escape-term [e]
  (let [rs (escape-chars (Character/toLowerCase (char e)))]
    (if (Character/isUpperCase (char e)) (apply regex/chars-except rs) (apply regex/chars rs))))

(defn- build
  "The term of a tree, by the constructors."
  [[kind & args]]
  (case kind
    :char (regex/chars (first args))
    :dot (regex/chars-except \newline)
    :empty regex/epsilon
    :escape (escape-term (first args))
    :class (let [[negated? items] args
                 members (apply regex/alt (for [[k x] items]
                                            (case k
                                              (:char :range) (regex/chars x)
                                              :escape (escape-term x))))]
             (if negated? (regex/intersect regex/any-char (regex/complement members)) members))
    :cat (apply regex/cat (map build args))
    :alt (apply regex/alt (map build args))
    :star (regex/star (build (first args)))
    :plus (regex/plus (build (first args)))
    :opt (regex/opt (build (first args)))
    :repeat (let [[t lo hi] args] (regex/repeat (build t) lo hi))))

(defn- class-chars
  "The characters of the alphabet and others a class tree holds."
  [[_ negated? items]]
  (let [in? (fn [c] (some (fn [[k x]]
                            (case k
                              :char (= c x)
                              :range (<= (x 0) c (x 1))
                              :escape (let [rs (escape-chars (Character/toLowerCase (char x)))
                                            hit (some (fn [[lo hi]] (<= lo c hi)) rs)]
                                        (if (Character/isUpperCase (char x)) (not hit) hit))))
                          items))]
    (filter #(if negated? (not (in? %)) (in? %)) (concat alphabet others))))

(defn- gen-member
  "A string of a tree's language, drawn by the tree (nil when the tree's
   classes hold none of the characters at hand)."
  [[kind & args :as t]]
  (let [pick (fn [cs] (if (seq cs) (gen/fmap vector (gen/elements cs)) (gen/return nil)))
        all-of (fn [gs] (gen/fmap #(when (every? some? %) (vec (apply concat %))) (apply gen/tuple gs)))
        reps (fn [t n] (all-of (repeat n (gen-member t))))]
    (case kind
      :char (gen/return [(first args)])
      :dot (pick (remove #{10} (concat alphabet others)))
      :empty (gen/return [])
      :escape (pick (class-chars [:class false [[:escape (first args)]]]))
      :class (pick (class-chars t))
      :cat (all-of (map gen-member args))
      :alt (gen/one-of (map gen-member args))
      (:star :plus :opt :repeat)
      (let [[lo hi] (case kind :star [0 3] :plus [1 3] :opt [0 1] :repeat [(second args) (or (nth args 2) (+ 2 (second args)))])]
        (gen/bind (gen/choose lo hi) #(reps (first args) %))))))

(defn- walk-member
  "A string the DFA accepts, by a random walk (from seed), or nil."
  [d ^long seed]
  (let [rnd (java.util.Random. seed)]
    (loop [s (dfa/start d) out [] steps 0]
      (cond (neg? s) nil
            (and (dfa/accepting? d s) (or (< 20 steps) (zero? (.nextInt rnd 3)))) (byte-array out)
            (< 40 steps) nil
            :else (let [bs (filterv #(<= 0 (dfa/step d s %)) (range 256))]
                    (when (seq bs)
                      (let [b (bs (.nextInt rnd (count bs)))]
                        (recur (dfa/step d s b) (conj out (unchecked-byte b)) (inc steps)))))))))

(def ^:private gen-tree-and-members
  (gen/let [t gen-tree
            members (gen/vector (gen-member t) 10)]
    [t (keep #(some-> % string-of) members)]))

(defspec the-syntax-agrees-with-java-util-regex 300
  (prop/for-all [[t members] gen-tree-and-members
                 strings (gen/vector (gen/fmap string-of (gen/vector (gen/elements (concat alphabet others)) 0 6)) 20)
                 seed gen/nat]
    (let [regex (render t)
          d (regex/dfa regex)
          walked (keep #(some-> (walk-member d (+ seed %)) (String. StandardCharsets/UTF_8)) (range 10))]
      (and (every? #(= (java-matches? regex %) (regex/matches? d %)) strings)
           (every? #(and (java-matches? regex %) (regex/matches? d %)) members)
           (every? #(java-matches? regex %) walked)))))

(defspec the-constructors-agree-with-the-syntax 200
  (prop/for-all [t gen-tree]
    (= (regex/dfa (render t)) (regex/dfa (build t)))))

;; ---------------------------------------------------------------------------
;; Characters as UTF-8
;; ---------------------------------------------------------------------------

(defn- decodes-to
  "The one character the bytes encode in UTF-8, or nil."
  [^bytes bs]
  (let [dec (-> StandardCharsets/UTF_8 .newDecoder
                (.onMalformedInput java.nio.charset.CodingErrorAction/REPORT)
                (.onUnmappableCharacter java.nio.charset.CodingErrorAction/REPORT))]
    (try (let [s (str (.decode dec (java.nio.ByteBuffer/wrap bs)))]
           (when (= 1 (.codePointCount s 0 (count s))) (.codePointAt s 0)))
         (catch java.nio.charset.CharacterCodingException _ nil))))

(def ^:private gen-code-range
  (gen/fmap sort (gen/vector (gen/one-of [(gen/choose 0 0x7FF) (gen/choose 0x800 0xFFFF) (gen/choose 0 0x10FFFF)]) 2)))

(defspec chars-accepts-exactly-the-utf8-of-its-characters 40
  (prop/for-all [[lo hi] gen-code-range
                 longer (gen/vector (gen/vector (gen/choose 0 255) 3 4) 300)
                 inside (gen/vector (gen/choose 0 0x10FFFF) 50)]
    (let [d (regex/dfa (regex/chars [lo hi]))
          ok? (fn [^bytes bs] (= (dfa/matches? d bs)
                                 (boolean (when-let [c (decodes-to bs)] (<= lo c hi)))))]
      (and (every? ok? (for [a (range 256)] (byte-array [(unchecked-byte a)])))
           (every? ok? (for [a (range 128 256) b (range 128 256)] (byte-array [(unchecked-byte a) (unchecked-byte b)])))
           (every? ok? (map #(byte-array (map unchecked-byte %)) longer))
           (every? #(or (<= 0xD800 % 0xDFFF)
                        (= (<= lo % hi) (regex/matches? d (string-of [%]))))
                   inside)))))

(deftest every-character-and-nothing-else
  (let [any (regex/dfa regex/any-char)]
    (is (every? #(regex/matches? any (string-of [%])) [0 10 127 128 0x7FF 0x800 0xD7FF 0xE000 0xFFFF 0x10000 0x10FFFF]))
    (is (not (dfa/matches? any (byte-array (map unchecked-byte [0xED 0xA0 0x80])))) "an encoded surrogate")
    (is (not (dfa/matches? any (byte-array (map unchecked-byte [0xC0 0x80])))) "an overlong encoding")
    (is (not (dfa/matches? any (byte-array (map unchecked-byte [0xF4 0x90 0x80 0x80])))) "past U+10FFFF")
    (is (not (dfa/matches? any (byte-array [])))))
  (is (regex/matches? (regex/text "naïve 𝄞") "naïve 𝄞"))
  (is (= (regex/dfa (regex/text "é")) (regex/dfa (regex/cat (regex/bytes-of [0xC3]) (regex/bytes-of [0xA9]))))))

;; ---------------------------------------------------------------------------
;; Intersection and complement, over byte strings
;; ---------------------------------------------------------------------------

(def ^:private gen-byte-term
  (gen/recursive-gen
   (fn [inner]
     (gen/one-of [(gen/fmap #(apply regex/cat %) (gen/vector inner 2))
                  (gen/fmap #(apply regex/alt %) (gen/vector inner 2))
                  (gen/fmap #(apply regex/intersect %) (gen/vector inner 2))
                  (gen/fmap regex/complement inner)
                  (gen/fmap regex/star inner)]))
   (gen/fmap regex/bytes-of (gen/set (gen/choose 0 3) {:min-elements 1}))))

(defspec intersect-and-complement-are-boolean 200
  (prop/for-all [r gen-byte-term
                 s gen-byte-term
                 strings (gen/vector (gen/vector (gen/choose 0 4) 0 5) 30)]
    (let [[dr ds di dc] (map regex/dfa [r s (regex/intersect r s) (regex/complement r)])
          in? (fn [d xs] (dfa/matches? d (byte-array (map unchecked-byte xs))))]
      (every? #(and (= (in? di %) (and (in? dr %) (in? ds %)))
                    (= (in? dc %) (not (in? dr %))))
              strings))))

(deftest text-not-containing-a-string
  ;; the tool-call value of GenMLX's toolcall: any text without the closing tag
  (let [no-close (regex/intersect (regex/star regex/any-char)
                                  (regex/complement (regex/cat (regex/star regex/any-byte) (regex/text "</p>")
                                                               (regex/star regex/any-byte))))
        d (regex/dfa (regex/cat no-close (regex/text "</p>")))]
    (is (regex/matches? d "héllo <b> </p>"))
    (is (not (regex/matches? d "a </p> b </p>")))
    (is (not (regex/matches? d "héllo")))))

;; ---------------------------------------------------------------------------
;; DFAs: minimal, canonical, from transition maps
;; ---------------------------------------------------------------------------

(defn- byte-reps
  "A byte of each class of bytes that d's states treat alike."
  [d]
  (map first (vals (group-by (fn [x] (mapv #(dfa/step d % x) (range (dfa/size d)))) (range 256)))))

(defn- distinguished?
  "Do states s and t have different languages (a breadth-first search over
   pairs for a string one accepts and the other not)?"
  [d reps s t]
  (loop [seen #{[s t]} queue [[s t]]]
    (when-let [[a b] (first queue)]
      (if (not= (dfa/accepting? d a) (dfa/accepting? d b))
        true
        (let [next (distinct (for [x reps] [(dfa/step d a x) (dfa/step d b x)]))
              new (remove seen next)]
          (recur (into seen new) (into (subvec queue 1) new)))))))

(defspec dfas-are-minimal 100
  (prop/for-all [t gen-tree]
    (let [d (regex/dfa (render t))
          n (dfa/size d)
          reps (byte-reps d)]
      (every? true? (for [s (range n) u (range (inc s) n)] (boolean (distinguished? d reps s u)))))))

(deftest equal-languages-have-equal-dfas
  (doseq [[a b] [["a|b" "[ab]"] ["(a|b)*" "[ab]*"] ["a*a*" "a*"] ["(ab)*a" "a(ba)*"] ["a{2,}" "aaa*"]
                 ["x?x?x?" "x{0,3}"] ["[0-9]" "\\d"] ["(?:a|ab)(?:c|bcd)" "ac|abcd|abc|abbcd"]
                 ["é|ë" "[éë]"]]]
    (is (= (regex/dfa a) (regex/dfa b)) [a b]))
  (is (not= (regex/dfa "a*") (regex/dfa "a+"))))

(def ^:private gen-transition-map
  (gen/let [n (gen/choose 1 6)
            edges (gen/vector (gen/tuple (gen/choose 0 (dec n)) (gen/choose 0 2) (gen/choose 0 (dec n))) 0 14)
            accept (gen/set (gen/choose 0 (dec n)))]
    {:start 0 :accept accept
     :transitions (reduce (fn [m [s b s']] (assoc-in m [s b] s')) {} edges)}))

(defn- map-accepts? [{:keys [start accept transitions]} xs]
  (contains? accept (reduce (fn [s x] (get-in transitions [s x] ::dead)) start xs)))

(defspec a-transition-map-keeps-its-language 200
  (prop/for-all [spec gen-transition-map
                 strings (gen/vector (gen/vector (gen/choose 0 3) 0 6) 40)]
    (let [d (dfa/dfa spec)
          n (dfa/size d)
          reps (byte-reps d)]
      (and (every? #(= (map-accepts? spec %) (dfa/matches? d (byte-array (map unchecked-byte %)))) strings)
           (every? true? (for [s (range n) u (range (inc s) n)] (boolean (distinguished? d reps s u))))
           (= d (dfa/dfa (dfa/->data d)))))))

(deftest every-state-is-live-and-reached
  (let [d (dfa/dfa {:start 0 :accept #{2} :transitions {0 {97 1 98 3} 1 {98 2} 3 {98 3} 4 {97 2}}})]
    (is (= 3 (dfa/size d)) "3 never accepts, 4 is never reached")
    (is (= -1 (dfa/step d (dfa/start d) 98)) "a byte into a dead state leads to -1")
    (is (= -1 (dfa/walk d (dfa/start d) (utf8 "bb"))))
    (is (= -1 (dfa/step d -1 97)) "-1 stays -1")
    (is (dfa/matches? d (utf8 "ab"))))
  (let [empty (dfa/dfa {:start 0 :accept #{} :transitions {0 {97 0}}})]
    (is (zero? (dfa/size empty)))
    (is (= -1 (dfa/start empty)))
    (is (not (dfa/matches? empty (utf8 ""))))
    (is (= empty (regex/dfa (regex/intersect (regex/text "a") (regex/text "b")))))))

;; ---------------------------------------------------------------------------
;; Misuse
;; ---------------------------------------------------------------------------

(deftest misuse-is-loud
  (doseq [bad ["(a" "a)" "[a" "[]" "[a-]b-a]" "a**" "a*+" "a{2}{3}" "*a" "a{" "a{x}" "a{3,2}" "a{,2}" "}" "]"
               "\\1" "\\b" "\\p{L}" "(?=a)" "(?i)a" "a^" "$a" "[\\d-z]" "[z-a]" "[a[b]]" "[a&&b]" "\\x{110000}"
               "\\uD800" "\\x4" "\\"]]
    (is (= :invalid-regex (bc/reason #(regex/parse bad))) bad))
  (is (= :invalid-regex (bc/reason #(regex/parse 3))))
  (is (= :too-many-states (bc/reason #(regex/dfa "a{0,500}" {:max-states 100}))))
  (doseq [[what f] [["a term" #(regex/cat "a")] ["bytes" #(regex/bytes-of [256])] ["bounds" #(regex/repeat regex/any-byte 3 2)]
                    ["a char" #(regex/chars :a)] ["a range" #(regex/chars [\z \a])] ["text" #(regex/text 3)]
                    ["a dfa's term" #(regex/dfa [:what])]]]
    (is (= :invalid-term (bc/reason f)) what))
  (doseq [bad [{:start 0 :accept #{} :transitions {0 {256 0}}} {:start 0 :accept [1] :transitions {}} 3]]
    (is (= :invalid-dfa (bc/reason #(dfa/dfa bad))) (pr-str bad))))

(deftest lazy-quantifiers-are-the-same-language
  (is (= (regex/dfa "a*?b+?c??d{1,2}?") (regex/dfa "a*b+c?d{1,2}"))))
