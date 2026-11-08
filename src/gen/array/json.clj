(ns gen.array.json
  "JSON (RFC 8259) read on the host, with no dependency: the headers of
   safetensors files and their checkpoints' JSON (an index's weight map,
   config.json) for gen.array.safetensors and gen.llm.models.checkpoint;
   tokenizer.json's reader can take it too.

   Values: an object as a map from its keys (strings) to its values, an
   array as a vector, an integer as a long (a BigInt past a long's range),
   any other number as a double, true and false as booleans, null as nil.
   NaN, Infinity and -Infinity read as doubles, as Python's json module
   reads them (it writes them in config.json files). Strict otherwise, and
   loud (:invalid-json, with the position): a key twice in an object,
   trailing commas, control characters in a string, comments, anything
   after the value but whitespace. Written in plain Clojure: it requires
   nothing of gen.array."
  {:api :internal}
  (:import [java.nio.charset StandardCharsets]
           [java.nio.file Files Path]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

(definterface ^:private ICursor
  (^long pos [])
  (^void advance [^long n]))

(deftype ^:private Cursor [^String s ^:unsynchronized-mutable ^long i]
  ICursor
  (pos [_] i)
  (advance [_ n] (set! i (+ i n))))

(defn- at ^long [^Cursor c] (.pos c))

(defn- advance! [^Cursor c ^long n] (.advance c n))

(defn- fail [^Cursor c what]
  (throw (ex-info (str "JSON: " what " at position " (at c)) {:reason :invalid-json :position (at c)})))

(defn- text ^String [^Cursor c] (.-s c))

(defn- char-at
  "The char at i of s as a long, or -1 past its end."
  ^long [^String s ^long i]
  (if (< i (.length s)) (long (int (.charAt s (int i)))) -1))

(defn- peek-char
  "The char at the cursor, or -1 at the end."
  ^long [^Cursor c]
  (char-at (text c) (at c)))

(defn- skip-ws! [^Cursor c]
  (loop []
    (let [ch (peek-char c)]
      (when (or (== ch 32) (== ch 9) (== ch 10) (== ch 13))
        (advance! c 1)
        (recur)))))

(defn- expect! [^Cursor c ^String word]
  (let [s (text c) i (at c)]
    (if (.startsWith s word (int i))
      (advance! c (.length word))
      (fail c (str "expected " word)))))

(declare read-value)

(defn- hex-digit ^long [^Cursor c ^long ch]
  (cond (<= 48 ch 57) (- ch 48)
        (<= 97 ch 102) (- ch 87)
        (<= 65 ch 70) (- ch 55)
        :else (fail c "a bad \\u escape")))

(defn- run-end
  "The end of the run of plain chars (no quote, backslash or control
   char) of s from i."
  ^long [^String s ^long i]
  (loop [j i]
    (let [ch (char-at s j)]
      (if (or (== ch 34) (== ch 92) (< ch 32)) j (recur (inc j))))))

(defn- read-string* ^String [^Cursor c]
  (advance! c 1)
  (let [s (text c)
        n (.length s)
        sb (StringBuilder.)]
    (loop []
      (let [i (at c)
            j (run-end s i)]
        (.append sb s (int i) (int j))
        (advance! c (- j i))
        (let [ch (char-at s j)]
          (cond
            (== ch 34) (do (advance! c 1) (.toString sb))
            (== ch 92) (do (case (char-at s (inc j))
                             34 (.append sb \")
                             92 (.append sb \\)
                             47 (.append sb \/)
                             98 (.append sb \backspace)
                             102 (.append sb \formfeed)
                             110 (.append sb \newline)
                             114 (.append sb \return)
                             116 (.append sb \tab)
                             117 (do (when (> (+ j 6) n) (fail c "a \\u escape cut short"))
                                     (.append sb (char (+ (* 4096 (hex-digit c (char-at s (+ j 2))))
                                                          (* 256 (hex-digit c (char-at s (+ j 3))))
                                                          (* 16 (hex-digit c (char-at s (+ j 4))))
                                                          (hex-digit c (char-at s (+ j 5))))))
                                     (advance! c 4))
                             (fail c "a bad escape"))
                           (advance! c 2)
                           (recur))
            (== ch -1) (fail c "a string runs past the end")
            :else (fail c "a control character in a string")))))))

(defn- digits!
  "Advances past the digits at the cursor; their count."
  ^long [^Cursor c]
  (loop [k 0]
    (let [ch (peek-char c)]
      (if (<= 48 ch 57)
        (do (advance! c 1) (recur (inc k)))
        k))))

(defn- read-number [^Cursor c]
  (let [start (at c)
        _ (when (== 45 (peek-char c)) (advance! c 1))
        lead (peek-char c)
        int-digits (digits! c)
        _ (when (zero? int-digits) (fail c "a number without digits"))
        _ (when (and (== lead 48) (> int-digits 1)) (fail c "a number with a leading zero"))
        frac? (== 46 (peek-char c))
        _ (when frac? (advance! c 1) (when (zero? (digits! c)) (fail c "a fraction without digits")))
        exp? (let [ch (peek-char c)] (or (== ch 101) (== ch 69)))
        _ (when exp?
            (advance! c 1)
            (let [ch (peek-char c)] (when (or (== ch 43) (== ch 45)) (advance! c 1)))
            (when (zero? (digits! c)) (fail c "an exponent without digits")))
        literal (subs (text c) start (at c))]
    (if (or frac? exp?)
      (Double/parseDouble literal)
      (try (Long/parseLong literal)
           (catch NumberFormatException _ (bigint (BigInteger. literal)))))))

(defn- read-array [^Cursor c]
  (advance! c 1)
  (skip-ws! c)
  (if (== 93 (peek-char c))
    (do (advance! c 1) [])
    (loop [out (transient [])]
      (let [out (conj! out (read-value c))]
        (skip-ws! c)
        (case (peek-char c)
          44 (do (advance! c 1) (recur out))
          93 (do (advance! c 1) (persistent! out))
          (fail c "expected , or ] in an array"))))))

(defn- read-object [^Cursor c]
  (advance! c 1)
  (skip-ws! c)
  (if (== 125 (peek-char c))
    (do (advance! c 1) {})
    (loop [out (transient {})]
      (skip-ws! c)
      (when-not (== 34 (peek-char c)) (fail c "expected a key (a string)"))
      (let [k (read-string* c)]
        (when (contains? out k) (fail c (str "the key " (pr-str k) " twice in an object")))
        (skip-ws! c)
        (when-not (== 58 (peek-char c)) (fail c "expected :"))
        (advance! c 1)
        (let [out (assoc! out k (read-value c))]
          (skip-ws! c)
          (case (peek-char c)
            44 (do (advance! c 1) (recur out))
            125 (do (advance! c 1) (persistent! out))
            (fail c "expected , or } in an object")))))))

(defn- read-value [^Cursor c]
  (skip-ws! c)
  (let [ch (peek-char c)]
    (cond
      (== ch 123) (read-object c)
      (== ch 91) (read-array c)
      (== ch 34) (read-string* c)
      (== ch 116) (do (expect! c "true") true)
      (== ch 102) (do (expect! c "false") false)
      (== ch 110) (do (expect! c "null") nil)
      (== ch 78) (do (expect! c "NaN") Double/NaN)
      (== ch 73) (do (expect! c "Infinity") Double/POSITIVE_INFINITY)
      (and (== ch 45) (.startsWith (text c) "-I" (int (at c)))) (do (expect! c "-Infinity") Double/NEGATIVE_INFINITY)
      (or (== ch 45) (<= 48 ch 57)) (read-number c)
      (== ch -1) (fail c "the text ends before a value")
      :else (fail c (str "unexpected " (pr-str (char ch)))))))

(defn read-str
  "The value of the JSON text s, a string (whitespace around it allowed,
   nothing else)."
  [^String s]
  (let [c (Cursor. s 0)
        v (read-value c)]
    (skip-ws! c)
    (when-not (== -1 (peek-char c)) (fail c "text after the value"))
    v))

(defn read-bytes
  "The value of the JSON text in bytes, UTF-8 (a byte order mark is not
   JSON: loud)."
  [^bytes bs]
  (read-str (String. bs StandardCharsets/UTF_8)))

(defn read-file
  "The value of the JSON file at path, UTF-8. Loud (:no-file) when there
   is none; its errors name the file."
  [path]
  (let [p (Path/of (str path) (make-array String 0))]
    (when-not (Files/isRegularFile p (make-array java.nio.file.LinkOption 0))
      (throw (ex-info (str "no file at " path) {:reason :no-file :path (str path)})))
    (try (read-bytes (Files/readAllBytes p))
         (catch clojure.lang.ExceptionInfo e
           (throw (ex-info (str path ": " (ex-message e)) (assoc (ex-data e) :path (str path))))))))
