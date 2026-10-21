(ns gen.llm.byte-tokenizer
  "A tokenizer (gen.llm.lm's map) over pieces of bytes, for the grammar and
   byte-level tests on the toy engines: token i is piece i (a string, as
   UTF-8, or a byte array, so a piece can be part of a character), then
   BOS (<s>) and EOS (</s>). encode is greedy, the longest piece first;
   decode concatenates the bytes, as UTF-8."
  (:import (java.io ByteArrayOutputStream)
           (java.nio.charset StandardCharsets)))

(set! *warn-on-reflection* true)

(defn- as-bytes ^bytes [piece]
  (if (string? piece) (.getBytes ^String piece StandardCharsets/UTF_8) (byte-array piece)))

(defn tokenizer
  "The tokenizer of pieces, plus BOS and EOS (its :bos and :eos, the last
   two tokens)."
  [pieces]
  (let [vocab (conj (mapv as-bytes pieces) (as-bytes "<s>") (as-bytes "</s>"))
        n (count vocab)
        bos (- n 2)
        eos (dec n)
        by-length (sort-by (fn [t] (- (alength ^bytes (vocab t)))) (range bos))]
    {:n-vocab n
     :bos bos
     :eos eos
     :eog #{eos}
     :add-bos? true
     :token-bytes (fn [t] (aclone ^bytes (vocab t)))
     :encode (fn [text {:keys [add-special?] :or {add-special? true}}]
               (let [^bytes bs (as-bytes text)]
                 (loop [i 0 out (if add-special? [bos] [])]
                   (if (== i (alength bs))
                     out
                     (let [t (first (filter (fn [t] (let [^bytes p (vocab t)]
                                                      (and (pos? (alength p)) (<= (+ i (alength p)) (alength bs))
                                                           (java.util.Arrays/equals p 0 (alength p) bs i (+ i (alength p))))))
                                            by-length))]
                       (when-not t
                         (throw (ex-info "the byte tokenizer has no token here" {:reason :untokenizable :at i})))
                       (recur (+ i (alength ^bytes (vocab t))) (conj out t)))))))
     :decode (fn [tokens]
               (let [out (ByteArrayOutputStream.)]
                 (doseq [t tokens] (.write out ^bytes (vocab t)))
                 (String. (.toByteArray out) StandardCharsets/UTF_8)))}))
