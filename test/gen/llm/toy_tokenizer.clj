(ns gen.llm.toy-tokenizer
  "A tokenizer (gen.llm.lm's map) for the toy engines' vocabularies of at
   least 97 tokens: tokens 0 to 94 are the printable ASCII characters 32 to
   126, one each; 95 is BOS (<s>), 96 EOS (</s>), and any further token t
   is <t>.")

(set! *warn-on-reflection* true)

(def bos 95)
(def eos 96)

(defn- piece ^String [^long t]
  (cond (< t 95) (str (char (+ t 32)))
        (= t bos) "<s>"
        (= t eos) "</s>"
        :else (str "<" t ">")))

(defn tokenizer
  [n-vocab]
  (when (< (long n-vocab) 97)
    (throw (ex-info "the toy tokenizer needs at least 97 tokens" {:n-vocab n-vocab})))
  {:n-vocab n-vocab
   :bos bos
   :eos eos
   :eog #{eos}
   :add-bos? true
   :token-bytes (fn [t] (.getBytes ^String (piece t) "UTF-8"))
   :encode (fn [^String text {:keys [add-special?] :or {add-special? true}}]
             (into (if add-special? [bos] [])
                   (map (fn [c]
                          (let [i (long (int c))]
                            (if (<= 32 i 126)
                              (- i 32)
                              (throw (ex-info (str "the toy tokenizer has no token for " (pr-str c))
                                              {:reason :untokenizable :char c}))))))
                   text))
   :decode (fn [tokens] (apply str (map piece tokens)))})
