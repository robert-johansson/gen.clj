(ns gen.llm.llama-lora-test
  "LoRA adapters on the llama.cpp engine (gen.llm.llama): models that
   engines share, adapters loaded for a model and set, with scales, on an
   engine's context, and their errors and lifetimes.

   The adapters are test/fixtures/lora's, for Qwen3-0.6B
   (test/fixtures/lora/generate.py): a rank-2 adapter of attn_q and
   attn_v in four layers, the same with B = 0, and an aLoRA. With another
   test model, llama.cpp refuses them (another architecture or other
   shapes), which is checked, and the rest is skipped with a note.

   Needs llama.cpp's libraries (native/build-llama.sh), the compiled
   binding (clojure -T:build java) and a model, given as
   -J-Dgen.llama.test-model=<a GGUF file> (or GEN_LLAMA_TEST_MODEL), as
   gen.llm.llama-test does; skipped with a note otherwise.

   The noise checks use the bands of gen.llm.llama.bands (0.1 nats on
   Metal), over the 64 most likely tokens of the rows after a prompt and
   after one more token, and the scores of a continuation. Measured on
   Metal (Mac mini M4, 2026-11-07, Qwen3-0.6B-Q8_0), |Δ log p| against
   the base: the B = 0 adapter 0.00 at scales 1 and 3; the rank-2 one
   5.95 at scale 1, 14.3 at -1, 0.00 at 0, and 7.50 between scales 0.5
   and 1; both adapters against the rank-2 one alone 0.00.

   Mutations of gen.llm.llama checked there (each planted with with-redefs
   in a JVM of its own, the tests run, the failures named):
   - adapters never set on the context: an-adapter-changes-the-reads-by-
     its-scale, two-adapters-add, engines-over-one-model-are-independent;
   - every scale taken as 1: an-adapter-changes-the-reads-by-its-scale;
   - only the first adapter set: two-adapters-add;
   - an adapter's close (and its Cleaner action) freeing nothing:
     closing-is-checked, unreachable-models-and-adapters-are-freed,
     bad-adapters-are-loud, models-and-adapters-describe-themselves;
   - an engine never deregistered from its model and adapters:
     closing-is-checked;
   - no checks of an engine's model and adapters: bad-adapters-are-loud.
   (The last two leave the model open at the JVM's exit, where ggml-metal
   aborts on the Metal buffers still alive: close models and engines
   before exiting.)"
  (:require [clojure.test :refer [deftest is testing use-fixtures]]
            [gen.array.memory :as reservations]
            [gen.llm.branching :as br]
            [gen.llm.branching-contract :as contract :refer [reason]]
            [gen.llm.llama.bands :as bands]
            [gen.test-support :as ts]))

(def ^:private model-path
  (or (System/getProperty "gen.llama.test-model") (System/getenv "GEN_LLAMA_TEST_MODEL")))

(def ^:private fixtures
  {:lora "test/fixtures/lora/qwen3-0.6b-lora.gguf"
   :zero "test/fixtures/lora/qwen3-0.6b-lora-zero.gguf"
   :alora "test/fixtures/lora/qwen3-0.6b-alora.gguf"})

(defn- llama
  "A function of gen.llm.llama, loaded only when the tests run (it imports
   the compiled binding)."
  [sym]
  @(requiring-resolve (symbol "gen.llm.llama" (name sym))))

(def ^:private available?
  (delay
    (let [skip (fn [why] (println "Skipping the llama.cpp LoRA tests:" why) (ts/native-unavailable! :llama why) false)]
      (cond
        (nil? model-path) (skip "no model (-J-Dgen.llama.test-model=<a GGUF file>, or GEN_LLAMA_TEST_MODEL)")
        (not (.isFile (java.io.File. ^String model-path))) (skip (str "no model file at " model-path))
        :else (try ((llama 'system-info)) true
                   (catch Throwable t
                     (skip (str (or (ex-message t) (.getName (class t)))
                                " (build with native/build-llama.sh and clojure -T:build java)"))))))))

(def ^:private shared (atom nil))

(defn- model
  "The test model, loaded once and shared by the tests' engines."
  []
  (or @shared (reset! shared ((llama 'model) model-path))))

(use-fixtures :once (fn [f]
                      (try (f)
                           (finally (when-let [^java.io.Closeable m @shared]
                                      (.close m)
                                      (reset! shared nil))))))

(def ^:private fits?
  "Do the fixtures fit the test model (Qwen3-0.6B's architecture and
   shapes)? With another model llama.cpp refuses them, loudly."
  (delay
    (let [x (try (.close ^java.io.Closeable ((llama 'adapter) (model) (:lora fixtures))) nil
                 (catch clojure.lang.ExceptionInfo x x))]
      (or (nil? x)
          (do (println "Skipping the LoRA tests that read: the fixtures are Qwen3-0.6B's, and llama.cpp refuses them for"
                       (:desc ((llama 'model-info) (model))) (str "(" (:llama-error (ex-data x)) ")"))
              (is (= :adapter-load-failed (:reason (ex-data x))) (pr-str x))
              false)))))

(defmacro ^:private when-lora
  "body when llama.cpp, the test model and the fixtures fit, else a
   skipped test (counted, gen.test-support/when-native) or a note."
  [& body]
  `(ts/when-native :llama available? (when @fits? ~@body)))

(defn- engine
  "An engine over the shared model, with adapters ({adapter scale})."
  ^java.io.Closeable [adapters]
  ((llama 'engine) {:model (model) :max-branches 4 :n-ctx 512 :adapters adapters}))

(defn- adapter ^java.io.Closeable [k] ((llama 'adapter) (model) (fixtures k)))

(defn- text [e]
  {:prompt ((llama 'tokenize) e "The capital of Sweden is a city called")
   :continuation ((llama 'tokenize) e " Stockholm, which is known for its many islands and" {:add-special? false})})

(defn- reads
  "What engine e reads: the distribution after the prompt, after one more
   token (a one-token step), and the continuation's scores."
  [e]
  (let [{:keys [prompt continuation]} (text e)
        b (br/open e prompt)
        [s] (br/advance [(br/fork b)] [(first continuation)])
        out {:open (br/logprobs b) :step (br/logprobs s) :scores (br/score b continuation)}]
    (br/dispose! [b s])
    out))

(defn- diff
  "The largest |Δ log p| between two engines' reads: over the 64 most
   likely tokens of x's rows, and over the continuation's scores."
  ^double [x y]
  (max (contract/max-diff (:open x) (:open y) 64)
       (contract/max-diff (:step x) (:step y) 64)
       (reduce max 0.0 (map #(Math/abs (- (double %1) (double %2))) (:scores x) (:scores y)))))

(defn- band
  "The engine's noise band, or nil (its note printed) where it has none."
  [e]
  (let [{:keys [band skip]} (bands/bands (br/info e))]
    (when-let [note (:band skip)] (println note))
    band))

(defn- with-engines
  "(f e1 e2 …) with engines of the given adapter sets, closed after."
  [adapter-sets f]
  (let [es (mapv engine adapter-sets)]
    (try (apply f es)
         (finally (run! #(.close ^java.io.Closeable %) es)))))

;; ---------------------------------------------------------------------------
;; Reads under adapters
;; ---------------------------------------------------------------------------

(deftest an-adapter-with-b-zero-reads-as-the-base-model
  ;; B (A x) = 0: llama.cpp adds exact zeros to each adapted product
  (when-lora
    (with-open [zero (adapter :zero)]
      (with-engines [{} {zero 1.0} {zero 3.0}]
        (fn [base z1 z3]
          (when-let [b (band base)]
            (let [r (reads base)]
              (doseq [[k e] {:scale-1 z1 :scale-3 z3}
                      :let [d (diff r (reads e))]]
                (println (format "lora %s: B = 0 against the base, |Δ log p| at most %.3g" k d))
                (is (<= d b) (str k ": " d))))))))))

(deftest an-adapter-changes-the-reads-by-its-scale
  ;; the rank-2 adapter moves the reads far outside the noise; its scale
  ;; matters, and scale 0 (which llama.cpp leaves out) is the base again
  (when-lora
    (with-open [lora (adapter :lora)]
      (with-engines [{} {lora 1.0} {lora 0.5} {lora 0.0} {lora -1.0}]
        (fn [base l1 l05 l0 l-1]
          (when-let [b (band base)]
            (let [r (reads base)
                  r1 (reads l1)
                  d1 (diff r r1)
                  d05 (diff r1 (reads l05))
                  d0 (diff r (reads l0))
                  d-1 (diff r (reads l-1))]
              (println (format "lora: |Δ log p| against the base %.3g at scale 1, %.3g at -1, %.3g at 0; scale 0.5 against 1 %.3g"
                               d1 d-1 d0 d05))
              (is (< (* 10 b) d1) "scale 1 is far from the base")
              (is (< (* 10 b) d-1) "and so is scale -1")
              (is (< b d05) "scale 0.5 is not scale 1")
              (is (<= d0 b) "scale 0 reads as the base"))))))))

(deftest two-adapters-add
  ;; the B = 0 adapter beside the rank-2 one adds nothing (both set, in one
  ;; call, the zero one first)
  (when-lora
    (with-open [lora (adapter :lora)
                zero (adapter :zero)]
      (with-engines [{} {lora 1.0} (array-map zero 1.0 lora 1.0)]
        (fn [base l1 both]
          (when-let [b (band base)]
            (let [r1 (reads l1)
                  d (diff r1 (reads both))]
              (println (format "two adapters: |Δ log p| against the rank-2 one alone %.3g" d))
              (is (<= d b))
              (is (< (* 10 b) (diff (reads base) (reads both))) "and the pair is not the base"))))))))

(deftest engines-over-one-model-are-independent
  ;; an engine's adapters are its context's: the base engine reads the same
  ;; before and after an engine with an adapter is made and read, and as an
  ;; engine that loaded the model itself; the adapted engine reads the same
  ;; after the base engine has read again
  (when-lora
    (with-open [lora (adapter :lora)
                base (engine {})]
      (when-let [b (band base)]
        (let [before (reads base)]
          (with-open [tuned (engine {lora 1.0})]
            (let [t1 (reads tuned)
                  after (reads base)
                  t2 (reads tuned)]
              (is (<= (diff before after) b) "the base engine is untouched")
              (is (<= (diff t1 t2) b) "and so is the adapted one")
              (is (< (* 10 b) (diff after t2)) "while they differ")))
          (with-open [own ((llama 'engine) {:model model-path :max-branches 4 :n-ctx 512})]
            (is (<= (diff before (reads own)) b) "an engine with a model of its own reads the same")))))))

;; ---------------------------------------------------------------------------
;; What models, adapters and engines describe
;; ---------------------------------------------------------------------------

(deftest models-and-adapters-describe-themselves
  (when-lora
    (let [m (model)
          r0 (reservations/reserved)]
      (with-open [lora (adapter :lora)
                  alora (adapter :alora)]
        (let [info ((llama 'adapter-info) lora)]
          (is (= (:lora fixtures) (:path info)))
          (is (= model-path (:path (:model info))))
          (is (= 4.0 (:alpha info)))
          (is (= {"adapter.type" "lora" "general.architecture" "qwen3" "general.type" "adapter"}
                 (select-keys (:meta info) ["adapter.type" "general.architecture" "general.type"])))
          (is (= [] (:alora-invocation-tokens info)))
          (is (pos? (:bytes (:footprint info))))
          (is (every? #(= :lora (:kind %)) (:buffers (:footprint info))))
          (is (= [151644 77091 198] (:alora-invocation-tokens ((llama 'adapter-info) alora))) "an aLoRA's invocation")
          (is (= (+ (:bytes (:footprint info)) (:bytes (:footprint ((llama 'adapter-info) alora))))
                 (- (reservations/reserved) r0))
              "each adapter reserves its tensors")
          (is (= 2 (:adapters ((llama 'model-info) m))))
          (let [r1 (reservations/reserved)]
            (with-open [e (engine {lora 0.5})]
              (let [{:keys [footprint adapters model]} (br/info e)
                    ctx-bytes (- (reservations/reserved) r1)]
                (is (= [{:path (:lora fixtures) :scale 0.5 :alpha 4.0}] adapters))
                (is (= model-path (:path model)))
                (is (= 1 (:engines ((llama 'adapter-info) lora))))
                (is (zero? (:engines ((llama 'adapter-info) alora))))
                (is (= 1 (:engines ((llama 'model-info) m))))
                (is (= (:bytes footprint)
                       (+ (:bytes (:footprint ((llama 'model-info) m))) (:bytes (:footprint info)) ctx-bytes))
                    "the engine uses the weights, the adapter and its context, and reserves only its context")
                (is (some #(= :lora (:kind %)) (:buffers footprint)))))
            (is (= r1 (reservations/reserved)) "the engine's context is released")
            (is (zero? (:engines ((llama 'model-info) m)))))))
      (is (= r0 (reservations/reserved)) "and the adapters' tensors")
      (is (zero? (:adapters ((llama 'model-info) m)))))))

;; ---------------------------------------------------------------------------
;; Loud errors
;; ---------------------------------------------------------------------------

(deftest bad-adapters-are-loud
  (ts/when-native :llama available?
    (let [m (model)]
      (is (= :no-adapter (reason #((llama 'adapter) m "/no/such-lora.gguf"))))
      (is (= :not-a-model (reason #((llama 'adapter) model-path (:lora fixtures)))))
      (let [junk (java.io.File/createTempFile "not-an-adapter" ".gguf")]
        (try
          (spit junk "not a GGUF file")
          (let [x (try ((llama 'adapter) m (str junk)) (catch Exception x x))]
            (is (= :adapter-load-failed (:reason (ex-data x))) (pr-str x))
            (is (string? (:llama-error (ex-data x))) "with llama.cpp's own message"))
          (finally (.delete junk))))
      (is (zero? (:adapters ((llama 'model-info) m))) "nothing is left loaded")
      (when @fits?
        (with-open [lora (adapter :lora)]
          (doseq [bad [{lora ##NaN} {lora ##Inf} {lora 1e300} {lora "1"} {"lora" 1.0} [[lora 1.0]]]]
            (testing (pr-str bad)
              (is (= :invalid-option (reason #(engine bad))))))
          (is (= :invalid-option (reason #((llama 'engine) {:model m :gpu-layers 0}))) ":gpu-layers is the model's")
          (is (= :adapter-for-another-model
                 (reason #((llama 'engine) {:model model-path :adapters {lora 1.0}})))
              "an engine that loads a model of its own")
          (with-open [other ((llama 'model) model-path)
                      theirs ((llama 'adapter) other (:lora fixtures))]
            (is (= :adapter-for-another-model (reason #(engine {theirs 1.0})))
                "an adapter of another model value, even of the same file"))
          (with-open [alora (adapter :alora)]
            (is (= :unsupported-adapter (reason #(engine {alora 1.0}))) "an aLoRA"))
          (is (zero? (:engines ((llama 'model-info) m))) "no engine was left open"))))))

(deftest closing-is-checked
  ;; an adapter or a model that an open engine uses does not close; a
  ;; closed one is refused; closing a model frees its adapters
  (when-lora
    (let [m (model)
          lora (adapter :lora)]
      (with-open [e (engine {lora 1.0})]
        (is (= :in-use (reason #(.close lora))))
        (is (= :in-use (reason #(.close ^java.io.Closeable m))))
        (is (not (:closed? ((llama 'adapter-info) lora))))
        (is (= 1 (br/position (doto (br/open e [1]) br/dispose!))) "and the engine still reads"))
      (.close lora)
      (.close lora)
      (is (:closed? ((llama 'adapter-info) lora)))
      (is (= :closed (reason #(engine {lora 1.0}))) "a closed adapter")
      (let [m2 ((llama 'model) model-path)
            a2 ((llama 'adapter) m2 (:lora fixtures))
            r (reservations/reserved)
            bytes (+ (:bytes (:footprint ((llama 'model-info) m2))) (:bytes (:footprint ((llama 'adapter-info) a2))))]
        (.close ^java.io.Closeable m2)
        (is (:closed? ((llama 'model-info) m2)))
        (is (:closed? ((llama 'adapter-info) a2)) "closing a model frees its adapters")
        (is (= (- r bytes) (reservations/reserved)) "and releases what they reserved")
        (.close a2)
        (is (= :closed (reason #((llama 'adapter) m2 (:lora fixtures)))) "a closed model loads no adapter")
        (is (= :closed (reason #((llama 'engine) {:model m2}))) "nor serves an engine"))
      (is (zero? (:engines ((llama 'model-info) m)))))))

(defn- held?
  "Does owner hold a reservation?"
  [owner]
  (contains? (reservations/reservations-held) owner))

(defn- collected?
  "Are the owners' reservations released, after up to 20 collections?"
  [owners]
  (loop [n 0]
    (cond (not-any? held? owners) true
          (= n 20) false
          :else (do (System/gc) (Thread/sleep 50) (recur (inc n))))))

(deftest unreachable-models-and-adapters-are-freed
  ;; dropped, an adapter is freed by the Cleaner, and a model with the
  ;; adapters it still holds (its books stay with the test, the values not)
  (when-lora
    (let [books #(clojure.lang.Reflector/getInstanceField % "books")
          dropped (books (adapter :lora))]
      (is (collected? [dropped]) "a dropped adapter")
      (is (:closed? @dropped))
      (is (zero? (:adapters ((llama 'model-info) (model)))))
      (let [[m-books a-books] (let [m2 ((llama 'model) model-path)]
                                [(books m2) (books ((llama 'adapter) m2 (:lora fixtures)))])]
        (is (collected? [m-books a-books]) "a dropped model and its adapter")
        (is (:closed? @m-books))
        (is (:closed? @a-books))))))
