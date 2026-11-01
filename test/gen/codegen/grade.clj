(ns gen.codegen.grade
  "Grading a generated model against a task (the codegen bake-off,
   genclj-ihc4): stages in order, the grading stopping at the first that
   fails.

     :read          the code reads (gen.codegen.candidate/read-forms), or,
                    with :repair-delimiters? (on), reads once its delimiters
                    are rebuilt from its indentation (parinfer's indent
                    mode: :repaired-delimiters); a leading ns form is
                    dropped (:dropped-ns), the prelude being the namespace
     :safe          it uses no forbidden name (candidate/forbidden)
     :evaluates     it evaluates under the prelude and defines model, a
                    generative function
     :simulates     model simulates on the task's arguments, its scores finite
     :accepts-data  generate with the task's observations: every observed
                    address visited (an unvisited one throws), the weight finite
     :laws          gen.laws's laws (laws/all) on the task's arguments, the
                    model's return value dropped (comb/dimap: a closure is
                    never = another). Advisory: a failure is a warning, the
                    grading goes on, until gen.laws' close? scales with
                    scores (correct models with scores in the thousands fail
                    it by 1e-11; genclj-ihc4's notes)
     :explains      its log marginal likelihood of the data, by importance
                    sampling from the prior, against the reference's (the
                    task's answer, graded alike): the gap is reported, and
                    falling more than the task's tolerance below is a
                    warning, or a failure with :gate-explains? (off by
                    default: a marginal likelihood also falls with wider
                    priors, ~0.7 nats a well-measured parameter whose range
                    doubles, so the cutoff is the owner's to set);
                    when either estimate spreads more than max-spread nats
                    across its repeats, the gap is reported with a warning,
                    not compared (the changepoint task's prior is too far from
                    its posterior: data-tempered SMC is the remedy)

   (grade task code) → {:task :passed (the stages passed, of 7) :stages
   [{:stage :ok :ms :error :warnings} …] :warnings (the laws that failed)
   :log-ml {:candidate :reference :gap} :tainted? (a stage timed out and its
   thread may still run: start another JVM)}.

   Every draw is from a fixed key, so a grading repeats. The log marginal
   likelihood is the median of :repeats estimates, each from :particles
   particles (their spread is in :estimates), batched when the model runs
   under the batch tracer and one by one when not, the same estimator
   either way: a model's choices are drawn from its prior, so a model whose
   posterior is far from its prior gets a noisy estimate; the reference
   pays the same.

   Run the grader in a JVM of its own, never the LLM's: a candidate is
   evaluated in this JVM (gen.codegen.candidate), so one that loops is
   abandoned, not stopped, and one that exhausts memory ends the JVM."
  (:require [clojure.string :as str]
            [gen.array :as array]
            [gen.codegen.candidate :as candidate]
            [gen.codegen.explain :as explain]
            [gen.codegen.tasks :as tasks]
            [gen.combinators :as comb]
            [gen.gfi :as gfi]
            [gen.inference.importance :as is]
            [gen.inference.tempering :as tempering]
            [gen.laws :as laws]
            [gen.random :as rng]))

(def defaults
  {:timeout-ms 300000 :particles 20000 :repeats 3 :law-tests 10 :seed 1 :gate-explains? false
   :repair-delimiters? true :device nil :estimator :checked :tempered-particles 200})

(def max-spread
  "Nats: estimates of a log marginal likelihood that spread more than this
   (the largest less the smallest) are too unreliable to compare."
  2.0)

(defn- host [x] (double (if (array/backend-array? x) (array/item x) x)))

(defn- finite? [x] (Double/isFinite (host x)))

(defn- message
  "e's message and its causes', joined: a compiler error's own message
   names only where it happened (Syntax error compiling at (5:25).), its
   cause what (No such var: d/point), and the repair loop sends both."
  [^Throwable e]
  (let [m (->> (take-while some? (iterate ex-cause e))
               (map #(or (ex-message %) (str (class %))))
               distinct
               (str/join " Caused by: "))]
    (subs m 0 (min 500 (count m)))))

(defn- median [xs] (nth (sort xs) (quot (count xs) 2)))

(defn- tempered-log-ml
  "The log marginal likelihood by tempered SMC (gen.inference.tempering):
   :repeats runs of :tempered-particles particles."
  [model task {:keys [repeats seed tempered-particles]}]
  (let [estimates (mapv (fn [r]
                          (:log-ml-estimate (tempering/smc model (:args task) (:observations task)
                                                           {:particles tempered-particles
                                                            :key (rng/fold-in (rng/key seed) (+ 100 r))})))
                        (range repeats))]
    {:median (median estimates) :estimates estimates :spread (- (apply max estimates) (apply min estimates))
     :path :tempered :particles tempered-particles}))

(declare importance-log-ml)

(defn log-ml
  "The log marginal likelihood of task's data under model, by :estimator:
   :importance (importance-log-ml), :tempered (tempered-log-ml), :auto
   (importance sampling, then tempered SMC when its estimates spread more
   than max-spread), or :checked (both: tempered SMC's when its estimates
   agree, else importance sampling's when its do, else tempered SMC's,
   unreliable; :tempered and :importance their medians, :disagree? when
   both are reliable and differ by more than max-spread: importance
   sampling's repeats can agree and be wrong alike, as on the outliers
   task, 4 nats low)."
  [model task {:keys [estimator] :or {estimator :checked} :as opts}]
  (case estimator
    :importance (importance-log-ml model task opts)
    :tempered (tempered-log-ml model task opts)
    :auto (let [r (importance-log-ml model task opts)]
            (if (> (:spread r) max-spread) (tempered-log-ml model task opts) r))
    :checked (let [t (tempered-log-ml model task opts)
                   i (importance-log-ml model task opts)
                   reliable? #(<= (:spread %) max-spread)
                   ;; tempered SMC when it is reliable; else importance sampling
                   ;; when it is (a discrete switch between structures, which
                   ;; single-site moves cross rarely); else tempered, unreliable
                   chosen (cond (reliable? t) t (reliable? i) i :else t)]
               (assoc chosen :tempered (:median t) :importance (:median i)
                      :disagree? (and (reliable? i) (reliable? t)
                                      (> (abs (- (:median t) (:median i))) max-spread))))))

(defn importance-log-ml
  "The log marginal likelihood of task's data under model: {:median
   :estimates :spread :path :particles}, :repeats importance-sampling
   estimates of :particles particles each (the task's :particles, if it
   has them), from keys fold-in'd from :seed. :path is how they ran: with
   :device (:jvm, say), as one compiled program over every core when the
   model stages (:device; gen.inference.device, ext/device on the
   classpath), else batched (:batched), else one particle at a time
   (:unbatched). :spread is the largest estimate less the smallest.
   :device is off by default: measured (2026-10-31), the compiled path is
   slower than the batched one here (sine's 200000 particles 2.0 s against
   0.96 s, its program cached), as it rebuilds a whole trace for every
   particle where a log marginal likelihood needs only the weights."
  [model task {:keys [particles repeats seed device]}]
  (let [n (or (:particles task) particles)
        run (fn [opts r]
              (is/importance-sampling model (:args task) (:observations task) n
                                      (assoc opts :key (rng/fold-in (rng/key seed) r))))
        first-device (when device
                       (let [r (run {:device device} 0)]
                         (when (:device r) r)))
        first-batched (when-not first-device (try (run {:batched? true} 0) (catch Exception _ nil)))
        [path opts first] (cond first-device [:device {:device device} first-device]
                                first-batched [:batched {:batched? true} first-batched]
                                :else [:unbatched {} (run {} 0)])
        estimates (into [(host (:log-ml-estimate first))]
                        (map #(host (:log-ml-estimate (run opts %))))
                        (range 1 repeats))]
    {:median (median estimates) :estimates estimates :spread (- (apply max estimates) (apply min estimates))
     :path path :particles n}))

(declare grade)

(def ^:private references (atom {}))

(defn reference-log-ml
  "The log marginal likelihood of task's data under its answer (cached by
   task and options)."
  [task opts]
  (let [k [(:id task) (select-keys opts [:particles :repeats :seed :device :estimator :tempered-particles])]]
    (or (get @references k)
        ;; the answer graded as code: never a candidate's model (grade-model's ::model)
        (let [g (grade task (:answer task) (-> opts (dissoc ::model) (assoc ::reference? true)))
              v (or (get-in g [:log-ml :candidate])
                    (throw (ex-info (str "the answer of " (:id task) " does not grade: " (pr-str (:stages g))) g)))]
          (swap! references assoc k v)
          v))))

(defn- stage
  "Runs check (a function of no arguments returning a map merged into the
   stage, or throwing) as stage name within timeout-ms; an error explained
   (gen.codegen.explain) in the namespace ns, when there is one."
  [name timeout-ms check ns]
  (let [t0 (System/nanoTime)
        r (try (merge {:ok true} (candidate/within timeout-ms check))
               (catch Throwable e
                 {:ok false :error (let [m (if (::explained (ex-data e)) (ex-message e) (explain/explain e (some-> ns find-ns)))]
                                     (subs m 0 (min 1600 (count m))))
                  :timeout? (boolean (::candidate/timeout (ex-data e)))}))]
    (assoc r :stage name :ms (/ (- (System/nanoTime) t0) 1e6))))

(defn grade-model
  "Grades model, a generative function already evaluated (a REPL session's,
   gen.codegen.repl), against task: grade's stages from :simulates on, the
   code's three passed as given (:given)."
  ([task model] (grade-model task model nil))
  ([task model opts] (grade task nil (assoc opts ::model model))))

(defn grade
  "Grades code (a reply's code, a string) against task; opts override
   defaults. See the namespace's docstring."
  ([task code] (grade task code nil))
  ([task code opts]
   (let [{:keys [timeout-ms law-tests seed] :as opts} (merge defaults opts)
         {:keys [args observations]} task
         state (volatile! {})
         stages [[:read #(let [[forms repaired?] (try [(candidate/read-forms code) false]
                                                      (catch Exception e
                                                        (if (:repair-delimiters? opts)
                                                          (let [fixed (try (candidate/read-forms (candidate/repair-delimiters code))
                                                                           (catch Exception _ (throw e)))]
                                                            [fixed true])
                                                          (throw e))))
                               ns? (and (seq? (first forms)) (= 'ns (ffirst forms)))]
                           ;; a leading ns form, the prelude copied, is dropped: the prelude is the namespace
                           (vswap! state assoc :forms (if ns? (vec (rest forms)) forms))
                           (cond-> {} ns? (assoc :dropped-ns true) repaired? (assoc :repaired-delimiters true)))]
                 [:safe #(let [f (candidate/forbidden (:forms @state))]
                           (when (seq f) (throw (ex-info (str "forbidden names: " (pr-str f)) {})))
                           {})]
                 [:evaluates #(let [[n model] (try (candidate/evaluate (:forms @state))
                                                   (catch Throwable e
                                                     ;; explained here, with clj-kondo's findings on the whole code
                                                     (throw (ex-info (str (explain/explain e nil)
                                                                          (when-let [l (explain/lint code)] (str "\nclj-kondo on this code:\n" l)))
                                                                     {::explained true}))))]
                                (vswap! state assoc :ns n :model model)
                                (when-not (gfi/generative-function? model)
                                  (throw (ex-info (str "model is not a generative function: " (class model)) {})))
                                {})]
                 [:simulates #(let [scores (mapv (fn [k] (:score (gfi/simulate (:model @state) args (rng/key k)))) (range 3))]
                                (when-not (every? finite? scores)
                                  (throw (ex-info (str "simulate's scores are not finite: " (mapv host scores)) {})))
                                {})]
                 [:accepts-data #(let [{:keys [weight]} (gfi/generate (:model @state) args observations (rng/key seed))]
                                   (when-not (finite? weight)
                                     (throw (ex-info (str "generate's weight is not finite: " (host weight)) {})))
                                   {})]
                 [:laws #(let [dropped (comb/dimap (:model @state) vector (constantly nil))
                               failures (laws/verify [[:candidate dropped args [args]]]
                                                     {:num-tests law-tests :seed seed})]
                           (if (seq failures) {:warnings (vec (keys failures))} {}))]
                 [:explains #(let [c (log-ml (:model @state) task opts)]
                               (if (::reference? opts)
                                 (do (vswap! state assoc :log-ml {:candidate c}) {})
                                 (let [r (reference-log-ml task opts)
                                       gap (- (:median c) (:median r))
                                       reliable? (<= (max (:spread c) (:spread r)) max-spread)]
                                   (vswap! state assoc :log-ml {:candidate c :reference r :gap gap :reliable? reliable?})
                                   (cond (not reliable?)
                                         {:warnings [(format "the log marginal likelihoods (gap %.1f nats) spread %.1f and %.1f nats: not compared"
                                                             gap (:spread c) (:spread r))]}
                                         (< gap (- (tasks/tolerance task)))
                                         (let [m (format "explains the data %.1f nats worse than the reference" (- gap))]
                                           (if (:gate-explains? opts) (throw (ex-info m {})) {:warnings [m]}))
                                         :else {}))))]]
         ;; a model given as a value (grade-model) passes the code's stages as given
         given (::model opts)
         _ (when given (vswap! state assoc :model given))
         results (reduce (fn [done [name check]]
                           (let [r (if (and given (#{:read :safe :evaluates} name))
                                     {:stage name :ok true :ms 0.0 :given true}
                                     (stage name timeout-ms check (:ns @state)))]
                             (if (:ok r) (conj done r) (reduced (conj done r)))))
                         [] stages)]
     (when-let [n (:ns @state)] (candidate/discard! n))
     {:task (:id task)
      :passed (count (filter :ok results))
      :stages results
      :warnings (vec (mapcat :warnings results))
      :log-ml (:log-ml @state)
      :tainted? (boolean (some :timeout? results))})))
