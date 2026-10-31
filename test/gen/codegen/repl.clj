(ns gen.codegen.repl
  "REPL sessions for the codegen agent (genclj-t1xh): a namespace that
   persists across an LLM's turns, under the bake-off's prelude
   (gen.codegen.candidate), holding args (the task's arguments) and
   observations (its data), and the tools the LLM calls on it. Sessions
   live in the grader's process (gen.codegen.server), never the LLM's JVM.

     (open! task-id)              → {:text :session}, the session id
     (eval! id code)              → the last form's value, or the error
     (simulate! id name args)     → a trace of the generative function name:
                                    its choices and score
     (check! id name)             → name run on the data: does it take it, a
                                    rough log marginal likelihood, the best of
                                    1000 particles' latent choices and its
                                    worst-explained observations
     (evidence! id name)          → the log marginal likelihood of the data
                                    (the grader's estimator)
     (submit! id name)            → name graded as the task's model
                                    (gen.codegen.grade/grade-model) with the
                                    session's program, and the session closed
     (close! id)

   Each returns {:text …}, what the LLM is shown, and the tools' results as
   data where they have them. Code is read with delimiter repair, checked
   for forbidden names, and kept when it evaluates: the session's program
   is its successful forms in order."
  (:require [clojure.string :as str]
            [gen.array :as array]
            [gen.choicemap :as cm]
            [gen.codegen.candidate :as candidate]
            [gen.codegen.grade :as grade]
            [gen.codegen.tasks :as tasks]
            [gen.gfi :as gfi]
            [gen.inference.importance :as is]
            [gen.random :as rng]
            [gen.selection :as sel]))

(defonce ^:private sessions (atom {}))

(defn- session [id]
  (or (@sessions id) (throw (ex-info (str "no session " id) {}))))

(defn- message [^Throwable e]
  (let [m (->> (take-while some? (iterate ex-cause e))
               (map #(or (ex-message %) (str (class %))))
               distinct
               (str/join " Caused by: "))]
    (subs m 0 (min 600 (count m)))))

(defn- clip [s n] (if (> (count s) n) (str (subs s 0 n) "…") s))

(defn- show
  "A value as the LLM sees it: numbers to 4 significant digits, arrays as
   their values (or shape when large), the rest printed and cut short."
  [v]
  (cond (instance? Double v) (let [d (double v)] (if (Double/isFinite d) (format "%.4g" d) (str d)))
        (number? v) (str v)
        (array/backend-array? v) (let [x (array/->clj v)]
                                   (if (and (sequential? x) (> (count (flatten [x])) 12))
                                     (str "<array of shape " (vec (array/shape v)) ">")
                                     (clip (pr-str x) 200)))
        (fn? v) "<function>"
        (gfi/generative-function? v) "<generative function>"
        :else (clip (pr-str v) 300)))

(defn open!
  "A new session for task-id."
  [task-id]
  (let [task (or (tasks/by-id task-id) (throw (ex-info (str "no task " task-id) {})))
        n (gensym "gen.codegen.session-")
        id (str n)]
    (binding [*ns* *ns*]
      (eval (list 'ns n candidate/prelude candidate/requires)))
    (intern n 'args (:args task))
    (intern n 'observations (:observations task))
    (swap! sessions assoc id {:ns n :task task :program [] :draws 0})
    {:text (str "Session open. args = " (:args-text task) "; observations holds the data.") :session id}))

(defn close! [id]
  (when-let [{n :ns} (@sessions id)]
    (remove-ns n)
    (swap! sessions dissoc id))
  {:text "closed"})

(defn- in-session [id f]
  (binding [*ns* (the-ns (:ns (session id)))] (f)))

(defn eval!
  "code's forms evaluated in the session, in order: the last one's value,
   with what they printed, or the first error (the forms before it stay
   evaluated)."
  [id code]
  (let [{:keys [ns]} (session id)
        [forms repaired?] (try [(candidate/read-forms code) false]
                               (catch Exception e
                                 (try [(candidate/read-forms (candidate/repair-delimiters code)) true]
                                      (catch Exception _ (throw e)))))
        forbidden (candidate/forbidden forms)]
    (when (seq forbidden)
      (throw (ex-info (str "forbidden names: " (pr-str forbidden)) {})))
    (let [out (java.io.StringWriter.)
          note (when repaired? "(The delimiters did not balance; they were rebuilt from the indentation.)\n")]
      (try
        (let [v (binding [*ns* (the-ns ns) *out* out]
                  (reduce (fn [_ f]
                            (let [v (eval f)]
                              (swap! sessions update-in [id :program] conj (pr-str f))
                              v))
                          nil forms))]
          {:text (str note (when (seq (str out)) (str "printed: " (clip (str out) 400) "\n")) "=> " (show v))})
        (catch Throwable e
          {:text (str note (when (seq (str out)) (str "printed: " (clip (str out) 400) "\n")) "error: " (message e))})))))

(defn- resolve-gf [id name]
  (let [v (in-session id #(ns-resolve *ns* (symbol name)))]
    (when-not v (throw (ex-info (str "no var " name " in the session: define it with eval") {})))
    (when-not (gfi/generative-function? @v)
      (throw (ex-info (str name " is not a generative function") {})))
    @v))

(defn- next-key [id]
  (let [k (:draws (session id))]
    (swap! sessions update-in [id :draws] inc)
    (rng/fold-in (rng/key 4242) k)))

(defn simulate!
  "A trace of the generative function name on the arguments args-code
   evaluates to in the session (args by default): its score, its choices
   (the first 40) and its return value."
  [id name args-code]
  (let [gf (resolve-gf id name)
        args (in-session id #(eval (read-string (if (str/blank? args-code) "args" args-code))))
        tr (gfi/simulate gf args (next-key id))
        leaves (cm/leaves (:choices tr))]
    {:text (str "score " (show (:score tr)) "; " (count leaves) " choices:\n"
                (str/join "\n" (map (fn [[p v]] (str (pr-str p) " " (show v))) (take 40 leaves)))
                (when (> (count leaves) 40) (str "\n… " (- (count leaves) 40) " more"))
                "\nreturn value: " (show (:retval tr)))}))

(defn check!
  "name run with the task's data as observations: importance sampling of
   1000 particles from its prior, the best particle's latent choices and
   its observations' log-probabilities, worst first; with ml?, the rough
   log marginal likelihood too."
  ([id name] (check! id name true))
  ([id name ml?]
   (let [gf (resolve-gf id name)
         {:keys [args observations]} (:task (session id))
         r (is/importance-sampling gf args observations 1000 {:key (next-key id)})
         ws (:log-weights r)
         best (nth (:traces r) (first (apply max-key second (map-indexed vector ws))))
         obs (set (map first (cm/leaves observations)))
         latents (remove (comp obs first) (cm/leaves (:choices best)))
         scored (sort-by second (map (fn [p] [p (double (array/item (gfi/project best (sel/from-paths [p]))))]) obs))]
     {:text (str "It takes the data." (when ml? (str " Rough log marginal likelihood (1000 particles): " (show (:log-ml-estimate r)) "."))
                 "\nThe best particle's latent choices:\n"
                 (str/join "\n" (map (fn [[p v]] (str (pr-str p) " " (show v))) (take 20 latents)))
                 (when (> (count latents) 20) (str "\n… " (- (count latents) 20) " more"))
                 "\nIts worst-explained observations (log-probability, observed value):\n"
                 (str/join "\n" (map (fn [[p s]] (str (pr-str p) " " (show s) " (observed " (show (cm/get-choice observations p)) ")"))
                                     (take 5 scored))))})))

(defn evidence!
  "The log marginal likelihood of the task's data under name, by the
   grader's estimator."
  [id name]
  (let [gf (resolve-gf id name)
        {:keys [median spread]} (grade/log-ml gf (:task (session id)) grade/defaults)]
    {:text (str "log marginal likelihood " (show median) " (estimates spread " (show spread) " nats)")}))

(defn submit!
  "name graded as the task's model, the session's program kept, and the
   session closed: {:text :grading :program}."
  [id name]
  (let [gf (resolve-gf id name)
        {:keys [task program]} (session id)
        g (grade/grade-model task gf)
        program (str (str/join "\n\n" program) "\n\n(def model " name ")\n")]
    (close! id)
    {:text "submitted" :grading g :program program}))
