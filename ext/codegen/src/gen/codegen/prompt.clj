(ns gen.codegen.prompt
  "The codegen bake-off's prompts (genclj-ihc4): what a model is told to
   write a gen.clj model from a task, and told after a failed grading.

     (system-message)            → the rules, the prelude, a cheat sheet of
                                   gen.clj and two worked examples
     (task-message task)         → the task: its description, interface,
                                   arguments and data
     (feedback-message grading)  → a failed grading as the next user turn
     (repl-system-message evidence?)  the REPL agent's instructions
     (repl-task-message task)    → its task
     (tools-for evidence?)       → its tools, as gen.llm.chat offers them

   gen.llm.chat renders them in Qwen's template.

   The worked examples (a Poisson regression and a hierarchical model
   through comb/map) are none of the tasks' answers; the combinators the
   tasks may use (unfold, scan) are described, not shown. examples holds
   them as code, so a test can check that they run."
  {:api :internal}
  (:require [clojure.string :as str]
            [gen.choicemap :as cm]
            [gen.codegen.candidate :as candidate]))

(def cheat-sheet
  "gen.clj in brief.

- (gen [args …] body) makes a generative function. Inside its body, (trace addr dist) draws a random choice
  from the distribution dist at the address addr and returns its value. An address is a keyword (:slope) or a
  vector path ([:y i], [:data i :y]). Each address is used at most once in a run.
- (splice addr gf args …) calls the generative function gf, its choices under addr (a keyword or a path);
  (splice gf args …) puts them at the top level. A generative function called as a plain function records
  nothing: always splice it.
- Observed data are random choices like any other: trace each observation at its address with the
  distribution it is drawn from. The data are supplied as constraints from outside; never write them into
  the model.
- All randomness comes from trace: never rand, rand-int or rand-nth.
- +, -, *, /, <, >, <=, >= and == are gen.array's: they take numbers and arrays alike. Other operations are
  array/…: exp, log, sqrt, square, pow, abs, sin, cos, tanh, sigmoid, sum, mean, logsumexp, matmul,
  (array/where c a b) to choose elementwise, (array/take xs i) for an element of an array, (array/array v) to
  make an array from a vector.
- Clojure's own forms work as usual around the draws: let, if, cond, doseq, dotimes, map-indexed, mapv,
  reduce, and helper functions defined with defn. if may test a traced boolean.
- Distributions, d/… (Gen.jl's names and parameters): (d/normal mu sd), (d/uniform low high),
  (d/gamma shape scale), (d/inv-gamma shape scale), (d/beta a b), (d/exponential rate),
  (d/laplace loc scale), (d/cauchy loc scale), (d/log-normal mu sigma), (d/student-t nu),
  (d/truncated-normal mu sd low high), (d/mvnormal mean cov), (d/dirichlet alpha); discrete:
  (d/bernoulli p) gives true or false, (d/categorical probs) gives an index 0 … k-1,
  (d/categorical-logits logits), (d/uniform-discrete low high), (d/poisson rate), (d/binom n p),
  (d/geometric p), (d/neg-binom r p).
- Combinators, comb/…; each makes a generative function from another:
  (comb/map gf) is called with vectors of arguments and runs gf on their i-th elements, for each i; element
  i's choices are at [i …].
  (comb/unfold step) is called with [n init & params] and runs (step t state & params) for t = 0 … n-1,
  each step returning the next state (the first step gets init); step t's choices are at [t …].
  (comb/scan step) is called with [init xs & params] and runs (step carry x & params) for each x of xs in
  order, each step returning [next-carry output]; step t's choices are at [t …].
  (comb/switch gf0 gf1 …) is called with [index & args] and runs the index'th generative function.
  Use a combinator's result as the model, or splice it.")

(def examples
  "The worked examples: [description, code]."
  [["Counts of events in eight periods, at exposures x, with a log rate linear in x. The model takes xs,
the exposures; count i is observed at [:count i]."
    "(def model
  (gen [xs]
    (let [a (trace :a (d/normal 0 2))
          b (trace :b (d/normal 0 1))]
      (doseq [[i x] (map-indexed vector xs)]
        (trace [:count i] (d/poisson (array/exp (+ a (* b x))))))
      [a b])))"]
   ["Test scores of students in several classes: each class's mean varies around the school's mean, and
the students' scores vary around their class's mean. The model takes sizes, each class's number of
students; student s of class c is observed at [:class c :students s :score]."
    "(def student
  (gen [class-mean sd]
    (trace :score (d/normal class-mean sd))))

(def model
  (gen [sizes]
    (let [school (trace :school (d/normal 50 20))
          spread (trace :spread (d/gamma 2 5))
          sd (trace :sd (d/gamma 2 5))]
      (doseq [[c n] (map-indexed vector sizes)]
        (let [class-mean (trace [:class c :mean] (d/normal school spread))]
          (splice [:class c :students] (comb/map student) (vec (repeat n class-mean)) (vec (repeat n sd)))))
      school)))"]])

(defn system-message []
  (str "You write probabilistic models in gen.clj, a Clojure port of Gen, from a description of a situation "
       "and its data. Reply with one ```clojure code block: helper definitions if you need them, then "
       "(def model …), the model as a generative function. Write no ns form: the code is evaluated under "
       "this one.\n\n```clojure\n" (candidate/prelude-text) "\n```\n\n"
       "Do not call the model, print, or add examples of its use: the model is run on the data from outside.\n\n"
       cheat-sheet
       "\n\nTwo worked examples.\n\n"
       (str/join "\n\n" (map-indexed (fn [i [d code]] (str "Example " (inc i) ". " d "\n\n```clojure\n" code "\n```"))
                                     examples))))

(defn- data-lines
  "The observations, one address and value a line, in address order."
  [observations]
  (str/join "\n" (map (fn [[p v]] (str (pr-str p) " " (pr-str v))) (cm/leaves observations))))

(defn task-message [task]
  (str (:description task) "\n\n" (:interface task)
       "\n\nThe model is called with the arguments " (:args-text task) "."
       "\n\nThe data, an address and its observed value a line:\n" (data-lines (:observations task))
       "\n\nWrite the model."))

(def ^:private stage-text
  {:read "The code does not read as Clojure"
   :safe "The code uses names a model may not use"
   :evaluates "Evaluating the code failed"
   :simulates "Running the model on the task's arguments failed"
   :accepts-data "Running the model with the data as observations failed"
   :laws "The model's operations are inconsistent"})

(defn feedback-message
  "A failed grading as the next user turn, nil when nothing failed."
  [grading]
  (when-let [{:keys [stage error]} (first (remove :ok (:stages grading)))]
    (str (when (:repaired-delimiters (first (:stages grading)))
           "Your code's parentheses and brackets did not balance; they were rebuilt from its indentation.\n\n")
         (if (= :explains stage)
           (str "The model runs, but " error " (its log marginal likelihood of the data). Reconsider its "
                "structure and its priors against the description.")
           (str (stage-text stage) ": " error))
         "\n\nFix the model and reply with the whole program again, in one ```clojure code block.")))

;; ---------------------------------------------------------------------------
;; The REPL agent's prompt (genclj-t1xh)
;; ---------------------------------------------------------------------------

(def tools
  "The REPL session's tools (gen.codegen.repl), as gen.llm.chat offers
   them."
  [{:name "eval" :description "Evaluate Clojure forms in the session's namespace (the prelude's names; args and observations are defined). Define one or two small things at a time. Returns the last form's value, or the error."
    :params [{:name "code" :description "One or a few Clojure forms."}]}
   {:name "simulate" :description "Run the generative function name on arguments and show a sampled trace: its score, its choices and its return value. Use it to see what a piece does."
    :params [{:name "name" :description "The name of a generative function defined in the session."}
             {:name "args" :description "A Clojure vector of its arguments, such as [] or [2.0]; the task's arguments when left out (args)." :required? false}]}
   {:name "check" :description "Run the generative function name on the task's arguments with the task's data as observations: whether it takes the data and, for the particle that explains the data best, its latent choices and its worst-explained observations."
    :params [{:name "name" :description "The name of the model, a generative function defined in the session."}]}
   {:name "evidence" :description "The log marginal likelihood of the task's data under the generative function name: higher explains the data better. Use it to compare variants."
    :params [{:name "name" :description "The name of the model."}]}
   {:name "submit" :description "Submit the generative function name as the model. This ends the session."
    :params [{:name "name" :description "The name of the model."}]}])

(defn repl-system-message
  "The REPL agent's instructions; without evidence? the evidence tool is
   not mentioned (nor offered: repl-head)."
  [evidence?]
  (str "You build a probabilistic model in gen.clj at a REPL, step by step, by calling the tools. Work as a careful "
       "developer does: define one small piece at a time with eval (the priors, one observation's likelihood, a helper), "
       "simulate it to see what it does, check the model against the data"
       (when evidence? ", compare variants with evidence")
       ", and submit the model when it explains the data: when check says no observation stands out, submit (refine "
       "only for a reason the description gives). Keep each eval to one or two definitions, and make one tool call a "
       "reply.\n\n"
       "The model is a generative function whose parameters are the task's arguments; in the session, args holds them "
       "(simulate and check use them) and observations holds the data. Trace each observation at its address. The "
       "session's namespace is this one:\n\n```clojure\n" (candidate/prelude-text) "\n```\n\n"
       cheat-sheet
       "\n\nTwo worked examples.\n\n"
       (str/join "\n\n" (map-indexed (fn [i [d code]] (str "Example " (inc i) ". " d "\n\n```clojure\n" code "\n```"))
                                     examples))))

(defn tools-for
  "The tools offered: all, or all but evidence. Without it, check gives no
   log marginal likelihood either (the grader's measure stays unseen, as
   in one shot and repair)."
  [evidence?]
  (if evidence? tools (filterv #(not= "evidence" (:name %)) tools)))

(defn repl-task-message
  "The REPL agent's task: the task's message, told to build the model with
   the tools."
  [task]
  (str/replace (task-message task) #"Write the model\.$" "Build the model with the tools, then submit it."))
