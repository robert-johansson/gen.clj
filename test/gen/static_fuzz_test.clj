(ns gen.static-fuzz-test
  "The static language over random static models
   (gen.random-static-models): the GFI laws, and every operation equal to
   the same body's without ^:static (laws/static); compiled HMC and MALA
   steps equal to plain ones, and a static model's steps equal to its
   dynamic body's; the generator reaching every feature; and a failing
   spec shrunk and printed as data and as the form it became."
  (:require [clojure.edn :as edn]
            [clojure.set :as set]
            [clojure.string :as str]
            [clojure.test :refer [deftest is]]
            [clojure.test.check :as tc]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [gen.choicemap :as cm]
            [gen.diff :as diff]
            [gen.gfi :as gfi]
            [gen.gradients :as grad]
            [gen.inference.mcmc :as mcmc]
            [gen.laws :as laws]
            [gen.random :as rng]
            [gen.random-static-models :as rsm]
            [gen.selection :as sel]
            [gen.test-support :as ts]))

(laws/deflaws static-random rsm/gen-model 300 laws/static)

;; the general laws, slow, as each case evaluates a form (the static law
;; above caught every planted bug they did); and the static law at length,
;; for a bug whose case is rarer
(laws/deflaws ^:slow static-random rsm/gen-model 100)
(defspec ^:slow static-random-static-matches-dynamic-at-length 1000
  (laws/static-matches-dynamic (laws/cases rsm/gen-model)))

;; ---------------------------------------------------------------------------
;; Compiled kernel steps
;; ---------------------------------------------------------------------------

(def ^:private gen-kernel
  "A kernel and its settings."
  (gen/hash-map :kernel (gen/elements [:hmc :mala])
                :L (gen/elements [1 3])
                :eps (gen/elements [0.05 0.2])
                :tau (gen/elements [0.05 0.2])))

(def ^:private gen-move
  "A step of a chain: one of the chain's kernels (an index), its selection
   (every continuous choice, or some), and new arguments first (an index
   into the model's alternatives) or none."
  (gen/hash-map :kernel gen/nat
                :selection (gen/elements [:all :some])
                :args (gen/one-of [(gen/return nil) gen/nat])))

(defn- selection
  "The trace's continuous choices, all or those picked (all when none is),
   as a selection; nil when it has none."
  [tr which picks]
  (let [all (filterv #(grad/continuous? (cm/get-choice (:choices tr) %)) (cm/addresses (:choices tr)))
        some (keep (fn [[p pick?]] (when pick? p)) (map vector all (cycle picks)))]
    (when (seq all) (sel/from-paths (if (and (= :some which) (seq some)) some all)))))

(defn- move
  "The kernel step's result, or its error's message."
  [tr s {:keys [kernel L eps tau]} compile? key]
  (try (case kernel
         :hmc (mcmc/hmc tr s {:L L :eps eps :compile? compile?} key)
         :mala (mcmc/mala tr s tau {:compile? compile? :key key}))
       (catch clojure.lang.ExceptionInfo e {:error (ex-message e)})))

(defn- near?
  "Equal up to rounding: the runs compute the same values, in other orders
   (the score fn's sum against the handler's) or as one compiled graph."
  [x y]
  (if (ts/f32?) (laws/close? x y) (ts/close? x y {:rtol 1e-9 :atol 1e-9})))

(defn- leaves [x] (if (sequential? x) (mapcat leaves x) [x]))

(defn- same-move? [a b]
  (if (or (:error a) (:error b))
    (= (:error a) (:error b))
    (let [ca (:choices (:trace a)) cb (:choices (:trace b))
          values (fn [c p] (leaves (ts/host (cm/get-choice c p))))]
      (and (= (:accepted? a) (:accepted? b))
           (= (:diverged? a) (:diverged? b))
           (near? (:log-alpha a) (:log-alpha b))
           (near? (:score (:trace a)) (:score (:trace b)))
           (= (set (cm/addresses ca)) (set (cm/addresses cb)))
           (every? (fn [p] (every? true? (map near? (values ca p) (values cb p)))) (cm/addresses ca))))))

(defspec compiled-steps-move-as-plain-steps 100
  ;; A chain of HMC and MALA steps over a random continuous static model;
  ;; at each, from the same trace and key, the step compiled and plain, on
  ;; the static model and on its body without ^:static, make the same move.
  ;; A chain takes one or two kernels, and its selections and arguments
  ;; change along it, so that a cached compiled step reused with a stale
  ;; value (other arguments, other choices, another step size) moves
  ;; otherwise; on the static model the gradients are the score fn's, on
  ;; the dynamic one the handler's.
  (prop/for-all [[_ gf args alt-args] rsm/gen-continuous-model
                 kernels (gen/vector gen-kernel 1 2)
                 moves (gen/vector gen-move 2 8)
                 picks (gen/vector gen/boolean 8)
                 seed gen/nat]
    (let [dynamic (assoc gf :static nil)]
      (loop [tr (gfi/simulate gf args (rng/key seed)) moves moves i 0]
        (if-let [[m & more] (seq moves)]
          (let [[k-args k-move] (rng/split (rng/fold-in (rng/key seed) (inc i)))
                tr (if-let [j (:args m)]
                     (let [args' (nth alt-args (mod j (count alt-args)))]
                       (:trace (gfi/update tr args' (vec (repeat (count args') diff/unknown-change)) cm/EMPTY k-args)))
                     tr)
                s (selection tr (:selection m) picks)
                kernel (nth kernels (mod (:kernel m) (count kernels)))
                [plain & others] (when s (for [g [gf dynamic] compile? [false true]]
                                           (move (assoc tr :gen-fn g) s kernel compile? k-move)))]
            (if (every? #(same-move? plain %) others)
              (recur (or (:trace plain) tr) more (inc i))
              false))
          true)))))

;; ---------------------------------------------------------------------------
;; The generator
;; ---------------------------------------------------------------------------

(deftest the-generator-reaches-every-feature
  ;; over a fixed sample, the graphs have every node kind, computed
  ;; addresses, destructuring, hoisted calls, parameter reads and every diff
  ;; rule the ops write, and the forms every kind of callee: a fuzzer that
  ;; silently stopped making one would test less
  (let [models (for [size (range 0 200 2) seed (range 3)] (gen/generate rsm/gen-model size (+ seed (* 3 size))))
        nodes (mapcat (fn [[_ gf]] (:nodes (:static gf))) models)
        symbols (set (mapcat (fn [[label]] (filter symbol? (tree-seq coll? seq (:form label)))) models))
        hoisted? (fn [{[name] :names}] (and name (re-find #"^(trace|splice)__" (str name))))]
    (is (= {:kinds #{:trace :splice :compute} :computed-address true :destructured true :hoisted true :param true}
           {:kinds (set (map :kind nodes))
            :computed-address (boolean (some :keys nodes))
            :destructured (boolean (some (comp not :symbol?) nodes))
            :hoisted (boolean (some hoisted? nodes))
            :param (boolean (some :param? nodes))}))
    (is (= #{} (set/difference '#{clojure.core/conj clojure.core/assoc clojure.core/vec clojure.core/nth clojure.core/get
                                  clojure.core/count clojure.core/inc clojure.core/identity gen.array/+ gen.array/-}
                               (set (keep #(some-> (:rule %) symbol) nodes))))
        "the rules no binding reached")
    (is (= #{} (set/difference '#{k0 k1 kernel comb/map comb/unfold branchy normal-at d/product dyn/with-params &} symbols))
        "the callees and forms no model reached")))

(deftest a-failing-spec-shrinks-and-prints-as-data-and-code
  ;; a property that fails on any model with an Unfold: test.check shrinks
  ;; the spec to that statement alone, and the case prints the spec as data
  ;; and the form it became, ^:static included, which reads back as the same
  ;; code and evaluates to the same static model
  (let [unfold? (fn [form] (some #{'comb/unfold} (tree-seq coll? seq form)))
        result (tc/quick-check 100 (prop/for-all [[label] rsm/gen-model] (not (unfold? (:form label)))) :seed 11)
        [[label gf args]] (:smallest (:shrunk result))
        printed (pr-str label)
        read-back (edn/read-string printed)
        static-params (some #(when (and (seq? %) (= 'gen (first %))) (meta (second %))) (tree-seq coll? seq (:form read-back)))]
    (is (false? (:pass? result)))
    (is (= [:unfold] (map first (:statements (:spec label)))) "shrunk to the one statement")
    (is (str/includes? printed "(gen ^{:static true} ["))
    (is (= label read-back))
    (is (= {:static true} static-params))
    (let [again (rsm/evaluate (:form read-back))]
      (is (some? (:static again)))
      (is (= (:choices (gfi/simulate gf args (rng/key 3))) (:choices (gfi/simulate again args (rng/key 3))))))))
