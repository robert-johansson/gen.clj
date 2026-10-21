(ns gen.genjl.static-dsl-test
  "Gen.jl 0.4.8's test/dsl/static_dsl.jl: the static modeling language,
   gen ^:static. Its graph is gen.dynamic.static/ir: the arguments' names,
   a node per binding ({:kind :trace, :splice or :compute, :addr, :names,
   :inputs, :form}) and the names the return expression reads. By the
   owner's decision (G15) Gen.jl's internal shape is not ported: one node
   per argument expression (so a choice's or call's inputs), argument,
   constant and parameter nodes, node identity, and the distribution or
   callee object; nor are types and grad flags. The assertions about
   names, addresses, kinds, counts and a computation's inputs port.

   Gen.jl rewrites a computed address into a call_at or choice_at node;
   gen.clj keeps it a node at its constant prefix, its further keys
   computed, and hoists a call nested where it is evaluated once (the
   owner's decision on G2, DESIGN.md §9)."
  (:refer-clojure :exclude [+ *])
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array :refer [+ *]]
            [gen.choicemap :as cm]
            [gen.combinators :as comb]
            [gen.dist :as d]
            [gen.dynamic :refer [gen]]
            [gen.dynamic.static :as static]
            [gen.genjl.support :as support :refer [approx?]]
            [gen.gfi :as gfi]
            [gen.random :as rng]
            [gen.test-support :as ts :refer [close?]]))

(defrecord Params [prob-outlier slope intercept inlier-std outlier-std])

(def datum
  "static_dsl.jl:15"
  (gen ^:static [x params]
    (let [is-outlier (trace :z (d/bernoulli (:prob-outlier params)))
          std (if is-outlier (:outlier-std params) (:inlier-std params))
          y (trace :y (d/normal (+ (* x (:slope params)) (:intercept params)) std))]
      y)))

(def data-fn (comb/map datum))

(def model
  "static_dsl.jl:27"
  (gen ^:static [xs]
    (let [n (count xs)
          inlier-std (trace :inlier-std (d/gamma 1 1))
          outlier-std (trace :outlier-std (d/gamma 1 1))
          slope (trace :slope (d/normal 0 2))
          intercept (trace :intercept (d/normal 0 2))
          params (->Params 0.5 slope intercept inlier-std outlier-std)
          ys (splice :data data-fn xs (vec (repeat n params)))]
      ys)))

(defn- of-kind
  "The graph's nodes of a kind: Gen.jl's choice_nodes (:trace) and
   call_nodes (:splice)."
  [ir kind]
  (filterv #(= kind (:kind %)) (:nodes ir)))

(defn- named
  "get_node_by_name: the one node that binds sym."
  [ir sym]
  (let [nodes (filter #(some #{sym} (:names %)) (:nodes ir))]
    (when-not (= 1 (count nodes))
      (throw (ex-info (str (count nodes) " nodes bind " sym) {:name sym :nodes nodes})))
    (first nodes)))

(defn- returned
  "ir.return_node: the node whose value the return expression is."
  [ir]
  (named ir (first (:inputs (:return ir)))))

;; static_dsl.jl:118
(deftest check-ir
  (testing "datum" ; static_dsl.jl:125
    (let [ir (static/ir datum)
          [x params :as args] (:fixed (:args ir))
          [is-outlier y :as choices] (of-kind ir :trace)
          std (named ir 'std)]
      (is (= 2 (count args)))
      (is (= '[x] x))
      (is (= '[params] params))
      (is (= 2 (count choices)))
      (is (= 0 (count (of-kind ir :splice))))
      (is (= :z (:addr is-outlier)))
      (is (= :compute (:kind std)))
      (is (= '[std] (:names std)))
      (is (= 2 (count (:inputs std))))
      (is (= '#{is-outlier params} (set (:inputs std))))
      (is (= :y (:addr y)))
      (is (= y (returned ir)))))
  (testing "model" ; static_dsl.jl:188
    (let [ir (static/ir model)
          args (:fixed (:args ir))
          [inlier-std outlier-std slope intercept :as choices] (of-kind ir :trace)
          [ys :as calls] (of-kind ir :splice)
          params (named ir 'params)
          n (named ir 'n)]
      (is (= 1 (count args)))
      (is (= '[xs] (first args)))
      (is (= 4 (count choices)))
      (is (= 1 (count calls)))
      (is (= :inlier-std (:addr inlier-std)))
      (is (= :outlier-std (:addr outlier-std)))
      (is (= :slope (:addr slope)))
      (is (= :intercept (:addr intercept)))
      (is (= :data (:addr ys)))
      (is (= :compute (:kind params)))
      (is (= '[params] (:names params)))
      (is (= 4 (count (:inputs params))))
      (doseq [s '[slope intercept inlier-std outlier-std]]
        (is (some #{s} (:inputs params)) (str s " in params.inputs")))
      (is (= :compute (:kind n)))
      (is (= '[n] (:names n)))
      (is (= 1 (count (:inputs n))))
      (is (= '[xs] (:inputs n)))
      (is (= ys (returned ir))))))

;; static_dsl.jl:266: a computed address's node sits at its first key
(deftest at-choice
  (doseq [gf [(gen ^:static [i] (let [ret (trace [:x i] (d/bernoulli 0.5))] ret))
              (gen ^:static [i] (let [ret (trace [:x i :y] (d/bernoulli 0.5))] ret))]]
    (is (= :x (:addr (named (static/ir gf) 'ret))))))

(def ^:private foo
  "static_dsl.jl:51"
  (gen [mu] (trace :y (d/normal mu 1))))

;; static_dsl.jl:299
(deftest at-call
  (doseq [gf [(gen ^:static [i] (let [mu 1.123 ret (splice [:x i] foo mu)] ret))
              (gen ^:static [i] (let [mu 1.123 ret (splice [:x i :y] foo mu)] ret))]]
    (is (= :x (:addr (named (static/ir gf) 'ret))))))

(def ^:private circle
  "static_dsl.jl:412: a vector of two traces."
  (gen ^:static [r t std]
    (let [pt [(trace :x (d/normal (* r (array/cos t)) std)) (trace :y (d/normal (* r (array/sin t)) std))]]
      pt)))

(def ^:private beta-bernoulli
  "static_dsl.jl:417: a trace in a distribution's argument."
  (gen ^:static [] (let [] (trace :x (d/bernoulli (trace :z (d/beta 1 1)))))))

;; static_dsl.jl:410
(deftest trace-within-expressions
  (is (thrown? clojure.lang.ExceptionInfo (circle 2 (/ Math/PI 2) 0))
      "D5: a normal's std 0 is an error, where Gen.jl's circle returns the means")
  (is (approx? (+ (support/logpdf d/normal 1.0 1 1) (support/logpdf d/normal 1.0 0 1))
               (:weight (gfi/generate circle [1 0 1] (cm/choicemap :x 1.0 :y 1.0) (rng/key 6)))))
  (doseq [[z x] [[1.0 true] [0.0 false]]]
    (let [tr (:trace (gfi/generate beta-bernoulli [] (cm/choicemap :z z) (rng/key 7)))]
      (is (= x (ts/host (tr :x)) (ts/host (:retval tr))) (str "z " z))))
  (is (approx? (+ (support/logpdf d/beta 0.5 1 1) (support/logpdf d/bernoulli true 0.5))
               (:weight (gfi/generate beta-bernoulli [] (cm/choicemap :z 0.5 :x true) (rng/key 8))))))

;; static_dsl.jl:367
(deftest trainable-parameters
  (let [ir (static/ir (gen ^:static [] (let [theta (param :theta)] theta)))]
    (is (= (named ir 'theta) (returned ir)))))

;; static_dsl.jl:381
(deftest returning-a-trace-directly
  (let [r1 (returned (static/ir (gen ^:static [] (let [x (trace :foo (d/normal 0 1))] x))))
        r2 (returned (static/ir (gen ^:static [] (let [] (trace :foo (d/normal 0 1))))))
        dist (fn [node] (last (:form node)))]
    (is (= (:kind r1) (:kind r2)))
    (is (= (dist r1) (dist r2)) "the same distribution")
    (is (not= (:names r1) (:names r2)))
    (is (= (:addr r1) (:addr r2)))))

;; static_dsl.jl:435
(deftest repeated-assignments
  (let [foo (gen ^:static [x] (let [x (+ x 1)] x))
        bar (gen ^:static [x] (let [y (+ x 1) y (+ y 1)] (+ y 1)))]
    (is (= :compute (:kind (returned (static/ir foo)))))
    (is (close? 1 (foo 0)))
    (is (close? 3 (bar 0)))))

;; static_dsl.jl:466: the generates (the IR's assertions are G15's)
(deftest tuple-assignments
  (let [foo (gen ^:static [params]
              (let [[p [mu1 sigma1] [mu2 sigma2]] params
                    [mu sigma] (if (trace :x (d/bernoulli p)) [mu1 sigma1] [mu2 sigma2])]
                (trace :y (d/normal mu sigma))))
        params [0.5 [0 1] [1 2]]]
    (doseq [[x mu sigma] [[true 0 1] [false 1 2]]]
      (let [{tr :trace w :weight} (gfi/generate foo [params] (cm/choicemap :x x :y 1.0) (rng/key 9))]
        (is (close? 1.0 (:retval tr)))
        (is (approx? (+ (support/logpdf d/bernoulli x 0.5) (support/logpdf d/normal 1.0 mu sigma)) w))))))

;; static_dsl.jl:513
(deftest return-nothing
  (is (nil? ((gen ^:static [] (let [] nil))))))

(def ^:private bar
  "static_dsl.jl:525"
  (gen ^:static [r] (let [a (trace :a (d/normal 0 1))] r)))

;; static_dsl.jl:523
(deftest getindex-trace
  (let [foo (gen ^:static []
              (let [x (splice :x bar 1)
                    yz (splice [:y :z] bar 2)
                    u (trace :u (d/normal 0 1))
                    vw (trace [:v :w] (d/normal 0 1))
                    ret 7]
                ret))
        tr (:trace (gfi/generate foo [] (cm/choicemap :u 1.1 [:v :w] 1.2 [:x :a] 1.3 [:y :z :a] 1.4) (rng/key 1)))]
    (testing "random choices"
      (is (close? 1.1 (tr :u)))
      (is (close? 1.2 (tr [:v :w])))
      (is (close? 1.3 (tr [:x :a])))
      (is (close? 1.4 (tr [:y :z :a]))))
    (testing "auxiliary state: a call's return value at its address"
      (is (close? 1 (tr :x)))
      (is (close? 2 (tr [:y :z]))))
    (testing "the return value"
      (is (= 7 (:retval tr))))))

(def ^:private bar2
  "static_dsl.jl:598"
  (gen ^:static [] (let [b (trace :b (d/normal 0 1))] b)))

(def ^:private bar1
  "static_dsl.jl:603"
  (gen ^:static [] (let [a (splice :a bar2) x (trace :x (d/normal 0 1))] x)))

(def ^:private baz
  "static_dsl.jl:619, its argument a trace"
  (gen ^:static [tr] (let [x (trace :x (d/normal (tr :x) 0.1))] x)))

;; static_dsl.jl:596
(deftest static-gen-function-choicemaps
  (let [tr (gfi/simulate bar1 [] (rng/key 2))
        ch (:choices tr)]
    (is (support/has-value? ch :x))
    (is (not (support/has-value? ch :y)))
    ;; static_dsl.jl:613's KeyError: the value, a choice map here (D4's difference)
    (is (= (cm/->Value (cm/get-choice ch :x)) (cm/get-submap ch :x)))
    (is (support/has-value? (cm/get-submap ch :a) :b))
    (is (= cm/EMPTY (cm/get-submap ch :y)))
    (is (= {:values 1 :submaps 1} (support/shallow ch)))
    (let [{ch :choices rv :retval} (gfi/propose baz [tr] (rng/key 3))]
      (is (support/has-value? ch :x))
      (is (close? rv (cm/get-choice ch :x)))
      (is (close? (cm/get-choice ch :x) ((:trace (gfi/generate bar1 [] ch (rng/key 4))) :x))))))

(defmacro ^:private make-foo-sml
  "static_dsl.jl:85: a macro defining a static function."
  []
  `(def ~'foo-in-macro (gen ~(with-meta [] {:static true}) (let [a# 1] (+ a# 1)))))

(make-foo-sml)

;; static_dsl.jl:633
(deftest returning-a-static-function-from-a-macro
  (is (close? 2 (:retval (gfi/simulate foo-in-macro [] (rng/key 5))))))

(def ^:private external-var 1)

(def uses-external
  "static_dsl.jl:99"
  (gen ^:static [] (let [] (+ external-var 1))))

;; static_dsl.jl:640
(deftest global-variables
  (is (close? 2 (uses-external))))

;; Not ported
;; - G15, the IR's internal shape (the owner's decision): check IR's 25
;;   assertions on argument positions, per-expression nodes, node identity
;;   and distribution and callee objects; trainable parameters' node kind
;;   (376); returning a trace directly's constant nodes (401-402);
;;   repeated assignments' argument nodes and node identity (451, 453-454,
;;   460-461); tuple assignments' IR (479-488); at_choice's and at_call's
;;   call_at and choice_at nodes, their combinators and argument nodes (272,
;;   274-279, 285, 287-295; 305, 307-312, 318, 320-329), where gen.clj's node
;;   keeps its computed keys (the owner's decision on G2).
;; - Types and grad flags: check IR's 19, and 406.
;; - The (static, diffs) and (static, nojuliacache) options (333, 349), which
;;   have no counterpart (D10).
;; - Julia only: 'end' (501), @doc (560), one-line definitions (573),
;;   modules (584).
