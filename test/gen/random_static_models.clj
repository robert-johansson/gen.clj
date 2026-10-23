(ns gen.random-static-models
  "Random static generative functions for property tests: a test.check
   generator of specs (data) and an interpreter that writes a spec as the
   form of a static gen, (gen ^:static [params] (let [bindings…] ret)), and
   evaluates it here. The laws compare each with the same body without the
   flag (gen.laws/static), so the score fn, the walk's skipping on update
   and regenerate, propagated diffs, hoisting, computed addresses,
   destructuring and parameters are checked against the dynamic handler over
   random bodies, as gen.random-models does for dynamic ones.

     (form spec)        the code, a static gen under its store, its callees
                        bound around it
     (evaluate form)    the generative function of a form, evaluated in this
                        namespace (a printed form may be pasted back)
     gen-model          model entries [label gf args alt-args] (gen.laws's),
                        label {:spec :form}: every family
     gen-continuous-model  the same with continuous families only, for
                        gradient-based kernels

   A spec:
     {:params [kind…]        the arguments: :real, :hinted (^double, given
                             an integer first), :vec (a nonempty vector of reals),
                             :index (0 to 2), :map ({:p :q}), :keys (a map
                             destructured), :pair (a vector destructured),
                             :rest last (& more, 0 to 2 reals)
      :args [value…]         the arguments
      :alt-args [[value…]…]  others, for update and regenerate
      :store {:w x} or nil   a parameter store (dyn/with-params)
      :callees [callee…]     static functions of one real, {:statements
                             :ret}, bound around the model as k0, k1…
      :statements [s…]       the let's bindings
      :ret r}                its return expression

   A statement names its inputs by picks, natural numbers counted back from
   the latest name in scope of the type it needs (types :real, :int (a
   discrete choice), :bool, :arr (a vector-valued choice), :vec, :vec0 (a
   vector that may be empty), :index (a host integer: an argument, a count),
   :map and :seq), so that any spec is well formed and shrinks to one. An
   input of a type no name in scope has is a constant (0.5, [0.5 -0.5], {:p
   0.5 :q 1.0}, 1), but a boolean, a discrete or vector-valued choice or
   the store, without which a computation gives way to the next in ops.
   The n-th statement's address is :bn.
     [:trace family x shadow]  a choice, bound to a fresh name or, given
                               shadow, to an earlier one of its type
     [:observe family x]       a choice bound to _
     [:at family x i]          at a computed address [:bn i]
     [:macro x]                a user macro at the head, expanding to a trace
     [:product x]              a product distribution's value, destructured
     [:compute op picks shadow]  a pure expression (ops)
     [:pair x y], [:bounds x]  a vector and a map, destructured
     [:hoist variant family x y]  a call nested in an expression, which the
                               analysis hoists into a node of its own
     [:call c x], [:call-at c x i]  a callee spliced, at :bn or [:bn i]
     [:call-hoisted c x]       a splice whose argument is a trace call
     [:map c v], [:unfold i x] a Map of a callee, an Unfold of step
     [:mapped op c v x]        a vector computed by op (:conj, :assoc, :vec
                               or :twice), and a Map over it
     [:branchy x]              a dynamic callee whose choices vary
   and the return expression is [:vector picks], [:name k], [:call family x],
   [:add x y], [:hoisted x] or [:splice c x].

   genclj-f5fy (generated code for simulate, generate, update and
   regenerate) needs no other generator: its code runs behind the same gen
   ^:static forms, so (laws/deflaws prefix gen-model n laws/static) and the
   general laws over gen-model check it against the same body without the
   flag."
  (:refer-clojure :exclude [+ - *])
  (:require [clojure.test.check.generators :as tcg]
            [clojure.walk :as walk]
            [gen.array :as array :refer [+ - *]]
            [gen.combinators :as comb]
            [gen.dist :as d]
            [gen.dynamic :as dyn :refer [gen]]))

;; (test.check's generators are tcg here: gen is the macro the forms call)

;; ---------------------------------------------------------------------------
;; What the forms call
;; ---------------------------------------------------------------------------

(defn positive "A positive parameter from any real." [x] (+ 0.5 (array/softplus x)))

(defn probability "A probability from any real." [x] (+ 0.05 (* 0.9 (array/sigmoid x))))

(defn as-real "A discrete value as a real." [n] (* 1.0 n))

(defmacro normal-at
  "(trace addr (d/normal x 1.0)): a user macro at a binding's head."
  [addr x]
  `(~'trace ~addr (d/normal ~x 1.0)))

(def kernel
  "The callee of a spec without callees."
  (gen ^:static [x] (let [v (trace :v (d/normal x 1.0))] v)))

(def step
  "Unfold's kernel, an AR(1) step."
  (gen ^:static [_t s] (let [s2 (trace :s (d/normal (* 0.9 s) 1.0))] s2)))

(def branchy
  "A dynamic callee whose choices depend on a choice: m, or a draw around
   it."
  (gen [m] (if (trace :c (d/bernoulli 0.5)) (trace :x (d/normal m 1.0)) m)))

;; ---------------------------------------------------------------------------
;; Families, computations and their types
;; ---------------------------------------------------------------------------

(def ^:private families
  "A family's value type and its distribution's form, % its real input."
  {:normal [:real '(d/normal % 1.0)]
   :gamma [:real '(d/gamma (positive %) 1.0)]
   :beta [:real '(d/beta (positive %) 2.0)]
   :exponential [:real '(d/exponential (positive %))]
   :laplace [:real '(d/laplace % 0.7)]
   :uniform [:real '(d/uniform % (+ % (positive %)))]
   :vector [:arr '(d/normal (+ % (array/array [0.0 1.0 -1.0])) 1.0)]
   :poisson [:int '(d/poisson (positive %))]
   :categorical [:int '(d/categorical [0.2 0.5 0.3])]
   :bernoulli [:bool '(d/bernoulli (probability %))]})

(def ^:private continuous-families [:normal :gamma :beta :exponential :laplace :uniform :vector])

(def ^:private ops
  "Pure expressions, in order: the name, the value's type, the inputs'
   types (or classes of them), the form with %1… for the inputs. One call
   of a var on names is a binding with a gen.diff/propagate rule: conj,
   assoc, vec, nth, get, count, inc, +, - and identity here."
  [[:conj :vec [:vecs :real] '(conj %1 %2)]
   [:assoc :vec [:vec :real] '(assoc %1 0 %2)]
   [:vec :vec [:vec] '(vec %1)]
   [:twice :vec [:vec] '(mapv (fn [y] (* 2.0 y)) %1)]
   [:nth :real [:vec] '(nth %1 0)]
   [:get :real [:vecs :key] '(get %1 %2 0.0)]
   [:count :index [:coll] '(count %1)]
   [:inc :index [:index] '(inc %1)]
   [:index-sum :int [:index :index] '(+ %1 %2)]
   [:map-assoc :map [:map :real] '(assoc %1 :p %2)]
   [:map-get :real [:map] '(get %1 :p 0.0)]
   [:keyword :real [:map] '(:q %1 0.0)]
   [:sum :real [:arr] '(array/sum %1)]
   [:total :real [:seq] '(reduce + 0.0 %1)]
   [:if :real [:bool :real :real] '(if %1 %2 %3)]
   [:as-real :real [:int] '(as-real %1)]
   [:add :real [:real :real] '(+ %1 %2)]
   [:sub :real [:real :real] '(- %1 %2)]
   [:scale :real [:real] '(* 0.5 %1)]
   [:sin :real [:real] '(array/sin %1)]
   [:alias :real [:real] '%1]
   [:identity :real [:real] '(identity %1)]
   [:param :real [:real] '(* %1 (param :w))]])

(def ^:private op-index (zipmap (map first ops) (range)))

(def ^:private discrete-ops #{:if :as-real})

(def ^:private classes
  "The types a class of inputs takes (a type takes itself)."
  {:vecs #{:vec :vec0}
   :key #{:index :int}
   :coll #{:vec :vec0 :map :seq}
   :any #{:real :int :bool :arr :vec :vec0 :index :map :seq}})

(defn- fill
  "template with its holes (%, %1, %2, %3) filled by xs."
  [template & xs]
  (walk/postwalk-replace (zipmap '[% %1 %2 %3] (cons (first xs) xs)) template))

(defn- dist [family x] (fill (second (families family)) x))

(defn- value-type [family] (first (families family)))

;; ---------------------------------------------------------------------------
;; The interpreter: a spec's form
;; ---------------------------------------------------------------------------

(defn- pick
  "The name k back from the latest in scope ([[name type]…], in binding
   order) whose type is in class (nil when there is none)."
  [scope class k]
  (let [types (classes class #{class})
        names (into [] (keep (fn [[name t]] (when (types t) name))) (rseq scope))]
    (when (seq names) (nth names (mod k (count names))))))

(defn- type-of [scope name] (some (fn [[n t]] (when (= n name) t)) scope))

(def ^:private stand-ins
  "The constant an input of a class is when no name in scope has it."
  {:real 0.5 :vec [0.5 -0.5] :vecs [0.5 -0.5] :coll [0.5 -0.5] :map {:p 0.5 :q 1.0} :index 1 :key 1})

(defn- input
  "The name k back of class in scope, or its stand-in (nil when it has
   none)."
  [scope class k]
  (or (pick scope class k) (stand-ins class)))

(defn- real [scope k] (input scope :real k))

(defn- address [n & suffix] (keyword (apply str "b" n suffix)))

(defn- local [n & suffix] (symbol (apply str "b" n suffix)))

(defn- callee [{:keys [callees]} c] (if (seq callees) (nth callees (mod c (count callees))) 'kernel))

(defmulti ^:private statement
  "{:pairs [[binding init]…] :bound [[name type]…]} of the n-th raw
   statement (ctx: :scope, :n, :callees, :store?)."
  (fn [_ctx raw] (first raw)))

(defmethod statement :trace [{:keys [scope n]} [_ family x shadow]]
  (let [t (value-type family)
        name (or (when shadow (pick scope t shadow)) (local n))]
    {:pairs [[name (list 'trace (address n) (dist family (real scope x)))]] :bound [[name t]]}))

(defmethod statement :observe [{:keys [scope n]} [_ family x]]
  {:pairs [['_ (list 'trace (address n) (dist family (real scope x)))]]})

(defmethod statement :at [{:keys [scope n]} [_ family x i]]
  {:pairs [[(local n) (list 'trace [(address n) (input scope :index i)] (dist family (real scope x)))]]
   :bound [[(local n) (value-type family)]]})

(defmethod statement :macro [{:keys [scope n]} [_ x]]
  {:pairs [[(local n) (list 'normal-at (address n) (real scope x))]] :bound [[(local n) :real]]})

(defmethod statement :product [{:keys [scope n]} [_ x]]
  (let [[a b] [(local n "x") (local n "y")]]
    {:pairs [[[a b] (list 'trace (address n) (list '(d/product d/normal d/gamma) [(real scope x) 1.0] [2.0 1.0]))]]
     :bound [[a :real] [b :real]]}))

(defmethod statement :compute [{:keys [scope n store?]} [_ op picks shadow]]
  (let [inputs (fn [[_ _ ins]] (map #(input scope %1 %2) ins (cycle picks)))
        usable? (fn [[name :as op]] (and (or store? (not= :param name)) (every? some? (inputs op))))
        ;; one without its inputs gives way to the next that has them
        [_ t _ template :as op] (first (filter usable? (take (count ops) (drop (op-index op) (cycle ops)))))
        name (or (when shadow (pick scope t shadow)) (local n))]
    {:pairs [[name (apply fill template (inputs op))]] :bound [[name t]]}))

(defmethod statement :pair [{:keys [scope n]} [_ x y]]
  (let [[a b] [(local n "x") (local n "y")]]
    {:pairs [[[a b] [(real scope x) (list '* 2.0 (real scope y))]]] :bound [[a :real] [b :real]]}))

(defmethod statement :bounds [{:keys [scope n]} [_ x]]
  (let [[lo hi] [(local n "lo") (local n "hi")]
        x (real scope x)]
    {:pairs [[{lo :lo hi :hi} {:lo (list '- x 1.0) :hi (list '+ x 1.0)}]] :bound [[lo :real] [hi :real]]}))

(defmethod statement :hoist [{:keys [scope n]} [_ variant family x y]]
  (let [x (real scope x) y (real scope y) b (local n) t (value-type family)]
    (case variant
      ;; a function call's argument
      :argument {:pairs [[b (list '+ x (list 'trace (address n) (list 'd/normal y 1.0)))]] :bound [[b :real]]}
      ;; a distribution's argument, in a trace node (a draw of family, but
      ;; a boolean's, as a mean)
      :distribution (let [family (if (= :bool t) :normal family)]
                      {:pairs [[b (list 'trace (address n) (list 'd/normal (list 'trace (address n "z") (dist family x)) 1.0))]]
                       :bound [[b (if (= :arr (value-type family)) :arr :real)]]})
      ;; a vector's elements, destructured
      :vector (let [[a c] [(local n "x") (local n "y")]]
                {:pairs [[[a c] [(list 'trace (address n "x") (dist family x)) (list 'trace (address n "y") (list 'd/normal y 1.0))]]]
                 :bound [[a t] [c :real]]})
      ;; an if's test
      :test {:pairs [[b (list 'if (list 'trace (address n) '(d/bernoulli 0.5)) x y)]] :bound [[b :real]]}
      ;; a map literal's value, the argument of a keyword's call
      :map-literal {:pairs [[b (list :v {:v (list 'trace (address n) (dist family x))})]] :bound [[b t]]})))

(defmethod statement :call [{:keys [scope n] :as ctx} [_ c x]]
  {:pairs [[(local n) (list 'splice (address n) (callee ctx c) (real scope x))]] :bound [[(local n) :real]]})

(defmethod statement :call-at [{:keys [scope n] :as ctx} [_ c x i]]
  {:pairs [[(local n) (list 'splice [(address n) (input scope :index i)] (callee ctx c) (real scope x))]]
   :bound [[(local n) :real]]})

(defmethod statement :call-hoisted [{:keys [scope n] :as ctx} [_ c x]]
  {:pairs [[(local n) (list 'splice (address n) (callee ctx c) (list 'trace (address n "z") (list 'd/normal (real scope x) 1.0)))]]
   :bound [[(local n) :real]]})

(defmethod statement :map [{:keys [scope n] :as ctx} [_ c v]]
  (let [v (input scope :vecs v)]
    {:pairs [[(local n) (list 'splice (address n) (list 'comb/map (callee ctx c)) v)]]
     :bound [[(local n) (if (symbol? v) (type-of scope v) :vec)]]}))

;; the rules exist to let a Map rerun only the elements that changed
(defmethod statement :mapped [{:keys [n] :as ctx} [_ op c v x]]
  (let [{[[b vector]] :pairs} (statement ctx [:compute op [v x] nil])
        m (local n "m")]
    {:pairs [[b vector] [m (list 'splice (address n) (list 'comb/map (callee ctx c)) b)]]
     :bound [[b :vec] [m :vec]]}))

(defmethod statement :unfold [{:keys [scope n]} [_ i x]]
  {:pairs [[(local n) (list 'splice (address n) '(comb/unfold step) (input scope :index i) (real scope x))]]
   :bound [[(local n) :vec0]]})

(defmethod statement :branchy [{:keys [scope n]} [_ x]]
  {:pairs [[(local n) (list 'splice (address n) 'branchy (real scope x))]] :bound [[(local n) :real]]})

(defn- return-form
  "The return expression's form (ctx: :scope, :callees, :ret-class)."
  [{:keys [scope ret-class] :as ctx} [kind & more]]
  (case kind
    :vector (vec (keep #(pick scope :any %) (first more)))
    :name (or (pick scope ret-class (first more)) 0.5)
    :call (let [[family x] more] (list 'trace :ret (dist family (real scope x))))
    :add (let [[x y] more] (list '+ (real scope x) (real scope y)))
    :hoisted (list '+ (real scope (first more)) '(trace :ret (d/normal 0.0 1.0)))
    :splice (let [[c x] more] (list 'splice :ret (callee ctx c) (real scope x)))))

(defn- signature
  "[the parameter vector, [[name type]…] it binds] of the argument kinds."
  [params]
  (reduce (fn [[pv scope] [i kind]]
            (let [named #(symbol (str % i))]
              (case kind
                :real [(conj pv (named "x")) (conj scope [(named "x") :real])]
                :hinted [(conj pv (with-meta (named "h") {:tag 'double})) (conj scope [(named "h") :real])]
                :vec [(conj pv (named "xs")) (conj scope [(named "xs") :vec])]
                :index [(conj pv (named "i")) (conj scope [(named "i") :index])]
                :map [(conj pv (named "m")) (conj scope [(named "m") :map])]
                :keys [(conj pv {(named "p") :p (named "q") :q}) (conj scope [(named "p") :real] [(named "q") :real])]
                :pair [(conj pv [(named "u") (named "v")]) (conj scope [(named "u") :real] [(named "v") :real])]
                :rest [(conj pv '& (named "more")) (conj scope [(named "more") :seq])])))
          [[] []]
          (map-indexed vector params)))

(defn- bind
  "scope with the names bound, each now holding its latest binding."
  [scope bound]
  (let [rebound (set (map first bound))]
    (into (filterv (comp not rebound first) scope) bound)))

(defn- static-gen
  "The form of a static gen of the argument kinds, statements and return
   expression."
  [params statements ret ctx]
  (let [[pv scope] (signature params)
        {:keys [pairs scope]} (reduce (fn [{:keys [pairs scope]} [n raw]]
                                        (let [s (statement (assoc ctx :scope scope :n n) raw)]
                                          {:pairs (into pairs (:pairs s)) :scope (bind scope (:bound s))}))
                                      {:pairs [] :scope scope}
                                      (map-indexed vector statements))]
    (list 'gen (with-meta pv {:static true})
          (list 'let (into [] cat pairs) (return-form (assoc ctx :scope scope) ret)))))

(defn form
  "The code of spec: a static gen, under its store, with its callees bound
   around it."
  [{:keys [params store callees statements ret]}]
  (let [ks (mapv #(symbol (str "k" %)) (range (count callees)))
        model (static-gen params statements ret {:callees ks :store? (some? store) :ret-class :any})
        model (if store (list 'dyn/with-params model store) model)]
    ;; without the templates' reader positions, which would print as code
    (walk/postwalk #(if (seq? %) (with-meta % nil) %)
                   (if (seq callees)
                     (list 'let (vec (mapcat (fn [k {:keys [statements ret]}]
                                               [k (static-gen [:real] statements ret {:store? (some? store) :ret-class :real})])
                                             ks callees))
                           model)
                     model))))

(defn evaluate
  "The generative function of a form, evaluated in this namespace."
  [form]
  (binding [*ns* (the-ns 'gen.random-static-models)] (eval form)))

(defn- code
  "form, printed with its metadata (gen's ^:static), through print-method's
   :type dispatch, so that a failing case shows the code it ran."
  [form]
  (vary-meta form assoc :type ::code))

(defmethod print-method ::code [form ^java.io.Writer w]
  (binding [*print-meta* true] (print-method (with-meta form nil) w)))

(defn- model
  "The gen.laws model entry of spec."
  [spec]
  (let [f (form spec)]
    [{:spec spec :form (code f)} (evaluate f) (:args spec) (:alt-args spec)]))

;; ---------------------------------------------------------------------------
;; The generator
;; ---------------------------------------------------------------------------

;; argument values: quarters from -2 to 2, short when printed
(def ^:private quarter (tcg/fmap #(/ % 4.0) (tcg/choose -8 8)))

(def ^:private kinds
  "Argument kinds: a generator of a value, and of another value given one
   (for update and regenerate)."
  {:real {:value quarter :other (fn [_] quarter)}
   :vec {:value (tcg/vector quarter 1 3)
         :other (fn [v] (tcg/one-of [(tcg/let [i (tcg/choose 0 (dec (count v))) x quarter] (assoc v i x))
                                     (tcg/fmap #(conj v %) quarter)
                                     (tcg/return (if (next v) (pop v) v))]))}
   ;; an integer, which a primitive hint coerces; its alternatives are
   ;; floats, whose argdiffs are unknown-change: an integer's int-diff meets
   ;; a known bug of gen.diff's + and - rules (gen.static-fuzz-test), and
   ;; they should be integers again once it is fixed
   :hinted {:value (tcg/choose -2 2) :other (fn [_] (tcg/fmap double (tcg/choose -2 2)))}
   :index {:value (tcg/choose 0 2) :other (fn [_] (tcg/choose 0 2))}
   :map {:value (tcg/hash-map :p quarter :q quarter)
         :other (fn [m] (tcg/one-of [(tcg/fmap #(assoc m :p %) quarter)
                                     (tcg/fmap #(assoc m :z %) quarter)
                                     (tcg/return (dissoc m :q))]))}
   :keys {:value (tcg/hash-map :p quarter :q quarter)
          :other (fn [m] (tcg/let [k (tcg/elements [:p :q]) x quarter] (assoc m k x)))}
   :pair {:value (tcg/vector quarter 2)
          :other (fn [v] (tcg/let [i (tcg/choose 0 1) x quarter] (assoc v i x)))}
   :rest {:value (tcg/vector quarter 0 2)
          :other (fn [v] (tcg/one-of [(tcg/return (if (seq v) (pop v) v)) (tcg/fmap #(conj v %) quarter) (tcg/vector quarter 0 2)]))}})

(defn- arguments
  "The argument vector of values, one per kind (the rest's spliced)."
  [params values]
  (if (= :rest (peek params)) (into (pop values) (peek values)) values))

(def ^:private gen-signature
  (tcg/let [fixed (tcg/vector (tcg/frequency (for [[w kind] [[3 :vec] [2 :real] [1 :hinted] [2 :index] [1 :map] [1 :keys] [1 :pair]]]
                                                [w (tcg/return kind)]))
                              0 3)
            rest? (tcg/frequency [[3 (tcg/return false)] [1 (tcg/return true)]])]
    (let [params (cond-> fixed rest? (conj :rest))]
      (tcg/let [values (apply tcg/tuple (map (comp :value kinds) params))
                ;; each argument kept (where shrinking goes) or changed
                alts (tcg/vector (apply tcg/tuple (map (fn [kind v] (tcg/one-of [(tcg/return v) ((:other (kinds kind)) v)]))
                                                       params values))
                                 1 2)]
        {:params params :args (arguments params values) :alt-args (mapv #(arguments params %) alts)}))))

(defn- gen-statement
  "A raw statement over families; with calls when calls? (a callee makes
   none), discrete-only kinds when discrete?."
  [{:keys [families calls? discrete?]}]
  (let [family (tcg/elements families)
        ;; the latest name of a type half the time, so that values flow
        ;; through chains of nodes
        k (tcg/one-of [(tcg/return 0) tcg/nat])
        shadow (tcg/frequency [[3 (tcg/return nil)] [1 tcg/nat]])
        ;; vectors (which Maps read) most often, then the other
        ;; computations of collections, indices and the store, most of them
        ;; a rule's, then those of reals
        op (tcg/frequency (for [[name t ins] ops :when (or discrete? (not (discrete-ops name)))]
                            [(cond (= :vec t) 4 (or (not-every? #{:real} (cons t ins)) (= :param name)) 2 :else 1)
                             (tcg/return name)]))
        variant (tcg/elements (cond-> [:argument :distribution :vector :map-literal] discrete? (conj :test)))
        tagged (fn [tag & gs] (apply tcg/tuple (tcg/return tag) gs))]
    (tcg/frequency
     (cond-> [[5 (tagged :trace family k shadow)]
              [1 (tagged :observe family k)]
              [1 (tagged :at family k k)]
              [1 (tagged :macro k)]
              [1 (tagged :product k)]
              [7 (tagged :compute op (tcg/vector k 3) shadow)]
              [1 (tagged :pair k k)]
              [1 (tagged :bounds k)]
              [2 (tagged :hoist variant family k k)]]
       calls? (into [[2 (tagged :call k k)]
                     [1 (tagged :call-at k k k)]
                     [1 (tagged :call-hoisted k k)]
                     [3 (tagged :map k k)]
                     [2 (tagged :mapped (tcg/elements [:conj :assoc :vec :twice]) k k k)]
                     [1 (tagged :unfold k k)]])
       (and calls? discrete?) (conj [1 (tagged :branchy k)])))))

(defn- gen-statements
  "1 to most raw statements, fewer at small sizes."
  [opts most]
  (tcg/sized (fn [size] (tcg/vector (gen-statement opts) 1 (inc (min (dec most) (quot size 12)))))))

(defn- gen-ret
  "A raw return expression; a callee's is a real."
  [{:keys [families]} callee?]
  (let [family (tcg/elements (filter #(= :real (value-type %)) families))
        k tcg/nat]
    (tcg/one-of (cond-> [(tcg/tuple (tcg/return :name) k)
                         (tcg/tuple (tcg/return :call) family k)
                         (tcg/tuple (tcg/return :add) k k)
                         (tcg/tuple (tcg/return :hoisted) k)]
                  (not callee?) (into [(tcg/tuple (tcg/return :vector) (tcg/vector k 1 3))
                                       (tcg/tuple (tcg/return :splice) k k)])))))

(defn- gen-spec
  "A spec over opts' families (gen-statement's opts)."
  [opts]
  (tcg/let [{:keys [params args alt-args]} gen-signature
            store (tcg/frequency [[2 (tcg/return nil)] [1 (tcg/fmap #(hash-map :w %) quarter)]])
            callees (tcg/vector (tcg/hash-map :statements (gen-statements (assoc opts :calls? false) 3)
                                              :ret (gen-ret opts true))
                                0 2)
            statements (gen-statements (assoc opts :calls? true) 12)
            ret (gen-ret opts false)]
    (array-map :params params :args args :alt-args alt-args :store store
               :callees callees :statements statements :ret ret)))

(def gen-model
  "A random static model, a gen.laws model entry [label gf args alt-args]
   with label {:spec :form}: every family, discrete ones too."
  (tcg/fmap model (gen-spec {:families (vec (keys families)) :discrete? true})))

(def gen-continuous-model
  "A random static model of continuous families only (with integer
   arguments, counts and computed addresses), for gradient-based kernels."
  (tcg/fmap model (gen-spec {:families continuous-families :discrete? false})))
