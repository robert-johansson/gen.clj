(ns gen.combinators.recurse
  "Recurse (gen.combinators): a tree grown from the root by the production
   kernel, then folded from the leaves by the aggregation kernel. Node i's
   production trace is at (:gen.combinators/productions tr), its
   aggregation's at (:gen.combinators/aggregations tr), their choices at
   [i :production] and [i :aggregation]."
  {:api :internal}
  (:refer-clojure :exclude [+ -])
  (:require [gen.array :as array :refer [+ -]]
            [gen.choicemap :as cm]
            [gen.combinators :as-alias comb]
            [gen.combinators.calls :as calls :refer [element-key stopped?]]
            [gen.diff :as diff]
            [gen.gfi :as gfi]
            [gen.selection :as sel]
            [gen.trace :as trace]))

(set! *warn-on-reflection* true)

(defn child-of
  "The node number of child j of node node, unchecked: Recurse's own
   loops, whose max-branch and number of children are checked already."
  [node j max-branch]
  (clojure.core/+ (* node max-branch) j 1))

(defn- put-part
  "choices with sub at [node part], unless sub is empty."
  [choices node part sub]
  (if (cm/empty? sub) choices (cm/set-choice choices [node part] sub)))

(defn- production-output
  "The production kernel's return value at node, checked."
  [gf node {out :retval}]
  (when-not (and (map? out) (contains? out :value) (sequential? (:children out)))
    (throw (ex-info (str "Recurse's production kernel returns {:value v :children [inputs …]}, not " (pr-str out)
                         " (node " node ")")
                    {:node node :retval out})))
  (when (> (count (:children out)) (:max-branch gf))
    (throw (ex-info (str "Recurse: node " node " has " (count (:children out)) " children, more than max-branch "
                         (:max-branch gf))
                    {:node node :children (count (:children out)) :max-branch (:max-branch gf)})))
  out)

(defn- recurse-targets
  "The nodes the constraints (update, generate) or the selection
   (regenerate) reach: {node #{:production :aggregation}}, or :all."
  [op cs]
  (cond
    (nil? cs) {}
    (= op :regenerate)
    (let [is (calls/selected-indices cs)]
      (if (= :all is) :all (into {} (map (fn [i] [i #{:production :aggregation}])) is)))
    :else
    (do (when (cm/has-value? cs)
          (throw (gfi/unvisited op [[]] "a single value was given where Recurse's nodes belong")))
        (into {} (map (fn [[i sub]]
                        (when-not (nat-int? i)
                          (throw (gfi/unvisited op [[i]] "Recurse's addresses are node indices")))
                        (when (cm/has-value? sub)
                          (throw (gfi/unvisited op [[i]] (str "a value at node " i
                                                              ": a node has :production and :aggregation"))))
                        (let [parts (set (map first (cm/submaps sub)))]
                          (when-let [bad (seq (remove #{:production :aggregation} parts))]
                            (throw (gfi/unvisited op (mapv #(vector i %) bad) "a node has :production and :aggregation")))
                          [i parts])))
              (cm/submaps cs)))))

(defn- targeted? [targets node part]
  (or (= :all targets) (contains? (get targets node) part)))

(defn- run-node
  "One kernel run at node: revised from old when given (update or
   regenerate), else new (generate, or simulate for regenerate and
   unconstrained runs). {:trace :weight :retdiff :discard}. cs is what
   reaches the node's part (calls/reaching), or nil."
  [op kernel old args argdiffs cs key at]
  (if old
    (calls/revise-at op kernel old args argdiffs cs key at)
    (assoc (calls/fresh-at kernel args (when (not= op :regenerate) cs) key at) :retdiff diff/unknown-change)))

(defn- production-phase
  "The productions from the root, in node order (parents before children):
   a retained node is revisited only when its input may have changed or it
   is targeted. {:prods :p-diffs :weight :discard :stopped}, :stopped true
   when a production stopped the run (stopped?): no node after it runs."
  [{:keys [gf old-p targets op node-cs node-key]} input input-diff]
  (let [{:keys [production max-branch]} gf]
    ;; weight starts boxed: it accumulates arrays on array backends
    (loop [todo (sorted-map 0 [input input-diff]) prods {} p-diffs {} weight (Double/valueOf 0.0) discard cm/EMPTY]
      (if-let [[node [in d]] (first todo)]
        (let [old (old-p node)
              revisit? (or (nil? old) (diff/changed? d) (targeted? targets node :production))
              r (if revisit?
                  (run-node op production old [in] [d] (node-cs node :production) (node-key node :production)
                            [node :production])
                  {:trace old :weight 0.0 :retdiff diff/no-change})
              prods (assoc prods node (:trace r))
              p-diffs (assoc p-diffs node (:retdiff r))
              weight (+ weight (:weight r))
              discard (put-part discard node :production (:discard r))]
          (if (stopped? (:trace r))
            {:prods prods :p-diffs p-diffs :weight weight :discard discard :stopped true}
            (let [out (production-output gf node (:trace r))
                  child-diff (if (diff/changed? (:retdiff r)) diff/unknown-change diff/no-change)]
              (recur (into (dissoc todo node)
                           (map-indexed (fn [j c] [(child-of node j max-branch) [c child-diff]]) (:children out)))
                     prods p-diffs weight discard))))
        {:prods prods :p-diffs p-diffs :weight weight :discard discard :stopped false}))))

(defn- aggregation-phase
  "The aggregations from the leaves (children before parents): a retained
   node is revisited when its value, its children or its target may have
   changed. {:aggs :a-diffs :weight :discard :stopped}, the weight and
   discard continuing the production phase's; none runs after a production
   phase that stopped, nor after an aggregation that stops the run."
  [{:keys [gf old-a targets op node-cs node-key]} {:keys [prods p-diffs weight discard stopped]}]
  (let [{:keys [aggregation max-branch]} gf
        start {:aggs {} :a-diffs {} :weight weight :discard discard :stopped stopped}]
    (if stopped
      start
      (reduce (fn [{:keys [aggs a-diffs weight discard]} node]
                (let [{v :value children :children} (:retval (prods node))
                      kids (map #(child-of node % max-branch) (range (count children)))
                      old (old-a node)
                      v-changed? (diff/changed? (p-diffs node))
                      ws-changed? (or (nil? old) (not= (count children) (count (second (:args old))))
                                      (some #(diff/changed? (a-diffs %)) kids))
                      r (if (or (nil? old) v-changed? ws-changed? (targeted? targets node :aggregation))
                          (run-node op aggregation old [v (mapv #(:retval (aggs %)) kids)]
                                    [(if v-changed? diff/unknown-change diff/no-change)
                                     (if ws-changed? diff/unknown-change diff/no-change)]
                                    (node-cs node :aggregation) (node-key node :aggregation) [node :aggregation])
                          {:trace old :weight 0.0 :retdiff diff/no-change})
                      acc {:aggs (assoc aggs node (:trace r)) :a-diffs (assoc a-diffs node (:retdiff r))
                           :weight (+ weight (:weight r)) :discard (put-part discard node :aggregation (:discard r))
                           :stopped false}]
                  (if (stopped? (:trace r)) (reduced (assoc acc :stopped true)) acc)))
              start
              (sort > (keys prods))))))

(defn- node-choices [choices node p a]
  (-> choices (put-part node :production (:choices p)) (put-part node :aggregation (:choices a))))

(defn- node-key [key node part] (element-key (element-key key node) (if (= part :production) 0 1)))

(defn check-args
  "Recurse's arguments, (input) the root's, as a vector."
  [args]
  (when-not (= 1 (count args))
    (throw (ex-info (str "Recurse takes one argument, the root's input; given " (count args)) {:args args})))
  (vec args))

(defn run
  "generate (tr nil), update or regenerate of a Recurse trace: the
   production phase, then the aggregation phase. Nodes no longer reached
   are dropped: update discards them and subtracts their scores; in
   regenerate they cancel against the reverse move."
  [gf tr [input] [input-diff] op cs key]
  (let [ctx {:gf gf :old-p (::comb/productions tr {}) :old-a (::comb/aggregations tr {}) :targets (recurse-targets op cs)
             :op op
             :node-cs (fn [node part] (when cs (calls/reaching op cs [node part])))
             :node-key (partial node-key key)}
        {:keys [prods] :as produced} (production-phase ctx input input-diff)
        ;; a run that stopped (stopped?) never reached the nodes after it
        _ (when (and cs (not= op :regenerate) (not (:stopped produced)))
            (doseq [node (keys (:targets ctx))]
              (when-not (contains? prods node)
                (throw (gfi/unvisited op [[node]] (str "the tree has nodes " (vec (sort (keys prods)))))))))
        {:keys [aggs a-diffs weight discard stopped]} (aggregation-phase ctx produced)
        {:keys [old-p old-a]} ctx
        ;; every node has both parts, unless the run stopped before a part
        gone-p (remove prods (keys old-p))
        gone-a (remove aggs (keys old-a))
        [weight discard] (if (= op :update)
                           [(as-> weight w
                              (reduce (fn [w node] (- w (:score (old-p node)))) w gone-p)
                              (reduce (fn [w node] (- w (:score (old-a node)))) w gone-a))
                            (as-> discard d
                              (reduce #(put-part %1 %2 :production (:choices (old-p %2))) d gone-p)
                              (reduce #(put-part %1 %2 :aggregation (:choices (old-a %2))) d gone-a))]
                           [weight discard])
        nodes (sort (keys prods))
        t (assoc (trace/make-trace {:gen-fn gf :args [input]
                                    :choices (reduce #(node-choices %1 %2 (prods %2) (aggs %2)) cm/EMPTY nodes)
                                    :retval (:retval (aggs 0))
                                    :score (reduce + 0.0 (concat (map :score (vals prods)) (map :score (vals aggs))))})
                 ::comb/productions prods ::comb/aggregations aggs)]
    (cond-> {:trace t :weight weight}
      (not= op :generate) (assoc :retdiff (if (and tr (not stopped)) (a-diffs 0) diff/unknown-change))
      (= op :update) (assoc :discard discard))))

(defn assess
  "assess of a Recurse on args: its productions from the root, then its
   aggregations from the leaves, until a part stops the run."
  [gf [input] choices key]
  (let [{:keys [production aggregation max-branch]} gf
        targets (recurse-targets :assess choices)
        ;; stopped: a part stopped the run (stopped?), and nothing after it runs
        [ps weight stopped] (loop [todo (sorted-map 0 input) ps {} weight (Double/valueOf 0.0)]
                              (if-let [[node in] (first todo)]
                                (let [r (calls/assess-at production [in] (cm/get-submap choices [node :production] cm/EMPTY)
                                                         (node-key key node :production) [node :production])
                                      weight (+ weight (:weight r))]
                                  (if (stopped? (:weight r))
                                    [ps weight true]
                                    (let [out (production-output gf node r)]
                                      (recur (into (dissoc todo node)
                                                   (map-indexed (fn [j c] [(child-of node j max-branch) c]) (:children out)))
                                             (assoc ps node out) weight))))
                                [ps weight false]))
        _ (when-not stopped
            (doseq [node (keys targets)]
              (when-not (contains? ps node)
                (throw (gfi/unvisited :assess [[node]] (str "the tree has nodes " (vec (sort (keys ps)))))))))
        [as weight] (if stopped
                      [{} weight]
                      (reduce (fn [[as weight] node]
                                (let [{v :value children :children} (ps node)
                                      ws (mapv #(as (child-of node % max-branch)) (range (count children)))
                                      r (calls/assess-at aggregation [v ws] (cm/get-submap choices [node :aggregation] cm/EMPTY)
                                                         (node-key key node :aggregation) [node :aggregation])
                                      acc [(assoc as node (:retval r)) (+ weight (:weight r))]]
                                  (if (stopped? (:weight r)) (reduced [{} (second acc)]) acc)))
                              [{} weight] (sort > (keys ps))))]
    {:weight weight :retval (as 0)}))

(defn project
  "The projection of a Recurse trace on the selection s: its parts' under
   their subselections."
  [tr s]
  (calls/selected-indices s) ; for its check: a negative node is loud, as in regenerate
  (let [part (fn [subs part]
               (reduce (fn [w [node sub]]
                         (+ w (gfi/-project (:gen-fn sub) sub (sel/get-subselection s [node part]))))
                       0.0 subs))]
    (+ (part (::comb/productions tr) :production) (part (::comb/aggregations tr) :aggregation))))
