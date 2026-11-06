(ns gen.combinators.switch
  "Switch (gen.combinators): its run by the branch its index names, and,
   in a batched run whose particles' indices differ, by partition: each
   branch on its own particles (gen.partition)."
  {:api :internal}
  (:require [gen.array :as array]
            [gen.array.batch :as batch]
            [gen.choicemap :as cm]
            [gen.combinators :as-alias comb]
            [gen.combinators.calls :as calls]
            [gen.diff :as diff]
            [gen.gfi :as gfi]
            [gen.partition :as part]
            [gen.trace :as trace]))

(set! *warn-on-reflection* true)

(declare own-constraints)

(defn- branch-index
  "The branch index as a host value: a one-element array is read. (A
   batched one runs by partition: index-keys.)"
  [idx]
  (if (array/backend-array? idx) (array/item idx) idx))

(defn- branch [gf k]
  (or (get (:branches gf) k)
      (throw (ex-info (str "Switch has no branch " (pr-str k)) {:index k :branches (count (:branches gf))}))))

(defn- unknown-diffs [args] (vec (repeat (count args) diff/unknown-change)))

(defn- switch-trace [gf args k sub]
  (assoc (trace/make-trace {:gen-fn gf :args args :choices (:choices sub) :retval (:retval sub) :score (:score sub)})
         ::comb/branch k
         ::comb/subtrace sub))

(defn- generate-branch
  "generate (constraints, or nil to simulate) of a Switch on args by its
   branch k."
  [gf args k constraints key]
  (let [{sub :trace w :weight} (calls/fresh-at (branch gf k) (vec (rest args))
                                               (when constraints (own-constraints constraints array/*backend*)) key nil)]
    {:trace (switch-trace gf args k sub) :weight w}))

(defn- revise-branch
  "update or regenerate of a Switch trace by the branch k: the branch at the
   new index runs against the old branch's trace, another branch by
   gfi/revise, so the addresses both make keep their values (Def 2.3.1), as
   the equivalent if would and as Gen.jl does."
  [gf tr args argdiffs op cs key k]
  (let [bargs (vec (rest args))
        b (branch gf k)
        bdiffs (if (= k (::comb/branch tr)) (vec (rest argdiffs)) (unknown-diffs bargs))
        r (calls/revise-at op b (::comb/subtrace tr) bargs bdiffs
                           (if (= op :update) (own-constraints cs array/*backend*) cs) key nil)]
    (assoc r :trace (switch-trace gf args k (:trace r)))))

;; A Switch whose index differs across the particles of a batched run runs
;; by partition (DESIGN.md §9): the particles are grouped by branch, each
;; branch runs batched over its group under a part's backend, which draws
;; for a particle what the batch's own run would, and the results are
;; scattered back (gen.partition). The trace holds each group's branch, rows
;; and subtrace (:gen.combinators/parts, a gen.array.batch/Parts), and its
;; choices by group, a Parts value at the call's address, which gen.batched
;; reads per particle. An update or regenerate runs together each old
;; group's particles that go to one branch (a cell), against their old
;; subtraces, and joins a branch's cells (gen.partition/merge-groups). Every
;; particle in one branch is the plain trace.

(defn- index-keys
  "The branch indices of the particles of the batch in effect, host values
   (as branch-index reads one), from a batched index."
  [idx]
  (let [s (array/shape idx)
        v (batch/value idx)]
    (when-not (== 1 (reduce * 1 s))
      (throw (ex-info (str "A Switch's index is one value, not an array of shape " s) {:shape s})))
    (array/->clj (array/reshape v [(first (array/shape v))]))))

(defn- one-key? [ks] (every? #(= (first ks) %) ks))

(defn- choices-by-part
  "The choices of groups whose data are choice maps: a Parts value, or EMPTY
   when there are none."
  [groups]
  (if (every? (comp cm/empty? :data) groups) cm/EMPTY (cm/->Value (batch/->Parts groups))))

(defn- parts-trace
  "The trace of a Switch on args whose particles ran in groups ({:key :rows
   :data}, data the group's subtrace), with its score and return value."
  [gf args groups score retval]
  (assoc (trace/make-trace {:gen-fn gf :args args :score score :retval retval
                            :choices (choices-by-part (mapv #(update % :data :choices) groups))})
         ::comb/parts (batch/->Parts groups)))

(defn- scattered-trace
  "parts-trace, its score and return value scattered back from the groups'
   onto the batch backend b."
  [gf args groups b]
  (let [rows (mapv :rows groups)
        subs (mapv :data groups)]
    (parts-trace gf args groups (batch/scatter (mapv :score subs) rows b) (part/scatter (mapv :retval subs) rows b))))

(defn- in-order
  "The order taking particles held at rows to the batch's own: nil when
   they are in it."
  [rows]
  (when-not (= rows (range (count rows)))
    (let [inv (long-array (count rows))]
      (dotimes [j (count rows)] (aset inv (long (rows j)) (long j)))
      (vec inv))))

(defn- own-constraints
  "constraints, for a run of the batch backend b's particles: choices held
   by part (a Parts value: a batched trace's choices given back, as a
   gradient's assess or a rebuild gives them) whose particles are all in
   one group are that group's, taken into b; any other as they are (where
   the particles' choices differ in structure, the branch's run fails,
   loudly)."
  [constraints b]
  (let [v (cm/get-value constraints nil)]
    (if (and (batch/parts? v) (== 1 (count (:groups v))))
      (let [{rows :rows data :data} (first (:groups v))]
        (part/gather-into data (in-order rows) b))
      constraints)))

(defn- in-part
  "(f pb) under pb, the part backend of the particles at rows of the batch
   backend b."
  [b rows f]
  (let [pb (batch/part-backend b rows)]
    (array/with-backend pb (f pb))))

(defn- generate-parts
  "generate (constraints, or nil to simulate) of a Switch on args whose
   index keys ks differ across the particles: each branch on its group."
  [gf args ks constraints key]
  (let [b array/*backend*
        gs (part/groups ks)
        rs (mapv (fn [[k rows]]
                   (in-part b rows
                            (fn [pb]
                              (let [bargs (part/gather-into (vec (rest args)) rows pb)]
                                (calls/fresh-at (branch gf k) bargs
                                                (when constraints (own-constraints (part/gather-into constraints rows pb) pb))
                                                key nil)))))
                 gs)]
    {:trace (scattered-trace gf args (mapv (fn [[k rows] r] {:key k :rows rows :data (:trace r)}) gs rs) b)
     :weight (batch/scatter (mapv :weight rs) (mapv second gs) b)}))

(defn- old-groups
  "The groups of a Switch trace of m particles: its parts, or its one
   branch's."
  [tr m]
  (if-let [ps (::comb/parts tr)]
    (:groups ps)
    [{:key (::comb/branch tr) :rows (vec (range m)) :data (::comb/subtrace tr)}]))

(defn- revise-parts
  "update or regenerate of a Switch trace whose particles' branches differ,
   before or after: each cell, the particles of an old group that go to one
   branch, revised by that branch against their old subtraces (another
   branch's by gfi/revise, as revise-branch), and a branch's cells joined."
  [gf tr args argdiffs op cs key ks]
  (let [b array/*backend*
        bargs (vec (rest args))
        cells (vec (for [{old :key rows :rows data :data} (old-groups tr (count ks))
                         [k places] (part/groups (mapv ks rows))]
                     {:key k :old old :rows (mapv rows places) :data data
                      :places (when-not (= places (range (count rows))) places)}))
        rs (mapv (fn [{k :key old :old rows :rows data :data places :places}]
                   (in-part b rows
                            (fn [pb]
                              (let [prev (part/gather-into data places pb)
                                    ba (part/gather-into bargs rows pb)
                                    bdiffs (if (= k old) (vec (rest argdiffs)) (unknown-diffs bargs))]
                                (calls/revise-at op (branch gf k) prev ba bdiffs
                                                 (if (= op :update) (own-constraints (part/gather-into cs rows pb) pb) cs)
                                                 key nil)))))
                 cells)
        groups (part/merge-groups (mapv (fn [c r] {:key (:key c) :rows (:rows c) :data (:trace r)}) cells rs) b)
        t (if (== 1 (count groups))
            (let [{k :key rows :rows data :data} (first groups)]
              (switch-trace gf args k (part/gather-into data (in-order rows) b)))
            (scattered-trace gf args groups b))]
    (cond-> {:trace t
             :weight (batch/scatter (mapv :weight rs) (mapv :rows cells) b)
             :retdiff (if (every? #(diff/no-change? (:retdiff %)) rs) diff/no-change diff/unknown-change)}
      (= op :update) (assoc :discard (choices-by-part (mapv (fn [c r] {:key [(:old c) (:key c)] :rows (:rows c) :data (:discard r)})
                                                            cells rs))))))

;; ---------------------------------------------------------------------------
;; The GFI's operations
;; ---------------------------------------------------------------------------

(defn generate
  "generate (constraints, or nil to simulate) of a Switch on args: by its
   branch, or by partition when the particles' branches differ."
  [gf [idx :as args] constraints key]
  (let [args (vec args)]
    (if (batch/batched? idx)
      (let [ks (index-keys idx)]
        (if (one-key? ks)
          (generate-branch gf args (first ks) constraints key)
          (generate-parts gf args ks constraints key)))
      (generate-branch gf args (branch-index idx) constraints key))))

(defn revise
  "update or regenerate of a Switch trace: by its branch, or by partition
   when the particles' branches differ, before or after."
  [gf tr [idx :as args] argdiffs op cs key]
  (let [args (vec args)
        ks (when (batch/batched? idx) (index-keys idx))]
    (if (or (::comb/parts tr) (and ks (not (one-key? ks))))
      (revise-parts gf tr args argdiffs op cs key
                    (or ks (vec (repeat (batch/size array/*backend*) (branch-index idx)))))
      (revise-branch gf tr args argdiffs op cs key (if ks (first ks) (branch-index idx))))))

(defn assess
  "assess of a Switch on args: by its branch, or by partition when the
   particles' branches differ."
  [gf [idx & bargs] choices key]
  (let [bargs (vec bargs)
        ks (when (batch/batched? idx) (index-keys idx))]
    (if (and ks (not (one-key? ks)))
      (let [b array/*backend*
            gs (part/groups ks)
            rs (mapv (fn [[k rows]]
                       (in-part b rows #(gfi/-assess (branch gf k) (part/gather-into bargs rows %)
                                                     (own-constraints (part/gather-into choices rows %) %) key)))
                     gs)]
        {:weight (batch/scatter (mapv :weight rs) (mapv second gs) b)
         :retval (part/scatter (mapv :retval rs) (mapv second gs) b)})
      (gfi/-assess (branch gf (if ks (first ks) (branch-index idx))) bargs (own-constraints choices array/*backend*) key))))

(defn project
  "The projection of a Switch trace on the selection: its branch's, or
   each group's."
  [tr selection]
  (if-let [ps (::comb/parts tr)]
    (let [b array/*backend*
          gs (:groups ps)]
      (batch/scatter (mapv (fn [{rows :rows sub :data}]
                             (array/with-backend (part/trace-backend sub (count rows) b)
                               (gfi/-project (:gen-fn sub) sub selection)))
                           gs)
                     (mapv :rows gs) b))
    (let [sub (::comb/subtrace tr)] (gfi/-project (:gen-fn sub) sub selection))))

(defn join
  "Switch traces of consecutive parts of a batch as one (gen.partition's
   IJoin): their groups merged by branch, the plain trace when every
   particle is in one."
  [gf trs ns b]
  (let [groups (part/merge-groups (part/consecutive (mapv old-groups trs ns) ns) b)
        args (part/join (mapv :args trs) ns b)]
    (if (== 1 (count groups))
      (let [{k :key data :data} (first groups)] (switch-trace gf args k data))
      (parts-trace gf args groups (part/join (mapv :score trs) ns b) (part/join (mapv :retval trs) ns b)))))
