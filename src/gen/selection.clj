(ns gen.selection
  "Selections: which addresses an operation (regenerate, project) acts on.
   They form a Boolean algebra (all, none, complement) over hierarchical
   address sets. As in Gen.jl, selecting an address selects everything under
   it.

   An address is any value except a vector; a vector is a path, as in
   gen.choicemap: selected?, get-subselection and set-subselection take
   either, and the empty path is the root. A selection implements
   ISelection one address at a time."
  (:refer-clojure :exclude [complement])
  (:require [gen.choicemap :as cm]))

(set! *warn-on-reflection* true)

(defprotocol ISelection
  (-selected? [s addr] "Is the leaf at the single address addr selected?")
  (-subselection [s addr] "The selection for the addresses under the single address addr."))

(defrecord All []
  ISelection
  (-selected? [_ _] true)
  (-subselection [this _] this))

(defrecord None []
  ISelection
  (-selected? [_ _] false)
  (-subselection [this _] this))

(defn selection? [x] (instance? gen.selection.ISelection x))

(def all "Selects every address." (->All))
(def none "Selects no address." (->None))

(defrecord Hierarchical [m]
  ISelection
  ;; A leaf at addr is selected only when its whole subtree is: an entry that
  ;; descends to select part of addr's subtree does not select addr itself.
  (-selected? [_ addr] (instance? All (get m addr none)))
  (-subselection [_ addr] (get m addr none)))

(defrecord Complement [inner]
  ISelection
  (-selected? [_ addr] (not (-selected? inner addr)))
  (-subselection [_ addr] (->Complement (-subselection inner addr))))

(defn root-selected?
  "Does s select the root of a trace whose choices are a single value (a
   distribution's trace)? Only all-selecting selections do."
  [s]
  (cond (instance? All s) true
        (instance? Complement s) (not (root-selected? (:inner s)))
        :else false))

(defn get-subselection
  "The selection for the addresses under an address or path (s itself at
   the empty path)."
  [s addr]
  (if (vector? addr) (reduce -subselection s (cm/path addr)) (-subselection s addr)))

(defn selected?
  "Is the leaf at an address or path selected? At the empty path, the
   root: root-selected?."
  [s addr]
  (cond (not (vector? addr)) (-selected? s addr)
        (empty? addr) (root-selected? s)
        :else (let [p (cm/path addr)] (-selected? (get-subselection s (pop p)) (peek p)))))

(defn- kind
  "s's kind, as the docstrings name it."
  [s]
  (cond (instance? All s) "all"
        (instance? Complement s) "a complement"
        :else (str "another kind of selection (" (.getSimpleName ^Class (class s)) ")")))

(defn set-subselection
  "s with sub the selection under an address or path (Gen.jl's
   set_subselection!, persistent; sub itself at the empty path). s and the
   selections on the path are none or hierarchical: setting inside all, a
   complement or another kind of selection is an error."
  [s addr sub]
  (when-not (selection? sub)
    (throw (ex-info (str "set-subselection takes a selection, not a " (type sub)) {:addr addr})))
  (let [p (cm/path addr)]
    ((fn put [s i]
       (cond (== i (count p)) sub
             (instance? None s) (->Hierarchical {(nth p i) (put none (inc i))})
             (instance? Hierarchical s) (let [m (:m s) a (nth p i)]
                                          (->Hierarchical (assoc m a (put (get m a none) (inc i)))))
             :else (throw (ex-info (str "set-subselection: the path " (pr-str addr) " goes inside "
                                        (kind s) ", which cannot be nested")
                                   {:path p :selection s}))))
     s 0)))

(defn- paths->selection [paths]
  (cond (clojure.core/empty? paths) none
        (some empty? paths) all
        :else (->Hierarchical
               (into {}
                     (map (fn [[addr group]] [addr (paths->selection (map #(subvec % 1) group))]))
                     (group-by first paths)))))

(defn from-paths
  "The selection of a collection of address paths (vectors). An empty path
   selects everything."
  [paths]
  (doseq [p paths]
    (when-not (vector? p)
      (throw (ex-info (str "from-paths takes paths (vectors), not " (pr-str p)) {:path p})))
    (cm/path p))
  (paths->selection paths))

(defn select
  "The selection of the given addresses or paths, each with everything under it.
   (select :slope [:y 0])"
  [& addrs]
  (from-paths (map #(if (vector? %) % [%]) addrs)))

(defn hierarchical
  "A selection from address-subselection pairs, an address or a path each
   (set-subselection's).
   (hierarchical :params (select :a) :noise all [:obs 0] all)"
  [& kvs]
  (when (odd? (count kvs))
    (throw (ex-info "hierarchical takes address-selection pairs" {:arg-count (count kvs)})))
  (reduce (fn [s [addr sub]] (set-subselection s addr sub)) none (partition 2 kvs)))

(defn complement
  "Every address s does not select."
  [s]
  (cond (instance? All s) none
        (instance? None s) all
        (instance? Complement s) (:inner s)
        :else (->Complement s)))

(defn from-choicemap
  "The selection of exactly the addresses present in a choice map.
   (complement (from-choicemap observations)) is every latent address."
  [choices]
  (from-paths (cm/addresses choices)))

(defn get-selected
  "The part of a choice map that s selects (Gen.jl's get_selected)."
  [choices s]
  (cond (instance? All s) choices
        (instance? None s) cm/EMPTY
        (cm/has-value? choices) (if (root-selected? s) choices cm/EMPTY)
        :else (reduce (fn [acc [addr sub]]
                        (let [kept (if (cm/has-value? sub)
                                     (when (-selected? s addr) sub)
                                     (get-selected sub (-subselection s addr)))]
                          (if (or (nil? kept) (cm/empty? kept)) acc (cm/set-submap acc addr kept))))
                      cm/EMPTY
                      (cm/submaps choices))))
