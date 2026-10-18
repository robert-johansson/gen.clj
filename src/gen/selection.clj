(ns gen.selection
  "Selections: which addresses an operation (regenerate, project) acts on.
   They form a Boolean algebra (all, none, complement) over hierarchical
   address sets. As in Gen.jl, selecting an address selects everything under
   it."
  (:refer-clojure :exclude [complement])
  (:require [gen.choicemap :as cm]))

(set! *warn-on-reflection* true)

(defprotocol ISelection
  (selected? [s addr] "Is the leaf at addr selected?")
  (get-subselection [s addr] "The selection for the addresses under addr."))

(defrecord All []
  ISelection
  (selected? [_ _] true)
  (get-subselection [this _] this))

(defrecord None []
  ISelection
  (selected? [_ _] false)
  (get-subselection [this _] this))

(def all "Selects every address." (->All))
(def none "Selects no address." (->None))

(defrecord Hierarchical [m]
  ISelection
  ;; A leaf at addr is selected only when its whole subtree is: an entry that
  ;; descends to select part of addr's subtree does not select addr itself.
  (selected? [_ addr] (instance? All (get m addr none)))
  (get-subselection [_ addr] (get m addr none)))

(defrecord Complement [inner]
  ISelection
  (selected? [_ addr] (not (selected? inner addr)))
  (get-subselection [_ addr] (->Complement (get-subselection inner addr))))

(defn from-paths
  "The selection of a collection of address paths (vectors). An empty path
   selects everything."
  [paths]
  (cond (clojure.core/empty? paths) none
        (some empty? paths) all
        :else (->Hierarchical
               (into {}
                     (map (fn [[addr group]] [addr (from-paths (map #(vec (rest %)) group))]))
                     (group-by first paths)))))

(defn select
  "The selection of the given addresses or paths, each with everything under it.
   (select :slope [:y 0])"
  [& addrs]
  (from-paths (map #(if (vector? %) % [%]) addrs)))

(defn hierarchical
  "A selection from address-subselection pairs.
   (hierarchical :params (select :a) :noise all)"
  [& kvs]
  (->Hierarchical (apply hash-map kvs)))

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
        (cm/has-value? choices) (if (selected? s []) choices cm/EMPTY)
        :else (reduce (fn [acc [addr sub]]
                        (let [kept (if (cm/has-value? sub)
                                     (when (selected? s addr) sub)
                                     (get-selected sub (get-subselection s addr)))]
                          (if (or (nil? kept) (cm/empty? kept)) acc (cm/set-submap acc addr kept))))
                      cm/EMPTY
                      (cm/submaps choices))))
