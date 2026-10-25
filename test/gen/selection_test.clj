(ns gen.selection-test
  "gen.selection's Boolean algebra: union and intersection beside
   complement, by their laws over random selections, and how selections
   print. Two selections are equal as sets when they agree at every path
   of up to three addresses of a small alphabet, the root's included:
   selected? at a path reads the subselections along it, so that compares
   them too. (gen.laws checks project over unions, intersections and
   complements on every kind of generative function.)"
  (:require [clojure.set :as set]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [gen.choicemap :as cm]
            [gen.selection :as sel]
            [gen.test-support :as ts]))

;; ---------------------------------------------------------------------------
;; Random selections, and where to compare them
;; ---------------------------------------------------------------------------

(def ^:private alphabet [:a :b 0 1])

(def ^:private paths
  "Every path of up to three addresses of the alphabet, the root's
   included: 85."
  (vec (concat [[]]
               (for [a alphabet] [a])
               (for [a alphabet b alphabet] [a b])
               (for [a alphabet b alphabet c alphabet] [a b c]))))

(def ^:private gen-path (gen/vector (gen/elements alphabet) 1 3))

(def gen-selection
  "Selections built by every constructor: all, none and selects at the
   leaves; complements, unions, intersections and hierarchical selections
   of them above."
  (gen/recursive-gen
   (fn [inner]
     (gen/one-of [(gen/fmap sel/complement inner)
                  (gen/fmap #(apply sel/union %) (gen/vector inner 0 3))
                  (gen/fmap #(apply sel/intersection %) (gen/vector inner 0 3))
                  (gen/fmap #(apply sel/hierarchical (apply concat %))
                            (gen/vector (gen/tuple (gen/elements alphabet) inner) 1 3))]))
   ;; mostly selects: all and none absorb or drop out of most unions and
   ;; intersections, which would leave them trivial
   (gen/frequency [[1 (gen/return sel/all)]
                   [1 (gen/return sel/none)]
                   [6 (gen/fmap #(apply sel/select %) (gen/vector gen-path 1 3))]])))

(def ^:private gen-choicemap
  "Choice maps over the alphabet, a value at the root among them, which
   only a root-selecting selection selects."
  (gen/fmap cm/from-map
            (gen/recursive-gen #(gen/map (gen/elements alphabet) % {:min-elements 1 :max-elements 3})
                               (gen/fmap double gen/small-integer))))

(defn- selects
  "The paths of `paths` s selects."
  [s]
  (set (filter #(sel/selected? s %) paths)))

(defn- kept
  "The addresses of c that s selects (get-selected)."
  [c s]
  (set (cm/addresses (sel/get-selected c s))))

;; ---------------------------------------------------------------------------
;; The laws
;; ---------------------------------------------------------------------------

;; Planted bugs (2026-10-25), each failing the three laws below that
;; compare sets: an intersection that selects an address when any part does
;; (some for every?); a union whose subselection is its first part's; a
;; union that selects the root only when every part does; the meet of two
;; hierarchical selections that keeps an address only one of them has; a
;; union that drops its complements. An intersection whose top addresses
;; are none when a part may select anything, which selected? never reads,
;; fails the-root-and-the-top-addresses (and gen.laws's project law on
;; Map and Unfold, which read them).

(defspec union-and-intersection-act-path-by-path 300
  (prop/for-all [s gen-selection t gen-selection u gen-selection]
    (and (= (selects (sel/union s t)) (set/union (selects s) (selects t)))
         (= (selects (sel/intersection s t)) (set/intersection (selects s) (selects t)))
         (= (selects (sel/union s t u)) (set/union (selects s) (selects t) (selects u)))
         (= (selects (sel/intersection s t u)) (set/intersection (selects s) (selects t) (selects u))))))

(defspec de-morgan-s-laws-hold-with-complement 300
  (prop/for-all [s gen-selection t gen-selection]
    (and (= (selects (sel/complement (sel/union s t)))
            (selects (sel/intersection (sel/complement s) (sel/complement t))))
         (= (selects (sel/complement (sel/intersection s t)))
            (selects (sel/union (sel/complement s) (sel/complement t))))
         (= (selects (sel/complement s)) (set/difference (set paths) (selects s))))))

(defspec get-selected-takes-a-union-or-an-intersection-by-its-parts 300
  (prop/for-all [c gen-choicemap s gen-selection t gen-selection]
    (and (= (kept c (sel/union s t)) (set/union (kept c s) (kept c t)))
         (= (kept c (sel/intersection s t)) (set/intersection (kept c s) (kept c t)))
         (= (kept c (sel/complement (sel/union s t)))
            (kept c (sel/intersection (sel/complement s) (sel/complement t)))))))

(defspec a-union-of-selects-is-the-select-of-their-paths 200
  ;; the hierarchical selections join into one, the same record that select
  ;; builds, so a union prints, nests and compares as a select does
  (prop/for-all [ps (gen/vector gen-path 0 4) qs (gen/vector gen-path 0 4)]
    (= (apply sel/select (concat ps qs)) (sel/union (apply sel/select ps) (apply sel/select qs)))))

(deftest identities-and-absorption
  (let [s (sel/select :x [:y 0])
        c (sel/complement s)]
    (is (= [sel/none sel/all] [(sel/union) (sel/intersection)]) "the empty union and intersection")
    (is (= [s s c c] [(sel/union s) (sel/intersection s) (sel/union c) (sel/intersection c)]))
    (is (= [s s] [(sel/union s sel/none) (sel/intersection s sel/all)]))
    (is (= [sel/all sel/none] [(sel/union c sel/all) (sel/intersection c sel/none)]))
    (is (= [c c] [(sel/union c c) (sel/intersection c c)]) "idempotent")
    (is (= (sel/union s c) (sel/union (sel/union s) (sel/union c))) "a union of unions is flat")
    (is (= (count (:parts (sel/union (sel/union c (sel/complement (sel/select :z))) (sel/select :w)))) 3))
    (is (= sel/none (sel/intersection (sel/select :x) (sel/select :y))) "disjoint selects meet in none")
    (is (= (sel/select [:y 0]) (sel/intersection s (sel/select :y))))))

(deftest the-root-and-the-top-addresses
  ;; a distribution's trace is a value at the root: root-selected? decides it
  (let [x (sel/select :x)]
    (is (= [true false] (map sel/root-selected? [(sel/union x (sel/complement x)) (sel/intersection x (sel/complement x))])))
    (is (sel/selected? (sel/union x (sel/complement x)) []))
    (testing "a superset of the top addresses under which something is selected"
      (is (= #{:x :y} (sel/top-addresses (sel/union x (sel/select [:y 0])))))
      (is (= :all (sel/top-addresses (sel/union x (sel/complement (sel/select :y))))))
      (is (= #{:x :y} (sel/top-addresses (sel/intersection (sel/select :x :y) (sel/complement x)))))
      (is (= :all (sel/top-addresses (sel/intersection (sel/complement x) (sel/complement (sel/select :y)))))))))

(deftest unions-and-intersections-are-loud
  (doseq [[why f] [["a union of a non-selection" #(sel/union (sel/select :x) :y)]
                   ["an intersection of a non-selection" #(sel/intersection nil)]
                   ["nesting inside a union" #(sel/set-subselection (sel/union (sel/select :x) (sel/complement (sel/select :y))) [:x :z] sel/all)]]]
    (is (thrown? clojure.lang.ExceptionInfo (f)) why))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"^union takes selections, not :y$"
                        (sel/union (sel/select :x) :y)))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"goes inside a union,"
                        (sel/set-subselection (sel/union (sel/select :x) (sel/complement (sel/select :y))) [:x :z] sel/all)))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"goes inside an intersection,"
                        (sel/set-subselection (sel/intersection (sel/select :x) (sel/complement (sel/select :y))) [:x :z] sel/all))))

;; ---------------------------------------------------------------------------
;; Printing
;; ---------------------------------------------------------------------------

(deftest selections-print-as-their-trees
  ;; as choice maps print (#gen/choicemap {:x 1.0}), not as records
  ;; (#gen.selection.Hierarchical{:m {...}})
  (let [s (sel/select :x [:y 0])]
    (is (= "#gen/selection {:x all, :y {0 all}}" (pr-str s)))
    (is (= ["#gen/selection all" "#gen/selection none"] (map pr-str [sel/all sel/none])))
    (is (= "#gen/selection (complement {:x all, :y {0 all}})" (pr-str (sel/complement s))))
    (is (= "#gen/selection (union {:z all} (complement {:x all, :y {0 all}}))"
           (pr-str (sel/union (sel/select :z) (sel/complement s)))))
    (is (= "#gen/selection (intersection {:z all} (complement {:x all, :y {0 all}}))"
           (pr-str (sel/intersection (sel/select :z) (sel/complement s)))))
    (is (= "#gen/selection {:params (complement {:a all})}"
           (pr-str (sel/hierarchical :params (sel/complement (sel/select :a))))))
    (is (= "[#gen/selection none {:s #gen/selection {:x all}}]" (pr-str [sel/none {:s (sel/select :x)}]))
        "inside other data")
    (let [custom (reify sel/ISelection (-selected? [_ _] false) (-subselection [this _] this))]
      (is (re-matches #"#gen/selection \(union \{:z all\} #object\[.*\]\)" (pr-str (sel/union (sel/select :z) custom)))
          "another kind of selection prints as it does"))))
