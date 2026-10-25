(ns gen.api-test
  "The API's tiers and the stable surface (gen.api): every namespace of
   src/ and ext/ carries a tier, and the stable surface is the committed
   snapshot (test/fixtures/api-snapshot.edn). And the checks can fail: a
   missing or unknown tier, an added, removed, deprecated or changed var,
   and a namespace that leaves the stable tier each fail them, a removal
   without a deprecation called out apart."
  (:require [clojure.edn :as edn]
            [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [gen.api :as api]
            [gen.array]
            [gen.diff]
            [gen.dynamic]
            [gen.gfi]
            [gen.selection]
            [gen.trace]))

(deftest every-namespace-has-a-tier
  (let [infos (api/namespaces)
        problems (api/tier-problems infos)]
    (is (< 60 (count infos)) "the namespaces of src/ and ext/ are found")
    (is (some #(= 'gen.dist.commons-math (:ns %)) infos) "ext/'s too")
    (is (empty? problems) (str/join "\n" problems))))

(deftest the-stable-surface-is-the-snapshot
  (let [d (api/diff (api/read-snapshot) (api/surface))]
    (is (empty? (:removed d)) (api/report (select-keys d [:removed])))
    (is (api/same? (dissoc d :removed)) (api/report (dissoc d :removed)))))

(defn- info [form]
  (let [i (api/ns-info form)]
    (assoc i :file "src/a/b.clj" :tier (get-in i [:attrs :api]))))

(deftest tier-marks-can-fail
  (is (empty? (api/tier-problems [(info '(ns a.b "Doc." {:api :stable}))
                                  (info '(ns a.b {:api :internal} (:require x)))
                                  (info '(ns a.b "Doc." {:api :stable :implementers "a note"}))])))
  (testing "a namespace without a tier"
    (is (= 1 (count (api/tier-problems [(info '(ns a.b "Doc." (:require x)))]))))
    (is (= 1 (count (api/tier-problems [(info '(ns a.b))])))))
  (testing "a tier outside the set"
    (is (= 1 (count (api/tier-problems [(info '(ns a.b "Doc." {:api :public}))])))))
  (testing "an :implementers note on a namespace not stable, or not a string"
    (is (= 1 (count (api/tier-problems [(info '(ns a.b {:api :experimental :implementers "a note"}))]))))
    (is (= 1 (count (api/tier-problems [(info '(ns a.b {:api :stable :implementers true}))])))))
  (testing "a file that does not start with an ns form"
    (is (= 1 (count (api/tier-problems [{:ns nil :file "src/a/b.clj"}])))))
  (testing "a real namespace's mark removed"
    (is (= 1 (count (api/tier-problems [(assoc (first (api/namespaces)) :tier nil)]))))))

(deftest entries-record-what-the-promise-covers
  (testing "a function's arglists, without type hints or a macro's generated names"
    (is (= {:arglists '([trace selection])} (api/entry #'gen.gfi/project)))
    (is (= {:arglists '([m])} (api/entry #'gen.trace/map->Trace)) "map->'s m__…__auto__"))
  (testing "a kind where arglists do not apply, or beside them"
    (is (= :macro (:kind (api/entry #'gen.dynamic/gen))))
    (is (= {:kind :protocol :methods '[-assess -generate -project -propose -regenerate -simulate -update]}
           (api/entry #'gen.gfi/IGenerativeFunction)))
    (is (= {:kind :multimethod} (api/entry #'gen.diff/propagate)))
    (is (= {:kind :var} (api/entry #'gen.selection/all)))
    (is (= {:kind :var :dynamic true} (api/entry #'gen.array/*backend*))))
  (testing "deprecation"
    (let [v (intern (create-ns 'gen.api-test.scratch) (with-meta 'old {:deprecated "1.1" :arglists '([x])}) identity)]
      (is (= {:arglists '([x]) :deprecated "1.1"} (api/entry v)))))
  (testing "^:no-doc and private vars are not on the surface"
    (is (not (api/published? #'gen.gfi/realize-limit)))
    (is (not (api/published? #'gen.api/as-edn)))
    (is (api/published? #'gen.gfi/simulate))))

(deftest the-snapshot-check-can-fail
  (let [s (api/surface '[gen.selection])
        d #(api/diff s %)]
    (is (api/same? (d s)))
    (is (= s (edn/read-string (api/snapshot-text s))) "the file's text reads back as the surface")
    (testing "an added var"
      (let [r (d (assoc-in s '[gen.selection extra] {:arglists '([x])}))]
        (is (= '[gen.selection/extra] (:added r)))
        (is (str/includes? (api/report r) api/write-command) "the failure names the command that writes the snapshot")))
    (testing "a removed var, called out apart from additions"
      (let [r (d (update s 'gen.selection dissoc 'select))]
        (is (= '[gen.selection/select] (:removed r)))
        (is (every? empty? (vals (dissoc r :removed))))
        (is (str/includes? (api/report r) "without a deprecation first"))))
    (testing "a var deprecated"
      (let [deprecated (assoc-in s '[gen.selection select :deprecated] "1.1")]
        (is (= '[gen.selection/select] (:deprecated (d deprecated))))
        (is (empty? (:changed (d deprecated))))
        (is (= [['gen.selection/select "1.1"]] (api/deprecated deprecated)))
        (testing "and then removed: a difference still, but no removal without deprecation"
          (let [r (api/diff deprecated (update s 'gen.selection dissoc 'select))]
            (is (= '[gen.selection/select] (:removed-deprecated r)))
            (is (empty? (:removed r)))))))
    (testing "changed arglists"
      (let [r (d (assoc-in s '[gen.selection select :arglists] '([a])))]
        (is (= '[gen.selection/select] (map :var (:changed r))))
        (is (str/includes? (api/report r) "([a])"))))
    (testing "a namespace that leaves the stable tier"
      (let [r (api/diff s {})]
        (is (= '[gen.selection] (:namespaces-removed r)))
        (is (= (count (s 'gen.selection)) (count (:removed r))))))))
