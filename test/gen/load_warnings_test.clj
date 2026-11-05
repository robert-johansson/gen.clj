(ns gen.load-warnings-test
  "The house rule that namespaces with *warn-on-reflection* and
   *unchecked-math* :warn-on-boxed load warning-free, as a test: a JVM of
   its own, on this classpath, loads every source namespace (src and each
   ext/*/src on the classpath), and the test fails on any reflection or
   boxed-math warning it prints. A JVM of its own, since this one has
   loaded them already and reloading would redefine protocols under the
   other tests."
  (:require [clojure.java.io :as io]
            [clojure.java.shell :as sh]
            [clojure.string :as str]
            [clojure.test :refer [deftest is]]))

(defn- source-dirs
  "The classpath's source directories: src and the extensions' src."
  []
  (->> (str/split (System/getProperty "java.class.path") (re-pattern java.io.File/pathSeparator))
       (filter #(or (= "src" %) (re-matches #"ext/[^/]+/src" %)))
       (filter #(.isDirectory (io/file %)))))

(defn- namespaces
  "The namespaces of the .clj files under dir, as symbols."
  [dir]
  (let [root (.toPath (io/file dir))]
    (for [^java.io.File f (file-seq (io/file dir))
          :when (str/ends-with? (.getName f) ".clj")]
      (-> (str (.relativize root (.toPath f)))
          (str/replace #"\.clj$" "")
          (str/replace java.io.File/separator ".")
          (str/replace "_" "-")
          symbol))))

(defn- warnings
  "The warning lines a fresh JVM prints loading nses, and its exit code."
  [nses]
  (let [java (str (io/file (System/getProperty "java.home") "bin" "java"))
        {:keys [exit out err]} (sh/sh java "--enable-native-access=ALL-UNNAMED"
                                      "-cp" (System/getProperty "java.class.path")
                                      "clojure.main" "-e" (pr-str `(doseq [n# '~nses] (require n#))))]
    {:exit exit
     :warnings (filter #(re-find #"(?i)reflection warning|boxed math warning" %)
                       (str/split-lines (str out err)))
     :err err}))

(deftest source-namespaces-load-without-warnings
  (let [nses (sort (mapcat namespaces (source-dirs)))
        {:keys [exit warnings err]} (warnings nses)]
    (is (< 50 (count nses)) "the source namespaces are found")
    (is (zero? exit) (str "every source namespace loads: " err))
    (is (empty? warnings) (str/join "\n" warnings))))
