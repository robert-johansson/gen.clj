(ns gen.device.module-test
  "ext/device as a dependent takes it (genclj-ri2d): its deps.edn, which a
   git dependency reaches with :deps/root \"ext/device\", brings gen.clj
   from the same checkout and nothing else, and its code needs no test or
   bench code. Planted bugs (2026-11-06): a require of gen.parity-models
   (test/) in gen.device.chain fails the second; a dependency added to the
   module's deps.edn fails the first."
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [clojure.test :refer [deftest is]]))

(defn- ns-file [dir sym]
  (io/file dir (str (-> (name sym) (str/replace "." "/") (str/replace "-" "_")) ".clj")))

(defn- requires
  "The namespaces a source file's ns form requires."
  [^java.io.File f]
  (let [[_ _ & clauses] (with-open [r (java.io.PushbackReader. (io/reader f))] (read r))]
    (for [clause clauses :when (and (seq? clause) (= :require (first clause)))
          spec (rest clause)]
      (if (vector? spec) (first spec) spec))))

(deftest the-module-s-deps
  (let [module (edn/read-string (slurp "ext/device/deps.edn"))
        root (edn/read-string (slurp "deps.edn"))]
    (is (= ["src"] (:paths module)))
    (is (= {'gen.clj/gen.clj {:local/root "../.."}} (:deps module)) "gen.clj from the same checkout, nothing else")
    (is (some #{"ext/device/src"} (get-in root [:aliases :device :extra-paths])) "the :device alias puts it on the classpath")))

(deftest the-module-needs-no-test-code
  (let [sources (filter #(str/ends-with? (.getName ^java.io.File %) ".clj") (file-seq (io/file "ext/device/src")))
        required (distinct (mapcat requires sources))
        found? #(or (str/starts-with? (name %) "clojure.")
                    (.exists ^java.io.File (ns-file "src" %))
                    (.exists ^java.io.File (ns-file "ext/device/src" %)))]
    (is (seq sources))
    (is (seq required))
    (is (empty? (remove found? required)))))
