(ns gen.codegen.process-test
  "Tools in a JVM of their own (gen.codegen.process), against a host
   (gen.codegen.echo-host) started from this JVM's classpath: a request
   answered with its :id stripped, a bare value as {:value}; what the
   handler printed in :out, never the protocol; a throw answered as an
   error; a response past the host's own limit tainted, and the host
   restarted; a call past its timeout, and a host that dies, restarted
   too; stop ending the process."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.codegen.process :as process]))

(defn- command []
  [(str (System/getProperty "java.home") "/bin/java") "-cp" (System/getProperty "java.class.path")
   "clojure.main" "-m" "gen.codegen.echo-host"])

(deftest ^:slow a-host
  (let [h (process/start (command))
        pid #(:pid (process/call h {:op :echo}))
        first-pid (pid)]
    (try
      (testing "answers"
        (is (= {:value [1 "two"] :out ""} (dissoc (process/call h {:op :echo :value [1 "two"]}) :pid)))
        (is (= {:value 42 :out ""} (process/call h {:op :bare}))))
      (testing "printing goes to :out"
        (is (= {:value :printed :out "printed\nto err\n"} (process/call h {:op :print}))))
      (testing "a throw is an error, the host kept"
        (is (= {:error "it broke" :text "error: it broke" :tainted? false :out ""} (process/call h {:op :throw})))
        (is (= first-pid (pid))))
      (testing "past the host's limit: tainted, restarted"
        (let [r (process/call h {:op :sleep :ms 3000})]
          (is (:tainted? r))
          (is (:restarted r)))
        (is (not= first-pid (pid))))
      (testing "past the call's timeout: restarted"
        (let [before (pid)
              r (process/call h {:op :sleep :ms 800} 100)]
          (is (= {:error "timed out" :tainted? true :restarted true} (select-keys r [:error :tainted? :restarted])))
          (is (not= before (pid)))))
      (testing "a host that dies: restarted"
        (let [before (pid)]
          (is (= {:error "the host exited" :restarted true} (select-keys (process/call h {:op :exit}) [:error :restarted])))
          (is (not= before (pid)))
          (is (= :x (:value (process/call h {:op :echo :value :x}))))))
      (finally
        (let [^Process p (:process @h)]
          (process/stop h)
          (is (not (.isAlive p)) "stopped"))))))
