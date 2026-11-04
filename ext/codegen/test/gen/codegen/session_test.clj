(ns gen.codegen.session-test
  "A REPL session for an LLM's code (gen.codegen.session): definitions kept
   across evals, the program the forms that evaluated; an error explained
   by :explain, given the throwable, the namespace and the forms, the forms
   before it kept; code that does not read refused, or repaired and said
   to be; :vet refusing before any form runs; :show; what the code
   printed; a timeout reported, not waited for; #= refused; the eval tool
   on a session or a state holding one; close! removing the namespace."
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [gen.codegen.session :as session]))

(defn- with-session [f]
  (let [s (session/open {:prelude '[(:refer-clojure :exclude [inc])] :vars {'data [1 2 3]}})]
    (try (f s) (finally (session/close! s)))))

(deftest definitions-kept-and-the-program
  (with-session
    (fn [s]
      (is (= "=> 6" (:text (session/eval! s "(def total (reduce + data)) total"))) "the vars, interned")
      (is (= "=> 7" (:text (session/eval! s "(def inc (fn [x] (+ x 1))) (inc total)"))) "the prelude's exclusions")
      (is (= 6 (session/lookup s 'total)))
      (is (nil? (session/lookup s "nope")))
      (is (= "(def total (reduce + data))\n\ntotal\n\n(def inc (fn [x] (+ x 1)))\n\n(inc total)" (session/program s))))))

(deftest errors-explained-and-what-came-before-kept
  (with-session
    (fn [s]
      (let [seen (atom nil)
            r (session/eval! s "(def a 1) (undefined-thing a)"
                             {:explain (fn [e ns forms] (reset! seen [(class e) (ns-name ns) forms]) "explained")})]
        (is (= {:text "error: explained" :ok? false} (select-keys r [:text :ok?])))
        (is (= [clojure.lang.Compiler$CompilerException (:ns s) '[(def a 1) (undefined-thing a)]] @seen)
            ":explain gets the throwable, the session's namespace and the forms")
        (is (= 1 (session/lookup s 'a)) "the form before the error stays")
        (is (= "(def a 1)" (session/program s)) "the failed form is not in the program"))
      (is (re-find #"^error: .*Unable to resolve symbol: nope" (:text (session/eval! s "(nope)"))) "the default: the causes"))))

(deftest reading-repair-and-vetting
  (with-session
    (fn [s]
      (is (str/starts-with? (:text (session/eval! s "(def b (+ 1 2)")) "error: the code does not read: "))
      (let [r (session/eval! s "(def b (+ 1 2)" {:repair #(str % ")")})]
        (is (:repaired? r))
        (is (str/starts-with? (:text r) "(The delimiters did not balance; they were rebuilt from the indentation.)\n=> #'")))
      (is (str/starts-with? (:text (session/eval! s "#=(+ 1 2)")) "error: the code does not read") "no read-time evaluation")
      (let [r (session/eval! s "(def c 1) (danger)" {:vet (fn [forms] (when (some #(= '(danger) %) forms) "no danger"))})]
        (is (= "error: no danger" (:text r)))
        (is (nil? (session/lookup s 'c)) "vetted before any form runs")))))

(deftest show-printed-and-timeouts
  (with-session
    (fn [s]
      (is (= "=> <3>" (:text (session/eval! s "3" {:show #(str "<" % ">")}))))
      (let [r (session/eval! s "(println \"hi\") 4")]
        (is (= "printed: hi\n\n=> 4" (:text r)))
        (is (= "hi\n" (:printed r))))
      (let [t0 (System/nanoTime)
            r (session/eval! s "(Thread/sleep 5000)" {:timeout-ms 100})]
        (is (:timeout? r))
        (is (str/includes? (:text r) "timed out after 100 ms"))
        (is (< (/ (- (System/nanoTime) t0) 1e6) 2000) "not waited for")))))

(deftest the-eval-tool
  (with-session
    (fn [s]
      (let [{f :fn :keys [params]} (session/eval-tool {:show pr-str})]
        (is (= ["code"] (map :name params)))
        (is (= {:text "=> 3" :ok? true :repaired? false} (f {"code" "(+ 1 2)"} s)))
        (is (= "=> 1" (:text (f {"code" "1"} {:session s}))) "a state holding the session")))))

(deftest closing-removes-the-namespace
  (let [s (session/open)]
    (session/eval! s "(def x 1)")
    (session/close! s)
    (is (nil? (find-ns (:ns s))))))
