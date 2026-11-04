(ns gen.codegen.process
  "Tools hosted in a JVM of their own: the process that evaluates an LLM's
   code (gen.codegen.session) apart from the one that runs the LLM, so a
   runaway evaluation, a crash or code that does harm costs that process,
   which is restarted, and never the LLM's state.

   The host's side:

     (serve handler)          its main loop: prints {:ready true}, then reads
     (serve handler opts)     requests from stdin, an EDN map a line, and
                              writes each response to stdout, an EDN map a
                              line, until stdin closes

   handler takes a request and returns its response, a map (another value
   v is {:value v}). The response gets the request's :id, and :out, what the
   handler printed (to *out* or *err*; never the protocol), cut to
   :out-limit characters (2000). A handler that throws answers {:error
   :text \"error: …\"}, with :tainted? true when it was a
   gen.codegen.session/within timeout: its thread may still run, and the
   caller should restart the host. opts: :timeout-ms, a limit on every
   request (none: the handler sets its own).

   The caller's side:

     (start command)          → a host: command (a vector of strings) started,
     (start command opts)       returned when it prints {:ready true}; opts:
                                :log, a file its stderr is appended to (else
                                inherited)
     (call host request)      → request's response (an :id is added), within
     (call host request ms)     ms when given
     (stop host)              → stdin closed, the process waited for (5 s),
                                then killed with its descendants

   call restarts the host when the response is :tainted?, when none comes
   within ms, or when the host dies; the response then says :restarted true
   (and what the host held, its sessions, is gone). Calls on one host are
   serialized.

     ;; the host: clojure -M:… -m my.tools
     (defn -main [& _] (process/serve handle) (shutdown-agents) (System/exit 0))
     ;; the caller
     (def h (process/start [\"clojure\" \"-M:…\" \"-m\" \"my.tools\"] {:log \"tools.log\"}))
     (process/call h {:op :eval :code \"(+ 1 2)\"} 60000)"
  {:api :experimental}
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [gen.codegen.session :as session]))

(set! *warn-on-reflection* true)

(defn- causes [^Throwable e]
  (->> (take-while some? (iterate ex-cause e))
       (map #(or (ex-message %) (str (class %))))
       distinct
       (str/join " Caused by: ")))

(defn- clip [s n] (if (> (count s) n) (subs s 0 n) s))

(defn serve
  "The host's main loop. See the namespace's docstring."
  ([handler] (serve handler nil))
  ([handler {:keys [timeout-ms out-limit] :or {out-limit 2000}}]
   (let [out *out*
         respond (fn [m] (binding [*out* out] (println (pr-str m)) (flush)))]
     (respond {:ready true})
     (loop []
       (when-let [line (read-line)]
         (let [request (try (edn/read-string line) (catch Exception _ nil))
               captured (java.io.StringWriter.)
               response (binding [*out* captured *err* captured]
                          (if (map? request)
                            (try (let [r (if timeout-ms (session/within timeout-ms #(handler request)) (handler request))]
                                   (if (map? r) r {:value r}))
                                 (catch Throwable e
                                   {:error (causes e) :text (str "error: " (causes e))
                                    :tainted? (boolean (::session/timeout (ex-data e)))}))
                            {:error (str "not an EDN map: " (clip line 200)) :text "error: not an EDN map"}))]
           (respond (assoc response :id (when (map? request) (:id request)) :out (clip (str captured) out-limit)))
           (recur)))))))

(defn- kill! [^Process p]
  (doseq [^java.lang.ProcessHandle h (iterator-seq (.iterator (.descendants p)))] (.destroyForcibly h))
  (.destroyForcibly p))

(defn- launch [command log]
  (let [pb (doto (ProcessBuilder. ^java.util.List (vec command))
             (.redirectError (if log
                               (java.lang.ProcessBuilder$Redirect/appendTo (io/file log))
                               java.lang.ProcessBuilder$Redirect/INHERIT)))
        p (.start pb)
        in (io/reader (.getInputStream p))
        out (io/writer (.getOutputStream p))]
    (loop []
      (let [l (.readLine ^java.io.BufferedReader in)]
        (cond (nil? l) (do (kill! p) (throw (ex-info "the host exited before it was ready" {:command command :log log})))
              (:ready (try (edn/read-string l) (catch Exception _ nil))) {:process p :in in :out out}
              :else (recur))))))

(defn start
  "A host, started and ready. See the namespace's docstring."
  ([command] (start command nil))
  ([command {:keys [log]}]
   (atom (assoc (launch command log) :command command :log log))))

(defn- restart! [host]
  (let [{:keys [process command log]} @host]
    (kill! process)
    (swap! host merge (launch command log))))

(defn call
  "request's response from host. See the namespace's docstring."
  ([host request] (call host request nil))
  ([host request timeout-ms]
   (locking host
     (let [id (str (random-uuid))
           {:keys [in out]} @host]
       (try (.write ^java.io.Writer out (str (pr-str (assoc request :id id)) "\n"))
            (.flush ^java.io.Writer out)
            (catch java.io.IOException _ nil))
       (let [f (future (loop []
                         (when-let [l (try (.readLine ^java.io.BufferedReader in) (catch java.io.IOException _ nil))]
                           (let [m (try (edn/read-string l) (catch Exception _ nil))]
                             (if (and (map? m) (= id (:id m))) m (recur))))))
             r (if timeout-ms (deref f timeout-ms ::timeout) @f)]
         (cond (= r ::timeout) (do (future-cancel f) (restart! host)
                                   {:error "timed out" :text "error: timed out" :tainted? true :restarted true})
               (nil? r) (do (restart! host) {:error "the host exited" :text "error: the host exited" :restarted true})
               (:tainted? r) (do (restart! host) (assoc (dissoc r :id) :restarted true))
               :else (dissoc r :id)))))))

(defn stop
  "Stops host. See the namespace's docstring."
  [host]
  (let [{:keys [^Process process out]} @host]
    (try (.close ^java.io.Writer out) (catch java.io.IOException _ nil))
    (when-not (.waitFor process 5 java.util.concurrent.TimeUnit/SECONDS)
      (kill! process))
    nil))
