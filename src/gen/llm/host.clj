(ns gen.llm.host
  "The LLM layer's host work off the calling thread: pools of daemon
   platform threads, and an ordered map over one. The llama.cpp engine
   normalizes a decode's rows on one (gen.llm.llama), token SMC draws its
   particles' tokens on another (gen.llm.smc). Each keeps a pool of its
   own: one pool shared by both could deadlock, a task waiting on tasks
   queued behind it.

     (pool name n)          → a fixed pool of n daemon threads, name-0, name-1, …
     (default-threads)      → the default size: the system property
                              gen.llm.host-threads, or half the processors up to 4
     (in-order pool f xs)   → (mapv f xs), the calls on pool"
  (:import [java.util Collection]
           [java.util.concurrent ExecutionException ExecutorService Executors Future]))

(set! *warn-on-reflection* true)

(defn pool
  "A fixed pool of n daemon platform threads, named name-0, name-1, …;
   shut it down when done, or let the JVM's exit end it."
  ^ExecutorService [^String name n]
  (when-not (pos-int? n)
    (throw (ex-info (str "host/pool: a pool has a positive number of threads, not " (pr-str n))
                    {:reason :invalid-args :n n})))
  (Executors/newFixedThreadPool (int n) (-> (Thread/ofPlatform) (.name name 0) (.daemon true) .factory)))

(defn- threads
  "The default size from the property's value (a string, nil when it is
   not set): a positive integer, loud otherwise; without it, half of
   processors, up to 4."
  ^long [property ^long processors]
  (if (nil? property)
    (max 1 (min 4 (quot processors 2)))
    (let [n (parse-long property)]
      (when-not (and n (pos? (long n)))
        (throw (ex-info (str "gen.llm.host-threads is a positive integer, not " (pr-str property))
                        {:reason :invalid-option :host-threads property})))
      (long n))))

(def ^:private default
  (delay (threads (System/getProperty "gen.llm.host-threads") (.availableProcessors (Runtime/getRuntime)))))

(defn default-threads
  "The default size of a host pool, read once: the system property
   gen.llm.host-threads (a positive integer; anything else is loud) or,
   without it, half the available processors, up to 4."
  ^long []
  @default)

(defn in-order
  "(f x) for each x of xs, in order, as a vector: on pool when there is
   one (an ExecutorService; false or nil: none) and xs has several, each
   call under the calling thread's bindings; else on the calling thread.
   What a call threw is rethrown as itself, once every call has ended."
  [pool f xs]
  (if (and pool (next xs))
    (let [f (bound-fn* f)]
      (mapv (fn [^Future fut]
              (try (.get fut)
                   (catch ExecutionException x (throw (.getCause x)))))
            (.invokeAll ^ExecutorService pool ^Collection (mapv (fn [x] (fn [] (f x))) xs))))
    (mapv f xs)))
