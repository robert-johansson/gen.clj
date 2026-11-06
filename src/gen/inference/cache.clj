(ns gen.inference.cache
  "The compiled code the inference namespaces keep, one bounded cache:
   gen.inference.mcmc's compiled kernel steps (:compile?) and
   gen.inference.device's compiled programs (:device). An entry pins what
   its code closes over, a model and its data arrays (on MLX, Metal
   buffers), so the cache holds at most limit entries in all, and past it
   the least recently used goes. Keys are Clojure values, each caller's
   its own (tagged with its namespace's keyword). Thread-safe; a value
   two threads make at once is made twice, and one is kept."
  {:api :internal})

(def ^:private limit
  "The most entries the cache holds."
  64)

(defonce ^:private store
  ;; {:entries {k [v last-use]} :clock n}
  (atom {:entries {} :clock 0}))

(defn lookup
  "The value at k, or nil (marked as used)."
  [k]
  (when-let [[v] (get (:entries @store) k)]
    (swap! store (fn [{:keys [entries clock] :as s}]
                   (if (contains? entries k)
                     {:entries (assoc-in entries [k 1] clock) :clock (inc (long clock))}
                     s)))
    v))

(defn store!
  "v at k, the least recently used entry dropped past the limit; v."
  [k v]
  (swap! store (fn [{:keys [entries clock]}]
                 (let [entries (assoc entries k [v clock])]
                   {:entries (if (> (count entries) (long limit))
                               (dissoc entries (key (apply min-key (fn [[_ [_ used]]] used) entries)))
                               entries)
                    :clock (inc (long clock))})))
  v)

(defn cached
  "The value at k, made by (make) and stored when absent."
  [k make]
  (or (lookup k) (store! k (make))))

(defn size
  "How many entries the cache holds (tests)."
  ^long []
  (count (:entries @store)))

(defn clear!
  "Forget every entry."
  []
  (reset! store {:entries {} :clock 0})
  nil)
