(ns gen.array.memory
  "Memory that native engines hold beside the array backends. On Apple
   silicon llama.cpp's weights and caches and MLX's arrays come out of one
   unified memory, and neither knows of the other: an engine reserves what
   it holds here (gen.llm.llama does, from llama.cpp's own report of its
   buffers) and releases it when it closes, and the MLX backend's budget
   (gen.backend.mlx/budget) leaves the reservations out.

   Plain numbers: nothing here allocates or frees, and requiring it loads
   no native library.")

(set! *warn-on-reflection* true)

;; {:total bytes, :held {owner {:bytes :what}}}, the total kept with the
;; entries so the budget reads it in constant time at every evaluation
(defonce ^:private reservations (atom {:total 0 :held {}}))

(defn- total [held]
  (reduce-kv (fn [^long n _ {:keys [^long bytes]}] (+ n bytes)) 0 held))

(defn reserve!
  "Reserve bytes (a non-negative integer) for owner, any value compared by
   =, what describing it; a second reservation for the same owner replaces
   the first. Returns owner."
  [owner bytes what]
  (when-not (and (int? bytes) (not (neg? bytes)))
    (throw (ex-info (str "reserve!: bytes is a non-negative integer, not " (pr-str bytes))
                    {:reason :invalid-reservation :bytes bytes})))
  (swap! reservations (fn [{:keys [held]}]
                        (let [held (assoc held owner {:bytes (long bytes) :what what})]
                          {:total (total held) :held held})))
  owner)

(defn release!
  "Release owner's reservation, if it holds one."
  [owner]
  (swap! reservations (fn [{:keys [held]}]
                        (let [held (dissoc held owner)]
                          {:total (total held) :held held})))
  nil)

(defn reserved
  "The bytes reserved by every owner together."
  ^long []
  (:total @reservations))

(defn reservations-held
  "Every reservation: {owner {:bytes :what}}."
  []
  (:held @reservations))
