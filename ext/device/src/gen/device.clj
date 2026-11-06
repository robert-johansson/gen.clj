(ns gen.device
  "The device lowerings behind one call (ext/device, experimental): a
   staged program (gen.array.stage) compiled for :jvm (generated JVM code)
   or :metal (one Metal kernel through MLX), and run once on a key or n
   times, run i on (rng/split root n)[i]. gen.inference.device calls it
   for kernel/run's and importance sampling's :device option (DESIGN §9).

   (compile-program device program) returns {:run-one (fn [key-words
   inputs]) :run-many (fn [n root-words inputs])}, which give the
   program's outputs (in its :outputs order) on the host, a run's [shape]
   or n runs' [n shape], each {:data :shape :dtype}: :data a flat Java
   array in row-major order (float[], double[], int[], long[] or
   boolean[]; :jvm's, and :metal's copied from the GPU when first forced,
   a delay), :dtype the gen.array dtype it stands for, so a
   caller makes arrays of the parts it keeps and no others. inputs are the
   program's data inputs after its key, in order. A device or form a lowering lacks
   throws (the caller falls back to the handler).

   The devices are :jvm and :metal. The CUDA C lowering (gen.device.cuda)
   is parked (DESIGN §9, 1.2.1's owner calls, genclj-z1gj): no consumer,
   not a device here, its tests that need a CUDA device skipped by name."
  {:api :experimental}
  (:require [gen.array :as array]
            [gen.device.jvm :as jvm]))

(set! *warn-on-reflection* true)

(defn- array-dtype
  "The gen.array dtype a lowering's output of dtype stands for: unsigned
   words as int64."
  [dtype]
  (case dtype (:uint32 :int64) :int64 dtype))

(defn- check-size!
  "Loud when n runs' outputs would pass a Java array's length (2^31 - 1
   elements), before anything is allocated."
  [n outs]
  (doseq [{:keys [shape]} outs
          :let [cells (* (long n) (long (reduce * 1 shape)))]]
    (when (> cells Integer/MAX_VALUE)
      (throw (ex-info (str "The outputs of " n " runs would hold " cells " elements, past a Java array's 2^31 - 1")
                      {:reason :outputs-too-large :runs n :shape shape})))))

(defmulti compile-program
  "program compiled for device: {:run-one :run-many}."
  (fn [device _] device))

(defmethod compile-program :jvm [_ program]
  (let [k (jvm/compile program)
        outs (let [eo (:error-output k)] (if eo (subvec (:outputs k) 0 eo) (:outputs k)))
        java-inputs (fn [inputs] (mapv (fn [x {:keys [dtype]}] (jvm/input-array x dtype)) inputs (:inputs k)))]
    {:run-one (fn [key-words inputs]
                (mapv (fn [a {:keys [shape dtype]}] {:data a :shape shape :dtype (array-dtype dtype)})
                      ((:run k) key-words (java-inputs inputs)) outs))
     :run-many (fn [n root-words inputs]
                 (check-size! n outs)
                 (mapv (fn [a {:keys [shape dtype]}] {:data a :shape (into [n] shape) :dtype (array-dtype dtype)})
                       (:outputs (jvm/run-many k n root-words {:inputs (java-inputs inputs)})) outs))}))

(defmethod compile-program :metal [_ program]
  (let [compile (requiring-resolve 'gen.device.metal/compile)
        run (requiring-resolve 'gen.device.metal/run)
        host-array (requiring-resolve 'gen.device.metal/host-array)
        k (compile program)]
    {:run-one (fn [_ _] (throw (ex-info "Device :metal runs many particles (one GPU thread a particle); a single chain runs on :jvm"
                                        {:reason :unsupported :device :metal})))
     :run-many (fn [n root-words inputs]
                 (check-size! n (:outputs k))
                 (let [;; MLX hands a kernel a 0-d input as a scalar, not a buffer:
                       ;; the generated code reads every input as one (in[0]), so [1]
                       as-input (fn [x] (let [a (if (array/backend-array? x) x (array/array x))]
                                          (if (empty? (array/shape a)) (array/reshape a [1]) a)))
                       xs (run k n root-words {:inputs (mapv as-input inputs)})]
                   (array/eval! xs)
                   ;; each read into a Java array only when a caller asks for its data
                   (mapv (fn [x] {:data (delay (host-array x)) :shape (vec (array/shape x)) :dtype (array-dtype (array/dtype x))}) xs)))}))

(defmethod compile-program :default [device _]
  (throw (ex-info (str "No device lowering " (pr-str device) " (the lowerings are :jvm and :metal; CUDA C,"
                       " gen.device.cuda, is parked)")
                  {:reason :unknown-device :device device})))
