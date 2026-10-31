(ns gen.device
  "The device lowerings behind one call (ext/device, experimental): a
   staged program (gen.array.stage) compiled for :jvm (generated JVM code)
   or :metal (one Metal kernel through MLX), and run once on a key or n
   times, run i on (rng/split root n)[i]. gen.inference.device calls it
   for kernel/run's and importance sampling's :device option (DESIGN §9).

   (compile-program device program) returns {:run-one (fn [key-words
   inputs]) :run-many (fn [n root-words inputs])}, each giving the
   program's outputs (in its :outputs order) as gen.array arrays of the
   current backend, a run's [shape] or n runs' [n shape]; inputs are the
   program's data inputs after its key, in order. A device or form a
   lowering lacks throws (the caller falls back to the handler)."
  {:api :experimental}
  (:require [gen.array :as array]
            [gen.device.jvm :as jvm]))

(set! *warn-on-reflection* true)

(defn- java->array
  "A Java array of a lowering's outputs as a gen.array array of shape."
  [a shape dtype]
  (let [v (vec a)
        dtype (case dtype (:uint32 :int64) :int64 dtype)]
    (array/reshape (array/array v {:dtype dtype}) shape)))

(defmulti compile-program
  "program compiled for device: {:run-one :run-many}."
  (fn [device _] device))

(defmethod compile-program :jvm [_ program]
  (let [k (jvm/compile program)
        outs (let [eo (:error-output k)] (if eo (subvec (:outputs k) 0 eo) (:outputs k)))
        java-inputs (fn [inputs] (mapv (fn [x {:keys [dtype]}] (jvm/input-array x dtype)) inputs (:inputs k)))]
    {:run-one (fn [key-words inputs]
                (mapv (fn [a {:keys [shape dtype]}] (java->array a shape dtype))
                      ((:run k) key-words (java-inputs inputs)) outs))
     :run-many (fn [n root-words inputs]
                 (mapv (fn [a {:keys [shape dtype]}] (java->array a (into [n] shape) dtype))
                       (:outputs (jvm/run-many k n root-words {:inputs (java-inputs inputs)})) outs))}))

(defmethod compile-program :metal [_ program]
  (let [compile (requiring-resolve 'gen.device.metal/compile)
        run (requiring-resolve 'gen.device.metal/run)
        k (compile program)]
    {:run-one (fn [_ _] (throw (ex-info "Device :metal runs many particles (one GPU thread a particle); a single chain runs on :jvm"
                                        {:reason :unsupported :device :metal})))
     :run-many (fn [n root-words inputs] (run k n root-words {:inputs (mapv #(if (array/backend-array? %) % (array/array %)) inputs)}))}))

(defmethod compile-program :default [device _]
  (throw (ex-info (str "No device lowering " (pr-str device) " (the lowerings are :jvm and :metal)")
                  {:reason :unknown-device :device device})))
