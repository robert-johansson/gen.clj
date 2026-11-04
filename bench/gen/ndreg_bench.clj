(ns gen.ndreg-bench
  "The parity suite's ndreg_is spec on gen.clj (bean genclj-4zmv): importance
   sampling, 1000 particles from the prior, on D-dimensional Bayesian linear
   regression with M observations, w ~ N(0, 10²) [D] and y ~ N(X·w, 1) [M],
   one vector site each, for the cells (D, M) in {10, 100, 1000} × {1000,
   10000}. The data are the spec's frozen files (the RTX-6000 machine's
   ~/mlx/genjax-parity/data, X [M D], y [M] as float32 safetensors), read
   from -Dgen.ndreg.dir.

   A timed run is batched importance-sampling's (gen.batched) from a key to
   the N weights on the host; the first of three warm-ups is reported apart,
   then up to 20 timed runs within gen.parity.budget-s (at least 3). Prints
   one EDN map per cell, with the log mean weight and the exact log Z.

   Run: clojure [-J-Dgen.backend=mlx] -J-Dgen.ndreg.dir=<dir> -M:bench -m gen.ndreg-bench [D,M …]"
  (:require [clojure.string :as str]
            [gen.array :as array]
            [gen.array.protocols :as p]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dynamic :refer [gen]]
            [gen.inference.importance :as is]
            [gen.random :as rng])
  (:import [java.nio ByteBuffer ByteOrder]
           [java.nio.file Files Path]))

(def cells [[10 1000] [100 1000] [1000 1000] [10 10000] [100 10000] [1000 10000]])

(def n-particles 1000)

(defn read-safetensors
  "A safetensors file's float32 tensors: {name {:shape :data float[]}}. The
   header is JSON; its entries are read by pattern, which serves the spec's
   flat headers (dtype, shape, data_offsets)."
  [path]
  (let [bytes (Files/readAllBytes (Path/of ^String path (make-array String 0)))
        bb (.order (ByteBuffer/wrap bytes) ByteOrder/LITTLE_ENDIAN)
        n (.getLong bb 0)
        header (String. bytes 8 (int n) "UTF-8")]
    (into {}
          (for [[_ name dtype shape from to]
                (re-seq #"\"(\w+)\":\{\"dtype\":\"(\w+)\",\"shape\":\[([\d,]*)\],\"data_offsets\":\[(\d+),(\d+)\]\}" header)]
            (do (when-not (= dtype "F32")
                  (throw (ex-info (str "ndreg: " name " is " dtype ", not F32") {:name name :dtype dtype})))
                (let [from (+ 8 n (Long/parseLong from))
                      k (quot (- (+ 8 n (Long/parseLong to)) from) 4)
                      out (float-array k)]
                  (.get (.asFloatBuffer (.position (.order (.duplicate bb) ByteOrder/LITTLE_ENDIAN) (int from))) out)
                  [name {:shape (mapv #(Long/parseLong %) (remove str/blank? (str/split shape #",")))
                         :data out}]))))))

(defn exact-log-z
  "The cell's exact log Z, from its .ref.json sidecar."
  [path]
  (some->> (slurp path) (re-find #"\"exact_logZ\":\s*(-?[\d.eE+-]+)") second Double/parseDouble))

(def model
  (gen [X]
    (let [w (trace :w (d/normal (array/zeros [(second (array/shape X))]) 10.0))]
      (trace :y (d/normal (array/matmul X w) 1.0)))))

(defn- doubles-of ^doubles [^floats xs]
  (let [out (double-array (alength xs))]
    (dotimes [i (alength xs)] (aset out i (double (aget xs i))))
    out))

(defn cell-data
  "[X y exact-log-z] of the cell (D, M), X and y arrays of the default
   backend."
  [dir D M]
  (let [stem (str dir "/ndreg_D" D "_M" M)
        {:strs [X y]} (read-safetensors (str stem ".safetensors"))]
    [(array/reshape (array/array (doubles-of (:data X)) {:dtype :float32}) (:shape X))
     (array/array (doubles-of (:data y)) {:dtype :float32})
     (exact-log-z (str stem ".ref.json"))]))

(defn- ms [^long t0] (/ (- (System/nanoTime) t0) 1e6))

(defn- quantile [sorted q]
  (nth sorted (min (dec (count sorted)) (long (Math/floor (* q (count sorted)))))))

(defn -main [& args]
  (let [dir (or (System/getProperty "gen.ndreg.dir") (throw (ex-info "-Dgen.ndreg.dir=<the spec's data>" {})))
        budget-ms (* 1000.0 (Double/parseDouble (System/getProperty "gen.parity.budget-s" "30")))
        chosen (if (seq args) (mapv (fn [a] (mapv #(Long/parseLong %) (str/split a #","))) args) cells)]
    (doseq [[D M] chosen]
      (let [[X y log-z] (cell-data dir D M)
            obs (cm/choicemap :y y)
            unit (fn [k] (is/importance-sampling model [X] obs n-particles {:key k :batched? :required}))
            t0 (System/nanoTime)
            _ (unit (rng/key 1000))
            warmup (ms t0)]
        (when (< warmup 5000.0) (unit (rng/key 1001)) (unit (rng/key 1002)))
        (let [[times r] (loop [i 0 times [] spent 0.0 r nil]
                          (if (or (< i 3) (and (< i 20) (< spent budget-ms)))
                            (let [t (System/nanoTime) r (unit (rng/key i)) dt (ms t)]
                              (recur (inc i) (conj times dt) (+ spent dt) r))
                            [times r]))
              s (vec (sort times))]
          (prn {:spec :ndreg_is :D D :M M :N n-particles :backend (name (p/-name array/*backend*))
                :warmup-ms warmup :median-ms (quantile s 0.5) :p10-ms (quantile s 0.1) :p90-ms (quantile s 0.9)
                :runs (count s) :log-ml-estimate (:log-ml-estimate r) :exact-log-z log-z})
          (flush)))))
  (shutdown-agents))
