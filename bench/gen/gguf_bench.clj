(ns gen.gguf-bench
  "The host reader's speed (gen.array.gguf/host-tensor): a GGUF file's
   tensors decoded and repacked on the host, MB/s of the file's bytes by
   tensor type, as MLX's backend reads a file holding K-quants. The file
   is read once first, so that its pages are in memory and the decoding
   is measured, not the disk.

     clojure -M:bench -m gen.gguf-bench <file.gguf> [MB of each type, default 2000]"
  (:require [gen.array.gguf :as gguf]))

(defn- seconds [f]
  (let [t0 (System/nanoTime)]
    (f)
    (/ (- (System/nanoTime) t0) 1e9)))

(defn -main [path & [mb]]
  (let [limit (* 1e6 (double (or (some-> mb parse-long) 2000)))
        infos (gguf/check-tensors (gguf/headers path))
        by-type (group-by :type infos)]
    (println "file" path)
    (doseq [[t ts] (sort-by key by-type)
            :let [;; the type's tensors, up to the limit's bytes
                  picked (reduce (fn [acc info]
                                   (if (< (reduce + 0 (map :bytes acc)) limit) (conj acc info) (reduced acc)))
                                 [] ts)
                  bytes (reduce + 0 (map :bytes picked))]]
      ;; warm: the pages read, the JIT compiled
      (dotimes [_ 2] (doseq [info picked] (gguf/host-tensor path info)))
      (let [s (seconds #(doseq [info picked] (gguf/host-tensor path info)))]
        (println (format "  %-5s %3d of %3d tensors, %8.1f MB: %6.3f s, %7.0f MB/s"
                         (name t) (count picked) (count ts) (/ bytes 1e6) s (/ bytes 1e6 s)))))
    (shutdown-agents)))
