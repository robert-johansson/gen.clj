(ns gen.codegen.report
  "The codegen bake-off's report (genclj-ihc4): a run's results.edn
   (gen.codegen.driver) summarized, per task and in total.

     clojure -M:test -m gen.codegen.report <results.edn> …

   A reply is correct when it passes every hard stage (it reads, is safe,
   evaluates, simulates and takes the data); its gap is its log marginal
   likelihood less the reference's (nats; reliable or not, as graded).
   One shot: how many of the K replies are correct, and the seconds of
   their decoding (together). Repair: the round the chain from the first
   reply became correct at (0: it was), or that it never did. Seconds per
   correct model: every second of decoding the mode spent, over its
   correct models."
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [gen.codegen.driver :as driver]
            [gen.codegen.tasks :as tasks]))

(defn read-results [file]
  (with-open [r (io/reader file)]
    (mapv #(edn/read-string {:default (fn [_ v] v)} %) (remove str/blank? (line-seq r)))))

(defn- reached
  "The first stage a grading failed at, :correct when it passed every hard
   stage."
  [grading]
  (or (some #(when (and (not (:ok %)) (driver/hard-stages (:stage %))) (:stage %)) (:stages grading))
      (when (:error grading) :error)
      :correct))

(defn- correct? [entry] (= :correct (reached (:grading entry))))

(defn- gap [entry] (get-in entry [:grading :log-ml :gap]))

(defn- fmt [x] (if (number? x) (format "%.1f" (double x)) "–"))

(defn- task-row [entries]
  (let [one (filter #(= :one-shot (:mode %)) entries)
        chain (sort-by :round (cons (first (filter #(zero? (:reply %)) one)) (filter #(= :repair (:mode %)) entries)))
        fixed-at (some #(when (correct? %) (:round %)) chain)
        gaps (keep #(when (correct? %) (gap %)) one)]
    {:task (:task (first entries))
     :one-shot (str (count (filter correct? one)) "/" (count one))
     :one-shot-s (some :seconds one)
     :failures (frequencies (remove #{:correct} (map (comp reached :grading) one)))
     :repair (cond (nil? fixed-at) (str "no (" (dec (count chain)) " rounds)") (zero? fixed-at) "first reply" :else (str "round " fixed-at))
     :repair-s (reduce + (keep :seconds (filter #(= :repair (:mode %)) entries)))
     :gap (when (seq gaps) (nth (sort gaps) (quot (count gaps) 2)))
     :reliable? (every? #(get-in % [:grading :log-ml :reliable?] true) (filter correct? one))
     :tokens (when (seq one) (Math/round (double (/ (reduce + (map :tokens one)) (count one)))))}))

(defn summary
  "The report of a run's entries, as Markdown."
  [entries]
  (let [by-task (group-by :task entries)
        rows (keep #(some-> (by-task (:id %)) task-row) tasks/tasks)
        one (filter #(= :one-shot (:mode %)) entries)
        one-s (reduce + (keep :one-shot-s rows))
        repair-s (reduce + (map :repair-s rows))
        chains-fixed (count (filter #(not (str/starts-with? (:repair %) "no")) rows))]
    (str "Model: " (:model (first entries)) "\n\n"
         "| task | one shot correct | its decoding s | tokens a reply | failures | repair chain correct at | repair s | gap (nats, median) |\n"
         "|---|---|---|---|---|---|---|---|\n"
         (str/join (for [r rows]
                     (format "| %s | %s | %s | %s | %s | %s | %s | %s%s |\n" (name (:task r)) (:one-shot r) (fmt (:one-shot-s r))
                             (or (:tokens r) "–") (if (seq (:failures r)) (pr-str (:failures r)) "") (:repair r) (fmt (:repair-s r))
                             (fmt (:gap r)) (if (:reliable? r) "" " (unreliable)"))))
         "\n"
         (format "One shot: %d of %d replies correct; %.0f s decoding, %.1f s a correct model.\n"
                 (count (filter correct? one)) (count one) one-s
                 (let [c (count (filter correct? one))] (if (pos? c) (/ one-s c) Double/NaN)))
         (format "Repair: %d of %d chains correct (the first reply, or after up to 3 rounds); %.0f s decoding in rounds.\n"
                 chains-fixed (count rows) repair-s))))

(defn -main [& files]
  (doseq [f files]
    (println (summary (read-results f))))
  (shutdown-agents))
