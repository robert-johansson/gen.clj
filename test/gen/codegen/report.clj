(ns gen.codegen.report
  "The codegen bake-off's report (genclj-ihc4): a run's results.edn
   (gen.codegen.driver) summarized, per task and in total.

     clojure -M:test -m gen.codegen.report <results.edn> …

   A reply runs when it passes every hard stage (it reads, is safe,
   evaluates, simulates and takes the data), and explains the data when,
   besides, its gap (its log marginal likelihood less the reference's,
   nats) is reliable and within the task's tolerance: running is not being
   the right model.
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

(defn- reliable? [entry] (get-in entry [:grading :log-ml :reliable?] false))

(defn close?
  "Does a correct reply also explain the data: a reliable gap within the
   task's tolerance below the reference? nil when its gap is unreliable."
  [entry]
  (when (and (correct? entry) (gap entry) (reliable? entry))
    (>= (gap entry) (- (tasks/tolerance (tasks/by-id (:task entry)))))))

(defn- fmt [x] (if (number? x) (format "%.1f" (double x)) "–"))

(defn- gap-text
  "A correct reply's gap: its nats, ? when unreliable."
  [entry]
  (str (fmt (gap entry)) (when-not (reliable? entry) "?")))

(defn- task-row [entries]
  (let [one (filter #(= :one-shot (:mode %)) entries)
        chain (sort-by :round (cons (first (filter #(zero? (:reply %)) one)) (filter #(= :repair (:mode %)) entries)))
        fixed (first (filter correct? chain))]
    {:task (:task (first entries))
     :runs (count (filter correct? one)) :n (count one)
     :close (count (filter close? one))
     :repaired (count (filter #(and (correct? %) (get-in % [:grading :stages 0 :repaired-delimiters])) one))
     :gaps (mapv gap-text (filter correct? one))
     :one-shot-s (some :seconds one)
     :failures (frequencies (remove #{:correct} (map (comp reached :grading) one)))
     :repair (cond (nil? fixed) (str "no (" (dec (count chain)) " rounds)")
                   (zero? (:round fixed)) "first reply"
                   :else (str "round " (:round fixed)))
     :repair-gap (when fixed (gap-text fixed))
     :repair-close? (when fixed (close? fixed))
     :repair-s (reduce + (keep :seconds (filter #(= :repair (:mode %)) entries)))
     :tokens (when (seq one) (Math/round (double (/ (reduce + (map :tokens one)) (count one)))))}))

(defn summary
  "The report of a run's entries, as Markdown."
  [entries]
  (let [by-task (group-by :task entries)
        rows (keep #(some-> (by-task (:id %)) task-row) tasks/tasks)
        one (filter #(= :one-shot (:mode %)) entries)
        one-s (reduce + (keep :one-shot-s rows))
        runs (count (filter correct? one))
        close (count (filter close? one))
        unreliable (count (filter #(and (correct? %) (not (reliable? %))) one))
        repaired (reduce + (map :repaired rows))]
    (str "Model: " (:model (first entries)) "; sampling " (pr-str (:sampling (first entries))) "\n\n"
         "A reply runs when it reads, is safe, evaluates, simulates and takes the data; it explains the data when, "
         "besides, its log marginal likelihood is reliably within the task's tolerance (5 nats) below the book "
         "model's. Gaps in nats; ? marks an unreliable estimate.\n\n"
         "| task | one shot: runs | explains | gaps of those that run | decoding s | tokens a reply | failures | repair chain runs at | its gap | repair s |\n"
         "|---|---|---|---|---|---|---|---|---|---|\n"
         (str/join (for [r rows]
                     (format "| %s | %d/%d | %d | %s | %s | %s | %s | %s | %s | %s |\n" (name (:task r)) (:runs r) (:n r) (:close r)
                             (str/join " " (:gaps r)) (fmt (:one-shot-s r)) (or (:tokens r) "–")
                             (if (seq (:failures r)) (pr-str (:failures r)) "") (:repair r) (or (:repair-gap r) "–")
                             (fmt (:repair-s r)))))
         "\n"
         (format "One shot: %d of %d replies run, %d of them explain the data (%d unreliable to judge); %.0f s decoding: %.1f s a running model, %s a model that explains the data.\n"
                 runs (count one) close unreliable one-s (if (pos? runs) (/ one-s runs) Double/NaN)
                 (if (pos? close) (format "%.1f s" (/ one-s close)) "–"))
         (format "Of the one-shot replies that run, %d did so with their delimiters rebuilt from their indentation.\n" repaired)
         (format "Repair: %d of %d chains run (the first reply, or after up to 3 rounds), %d of them explain the data.\n"
                 (count (filter #(not (str/starts-with? (:repair %) "no")) rows)) (count rows)
                 (count (filter :repair-close? rows))))))

(defn -main [& files]
  (doseq [f files]
    (println (summary (read-results f))))
  (shutdown-agents))
