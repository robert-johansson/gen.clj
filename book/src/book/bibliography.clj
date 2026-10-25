;; The code of book/066-bibliography.md. Each region, from a `;; book:` line
;; to the next, is included there by name (gen.book).

;; book: ns
(ns book.bibliography
  (:require [clojure.java.io :as io]
            [clojure.string :as str]))

;; book: pages
(defn prose
  "A page's prose, as the checks read it: its Markdown without code blocks,
   inline code and comments, its lines joined, so that a citation broken
   across two lines reads whole."
  [md]
  (-> md
      (str/replace #"(?ms)^```.*?^```$" " ")
      (str/replace #"`[^`]*`" " ")
      (str/replace #"(?s)<!--.*?-->" " ")
      (str/replace #"\s+" " ")))

(defn title
  "A page's title, less a subtitle after a colon: Getting Started, for
   Getting Started: Linear Regression."
  [md]
  (first (str/split (second (re-find #"(?m)^# (.+)$" md)) #": ")))

(def pages
  "The pages that cite, in the book's order, as {:title :text}: every
   numbered Markdown file but a Part's heading, The Public API (the
   docstrings', which cite as the code does) and this appendix."
  (->> (.listFiles (io/file "book"))
       (map #(.getName ^java.io.File %))
       (filter #(re-matches #"\d{3}-.*\.md" %))
       (remove #(re-matches #"\d\d0-.*|063-.*|066-.*" %))
       sort
       (mapv (fn [file]
               (let [md (slurp (str "book/" file))]
                 {:title (title md) :text (prose md)})))))

;; book: pattern
(def surname
  "A surname: a capital, then letters and hyphens, ending in a small letter
   (so not MCMC), perhaps after a particle (Del Moral)."
  "(?:(?:Del|De|van|von) )?\\p{Lu}[\\p{L}-]*\\p{Ll}")

(def citation-re
  "An author-year citation. An opening parenthesis or semicolon, if any;
   the authors: a surname, two joined by and, more with commas, or one with
   et al.; then the year: bare, after a comma, in parentheses, or after a
   possessive and a title, as in Cusumano-Towner's thesis (2020) and Saad
   et al.'s *Title* (POPL 2019)."
  (re-pattern
   (str "(\\(|; )?(?<![\\p{L}-])"
        "(" surname "(?:(?:, " surname ")* and " surname "| et al\\.)?)"
        "('s (?:thesis |\\*[^*]+\\* )?\\((?:\\p{Lu}+ )?|'s |,? \\(?)"
        "((?:19|20)\\d\\d)(?!\\d)")))

(defn citations
  "The author-year citations of a text, as keys: authors, a space, the
   year. One surname before a bare year is a citation only inside
   parentheses, as in (Green 1995), so that a sentence that begins In 2000
   tries is not one."
  [text]
  (vec (for [[_ opener authors link year] (re-seq citation-re text)
             :when (or opener
                       (re-find #" and | et al\." authors)
                       (re-find #"'s|\(" link))]
         (str authors " " year))))

;; book: sample
(citations (str "the proposal's efficiency (Kong, 1992), Gilks and "
                "Berzuini (2001) and Saad et al.'s *Bayesian Synthesis* "
                "(POPL 2019). In 2000 tries, Cusumano-Towner's thesis"))
;=> ["Kong 1992" "Gilks and Berzuini 2001" "Saad et al. 2019"]

;; book: entries
(defn items
  "The list items of Markdown md that begin with a key in bold, each on
   one line."
  [md]
  (map #(str/replace % #"\s+" " ")
       (re-seq #"(?ms)^- \*\*.*?(?=\n- |\n\n|\z)" md)))

(defn entry
  "An entry of the bibliography from its list item: {:key :alias :cited}.
   The item begins with the key in bold, and perhaps an alias in italics
   and parentheses, (*the thesis*); it ends with Cited in and the citing
   pages' titles in italics, or with Cited throughout."
  [item]
  (let [[_ k alias] (re-find #"^- \*\*([^*]+)\*\*(?: \(\*([^*]+)\*\))?\. "
                             item)]
    {:key k
     :alias alias
     :cited (if (str/ends-with? item "Cited throughout.")
              :throughout
              (set (map second (re-seq #"\*([^*]+)\*"
                                       (second (re-find #"Cited in (.+)\.$"
                                                        item))))))}))

(defn tutorials
  "The rows of the table of tutorials in md, [title cited]: the rows whose
   first cell is a title in italics, cited being the titles in their last
   cell."
  [md]
  (for [line (str/split-lines md)
        :let [[_ t cited] (re-matches #"\| \*([^*]+)\* \|.*\| (.+) \|" line)]
        :when t]
    [t (set (map second (re-seq #"\*([^*]+)\*" cited)))]))

;; book: names
(defn year-key?
  "Whether an entry's key ends in a year: Green 1995, not Stan."
  [k]
  (boolean (re-find #" \d{4}$" k)))

(defn names
  "The names that cite each entry, as [name key exact?]: an author-year
   entry's authors, as written (exact), and any other key, or an alias, in
   either case (the GenMLX book, genlm). An alias drops its article: the
   chapters write the thesis, the thesis's and thesis §4.1."
  [entries]
  (for [{k :key alias :alias} entries
        [n exact?] (cond-> [(if (year-key? k)
                              [(str/replace k #" \d{4}$" "") true]
                              [k false])]
                     alias (conj [(str/replace alias #"^the " "") false]))]
    [n k exact?]))

(defn cited-by-name
  "The keys of the entries whose names appear in a text. Longer names are
   found first, then blanked out, so that Cusumano-Towner and Mansinghka
   does not also count as the thesis's Cusumano-Towner, nor the GenMLX
   book as GenMLX."
  [text names]
  (loop [text text
         [[n k exact?] & more] (sort-by (comp count first) > names)
         found #{}]
    (if-not n
      found
      (let [re (re-pattern (str (when-not exact? "(?i)")
                                "(?<![\\p{L}\\p{N}.-])"
                                (java.util.regex.Pattern/quote n)
                                "(?![\\p{L}\\p{N}-])"))]
        (recur (str/replace text re #(apply str (repeat (count %) "~")))
               more
               (cond-> found (re-find re text) (conj k)))))))

;; book: resolve
(defn first-author
  "A key's first author: Andrieu, of Andrieu, Doucet and Holenstein 2010."
  [k]
  (re-find #"^.+?(?=,| and | et al\.| \d{4}$)" k))

(defn entry-of
  "The key of the entry a citation names: its own key, or, for X et al. Y,
   the one entry whose first author is X and whose year is Y, as Andrieu
   et al. 2010 names Andrieu, Doucet and Holenstein 2010."
  [ks citation]
  (or (some #{citation} ks)
      (when (str/includes? citation " et al. ")
        (let [year #(re-find #"\d{4}$" %)
              same (filter #(and (= (first-author %) (first-author citation))
                                 (= (year %) (year citation)))
                           ks)]
          (when (= 1 (count same)) (first same))))))

;; book: checks
(def bibliography (slurp "book/066-bibliography.md"))

(def entries (map entry (items bibliography)))

(def ks (map :key entries))

(def cited-in
  "Each entry's citing pages, by title: those with a citation that names
   it, and those that write one of its names."
  (let [all-names (names entries)]
    (apply merge-with into (zipmap ks (repeat #{}))
           (for [{:keys [title text]} pages
                 k (concat (keep #(entry-of ks %) (citations text))
                           (cited-by-name text all-names))]
             {k #{title}}))))

(defn listed-right?
  "Whether an entry lists the pages that cite it: all of them, or Cited
   throughout when more than half the pages do."
  [{k :key cited :cited}]
  (if (= cited :throughout)
    (> (count (cited-in k)) (quot (count pages) 2))
    (= cited (cited-in k))))

[(for [{:keys [title text]} pages
       c (citations text)
       :when (not (entry-of ks c))]
   [title c])
 (remove (comp seq cited-in) ks)
 (map :key (remove listed-right? entries))]
;=> [[] [] []]

;; book: tutorials-check
(def text-of (into {} (map (juxt :title :text) pages)))

(for [[t cited] (tutorials bibliography)
      page cited
      :when (not (str/includes? (text-of page "") (str "*" t "*")))]
  [t page])
;=> []
