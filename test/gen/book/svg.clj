(ns gen.book.svg
  "The book's figures as SVG text, from the book's own code (book/): no
   plotting dependency, and the same text for the same data, so a test can
   tell a committed figure from a stale one (gen.book). Values are read to
   the host here, so a chapter passes arrays of any backend."
  (:require [clojure.string :as str]
            [gen.array :as array]))

(def ^:private width 480)
(def ^:private height 300)
(def ^:private margin {:left 52 :right 16 :bottom 40})

(defn- host [x] (double (array/item x)))

(defn- fmt [^double x] (String/format java.util.Locale/ROOT "%.1f" (object-array [x])))

(defn- ticks
  "About five round values spanning [lo hi]."
  [^double lo ^double hi]
  (let [span (- hi lo)
        raw (/ span 5.0)
        mag (Math/pow 10.0 (Math/floor (Math/log10 raw)))
        step (* mag (some #(when (<= raw (* % mag)) %) [1.0 2.0 2.5 5.0 10.0]))]
    (->> (iterate #(+ % step) (* step (Math/ceil (/ lo step))))
         (take-while #(<= % (+ hi (* 1e-9 span))))
         (mapv #(if (< (Math/abs (double %)) (* 1e-9 span)) 0.0 %)))))

(defn- label [^double v]
  (let [s (String/format java.util.Locale/ROOT "%.2f" (object-array [v]))]
    (-> s (str/replace #"0+$" "") (str/replace #"\.$" ""))))

(defn- frame
  "The SVG of axes over the box [x0 x1] × [y0 y1], its marks inside: marks
   is a function of the data-to-pixel maps px and py."
  [{:keys [title x-label y-label]} [x0 x1] [y0 y1] marks]
  (let [{:keys [left right bottom]} margin
        top (if title 34 14)
        pw (- width left right)
        ph (- height top bottom)
        px (fn [x] (+ left (* pw (/ (- (double x) x0) (- x1 x0)))))
        py (fn [y] (+ top (* ph (- 1.0 (/ (- (double y) y0) (- y1 y0))))))]
    (str/join
     "\n"
     (concat
      [(str "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" width "\" height=\"" height
            "\" viewBox=\"0 0 " width " " height "\" font-family=\"sans-serif\" font-size=\"11\">")
       "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>"]
      (when title [(str "<text x=\"" (fmt (/ width 2.0)) "\" y=\"20\" text-anchor=\"middle\" font-size=\"13\">" title "</text>")])
      (for [t (ticks x0 x1)]
        (str "<g><line x1=\"" (fmt (px t)) "\" y1=\"" (fmt (+ top ph)) "\" x2=\"" (fmt (px t)) "\" y2=\"" (fmt (+ top ph 4.0))
             "\" stroke=\"black\"/><text x=\"" (fmt (px t)) "\" y=\"" (fmt (+ top ph 16.0)) "\" text-anchor=\"middle\">" (label t) "</text></g>"))
      (for [t (ticks y0 y1)]
        (str "<g><line x1=\"" (fmt (- left 4.0)) "\" y1=\"" (fmt (py t)) "\" x2=\"" left "\" y2=\"" (fmt (py t))
             "\" stroke=\"black\"/><text x=\"" (fmt (- left 7.0)) "\" y=\"" (fmt (+ (py t) 4.0)) "\" text-anchor=\"end\">" (label t) "</text></g>"))
      [(str "<rect x=\"" left "\" y=\"" top "\" width=\"" pw "\" height=\"" ph "\" fill=\"none\" stroke=\"black\"/>")]
      (when x-label [(str "<text x=\"" (fmt (+ left (/ pw 2.0))) "\" y=\"" (- height 6) "\" text-anchor=\"middle\">" x-label "</text>")])
      (when y-label [(str "<text transform=\"translate(14 " (fmt (+ top (/ ph 2.0))) ") rotate(-90)\" text-anchor=\"middle\">" y-label "</text>")])
      (marks px py)
      ["</svg>"]))))

(defn- padded [lo hi]
  (let [pad (* 0.05 (max (- hi lo) 1e-9))] [(- lo pad) (+ hi pad)]))

(defn lines
  "The points (xs, ys) and the lines y = slope·x + intercept, one for each
   [slope intercept] of lines, across xs's range. Options: :title,
   :x-label, :y-label."
  [xs ys lines & [opts]]
  (let [xs (map host xs)
        ys (map host ys)
        lo (apply min xs)
        hi (apply max xs)
        ends (for [[slope intercept] lines
                   :let [s (host slope) b (host intercept)]]
               [[lo (+ (* s lo) b)] [hi (+ (* s hi) b)]])
        all-y (concat ys (map second (apply concat ends)))]
    (frame (merge {:x-label "x" :y-label "y"} opts)
           (padded lo hi) (padded (apply min all-y) (apply max all-y))
           (fn [px py]
             (concat
              (for [[[xa ya] [xb yb]] ends]
                (str "<line x1=\"" (fmt (px xa)) "\" y1=\"" (fmt (py ya)) "\" x2=\"" (fmt (px xb)) "\" y2=\"" (fmt (py yb))
                     "\" stroke=\"steelblue\" stroke-opacity=\"0.6\" stroke-width=\"1.5\"/>"))
              (for [[x y] (map vector xs ys)]
                (str "<circle cx=\"" (fmt (px x)) "\" cy=\"" (fmt (py y)) "\" r=\"3\" fill=\"black\"/>")))))))

(defn series
  "One quantity's values against their index: a chain's draws. Options:
   :title, :x-label, :y-label."
  [values & [opts]]
  (let [vs (mapv host values)
        n (count vs)]
    (frame (merge {:x-label "iteration" :y-label "value"} opts)
           [0.0 (double (max 1 (dec n)))] (padded (apply min vs) (apply max vs))
           (fn [px py]
             [(str "<polyline fill=\"none\" stroke=\"steelblue\" stroke-width=\"1\" points=\""
                   (str/join " " (map-indexed (fn [i v] (str (fmt (px i)) "," (fmt (py v)))) vs))
                   "\"/>")]))))

(defn histogram
  "The values' histogram as a density (each bar's area is its fraction of
   the values), with the curve of the density f across their range when f
   is given (a function of a host double; nil for none). Options: :bins
   (30), :title, :x-label, :y-label."
  [values f & [opts]]
  (let [vs (mapv host values)
        bins (long (:bins opts 30))
        lo (apply min vs)
        hi (apply max vs)
        step (/ (- hi lo) bins)
        counts (reduce (fn [acc v] (update acc (min (dec bins) (long (/ (- v lo) step))) inc))
                       (vec (repeat bins 0)) vs)
        heights (mapv #(/ (double %) (* (count vs) step)) counts)
        curve (when f
                (mapv (fn [i] (let [x (+ lo (* (- hi lo) (/ i 100.0)))] [x (host (f x))])) (range 101)))
        top (apply max (concat heights (map second curve)))]
    (frame (merge {:x-label "value" :y-label "density"} opts)
           (padded lo hi) [0.0 (* 1.05 top)]
           (fn [px py]
             (concat
              (for [[i h] (map-indexed vector heights)
                    :let [x0 (+ lo (* i step))]]
                (str "<rect x=\"" (fmt (px x0)) "\" y=\"" (fmt (py h)) "\" width=\"" (fmt (- (px (+ x0 step)) (px x0)))
                     "\" height=\"" (fmt (- (py 0.0) (py h))) "\" fill=\"steelblue\" fill-opacity=\"0.5\" stroke=\"white\"/>"))
              (when curve
                [(str "<polyline fill=\"none\" stroke=\"black\" stroke-width=\"1.5\" points=\""
                      (str/join " " (map (fn [[x y]] (str (fmt (px x)) "," (fmt (py y)))) curve))
                      "\"/>")]))))))
