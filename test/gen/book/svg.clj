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
  "Histograms of samples on one axis, each scaled to a density (its bars'
   area is its weight in range over its total weight), each with an optional
   density curve to compare it with. layers is a seq of {:values, and
   optionally :weights (nonnegative, any scale), :density (a function of
   x) and :label (shown in a legend)}. Options: :bins (default 40), :range
   [lo hi] (default the values'), :x-label, :y-label."
  [layers & [opts]]
  (let [{:keys [bins range] :or {bins 40}} opts
        colors ["steelblue" "darkorange" "seagreen" "firebrick"]
        layers (mapv (fn [{:keys [values weights] :as l} color]
                       (let [vs (mapv host values)]
                         (assoc l :color color :vs vs
                                :ws (if weights (mapv host weights) (vec (repeat (count vs) 1.0))))))
                     layers colors)
        all-vs (mapcat :vs layers)
        [lo hi] (mapv double (or range [(apply min all-vs) (apply max all-vs)]))
        step (/ (- hi lo) bins)
        bin (fn [v] (if (== v hi) (dec bins) (long (Math/floor (/ (- v lo) step)))))
        heights (fn [{:keys [vs ws]}]
                  (let [total (reduce + ws)]
                    (->> (map vector vs ws)
                         (reduce (fn [acc [v w]] (if (<= lo v hi) (update acc (bin v) + w) acc))
                                 (vec (repeat bins 0.0)))
                         (mapv #(/ % (* total step))))))
        curve (fn [f] (mapv (fn [i] (let [x (+ lo (* i (/ (- hi lo) 200.0)))] [x (host (f x))]))
                            (clojure.core/range 201)))
        layers (mapv #(cond-> (assoc % :heights (heights %)) (:density %) (assoc :curve (curve (:density %))))
                     layers)
        top (apply max (concat (mapcat :heights layers) (map second (mapcat :curve layers))))]
    (frame (merge {:x-label "value" :y-label "density"} opts)
           [lo hi] [0.0 (* 1.05 top)]
           (fn [px py]
             (concat
              (for [{:keys [heights color]} layers]
                (str "<path fill=\"" color "\" fill-opacity=\"0.25\" stroke=\"" color "\" stroke-width=\"1\" d=\"M"
                     (fmt (px lo)) "," (fmt (py 0.0))
                     (str/join (map-indexed (fn [i h] (str " L" (fmt (px (+ lo (* i step)))) "," (fmt (py h))
                                                           " L" (fmt (px (+ lo (* (inc i) step)))) "," (fmt (py h))))
                                            heights))
                     " L" (fmt (px hi)) "," (fmt (py 0.0)) " Z\"/>"))
              (for [{:keys [curve color]} layers :when curve]
                (str "<polyline fill=\"none\" stroke=\"" color "\" stroke-width=\"2\" points=\""
                     (str/join " " (map (fn [[x y]] (str (fmt (px x)) "," (fmt (py y)))) curve)) "\"/>"))
              (for [[i {:keys [label color]}] (map-indexed vector (filter :label layers))
                    :let [y (+ 30.0 (* 16.0 i))]]
                (str "<rect x=\"" (- width 120) "\" y=\"" (fmt (- y 9.0)) "\" width=\"12\" height=\"10\" fill=\""
                     color "\" fill-opacity=\"0.6\"/><text x=\"" (- width 102) "\" y=\"" (fmt y) "\">" label
                     "</text>")))))))
(defn tree
  "The address tree of paths (a choice map's addresses, or any vectors of
   addresses): a node for each distinct prefix, the root at the top, and the
   paths' last addresses, the choices, shaded at the ends of their branches.
   Children keep the order of the paths. Options: :root, the root's label
   (default \"trace\")."
  [paths & [opts]]
  (let [root (get opts :root "trace")
        ;; [[addr subtree] …] for paths sharing their first addresses
        grow (fn grow [ps]
               (let [ps (remove empty? ps)]
                 (vec (for [a (distinct (map first ps))]
                        [a (grow (map rest (filter #(= a (first %)) ps)))]))))
        nodes (grow (map seq paths))
        depth (fn depth [kids] (if (empty? kids) 0 (inc (apply max (map (comp depth second) kids)))))
        levels (max 1 (depth nodes))
        leaves (fn leaves [kids] (if (empty? kids) 1 (reduce + (map (comp leaves second) kids))))
        n (leaves nodes)
        row 56.0
        h (long (+ 40.0 (* row levels)))
        dx (/ (double width) n)
        ;; [x y label leaf? parent-x parent-y] of every node below the root
        place (fn place [kids first-leaf level px py]
                (loop [kids kids i first-leaf out []]
                  (if-let [[a sub] (first kids)]
                    (let [k (leaves sub)
                          x (* dx (+ i (/ k 2.0)))
                          y (+ 20.0 (* row level))]
                      (recur (rest kids) (+ i k)
                             (into (conj out [x y (pr-str a) (empty? sub) px py])
                                   (place sub i (inc level) x y))))
                    out)))
        rx (/ (double width) 2.0)
        placed (place nodes 0 1 rx 20.0)
        box (fn [x y label leaf?]
              (let [w (+ 10.0 (* 6.5 (count label)))]
                (str "<g><rect x=\"" (fmt (- x (/ w 2.0))) "\" y=\"" (fmt (- y 9.0)) "\" width=\"" (fmt w)
                     "\" height=\"18\" rx=\"3\" fill=\"" (if leaf? "#dbe7f3" "white")
                     "\" stroke=\"steelblue\"/><text x=\"" (fmt x) "\" y=\"" (fmt (+ y 4.0))
                     "\" text-anchor=\"middle\">" label "</text></g>")))]
    (str/join
     "\n"
     (concat
      [(str "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" width "\" height=\"" h
            "\" viewBox=\"0 0 " width " " h "\" font-family=\"sans-serif\" font-size=\"11\">")
       "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>"]
      (for [[x y _ _ px py] placed]
        (str "<line x1=\"" (fmt px) "\" y1=\"" (fmt (+ py 9.0)) "\" x2=\"" (fmt x) "\" y2=\"" (fmt (- y 9.0))
             "\" stroke=\"gray\"/>"))
      [(box rx 20.0 root false)]
      (for [[x y label leaf?] placed] (box x y label leaf?))
      ["</svg>"]))))
(defn plot
  "Layers of points and curves in one box: a model's traces over data. A
   layer is a map, {:points [[x y] …]} for dots or {:curves [f …]} for the
   graphs of functions of x across the box, with optional :color (default
   black for dots, steelblue for curves), :opacity (default 1 for dots, 0.7
   for curves), :r (a dot's radius, default 3) and :width (a curve's
   stroke, default 1.5). Options: :x-range and :y-range (default: the dots'
   span, padded, or the curves' when there are no dots), :title, :x-label,
   :y-label. Marks are clipped to the box."
  [layers & [opts]]
  (let [dots (for [l layers [x y] (:points l)] [(host x) (host y)])
        span (fn [vs] (padded (apply min vs) (apply max vs)))
        [x0 x1] (mapv double (or (:x-range opts)
                                 (some-> (seq (map first dots)) span)
                                 (throw (ex-info "plot: curves without dots need an :x-range" {}))))
        grid (mapv #(+ x0 (* (- x1 x0) (/ % 400.0))) (range 401))
        graph (fn [f] (mapv (fn [x] [x (host (f x))]) grid))
        layers (mapv #(assoc % ::graphs (mapv graph (:curves %))) layers)
        [y0 y1] (mapv double (or (:y-range opts)
                                 (span (or (seq (map second dots))
                                           (for [l layers g (::graphs l) [_ y] g] y)))))]
    (frame (merge {:x-label "x" :y-label "y"} opts) [x0 x1] [y0 y1]
           (fn [px py]
             (concat
              [(str "<clipPath id=\"box\"><rect x=\"" (fmt (px x0)) "\" y=\"" (fmt (py y1))
                    "\" width=\"" (fmt (- (px x1) (px x0))) "\" height=\"" (fmt (- (py y0) (py y1)))
                    "\"/></clipPath>")
               "<g clip-path=\"url(#box)\">"]
              (for [{:keys [points color opacity r width] ::keys [graphs]} layers
                    mark (concat
                          (for [g graphs]
                            (str "<polyline fill=\"none\" stroke=\"" (or color "steelblue")
                                 "\" stroke-opacity=\"" (or opacity 0.7) "\" stroke-width=\"" (or width 1.5)
                                 "\" points=\"" (str/join " " (map (fn [[x y]] (str (fmt (px x)) "," (fmt (py y)))) g))
                                 "\"/>"))
                          (for [[x y] points]
                            (str "<circle cx=\"" (fmt (px (host x))) "\" cy=\"" (fmt (py (host y))) "\" r=\"" (or r 3)
                                 "\" fill=\"" (or color "black") "\" fill-opacity=\"" (or opacity 1) "\"/>")))]
                mark)
              ["</g>"])))))
(defn heatmap
  "A table of values as a grid of shaded cells: a 2-d posterior. rows holds
   a row of values for each of ys, the rows' centers from the bottom up,
   and each row a value for each of xs, the columns' centers; both evenly
   spaced. A cell's shade runs from white (0) to dark blue (the largest
   value). Options: :labels? (each cell's value printed, to 3 decimals),
   :mark [x y] (the true values: a red cross, or with labels, a red outline
   of the cell holding them), :marginals? (bars of the column sums above the
   grid and of the row sums to its right), :x-label, :y-label."
  [xs ys rows & [opts]]
  (let [{:keys [labels? mark marginals? x-label y-label]} opts
        xs (mapv host xs)
        ys (mapv host ys)
        vs (mapv #(mapv host %) rows)
        most (apply max 1e-300 (apply concat vs))
        spacing (fn [cs] (if (> (count cs) 1) (- (cs 1) (cs 0)) 1.0))
        dx (spacing xs)
        dy (spacing ys)
        [x0 x1] [(- (first xs) (/ dx 2.0)) (+ (peek xs) (/ dx 2.0))]
        [y0 y1] [(- (first ys) (/ dy 2.0)) (+ (peek ys) (/ dy 2.0))]
        strip (if marginals? 56.0 0.0)
        h 400.0
        left 52.0
        top (+ 14.0 strip)
        pw (- width left 16.0 strip)
        ph (- h top 40.0)
        px (fn [x] (+ left (* pw (/ (- x x0) (- x1 x0)))))
        py (fn [y] (+ top (* ph (- 1.0 (/ (- y y0) (- y1 y0))))))
        rect (fn [x y w hh fill]
               (str "<rect x=\"" (fmt x) "\" y=\"" (fmt y) "\" width=\"" (fmt w) "\" height=\"" (fmt hh)
                    "\" fill=\"" fill "\"/>"))
        shade (fn [v] (let [t (/ v most)
                            c (fn [hi lo] (Math/round (+ hi (* t (- lo hi)))))]
                        (str "rgb(" (c 255.0 8.0) "," (c 255.0 48.0) "," (c 255.0 107.0) ")")))
        cols (apply map + vs)
        row-sums (map #(reduce + %) vs)]
    (str/join
     "\n"
     (concat
      [(str "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" width "\" height=\"" (long h)
            "\" viewBox=\"0 0 " width " " (long h) "\" font-family=\"sans-serif\" font-size=\"11\">")
       "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>"]
      (for [[j y] (map-indexed vector ys) [i x] (map-indexed vector xs)
            :let [v (get-in vs [j i])
                  cx (px (- x (/ dx 2.0)))
                  cy (py (+ y (/ dy 2.0)))]]
        (str (rect cx cy (- (px (+ x (/ dx 2.0))) cx) (- (py (- y (/ dy 2.0))) cy) (shade v))
             (when labels?
               (str "<text x=\"" (fmt (px x)) "\" y=\"" (fmt (+ (py y) 4.0)) "\" text-anchor=\"middle\" fill=\""
                    (if (> (/ v most) 0.5) "white" "black") "\">"
                    (String/format java.util.Locale/ROOT "%.3f" (object-array [v])) "</text>"))))
      (when marginals?
        (let [cmax (apply max 1e-300 cols)
              rmax (apply max 1e-300 row-sums)]
          (concat
           (for [[x c] (map vector xs cols)
                 :let [bh (* (- strip 8.0) (/ c cmax))]]
             (rect (+ (px (- x (/ dx 2.0))) 0.5) (- top 4.0 bh) (max 0.0 (- (* dx (/ pw (- x1 x0))) 1.0)) bh
                   "steelblue"))
           (for [[y r] (map vector ys row-sums)
                 :let [bw (* (- strip 8.0) (/ r rmax))]]
             (rect (+ left pw 4.0) (+ (py (+ y (/ dy 2.0))) 0.5) bw (max 0.0 (- (* dy (/ ph (- y1 y0))) 1.0))
                   "steelblue")))))
      (for [t (ticks x0 x1)]
        (str "<g><line x1=\"" (fmt (px t)) "\" y1=\"" (fmt (+ top ph)) "\" x2=\"" (fmt (px t)) "\" y2=\""
             (fmt (+ top ph 4.0)) "\" stroke=\"black\"/><text x=\"" (fmt (px t)) "\" y=\"" (fmt (+ top ph 16.0))
             "\" text-anchor=\"middle\">" (label t) "</text></g>"))
      (for [t (ticks y0 y1)]
        (str "<g><line x1=\"" (fmt (- left 4.0)) "\" y1=\"" (fmt (py t)) "\" x2=\"" (fmt left) "\" y2=\"" (fmt (py t))
             "\" stroke=\"black\"/><text x=\"" (fmt (- left 7.0)) "\" y=\"" (fmt (+ (py t) 4.0))
             "\" text-anchor=\"end\">" (label t) "</text></g>"))
      [(str "<rect x=\"" (fmt left) "\" y=\"" (fmt top) "\" width=\"" (fmt pw) "\" height=\"" (fmt ph)
            "\" fill=\"none\" stroke=\"black\"/>")]
      (when x-label
        [(str "<text x=\"" (fmt (+ left (/ pw 2.0))) "\" y=\"" (fmt (- h 6.0)) "\" text-anchor=\"middle\">"
              x-label "</text>")])
      (when y-label
        [(str "<text transform=\"translate(14 " (fmt (+ top (/ ph 2.0))) ") rotate(-90)\" text-anchor=\"middle\">"
              y-label "</text>")])
      (when-let [[mx my] mark]
        (let [mx (host mx)
              my (host my)
              x (px mx)
              y (py my)
              cross (fn [color w]
                      (str "<path d=\"M" (fmt (- x 7.0)) "," (fmt (- y 7.0)) " L" (fmt (+ x 7.0)) "," (fmt (+ y 7.0))
                           " M" (fmt (- x 7.0)) "," (fmt (+ y 7.0)) " L" (fmt (+ x 7.0)) "," (fmt (- y 7.0))
                           "\" stroke=\"" color "\" stroke-width=\"" w "\"/>"))
              ;; with labels, the outline of the cell holding the mark
              nearest (fn [cs v] (apply min-key #(Math/abs (- (double %) v)) cs))
              cx (nearest xs mx)
              cy (nearest ys my)]
          (if labels?
            [(str "<rect x=\"" (fmt (+ (px (- cx (/ dx 2.0))) 1.5)) "\" y=\"" (fmt (+ (py (+ cy (/ dy 2.0))) 1.5))
                  "\" width=\"" (fmt (- (* dx (/ pw (- x1 x0))) 3.0)) "\" height=\"" (fmt (- (* dy (/ ph (- y1 y0))) 3.0))
                  "\" fill=\"none\" stroke=\"red\" stroke-width=\"3\"/>")]
            [(cross "white" 4) (cross "red" 2)])))
      ["</svg>"]))))
