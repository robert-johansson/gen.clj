(ns gen.dist.host
  "Distributions backed by a host library, as GenDistributions'
   DistributionsBacked wraps Distributions.jl's: a family of univariate
   distributions given by a host object built from the parameters, its
   quantile function (inverse CDF) and its log density.

     (def weibull
       (host/family {:name \"weibull\"
                     :make (fn [k lambda] {:k k :lambda lambda})
                     :quantile (fn [{:keys [k lambda]} u] (* lambda (Math/pow (- (Math/log1p (- u))) (/ 1.0 k))))
                     :log-density (fn [{:keys [k lambda]} x] …)}))
     (trace :t (weibull 1.5 2.0))

   :make builds the object from host parameters (and validates them; an
   exception there is reported as the family's); :discrete? marks integer
   families, whose quantile returns integers; :support (fn [object] →
   values or nil) gives a finite support for enumeration.

   A draw is the quantile of a uniform in (0, 1) from the key's Threefry
   bits (gen.random.threefry), so it is a pure function of the key and the
   same on every backend. Parameters and values are read on the host, as
   Clojure code branching on a value reads it: a batched one raises the
   batched fallback (the run continues one particle at a time), one being
   differentiated is an error (the library gives no gradient), and a lazy
   backend's array is evaluated to be read.

   gen.dist.commons-math (the :commons-math alias) makes families of Apache
   Commons Math's distributions."
  (:require [gen.array :as array]
            [gen.array.batch :as batch]
            [gen.array.shape :as sh]
            [gen.array.tape :as tape]
            [gen.dist.core :as dc]
            [gen.random.threefry :as threefry]))

(set! *warn-on-reflection* true)

(defn- host-number
  "x, a parameter or value of family, as a host number."
  [family what x]
  (cond
    (batch/batched? x)
    (throw (array/batched-branch (str "A " (:name family) " (host library) " what " in a batched run") {}))
    (tape/tracked? x)
    (throw (ex-info (str "A " (:name family) " distribution is backed by a host library, which gives no gradient: its "
                         what " is being differentiated")
                    {:family (:name family)}))
    (array/host? x) x
    (and (array/array? x) (== 1 (array/size x))) (array/item x)
    :else (throw (ex-info (str "A " (:name family) " " what " must be a number, got " (pr-str x))
                          {:family (:name family) :value x}))))

(defn family
  "A constructor of distributions from a host library (see the namespace
   doc): {:name :make :quantile :log-density}, and :discrete? and :support."
  [{:keys [name make quantile log-density] :as spec}]
  (doseq [[k f] {:make make :quantile quantile :log-density log-density}]
    (when-not (ifn? f)
      (throw (ex-info (str "host/family needs " k ", a function") {:name name :missing k}))))
  (let [family (assoc spec :name (or name "host"))]
    (fn [& params]
      (let [ps (mapv #(host-number family "parameter" %) params)
            _ (when (some #(and (float? %) (Double/isNaN (double %))) ps)
                (throw (ex-info (str (:name family) ": a parameter is NaN") {:family (:name family) :params ps})))
            object (try (apply make ps)
                        (catch clojure.lang.ExceptionInfo e (throw e))
                        (catch Exception e
                          (throw (ex-info (str (:name family) ": invalid parameters " (pr-str ps) ": " (.getMessage e))
                                          {:family (:name family) :params ps} e))))]
        (dc/->Distribution ::host {:family family :params ps :object object})))))

;; uniforms in (0, 1): quantiles at 0 or 1 are the support's bounds, which
;; may be infinite
(def ^:private half-ulp (Math/pow 2.0 -53))

(defn- draw [{:keys [family object]} u]
  (let [x ((:quantile family) object (+ u half-ulp))]
    (if (:discrete? family) (long x) (double x))))

(defmethod dc/-sample ::host [d key batch-shape]
  (let [p (:params d)
        ^doubles us (threefry/uniform-doubles key (sh/size batch-shape))]
    (if (= [] batch-shape)
      (draw p (aget us 0))
      (array/reshape (array/array (mapv #(draw p %) us)) batch-shape))))

(defn- log-density [{:keys [family object]} x]
  (let [x (host-number family "value" x)]
    (if (and (:discrete? family) (not (== x (Math/rint (double x)))))
      ##-Inf
      (double ((:log-density family) object x)))))

(defmethod dc/-log-prob ::host [d v]
  (let [p (:params d)]
    (if (or (array/host? v) (batch/batched? v) (tape/tracked? v) (= [] (array/shape v)))
      (log-density p v)
      (array/reshape (array/array (mapv #(log-density p %) (array/->clj (array/reshape v [(array/size v)]))))
                     (array/shape v)))))

(defmethod dc/-discrete? ::host [d] (boolean (:discrete? (:family (:params d)))))

(defmethod dc/-support ::host [d]
  (let [{:keys [family object]} (:params d)
        s (when-let [f (:support family)] (f object))]
    (or (some-> s vec)
        (throw (ex-info (str (:name family) " has no enumerable support") {:family (:name family)})))))

(defmethod dc/-value-shape ::host [_] [])
