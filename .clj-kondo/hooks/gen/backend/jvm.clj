(ns hooks.gen.backend.jvm
  "The JVM backend's kernel macros: each defines a private fn of the
   kernel's arity, its expression in the scope of the element names (bound
   to doubles), and fold-range and at as they expand."
  (:require [hooks.gen.util :refer [used]]))

(defmacro defunary [fname [x] expr]
  `(defn- ~fname [~(used 'a)] (let [~x 0.0] (double ~expr))))

(defmacro defbinary [fname [x y] expr]
  `(defn- ~fname [~@(map used '[a sa b sb os])] (let [~x 0.0 ~y 0.0] (double ~expr))))

(defmacro defreduce [fname [acc v] expr]
  `(defn- ~fname [~@(map used '[src outer len inner init])] (let [~acc 0.0 ~v 0.0] (double ~expr))))

(defmacro defscan [fname [acc v] expr]
  `(defn- ~fname [~@(map used '[src outer len inner reverse?])] (let [~acc 0.0 ~v 0.0] (double ~expr))))

(defmacro fold-range [[k start end] [acc init] expr]
  `(let [cell# (double-array 1)]
     (aset cell# 0 (double ~init))
     (loop [~k (long ~start)]
       (when (< ~k (long ~end))
         (let [~acc (aget cell# 0)] (aset cell# 0 (double ~expr)))
         (recur (inc ~k))))
     (aget cell# 0)))

;; off and n are the caller's locals
(defmacro at [i j] `(+ ~'off (* ~i ~'n) ~j))
