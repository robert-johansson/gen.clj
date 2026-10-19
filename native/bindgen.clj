;; Generates the Java binding of mlx-c from its headers (bean genclj-izym):
;;
;;   src-java/gen/mlx/MlxRaw.java  one static method per C function, called with
;;                                 invokeExact, C types mapped mechanically
;;   src-java/gen/mlx/Mlx.java     checked wrappers for the common shapes: the
;;                                 status is checked (MlxException), out-params
;;                                 become return values, int arrays and strings
;;                                 are copied to per-thread scratch memory
;;
;; Every mlx-c handle, a struct { void* ctx; } passed by value, is lowered to a
;; long: one-pointer structs are classified as pointers on AArch64 and SysV
;; x86-64 (the ABI test in test/gen/backend/mlx_abi_test.clj checks this).
;; Other by-value structs (the mlx_optional_* types) keep their FFM layouts.
;;
;; Usage: clojure -M native/bindgen.clj [include-dir]
;;   include-dir defaults to native/build/<platform>/include (from
;;   native/build-mlx.sh), so the binding matches the built library.

(require '[clojure.java.io :as io]
         '[clojure.string :as str])

(def include-dir
  (or (first *command-line-args*)
      (let [os (if (str/includes? (str/lower-case (System/getProperty "os.name")) "mac") "macos" "linux")
            arch (case (System/getProperty "os.arch") ("aarch64" "arm64") "arm64" "x86_64")]
        (str "native/build/" os "-" (if (= os "macos") arch (str arch "-cpu")) "/include"))))

(def out-dir "src-java/gen/mlx")

;; ---------------------------------------------------------------------------
;; Parsing
;; ---------------------------------------------------------------------------

(defn strip-comments [s]
  (-> s
      (str/replace #"(?s)/\*.*?\*/" " ")
      (str/replace #"//[^\n]*" "")))

(defn headers []
  (->> (.listFiles (io/file include-dir "mlx" "c"))
       (filter #(str/ends-with? (.getName ^java.io.File %) ".h"))
       (sort-by #(.getName ^java.io.File %))))

(def sources (delay (mapv (fn [f] [(.getName ^java.io.File f) (strip-comments (slurp f))]) (headers))))

(defn handle-types
  "Names of the typedef'd structs that hold just void* ctx."
  [src]
  (set (map second (re-seq #"typedef\s+struct\s+\w+\s*\{\s*void\s*\*\s*ctx\s*;\s*\}\s*(\w+)\s*;" src))))

(defn enum-types [src]
  (set (map second (re-seq #"typedef\s+enum\s+\w+\s*\{[^}]*\}\s*(\w+)\s*;" src))))

(defn fnptr-types
  "Names of typedef'd function pointer types."
  [src]
  (set (map second (re-seq #"typedef\s+[\w\s\*]+\(\s*\*\s*(\w+)\s*\)\s*\(" src))))

(def all-src (delay (str/join "\n" (map second @sources))))
(def handles (delay (handle-types @all-src)))
(def enums (delay (enum-types @all-src)))
(def fnptrs (delay (fnptr-types @all-src)))

;; by-value structs other than handles, with their FFM layouts
(def struct-layouts
  {"mlx_optional_int" {:layout "MlxRuntime.OPTIONAL_INT" :java "Integer" :value "JAVA_INT" :set "setInt"}
   "mlx_optional_float" {:layout "MlxRuntime.OPTIONAL_FLOAT" :java "Float" :value "JAVA_FLOAT" :set "setFloat"}
   "mlx_optional_dtype" {:layout "MlxRuntime.OPTIONAL_INT" :java "Integer" :value "JAVA_INT" :set "setInt"}})

(defn split-params
  "Split a parameter list at top-level commas (function pointer parameters
   contain commas of their own)."
  [s]
  (loop [cs (seq s) depth 0 cur (StringBuilder.) out []]
    (if-let [c (first cs)]
      (cond (= c \() (recur (next cs) (inc depth) (.append cur c) out)
            (= c \)) (recur (next cs) (dec depth) (.append cur c) out)
            (and (= c \,) (zero? depth)) (recur (next cs) depth (StringBuilder.) (conj out (str cur)))
            :else (recur (next cs) depth (.append cur c) out))
      (let [last (str/trim (str cur))]
        (if (str/blank? last) out (conj out (str cur)))))))

(defn parse-param
  "{:type :name :fnptr?} of one C parameter."
  [p]
  (let [p (str/replace (str/trim p) #"\s+" " ")]
    (cond
      (= p "...") {:varargs? true}
      (str/includes? p "(*")
      {:name (second (re-find #"\(\*\s*(\w+)\s*\)" p)) :type "fnptr" :fnptr? true}
      :else
      (let [[_ t n] (re-find #"^(.*?)\s*\b(\w+)$" p)
            t (str/trim (str/replace t #"\s*\*" "*"))]
        (if (and t (or (str/ends-with? t "*") (not (str/blank? t))))
          {:type t :name n}
          ;; an unnamed parameter: the whole thing is the type
          {:type p :name nil})))))

(defn functions
  "Every function declaration: {:header :ret :name :params}."
  []
  (for [[header src] @sources
        m (re-seq #"(?m)^([A-Za-z_][\w \t\*]*?)\s*\b((?:_?mlx_)\w+)\s*\(([^;{]*?)\)\s*;" src)
        :let [[_ ret name params] m
              ret (str/trim (str/replace ret #"\s*\*" "*"))]
        :when (not (re-find #"\b(typedef|static|return)\b" ret))]
    {:header header
     :ret (str/replace ret #"\s+" " ")
     :name name
     :params (let [ps (mapv parse-param (split-params params))]
               (if (= [{:type "void" :name nil}] ps) [] (vec (remove #(= "void" (:type %)) ps))))}))

;; ---------------------------------------------------------------------------
;; C types → Java
;; ---------------------------------------------------------------------------

(defn base-type [t] (-> t (str/replace #"^const " "") str/trim))

(defn ctype
  "{:java :layout} for a C parameter or return type, or nil when unsupported."
  [t]
  (let [b (base-type t)]
    (cond
      (or (= t "fnptr") (@fnptrs b)) {:java "MemorySegment" :layout "ADDRESS"}
      (str/ends-with? b "*") {:java "MemorySegment" :layout "ADDRESS"}
      (@handles b) {:java "long" :layout "JAVA_LONG"}
      (@enums b) {:java "int" :layout "JAVA_INT"}
      (struct-layouts b) {:java "MemorySegment" :layout (:layout (struct-layouts b))}
      :else (case b
              "void" {:java "void" :layout nil}
              ("int" "int32_t" "uint32_t") {:java "int" :layout "JAVA_INT"}
              ("size_t" "uint64_t" "int64_t" "uintptr_t") {:java "long" :layout "JAVA_LONG"}
              "bool" {:java "boolean" :layout "JAVA_BOOLEAN"}
              "float" {:java "float" :layout "JAVA_FLOAT"}
              "double" {:java "double" :layout "JAVA_DOUBLE"}
              nil))))

(defn java-name [n]
  ;; parameter names that are Java keywords
  (case n
    ("default" "new" "class" "char" "int" "float" "double" "long" "short" "byte" "boolean"
     "this" "final" "static" "public" "private" "native" "package" "import" "return" "switch"
     "case" "for" "while" "do" "if" "else" "try" "catch" "throw" "throws" "goto" "const" "volatile"
     "transient" "abstract" "interface" "extends" "implements" "super" "synchronized" "assert"
     "enum" "strictfp" "instanceof" "void") (str n "_")
    n))

(defn supported? [{:keys [ret params]}]
  (and (ctype ret)
       (every? #(and (not (:varargs? %)) (ctype (:type %))) params)))

;; ---------------------------------------------------------------------------
;; Raw layer
;; ---------------------------------------------------------------------------

(defn descriptor [{:keys [ret params]}]
  (let [r (:layout (ctype ret))
        ps (map #(:layout (ctype (:type %))) params)]
    (if r
      (str "FunctionDescriptor.of(" (str/join ", " (cons r ps)) ")")
      (str "FunctionDescriptor.ofVoid(" (str/join ", " ps) ")"))))

(defn param-names
  "Java names for the parameters, unique, unnamed ones numbered."
  [params]
  (vec (map-indexed (fn [i p] (java-name (or (:name p) (str "arg" i)))) params)))

(defn raw-method [{:keys [name ret params] :as f}]
  (let [rj (:java (ctype ret))
        names (param-names params)
        decl (str/join ", " (map (fn [p n] (str (:java (ctype (:type p))) " " n)) params names))
        call (str name ".H.invokeExact(" (str/join ", " names) ")")]
    (str "  private static final class " name " {\n"
         "    static final MethodHandle H = MlxRuntime.downcall(\"" name "\", " (descriptor f) ");\n"
         "  }\n\n"
         "  public static " rj " " name "(" decl ") {\n"
         "    try {\n"
         "      " (if (= rj "void") "" (str "return (" rj ") ")) call ";\n"
         "    } catch (Throwable t) {\n"
         "      throw MlxRuntime.unreachable(t);\n"
         "    }\n"
         "  }\n")))

;; ---------------------------------------------------------------------------
;; Checked layer
;; ---------------------------------------------------------------------------

(def out-kinds
  "Out-parameter base types of the checked layer → [java type, read expression]."
  {"bool" ["boolean" "JAVA_BOOLEAN"]
   "int" ["int" "JAVA_INT"]
   "size_t" ["long" "JAVA_LONG"]
   "float" ["float" "JAVA_FLOAT"]
   "double" ["double" "JAVA_DOUBLE"]
   "uint32_t" ["int" "JAVA_INT"]
   "int32_t" ["int" "JAVA_INT"]
   "int64_t" ["long" "JAVA_LONG"]
   "uint64_t" ["long" "JAVA_LONG"]})

(defn out-param?
  "Is p an out-parameter: a pointer to a handle or a scalar? (Only leading
   parameters, and not in the _set functions, where it is assigned into.)"
  [p]
  (let [t (:type p)]
    (and (str/ends-with? t "*")
         (not (str/starts-with? t "const"))
         (let [b (subs t 0 (dec (count t)))]
           (or (@handles b) (out-kinds b))))))

(defn status-function?
  "Does the function return a status (mlx-c's convention for int)? A few
   return a value instead."
  [{:keys [name ret]}]
  (and (= ret "int")
       (not (#{"mlx_array_dim" "mlx_distributed_group_rank" "mlx_distributed_group_size"} name))))

(defn outs-of [{:keys [name params]}]
  (if (and (re-find #"_set(_|$)" name)
           (not (re-matches #"res(_\d+)?" (str (:name (first params))))))
    []
    (vec (take-while out-param? params))))

(defn out-java [p]
  (let [b (subs (:type p) 0 (dec (count (:type p))))]
    (if (@handles b) ["long" "JAVA_LONG"] (out-kinds b))))

(defn checked-params
  "The checked layer's view of the in-parameters: a vector of
   {:decl java-decl :args [raw-arg-exprs] :prep [statements]}, consuming
   (array, count) pairs as one Java array."
  [ins]
  (loop [ps ins out [] i 0]
    (if-let [p (first ps)]
      (let [n (java-name (or (:name p) (str "arg" i)))
            nxt (second ps)
            t (:type p)
            b (base-type t)
            count-of? (and nxt (= (:name nxt) (str (:name p) "_num")) (= "size_t" (:type nxt)))]
        (cond
          (and (= t "const int*") count-of?)
          (recur (nnext ps)
                 (conj out {:decl (str "int[] " n)
                            :prep [(str "MemorySegment " n "$ = scratch$.ints(" n ");")]
                            :args [(str n "$") (str "(long) " n ".length")]})
                 (+ i 2))

          (and (= t "const int64_t*") count-of?)
          (recur (nnext ps)
                 (conj out {:decl (str "long[] " n)
                            :prep [(str "MemorySegment " n "$ = scratch$.longs(" n ");")]
                            :args [(str n "$") (str "(long) " n ".length")]})
                 (+ i 2))

          (and (re-matches #"const (mlx_\w+)\*" t) (@handles (subs b 0 (dec (count b)))) count-of?)
          (recur (nnext ps)
                 (conj out {:decl (str "long[] " n)
                            :prep [(str "MemorySegment " n "$ = scratch$.longs(" n ");")]
                            :args [(str n "$") (str "(long) " n ".length")]})
                 (+ i 2))

          (= t "const char*")
          (recur (next ps)
                 (conj out {:decl (str "String " n)
                            :prep [(str "MemorySegment " n "$ = scratch$.string(" n ");")]
                            :args [(str n "$")]})
                 (inc i))

          (struct-layouts b)
          (let [{:keys [java value set]} (struct-layouts b)]
            (recur (next ps)
                   (conj out {:decl (str java " " n)
                              :prep [(str "MemorySegment " n "$ = scratch$.optional(" n " == null ? 0 : " n ", " n " != null);")]
                              :args [(str n "$")]})
                   (inc i)))

          :else
          (recur (next ps)
                 (conj out {:decl (str (:java (ctype t)) " " n) :prep [] :args [n]})
                 (inc i))))
      out)))

(defn checked-method
  "The checked wrapper of an int-returning function, or nil."
  [{:keys [name ret params] :as f}]
  (when (status-function? f)
    (let [outs (outs-of f)
          ins (vec (drop (count outs) params))
          cps (checked-params ins)
          decl (str/join ", " (map :decl cps))
          [rj read] (case (count outs)
                      0 ["void" nil]
                      1 (let [[j l] (out-java (first outs))] [j (str "out$.get(" l ", 0)")])
                      (when (every? #(= "long" (first (out-java %))) outs)
                        ["long[]" (str "new long[] {"
                                       (str/join ", " (map #(str "out$.get(JAVA_LONG, " (* 8 %) ")") (range (count outs))))
                                       "}")]))]
      (when rj
        (str "  public static " rj " " name "(" decl ") {\n"
             "    MlxRuntime.Scratch scratch$ = MlxRuntime.scratch();\n"
             (when (seq outs) (str "    MemorySegment out$ = scratch$.out(" (count outs) ");\n"))
             (apply str (for [cp cps st (:prep cp)] (str "    " st "\n")))
             "    int status$ = MlxRaw." name "("
             (str/join ", " (concat (map-indexed (fn [i _] (if (zero? i) "out$" (str "out$.asSlice(" (* 8 i) ")"))) outs)
                                    (mapcat :args cps)))
             ");\n"
             "    if (status$ != 0) throw MlxRuntime.error(\"" name "\");\n"
             (when read (str "    return " read ";\n"))
             "  }\n")))))

;; ---------------------------------------------------------------------------
;; Output
;; ---------------------------------------------------------------------------

(def preamble
  "// Generated by native/bindgen.clj from the mlx-c headers. Do not edit.
package gen.mlx;

import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import static java.lang.foreign.ValueLayout.*;
")

(defn pass-through
  "A Mlx method delegating to MlxRaw, for functions without a checked shape."
  [{:keys [name ret params]}]
  (let [rj (:java (ctype ret))
        names (param-names params)
        decl (str/join ", " (map (fn [p n] (str (:java (ctype (:type p))) " " n)) params names))]
    (str "  public static " rj " " name "(" decl ") {\n"
         "    " (if (= rj "void") "" "return ") "MlxRaw." name "(" (str/join ", " names) ");\n"
         "  }\n")))

(defn -main []
  (let [fs (functions)
        ok (filter supported? fs)
        skipped (remove supported? fs)
        checked (keep #(when-let [m (checked-method %)] [% m]) ok)
        checked-names (set (map (comp :name first) checked))
        passed (keep #(when-not (checked-names (:name %)) [% (pass-through %)]) ok)]
    (.mkdirs (io/file out-dir))
    (spit (io/file out-dir "MlxRaw.java")
          (str preamble
               "\n/** Raw mlx-c: one static method per C function (" (count ok) " of " (count fs) "). Handles are\n"
               " *  longs; pointers are MemorySegments; statuses are returned. */\n"
               "public final class MlxRaw {\n  private MlxRaw() {}\n\n"
               (str/join "\n" (map raw-method (sort-by :name ok)))
               "}\n"))
    (spit (io/file out-dir "Mlx.java")
          (str preamble
               "\n/** Checked mlx-c: the status is checked (MlxException on failure) and out-parameters\n"
               " *  are return values (" (count checked) " functions); the other " (count passed) " delegate to MlxRaw. */\n"
               "public final class Mlx {\n  private Mlx() {}\n\n"
               (str/join "\n" (map second (sort-by (comp :name first) (concat checked passed))))
               "}\n"))
    (println "functions:" (count fs) "raw:" (count ok) "checked:" (count checked))
    (println "skipped (unsupported types):" (str/join ", " (map :name skipped)))))

(-main)
