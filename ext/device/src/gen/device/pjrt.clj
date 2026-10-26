(ns gen.device.pjrt
  "XLA's PJRT C API (native/pjrt_c_api.h, API 0.76) by FFM: load a PJRT
   plugin, make a client, compile StableHLO, move arrays to and from the
   device, and execute. A research prototype (bean genclj-a1hu), outside the
   core.

     (with-open [c (pjrt/client)
                 exe (pjrt/compile c mlir)
                 a (pjrt/->device c (float-array [1 2]) [2])
                 b (pjrt/->device c (float-array [3 4]) [2])]
       (let [[out] (pjrt/execute exe [a b])]
         (with-open [out out] (pjrt/->host out))))
     ;; => {:dtype :f32, :shape [2], :data #floats [4.0 6.0]}

   The plugin is a PJRT plugin's shared library (CUDA: jax-cuda13-plugin's
   xla_cuda_plugin.so), given as -Dgen.xla.plugin=<path> or to `plugin`. It
   is loaded and initialized once per JVM.

   Every API function takes one args struct (struct_size first, then
   extension_start, here NULL) and returns a PJRT_Error*, NULL on success.
   An error is thrown as ex-info carrying PJRT's message, :pjrt/code (a
   keyword, :invalid-argument …) and :pjrt/fn. The structs' layouts are the
   C compiler's (`layout`, from native/pjrt_offsets.c), never counted by
   hand.

   Clients, executables and buffers are Closeable handles of native
   resources: close each (with-open), buffers and executables before their
   client. Using a closed handle throws. Host arrays are float[] (:f32),
   double[] (:f64), int[] (:i32, or the bits of :u32) and boolean[] (:bool),
   row-major."
  {:api :internal}
  (:refer-clojure :exclude [compile])
  (:import [java.io Closeable Writer]
           [java.lang.foreign Arena FunctionDescriptor Linker Linker$Option MemoryLayout MemorySegment
            SymbolLookup ValueLayout]
           [java.lang.invoke MethodHandle]
           [java.nio.charset StandardCharsets]
           [java.nio.file Path]
           [java.util HexFormat]
           [java.util.concurrent.atomic AtomicReference]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

;; ---------------------------------------------------------------------------
;; Layout
;; ---------------------------------------------------------------------------

;; The output of native/pjrt_offsets.c, compiled with gcc 13.3 on fenrir
;; (x86_64 Linux) against native/pjrt_c_api.h (2026-10-26); clang on arm64
;; macOS prints the same. A struct's :size is its *_STRUCT_SIZE (the value of
;; struct_size), :sizeof what to allocate; a field is [offset size kind].
;; Regenerate after changing the header, or to use another field.
(def ^:private layout
  '{:header [0 76]
    :structs
    {:PJRT_Api
     {:size 960 :sizeof 960
      :fields {:pjrt_api_version [16 24 :struct]
               :PJRT_Error_Destroy [40 8 :ptr]
               :PJRT_Error_Message [48 8 :ptr]
               :PJRT_Error_GetCode [56 8 :ptr]
               :PJRT_Plugin_Initialize [64 8 :ptr]
               :PJRT_Event_Destroy [80 8 :ptr]
               :PJRT_Event_Await [104 8 :ptr]
               :PJRT_Client_Create [120 8 :ptr]
               :PJRT_Client_Destroy [128 8 :ptr]
               :PJRT_Client_PlatformName [136 8 :ptr]
               :PJRT_Client_PlatformVersion [152 8 :ptr]
               :PJRT_Client_AddressableDevices [168 8 :ptr]
               :PJRT_Client_Compile [200 8 :ptr]
               :PJRT_Client_BufferFromHostBuffer [216 8 :ptr]
               :PJRT_DeviceDescription_Kind [248 8 :ptr]
               :PJRT_Device_GetDescription [272 8 :ptr]
               :PJRT_Executable_Destroy [360 8 :ptr]
               :PJRT_Executable_NumOutputs [392 8 :ptr]
               :PJRT_LoadedExecutable_Destroy [440 8 :ptr]
               :PJRT_LoadedExecutable_GetExecutable [448 8 :ptr]
               :PJRT_LoadedExecutable_Execute [480 8 :ptr]
               :PJRT_Buffer_Destroy [504 8 :ptr]
               :PJRT_Buffer_ElementType [512 8 :ptr]
               :PJRT_Buffer_Dimensions [520 8 :ptr]
               :PJRT_Buffer_ToHostBuffer [600 8 :ptr]}}
     :PJRT_Api_Version
     {:size 24 :sizeof 24
      :fields {:major_version [16 4 :int]
               :minor_version [20 4 :int]}}
     :PJRT_Error_Destroy_Args
     {:size 24 :sizeof 24
      :fields {:error [16 8 :ptr]}}
     :PJRT_Error_Message_Args
     {:size 40 :sizeof 40
      :fields {:error [16 8 :ptr]
               :message [24 8 :ptr]
               :message_size [32 8 :int]}}
     :PJRT_Error_GetCode_Args
     {:size 28 :sizeof 32
      :fields {:error [16 8 :ptr]
               :code [24 4 :int]}}
     :PJRT_Plugin_Initialize_Args
     {:size 16 :sizeof 16
      :fields {}}
     :PJRT_Event_Destroy_Args
     {:size 24 :sizeof 24
      :fields {:event [16 8 :ptr]}}
     :PJRT_Event_Await_Args
     {:size 24 :sizeof 24
      :fields {:event [16 8 :ptr]}}
     :PJRT_NamedValue
     {:size 56 :sizeof 56
      :fields {:name [16 8 :ptr]
               :name_size [24 8 :int]
               :type [32 4 :int]
               :string_value [40 8 :ptr]
               :int64_value [40 8 :int]
               :int64_array_value [40 8 :ptr]
               :float_value [40 4 :float]
               :bool_value [40 1 :int]
               :value_size [48 8 :int]}}
     :PJRT_Client_Create_Args
     {:size 88 :sizeof 88
      :fields {:create_options [16 8 :ptr]
               :num_options [24 8 :int]
               :client [64 8 :ptr]}}
     :PJRT_Client_Destroy_Args
     {:size 24 :sizeof 24
      :fields {:client [16 8 :ptr]}}
     :PJRT_Client_PlatformName_Args
     {:size 40 :sizeof 40
      :fields {:client [16 8 :ptr]
               :platform_name [24 8 :ptr]
               :platform_name_size [32 8 :int]}}
     :PJRT_Client_PlatformVersion_Args
     {:size 40 :sizeof 40
      :fields {:client [16 8 :ptr]
               :platform_version [24 8 :ptr]
               :platform_version_size [32 8 :int]}}
     :PJRT_Client_AddressableDevices_Args
     {:size 40 :sizeof 40
      :fields {:client [16 8 :ptr]
               :addressable_devices [24 8 :ptr]
               :num_addressable_devices [32 8 :int]}}
     :PJRT_Device_GetDescription_Args
     {:size 32 :sizeof 32
      :fields {:device [16 8 :ptr]
               :device_description [24 8 :ptr]}}
     :PJRT_DeviceDescription_Kind_Args
     {:size 40 :sizeof 40
      :fields {:device_description [16 8 :ptr]
               :device_kind [24 8 :ptr]
               :device_kind_size [32 8 :int]}}
     :PJRT_Program
     {:size 48 :sizeof 48
      :fields {:code [16 8 :ptr]
               :code_size [24 8 :int]
               :format [32 8 :ptr]
               :format_size [40 8 :int]}}
     :PJRT_Client_Compile_Args
     {:size 56 :sizeof 56
      :fields {:client [16 8 :ptr]
               :program [24 8 :ptr]
               :compile_options [32 8 :ptr]
               :compile_options_size [40 8 :int]
               :executable [48 8 :ptr]}}
     :PJRT_LoadedExecutable_Destroy_Args
     {:size 24 :sizeof 24
      :fields {:executable [16 8 :ptr]}}
     :PJRT_LoadedExecutable_GetExecutable_Args
     {:size 32 :sizeof 32
      :fields {:loaded_executable [16 8 :ptr]
               :executable [24 8 :ptr]}}
     :PJRT_Executable_Destroy_Args
     {:size 24 :sizeof 24
      :fields {:executable [16 8 :ptr]}}
     :PJRT_Executable_NumOutputs_Args
     {:size 32 :sizeof 32
      :fields {:executable [16 8 :ptr]
               :num_outputs [24 8 :int]}}
     :PJRT_ExecuteOptions
     {:size 88 :sizeof 88
      :fields {:launch_id [48 4 :int]}}
     :PJRT_LoadedExecutable_Execute_Args
     {:size 80 :sizeof 80
      :fields {:executable [16 8 :ptr]
               :options [24 8 :ptr]
               :argument_lists [32 8 :ptr]
               :num_devices [40 8 :int]
               :num_args [48 8 :int]
               :output_lists [56 8 :ptr]
               :device_complete_events [64 8 :ptr]
               :execute_device [72 8 :ptr]}}
     :PJRT_Client_BufferFromHostBuffer_Args
     {:size 120 :sizeof 120
      :fields {:client [16 8 :ptr]
               :data [24 8 :ptr]
               :type [32 4 :int]
               :dims [40 8 :ptr]
               :num_dims [48 8 :int]
               :byte_strides [56 8 :ptr]
               :num_byte_strides [64 8 :int]
               :host_buffer_semantics [72 4 :int]
               :device [80 8 :ptr]
               :memory [88 8 :ptr]
               :device_layout [96 8 :ptr]
               :done_with_host_buffer [104 8 :ptr]
               :buffer [112 8 :ptr]}}
     :PJRT_Buffer_Destroy_Args
     {:size 24 :sizeof 24
      :fields {:buffer [16 8 :ptr]}}
     :PJRT_Buffer_ElementType_Args
     {:size 28 :sizeof 32
      :fields {:buffer [16 8 :ptr]
               :type [24 4 :int]}}
     :PJRT_Buffer_Dimensions_Args
     {:size 40 :sizeof 40
      :fields {:buffer [16 8 :ptr]
               :dims [24 8 :ptr]
               :num_dims [32 8 :int]}}
     :PJRT_Buffer_ToHostBuffer_Args
     {:size 56 :sizeof 56
      :fields {:src [16 8 :ptr]
               :host_layout [24 8 :ptr]
               :dst [32 8 :ptr]
               :dst_size [40 8 :int]
               :event [48 8 :ptr]}}}
    :enums
    {:PJRT_Buffer_Type_PRED 1
     :PJRT_Buffer_Type_S32 4
     :PJRT_Buffer_Type_U32 8
     :PJRT_Buffer_Type_F32 11
     :PJRT_Buffer_Type_F64 12
     :PJRT_NamedValue_kString 0
     :PJRT_NamedValue_kInt64 1
     :PJRT_NamedValue_kInt64List 2
     :PJRT_NamedValue_kFloat 3
     :PJRT_NamedValue_kBool 4
     :PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes 1}})

(defn- struct-layout [s]
  (or (get-in layout [:structs s])
      (throw (ex-info (str "No layout for " s " (native/pjrt_offsets.c)") {:struct s}))))

(defn- field [s f]
  (or (get-in (struct-layout s) [:fields f])
      (throw (ex-info (str "No field " f " of " s " in the layout (native/pjrt_offsets.c)")
                      {:struct s :field f}))))

(defn- enum ^long [e] (long (get-in layout [:enums e])))

(def ^:private codes
  "PJRT_Error_Code: abseil's status codes, numbered so in the header."
  [nil :cancelled :unknown :invalid-argument :deadline-exceeded :not-found
   :already-exists :permission-denied :resource-exhausted :failed-precondition
   :aborted :out-of-range :unimplemented :internal :unavailable :data-loss
   :unauthenticated])

;; ---------------------------------------------------------------------------
;; Structs
;; ---------------------------------------------------------------------------

(defn- null? [^MemorySegment p] (zero? (.address p)))

(defn- put! [^MemorySegment seg [off size kind] v]
  (let [off (long off)]
    (case kind
      :ptr (.set seg ValueLayout/ADDRESS off ^MemorySegment v)
      :float (.set seg ValueLayout/JAVA_FLOAT off (float v))
      :int (let [v (if (boolean? v) (if v 1 0) v)]
             (case (long size)
               8 (.set seg ValueLayout/JAVA_LONG off (long v))
               4 (.set seg ValueLayout/JAVA_INT off (int v))
               1 (.set seg ValueLayout/JAVA_BYTE off (byte v)))))))

(defn- fetch [^MemorySegment seg [off size kind]]
  (let [off (long off)]
    (case kind
      :ptr (.get seg ValueLayout/ADDRESS off)
      :int (case (long size)
             8 (.get seg ValueLayout/JAVA_LONG off)
             4 (long (.get seg ValueLayout/JAVA_INT off))))))

(defn- write!
  "Writes struct s into seg: struct_size, the fields of `in` ({field value}),
   and zero (NULL) for the rest."
  ^MemorySegment [^MemorySegment seg s in]
  (.fill seg (byte 0))
  (.set seg ValueLayout/JAVA_LONG 0 (long (:size (struct-layout s))))
  (reduce-kv (fn [_ f v] (put! seg (field s f) v)) nil in)
  seg)

(defn- struct! ^MemorySegment [^Arena arena s in]
  (write! (.allocate arena (long (:sizeof (struct-layout s))) 8) s in))

(defn- utf8
  "s as a C string in arena: [pointer, its length in bytes without the NUL]."
  [^Arena arena ^String s]
  (let [b (.getBytes s StandardCharsets/UTF_8)]
    [(.allocateFrom arena s) (alength b)]))

(defn- string [^MemorySegment p ^long n]
  (String. ^bytes (.toArray (.reinterpret p n) ValueLayout/JAVA_BYTE) StandardCharsets/UTF_8))

;; ---------------------------------------------------------------------------
;; Calls
;; ---------------------------------------------------------------------------

(def ^:private void-fns
  "The API functions that return nothing (the rest return a PJRT_Error*)."
  #{:PJRT_Error_Destroy :PJRT_Error_Message})

(defn- call [plugin f ^MemorySegment args]
  (.invokeWithArguments ^MethodHandle (get-in plugin [:fns f]) ^objects (object-array [args])))

(defn- error
  "The PJRT_Error e as an ex-info; destroys e."
  [plugin f ^MemorySegment e]
  (with-open [arena (Arena/ofConfined)]
    (let [m (struct! arena :PJRT_Error_Message_Args {:error e})
          _ (call plugin :PJRT_Error_Message m)
          message (string (fetch m (field :PJRT_Error_Message_Args :message))
                          (fetch m (field :PJRT_Error_Message_Args :message_size)))
          g (struct! arena :PJRT_Error_GetCode_Args {:error e})
          _ (call plugin :PJRT_Error_GetCode g)
          code (fetch g (field :PJRT_Error_GetCode_Args :code))]
      (call plugin :PJRT_Error_Destroy (struct! arena :PJRT_Error_Destroy_Args {:error e}))
      (ex-info (str (name f) ": " message)
               {:pjrt/fn f :pjrt/code (get codes code code) :pjrt/message message}))))

(defn- invoke!
  "Calls the API function f with its args struct (f_Args) in arena, the
   fields of `in` written and the rest zero; throws the error it returns.
   Returns the fields `out` read back, as a map."
  ([plugin arena f in] (invoke! plugin arena f in []))
  ([plugin arena f in out]
   (let [s (keyword (str (name f) "_Args"))
         args (struct! arena s in)
         ^MemorySegment e (call plugin f args)]
     (when-not (null? e) (throw (error plugin f e)))
     (into {} (map (fn [o] [o (fetch args (field s o))])) out))))

(defn- await!
  "Waits for the event (if any), then destroys it; throws the error it ends
   with."
  [plugin arena event]
  (when-not (null? event)
    (try (invoke! plugin arena :PJRT_Event_Await {:event event})
         (finally (invoke! plugin arena :PJRT_Event_Destroy {:event event})))))

;; ---------------------------------------------------------------------------
;; The plugin
;; ---------------------------------------------------------------------------

(defn- downcall ^MethodHandle [^MemorySegment address ^FunctionDescriptor descriptor]
  (.downcallHandle (Linker/nativeLinker) address descriptor (make-array Linker$Option 0)))

(defn- load-plugin [^String path]
  (let [lookup (SymbolLookup/libraryLookup (Path/of path (make-array String 0)) (Arena/global))
        get-api (downcall (.orElseThrow (.find lookup "GetPjrtApi"))
                          (FunctionDescriptor/of ValueLayout/ADDRESS (make-array MemoryLayout 0)))
        api-layout (struct-layout :PJRT_Api)
        api (.reinterpret ^MemorySegment (.invokeWithArguments get-api ^objects (object-array 0))
                          (long (:sizeof api-layout)))
        struct-size (.get api ValueLayout/JAVA_LONG 0)
        [vo] (field :PJRT_Api :pjrt_api_version)
        version [(.get api ValueLayout/JAVA_INT (+ (long vo) (long (first (field :PJRT_Api_Version :major_version)))))
                 (.get api ValueLayout/JAVA_INT (+ (long vo) (long (first (field :PJRT_Api_Version :minor_version)))))]
        fns (dissoc (:fields api-layout) :pjrt_api_version)
        arg (into-array MemoryLayout [ValueLayout/ADDRESS])]
    (when-not (= (first version) (first (:header layout)))
      (throw (ex-info (str "The plugin's PJRT API is " version ", the header's " (:header layout))
                      {:plugin path :version version :header (:header layout)})))
    (doseq [[f [off]] fns
            :when (> (+ (long off) 8) struct-size)]
      (throw (ex-info (str "The plugin's PJRT_Api (" struct-size " bytes) has no " f)
                      {:plugin path :version version :fn f})))
    (let [handles (into {}
                        (for [[f [off]] fns]
                          [f (downcall (.get api ValueLayout/ADDRESS (long off))
                                       (if (void-fns f)
                                         (FunctionDescriptor/ofVoid arg)
                                         (FunctionDescriptor/of ValueLayout/ADDRESS arg)))]))
          plugin {:path path :version version :struct-size struct-size :fns handles}]
      (with-open [arena (Arena/ofConfined)]
        (invoke! plugin arena :PJRT_Plugin_Initialize {}))
      plugin)))

(defonce ^:private plugins (atom {}))

(defn plugin
  "The PJRT plugin at path (default: -Dgen.xla.plugin), loaded and initialized
   once per JVM: a map with :path, :version (the plugin's [major minor] PJRT
   API) and the API's functions."
  ([] (plugin (or (System/getProperty "gen.xla.plugin")
                  (throw (ex-info "No PJRT plugin: -Dgen.xla.plugin=<path to the plugin's .so>" {})))))
  ([path]
   (let [path (str (.toRealPath (Path/of ^String path (make-array String 0))
                                (make-array java.nio.file.LinkOption 0)))]
     @(get (swap! plugins update path #(or % (delay (load-plugin path)))) path))))

;; ---------------------------------------------------------------------------
;; Handles
;; ---------------------------------------------------------------------------

(def ^:private destructors
  {:client [:PJRT_Client_Destroy :client]
   :executable [:PJRT_LoadedExecutable_Destroy :executable]
   :buffer [:PJRT_Buffer_Destroy :buffer]})

(defn- destroy! [plugin kind p]
  (let [[f arg] (destructors kind)]
    (with-open [arena (Arena/ofConfined)]
      (invoke! plugin arena f {arg p}))))

(deftype Handle [plugin kind info ^AtomicReference ref]
  Closeable
  (close [_]
    (when-let [p (.getAndSet ref nil)]
      (destroy! plugin kind p)))
  clojure.lang.ILookup
  (valAt [_ k] (get info k))
  (valAt [_ k not-found] (get info k not-found)))

(defmethod print-method Handle [^Handle h ^Writer w]
  (.write w (str "#<pjrt " (name (.-kind h)) " " (pr-str (.-info h))
                 (when-not (.get ^AtomicReference (.-ref h)) " closed") ">")))

(defn- handle [plugin kind info p]
  (Handle. plugin kind info (AtomicReference. p)))

(defn- ptr ^MemorySegment [^Handle h]
  (or (.get ^AtomicReference (.-ref h))
      (throw (ex-info (str "The " (name (.-kind h)) " is closed") {:kind (.-kind h) :info (.-info h)}))))

(defn close
  "Destroys the handle's native resource (as .close does); closing twice does
   nothing."
  [^Handle h]
  (.close h))

;; ---------------------------------------------------------------------------
;; Clients
;; ---------------------------------------------------------------------------

(def default-options
  "Client options that take device memory as needed (XLA's BFC allocator
   growing), up to a tenth of the device's. XLA's default preallocates 75 %."
  {"allocator" "bfc" "preallocate" false "memory_fraction" 0.1})

(defn- named-values
  "options ({name value}) as an array of PJRT_NamedValue in arena: a value is
   a string, a boolean, an integer, a float or a sequence of integers."
  ^MemorySegment [^Arena arena options]
  (let [size (long (:sizeof (struct-layout :PJRT_NamedValue)))
        array (.allocate arena (max 1 (* size (count options))) 8)]
    (doseq [[^long i [k v]] (map-indexed vector options)
            :let [[nm n] (utf8 arena (name k))]]
      (write! (.asSlice array (* i size) size) :PJRT_NamedValue
              (merge {:name nm :name_size n :value_size 1}
                     (cond
                       (string? v) (let [[s n] (utf8 arena v)]
                                     {:type (enum :PJRT_NamedValue_kString) :string_value s :value_size n})
                       (boolean? v) {:type (enum :PJRT_NamedValue_kBool) :bool_value v}
                       (integer? v) {:type (enum :PJRT_NamedValue_kInt64) :int64_value v}
                       (float? v) {:type (enum :PJRT_NamedValue_kFloat) :float_value v}
                       (sequential? v) {:type (enum :PJRT_NamedValue_kInt64List)
                                        :int64_array_value (.allocateFrom arena ValueLayout/JAVA_LONG
                                                                          ^longs (long-array v))
                                        :value_size (count v)}
                       :else (throw (ex-info (str "A client option of no PJRT type: " k " " (pr-str v))
                                             {:option k :value v}))))))
    array))

(defn- client-info [plugin ^Arena arena c]
  (let [{name-p :platform_name name-n :platform_name_size}
        (invoke! plugin arena :PJRT_Client_PlatformName {:client c} [:platform_name :platform_name_size])
        {version-p :platform_version version-n :platform_version_size}
        (invoke! plugin arena :PJRT_Client_PlatformVersion {:client c} [:platform_version :platform_version_size])
        {:keys [addressable_devices num_addressable_devices]}
        (invoke! plugin arena :PJRT_Client_AddressableDevices {:client c}
                 [:addressable_devices :num_addressable_devices])
        _ (when (zero? (long num_addressable_devices))
            (throw (ex-info "The PJRT client has no addressable device" {})))
        device (.getAtIndex (.reinterpret ^MemorySegment addressable_devices 8) ValueLayout/ADDRESS 0)
        {:keys [device_description]}
        (invoke! plugin arena :PJRT_Device_GetDescription {:device device} [:device_description])
        {:keys [device_kind device_kind_size]}
        (invoke! plugin arena :PJRT_DeviceDescription_Kind {:device_description device_description}
                 [:device_kind :device_kind_size])]
    {:platform (string name-p name-n)
     :platform-version (string version-p version-n)
     :devices num_addressable_devices
     :device-kind (string device_kind device_kind_size)
     :device device}))

(defn client
  "A PJRT client of the plugin (default `(plugin)`), with create options
   ({name value}, default `default-options`). (:device c) is its first
   addressable device, where it compiles, places buffers and runs;
   :platform, :platform-version, :device-kind and :devices describe it."
  ([] (client (plugin) default-options))
  ([options] (client (plugin) options))
  ([plugin options]
   (with-open [arena (Arena/ofConfined)]
     (let [{c :client} (invoke! plugin arena :PJRT_Client_Create
                                {:create_options (named-values arena options) :num_options (count options)}
                                [:client])]
       (try (handle plugin :client (assoc (client-info plugin arena c) :options options) c)
            (catch Throwable t (destroy! plugin :client c) (throw t)))))))

;; ---------------------------------------------------------------------------
;; Executables
;; ---------------------------------------------------------------------------

(def ^:private compile-options
  "A serialized xla.CompileOptionsProto: executable_build_options
   {device_ordinal -1, num_replicas 1, num_partitions 1}."
  (.parseHex (HexFormat/of) "1a0f08ffffffffffffffffff0120012801"))

(defn- num-outputs ^long [plugin arena loaded]
  (let [{exe :executable} (invoke! plugin arena :PJRT_LoadedExecutable_GetExecutable
                                   {:loaded_executable loaded} [:executable])]
    (try (:num_outputs (invoke! plugin arena :PJRT_Executable_NumOutputs {:executable exe} [:num_outputs]))
         (finally (invoke! plugin arena :PJRT_Executable_Destroy {:executable exe})))))

(defn compile
  "Compiles a StableHLO module (MLIR text) for the client's device: an
   executable, whose (:outputs exe) is its number of outputs. A module XLA
   cannot parse or compile throws its message."
  [^Handle client ^String mlir]
  (let [plugin (.-plugin client)]
    (with-open [arena (Arena/ofConfined)]
      (let [[code code-n] (utf8 arena mlir)
            [fmt fmt-n] (utf8 arena "mlir")
            {exe :executable}
            (invoke! plugin arena :PJRT_Client_Compile
                     {:client (ptr client)
                      :program (struct! arena :PJRT_Program
                                        {:code code :code_size code-n :format fmt :format_size fmt-n})
                      :compile_options (.allocateFrom arena ValueLayout/JAVA_BYTE ^bytes compile-options)
                      :compile_options_size (alength ^bytes compile-options)}
                     [:executable])]
        (try (handle plugin :executable {:outputs (num-outputs plugin arena exe)} exe)
             (catch Throwable t (destroy! plugin :executable exe) (throw t)))))))

;; ---------------------------------------------------------------------------
;; Buffers
;; ---------------------------------------------------------------------------

(def ^:private dtypes
  {:f32 {:type :PJRT_Buffer_Type_F32 :bytes 4 :array float/1}
   :f64 {:type :PJRT_Buffer_Type_F64 :bytes 8 :array double/1}
   :i32 {:type :PJRT_Buffer_Type_S32 :bytes 4 :array int/1}
   :u32 {:type :PJRT_Buffer_Type_U32 :bytes 4 :array int/1}
   :bool {:type :PJRT_Buffer_Type_PRED :bytes 1 :array boolean/1}})

(def ^:private dtype-of-type
  (into {} (for [[d {:keys [type]}] dtypes] [(enum type) d])))

(defn- host-segment
  "The host array's elements, as the device stores them, in arena."
  ^MemorySegment [^Arena arena data]
  (condp instance? data
    float/1 (.allocateFrom arena ValueLayout/JAVA_FLOAT ^float/1 data)
    double/1 (.allocateFrom arena ValueLayout/JAVA_DOUBLE ^double/1 data)
    int/1 (.allocateFrom arena ValueLayout/JAVA_INT ^int/1 data)
    boolean/1 (let [^boolean/1 b data
                    n (alength b)
                    bytes (byte-array n)]
                (dotimes [i n] (when (aget b i) (aset bytes i (byte 1))))
                (.allocateFrom arena ValueLayout/JAVA_BYTE bytes))))

(defn- host-array
  "n elements of dtype from seg, as a host array."
  [^MemorySegment seg dtype ^long n]
  (case dtype
    :f32 (.toArray seg ValueLayout/JAVA_FLOAT)
    :f64 (.toArray seg ValueLayout/JAVA_DOUBLE)
    (:i32 :u32) (.toArray seg ValueLayout/JAVA_INT)
    :bool (let [^byte/1 bytes (.toArray seg ValueLayout/JAVA_BYTE)
                b (boolean-array n)]
            (dotimes [i n] (aset b i (not (zero? (aget bytes i)))))
            b)))

(defn- elements ^long [shape] (reduce (fn [^long n ^long d] (* n d)) 1 shape))

(defn- default-dtype [data]
  (or (some (fn [d] (when (instance? (:array (dtypes d)) data) d)) [:f32 :f64 :i32 :bool])
      (throw (ex-info (str "No PJRT dtype for host data of " (class data)) {:class (class data)}))))

(defn ->device
  "Copies a host array (row-major, of the given shape: a vector, [] for a
   scalar) to the client's device: a buffer. dtype is :f32, :f64, :i32,
   :u32 or :bool; it defaults from the array (float[], double[], int[],
   boolean[]), and :u32 takes an int[] of its bits."
  ([client data shape] (->device client data shape (default-dtype data)))
  ([^Handle client data shape dtype]
   (let [{:keys [type array]} (or (dtypes dtype)
                                  (throw (ex-info (str "Not a PJRT dtype: " dtype) {:dtype dtype})))
         plugin (.-plugin client)
         shape (vec shape)]
     (when-not (instance? array data)
       (throw (ex-info (str "A " dtype " buffer takes a " (.getSimpleName ^Class array) ", not a "
                            (.getSimpleName (class data)))
                       {:dtype dtype :class (class data)})))
     (let [n (java.lang.reflect.Array/getLength data)]
       (when-not (== (elements shape) n)
         (throw (ex-info (str "Shape " shape " needs " (elements shape) " elements, the array has " n)
                         {:shape shape :length n}))))
     (with-open [arena (Arena/ofConfined)]
       (let [{:keys [buffer done_with_host_buffer]}
             (invoke! plugin arena :PJRT_Client_BufferFromHostBuffer
                      {:client (ptr client)
                       :data (host-segment arena data)
                       :type (enum type)
                       :dims (.allocateFrom arena ValueLayout/JAVA_LONG ^longs (long-array shape))
                       :num_dims (count shape)
                       :host_buffer_semantics (enum :PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes)
                       :device (:device client)}
                      [:buffer :done_with_host_buffer])]
         (try (await! plugin arena done_with_host_buffer)
              (catch Throwable t (destroy! plugin :buffer buffer) (throw t)))
         (handle plugin :buffer {:dtype dtype :shape shape} buffer))))))

(defn- describe* [plugin arena ^Handle b]
  (let [{:keys [type]} (invoke! plugin arena :PJRT_Buffer_ElementType {:buffer (ptr b)} [:type])
        {:keys [dims num_dims]} (invoke! plugin arena :PJRT_Buffer_Dimensions {:buffer (ptr b)}
                                         [:dims :num_dims])
        n (long num_dims)
        dtype (dtype-of-type type)]
    (when-not dtype
      (throw (ex-info (str "A buffer of PJRT element type " type ", which gen.device.pjrt does not read")
                      {:type type})))
    {:dtype dtype
     :shape (if (zero? n) [] (vec (.toArray (.reinterpret ^MemorySegment dims (* 8 n)) ValueLayout/JAVA_LONG)))}))

(defn describe
  "The buffer's {:dtype :shape}, as PJRT has them."
  [^Handle b]
  (with-open [arena (Arena/ofConfined)]
    (describe* (.-plugin b) arena b)))

(defn ->host
  "Copies a buffer to the host: {:dtype :shape :data}, data a host array
   (float[], double[], int[] for :i32 and :u32's bits, boolean[])."
  [^Handle b]
  (let [plugin (.-plugin b)]
    (with-open [arena (Arena/ofConfined)]
      (let [{:keys [dtype shape]} (describe* plugin arena b)
            n (elements shape)
            size (* n (long (:bytes (dtypes dtype))))
            dst (.allocate arena (max 1 size) 8)
            {:keys [event]} (invoke! plugin arena :PJRT_Buffer_ToHostBuffer
                                     {:src (ptr b) :dst dst :dst_size size} [:event])]
        (await! plugin arena event)
        {:dtype dtype :shape shape :data (host-array (.asSlice dst 0 size) dtype n)}))))

;; ---------------------------------------------------------------------------
;; Execution
;; ---------------------------------------------------------------------------

(defn execute
  "Runs the executable on its client's device with argument buffers, and
   waits for it: its output buffers, a vector."
  [^Handle exe args]
  (let [plugin (.-plugin exe)
        n (long (:outputs exe))]
    (with-open [arena (Arena/ofConfined)]
      (let [arg-array (.allocate arena ValueLayout/ADDRESS (max 1 (count args)))
            _ (doseq [[^long i b] (map-indexed vector args)]
                (.setAtIndex arg-array ValueLayout/ADDRESS i (ptr b)))
            out-array (.allocate arena ValueLayout/ADDRESS (max 1 n))
            events (.allocate arena ValueLayout/ADDRESS 1)]
        (invoke! plugin arena :PJRT_LoadedExecutable_Execute
                 {:executable (ptr exe)
                  :options (struct! arena :PJRT_ExecuteOptions {})
                  :argument_lists (.allocateFrom arena ValueLayout/ADDRESS arg-array)
                  :num_devices 1
                  :num_args (count args)
                  :output_lists (.allocateFrom arena ValueLayout/ADDRESS out-array)
                  :device_complete_events events})
        (let [outs (mapv #(handle plugin :buffer {} (.getAtIndex out-array ValueLayout/ADDRESS (long %)))
                         (range n))]
          (try (await! plugin arena (.getAtIndex events ValueLayout/ADDRESS 0))
               (catch Throwable t (run! close outs) (throw t)))
          outs)))))
