package gen.mlx;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.StructLayout;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.lang.ref.PhantomReference;
import java.lang.ref.Reference;
import java.lang.ref.ReferenceQueue;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import static java.lang.foreign.ValueLayout.*;

/** The hand-written part of the mlx-c binding: loading the library, errors,
 *  per-thread scratch memory and closures (Java functions MLX can call). The
 *  generated classes Mlx and MlxRaw call into it. */
public final class MlxRuntime {
  private MlxRuntime() {}

  // ---- the library ------------------------------------------------------------

  static final Linker LINKER = Linker.nativeLinker();
  static final SymbolLookup LIB = load();

  /** The directory holding libmlxc: the system property gen.mlx.lib, the
   *  environment variable GEN_MLX_LIB, or native/build/<platform>/lib under
   *  the working directory (where native/build-mlx.sh puts it). */
  public static Path libraryDirectory() {
    String dir = System.getProperty("gen.mlx.lib");
    if (dir == null) dir = System.getenv("GEN_MLX_LIB");
    if (dir != null) return Path.of(dir);
    String os = System.getProperty("os.name").toLowerCase();
    String arch = System.getProperty("os.arch");
    arch = (arch.equals("aarch64") || arch.equals("arm64")) ? "arm64" : "x86_64";
    String platform = os.contains("mac") ? "macos-" + arch
        : "linux-" + (arch.equals("arm64") ? "aarch64" : "x86_64") + "-cpu";
    return Path.of(System.getProperty("user.dir"), "native", "build", platform, "lib");
  }

  private static SymbolLookup load() {
    String name = System.getProperty("os.name").toLowerCase().contains("mac") ? "libmlxc.dylib" : "libmlxc.so";
    Path lib = libraryDirectory().resolve(name);
    if (!Files.exists(lib)) {
      throw new UnsatisfiedLinkError("mlx-c not found at " + lib
          + " (build it with native/build-mlx.sh, or point -Dgen.mlx.lib at its directory)");
    }
    return SymbolLookup.libraryLookup(lib, Arena.global());
  }

  /** The symbols of libmlxc. */
  public static SymbolLookup library() {
    return LIB;
  }

  public static MethodHandle downcall(String name, FunctionDescriptor fd) {
    MemorySegment f = LIB.find(name).orElseThrow(() -> new UnsatisfiedLinkError("mlx-c has no symbol " + name));
    return LINKER.downcallHandle(f, fd);
  }

  /** invokeExact on a handle of the right type cannot fail but for a bug in
   *  the binding (upcalls catch everything). */
  static AssertionError unreachable(Throwable t) {
    return new AssertionError("mlx-c call failed in the binding", t);
  }

  // ---- by-value structs ----------------------------------------------------------

  /** mlx_optional_int and mlx_optional_dtype: { int value; bool has_value; }. */
  public static final StructLayout OPTIONAL_INT =
      MemoryLayout.structLayout(JAVA_INT.withName("value"), JAVA_BOOLEAN.withName("has_value"), MemoryLayout.paddingLayout(3));
  /** mlx_optional_float: { float value; bool has_value; }. */
  public static final StructLayout OPTIONAL_FLOAT =
      MemoryLayout.structLayout(JAVA_FLOAT.withName("value"), JAVA_BOOLEAN.withName("has_value"), MemoryLayout.paddingLayout(3));

  // ---- errors ----------------------------------------------------------------------

  private static final ThreadLocal<String> LAST_ERROR = new ThreadLocal<>();
  private static final ThreadLocal<Throwable> UPCALL_ERROR = new ThreadLocal<>();

  /** mlx-c's error handler (the default prints and calls exit(-1)): keeps
   *  the message for the failing call to throw. */
  private static void onError(MemorySegment msg, MemorySegment data) {
    try {
      LAST_ERROR.set(msg.reinterpret(Long.MAX_VALUE).getString(0));
    } catch (Throwable t) {
      // nothing may escape an upcall
    }
  }

  /** The exception for a failed call to the function named: what a Java
   *  function MLX called threw, if that is why it failed, or an
   *  MlxException with MLX's message. */
  public static RuntimeException error(String function) {
    Throwable t = UPCALL_ERROR.get();
    String msg = LAST_ERROR.get();
    UPCALL_ERROR.remove();
    LAST_ERROR.remove();
    if (t != null) {
      if (t instanceof RuntimeException r) return r;
      if (t instanceof Error e) throw e;
      return new RuntimeException(t);
    }
    return new MlxException(function, msg == null ? "unknown error" : msg);
  }

  // ---- scratch memory for arguments and out-parameters --------------------------

  /** Per-thread native memory for one call's out-parameters and copied
   *  arguments. A Java function MLX calls back runs with its own (child)
   *  scratch, since the call it runs inside still owns the outer one. */
  public static final class Scratch {
    private final Arena arena = Arena.ofAuto();
    private final MemorySegment out = arena.allocate(64, 8);
    private MemorySegment buf = arena.allocate(1024, 8);
    private long pos;
    private Scratch child;

    Scratch reset() {
      pos = 0;
      return this;
    }

    Scratch child() {
      if (child == null) child = new Scratch();
      return child;
    }

    /** The out-parameter block, its first n slots zeroed: mlx-c assigns into
     *  a non-null handle rather than creating one. */
    public MemorySegment out(int n) {
      out.asSlice(0, 8L * n).fill((byte) 0);
      return out;
    }

    private MemorySegment bump(long bytes) {
      long aligned = (bytes + 7) & ~7L;
      if (pos + aligned > buf.byteSize()) {
        if (aligned > buf.byteSize() / 2) return arena.allocate(aligned, 8);
        buf = arena.allocate(buf.byteSize() * 2, 8);
        pos = 0;
      }
      MemorySegment s = buf.asSlice(pos, aligned);
      pos += aligned;
      return s;
    }

    public MemorySegment ints(int[] a) {
      MemorySegment s = bump(4L * Math.max(1, a.length));
      MemorySegment.copy(a, 0, s, JAVA_INT, 0, a.length);
      return s;
    }

    public MemorySegment longs(long[] a) {
      MemorySegment s = bump(8L * Math.max(1, a.length));
      MemorySegment.copy(a, 0, s, JAVA_LONG, 0, a.length);
      return s;
    }

    public MemorySegment string(String str) {
      byte[] b = str.getBytes(StandardCharsets.UTF_8);
      MemorySegment s = bump(b.length + 1);
      MemorySegment.copy(b, 0, s, JAVA_BYTE, 0, b.length);
      s.set(JAVA_BYTE, b.length, (byte) 0);
      return s;
    }

    public MemorySegment optional(int value, boolean has) {
      MemorySegment s = bump(8);
      s.set(JAVA_INT, 0, value);
      s.set(JAVA_BOOLEAN, 4, has);
      return s.asSlice(0, 8);
    }

    public MemorySegment optional(float value, boolean has) {
      MemorySegment s = bump(8);
      s.set(JAVA_FLOAT, 0, value);
      s.set(JAVA_BOOLEAN, 4, has);
      return s.asSlice(0, 8);
    }
  }

  private static final ThreadLocal<Scratch> SCRATCH = ThreadLocal.withInitial(Scratch::new);

  public static Scratch scratch() {
    return SCRATCH.get().reset();
  }

  // ---- array lifetime --------------------------------------------------------------

  /** A phantom reference to a Java value owning an MLX array handle, kept
   *  reachable in an intrusive doubly-linked list (as the JDK's Cleaner
   *  does: no hashing, O(1) insert and remove). */
  private static final class Owned extends PhantomReference<Object> {
    final long handle;
    Owned prev, next;

    Owned(Object owner, long handle) {
      super(owner, OWNED_QUEUE);
      this.handle = handle;
    }
  }

  private static final ReferenceQueue<Object> OWNED_QUEUE = new ReferenceQueue<>();
  private static final Object OWNED_LOCK = new Object();
  private static final Owned OWNED_HEAD = new Owned(null, 0);
  static {
    OWNED_HEAD.prev = OWNED_HEAD;
    OWNED_HEAD.next = OWNED_HEAD;
  }
  private static final AtomicLong LIVE = new AtomicLong();
  private static final AtomicLong ADOPTED = new AtomicLong();

  /** Collect when this many arrays are live (Metal allows about 500,000
   *  buffers in all; each array with data holds one): the system property
   *  gen.mlx.gc-arrays, default 100,000. The threshold rises to twice the
   *  arrays still live after a collection, so a large working set does not
   *  collect on every array. */
  private static final long BASE_ARRAYS = Long.getLong("gen.mlx.gc-arrays", 100_000L);
  /** Collect when MLX's active memory exceeds this many bytes: gen.mlx.gc-bytes,
   *  default a quarter of the physical memory. */
  private static final long BASE_BYTES = Long.getLong("gen.mlx.gc-bytes", physicalMemory() / 4);
  private static volatile long arrayThreshold = BASE_ARRAYS;
  private static volatile long byteThreshold = BASE_BYTES;
  private static final AtomicLong COLLECTIONS = new AtomicLong();

  private static long physicalMemory() {
    try {
      return ((com.sun.management.OperatingSystemMXBean)
          java.lang.management.ManagementFactory.getOperatingSystemMXBean()).getTotalMemorySize();
    } catch (Throwable t) {
      return 8L << 30;
    }
  }

  /** Make owner (a Java value) own the array handle: it is freed once owner
   *  is unreachable, by whichever thread next makes an array or collects.
   *  Frees only drop a reference: a graph that still needs the array keeps
   *  it. Collects (System.gc) under array or memory pressure, since the GC
   *  does not see MLX's native memory. */
  public static void adopt(Object owner, long handle) {
    Owned o = new Owned(owner, handle);
    synchronized (OWNED_LOCK) {
      o.next = OWNED_HEAD.next;
      o.prev = OWNED_HEAD;
      OWNED_HEAD.next.prev = o;
      OWNED_HEAD.next = o;
    }
    long live = LIVE.incrementAndGet();
    long n = ADOPTED.incrementAndGet();
    if ((n & 255) == 0) drain();
    if (live > arrayThreshold || ((n & 4095) == 0 && activeMemory() > byteThreshold)) collect();
  }

  /** Free the handles of owners found unreachable; returns how many. */
  public static int drain() {
    int freed = 0;
    Reference<?> r;
    while ((r = OWNED_QUEUE.poll()) != null) freed += release((Owned) r);
    return freed;
  }

  private static int release(Owned o) {
    synchronized (OWNED_LOCK) {
      if (o.next == null) return 0;
      o.prev.next = o.next;
      o.next.prev = o.prev;
      o.next = null;
      o.prev = null;
    }
    MlxRaw.mlx_array_free(o.handle);
    LIVE.decrementAndGet();
    return 1;
  }

  /** Run the GC and free what it found unreachable, then reset the
   *  thresholds above what is still live (but below Metal's limit). */
  public static synchronized void collect() {
    if (LIVE.get() <= arrayThreshold && activeMemory() <= byteThreshold) return;
    COLLECTIONS.incrementAndGet();
    System.gc();
    // the reference handler enqueues what the GC found after the GC returns:
    // free until the queue stays empty for a moment
    try {
      Reference<?> r;
      while ((r = OWNED_QUEUE.remove(10)) != null) {
        release((Owned) r);
        drain();
      }
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
    }
    arrayThreshold = Math.min(MAX_ARRAYS, Math.max(BASE_ARRAYS, 2 * LIVE.get()));
    byteThreshold = Math.max(BASE_BYTES, 2 * activeMemory());
  }

  /** The highest array threshold: 60% of Metal's buffer limit (the device's
   *  resource_limit; MLX on macOS allocates every buffer through Metal, also
   *  on the CPU device), or 300,000 where there is none. */
  private static final long MAX_ARRAYS = maxArrays();

  private static long maxArrays() {
    try {
      Scratch s = scratch();
      long dev = MlxRaw.mlx_device_new_type(1, 0);
      MemorySegment info = s.out(1);
      if (MlxRaw.mlx_device_info_get(info, dev) == 0) {
        long handle = info.get(JAVA_LONG, 0);
        MemorySegment key = s.string("resource_limit");
        MemorySegment has = s.out(1);
        if (MlxRaw.mlx_device_info_has_key(has, handle, key) == 0 && has.get(JAVA_BOOLEAN, 0)) {
          MemorySegment v = s.out(1);
          if (MlxRaw.mlx_device_info_get_size(v, handle, key) == 0) {
            return Math.max(BASE_ARRAYS, (long) (0.6 * v.get(JAVA_LONG, 0)));
          }
        }
      }
    } catch (Throwable t) {
      // no GPU device: no Metal limit
    }
    return Math.max(BASE_ARRAYS, 300_000L);
  }

  private static long activeMemory() {
    MemorySegment o = scratch().out(1);
    return MlxRaw.mlx_get_active_memory(o) == 0 ? o.get(JAVA_LONG, 0) : 0;
  }

  /** {live arrays, collections so far}, for tests and diagnostics. */
  public static long[] lifetimeStats() {
    return new long[] {LIVE.get(), COLLECTIONS.get()};
  }

  // ---- closures: Java functions MLX calls --------------------------------------------

  /** A function of arrays to arrays (handles), for mlx_closure. */
  @FunctionalInterface
  public interface Closure {
    long[] apply(long[] inputs) throws Throwable;
  }

  /** A custom VJP, for mlx_closure_custom: cotangents of the primals from
   *  the primals, the output cotangents and the outputs. */
  @FunctionalInterface
  public interface CustomVjp {
    long[] apply(long[] primals, long[] cotangents, long[] outputs) throws Throwable;
  }

  private static final ConcurrentHashMap<Long, Object> REGISTRY = new ConcurrentHashMap<>();
  private static final AtomicLong NEXT_ID = new AtomicLong(1);

  /** The handles of the arrays in an mlx_vector_array (new references). */
  public static long[] vectorArrays(long vec) {
    int n = (int) MlxRaw.mlx_vector_array_size(vec);
    long[] r = new long[n];
    for (int i = 0; i < n; i++) r[i] = Mlx.mlx_vector_array_get(vec, (long) i);
    return r;
  }

  /** A new mlx_vector_array of the arrays (MLX takes its own references). */
  public static long vectorOf(long[] arrays) {
    Scratch s = scratch();
    return MlxRaw.mlx_vector_array_new_data(s.longs(arrays), (long) arrays.length);
  }

  private static void setResult(MemorySegment res, long[] outs) {
    Scratch s = scratch();
    int status = MlxRaw.mlx_vector_array_set_data(res, s.longs(outs), (long) outs.length);
    if (status != 0) throw error("mlx_vector_array_set_data");
  }

  /** int fun(mlx_vector_array* res, const mlx_vector_array input, void* payload) */
  private static int closureCall(MemorySegment res, long input, MemorySegment payload) {
    Scratch outer = SCRATCH.get();
    SCRATCH.set(outer.child());
    try {
      Closure f = (Closure) REGISTRY.get(payload.address());
      setResult(res.reinterpret(8), f.apply(vectorArrays(input)));
      return 0;
    } catch (Throwable t) {
      UPCALL_ERROR.set(t);
      return 1;
    } finally {
      SCRATCH.set(outer);
    }
  }

  /** int fun(mlx_vector_array* res, primals, cotangents, outputs, void* payload) */
  private static int customVjpCall(MemorySegment res, long primals, long cotangents, long outputs, MemorySegment payload) {
    Scratch outer = SCRATCH.get();
    SCRATCH.set(outer.child());
    try {
      CustomVjp f = (CustomVjp) REGISTRY.get(payload.address());
      setResult(res.reinterpret(8), f.apply(vectorArrays(primals), vectorArrays(cotangents), vectorArrays(outputs)));
      return 0;
    } catch (Throwable t) {
      UPCALL_ERROR.set(t);
      return 1;
    } finally {
      SCRATCH.set(outer);
    }
  }

  /** The closure's payload is released (MLX holds it as long as any compiled
   *  or transformed function needs it). */
  private static void release(MemorySegment payload) {
    try {
      REGISTRY.remove(payload.address());
    } catch (Throwable t) {
      // nothing may escape an upcall
    }
  }

  private static final MemorySegment CLOSURE_STUB;
  private static final MemorySegment CUSTOM_VJP_STUB;
  private static final MemorySegment RELEASE_STUB;

  static {
    try {
      MethodHandles.Lookup lk = MethodHandles.lookup();
      MemorySegment errorStub = LINKER.upcallStub(
          lk.findStatic(MlxRuntime.class, "onError",
              MethodType.methodType(void.class, MemorySegment.class, MemorySegment.class)),
          FunctionDescriptor.ofVoid(ADDRESS, ADDRESS), Arena.global());
      MlxRaw.mlx_set_error_handler(errorStub, MemorySegment.NULL, MemorySegment.NULL);
      CLOSURE_STUB = LINKER.upcallStub(
          lk.findStatic(MlxRuntime.class, "closureCall",
              MethodType.methodType(int.class, MemorySegment.class, long.class, MemorySegment.class)),
          FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS), Arena.global());
      CUSTOM_VJP_STUB = LINKER.upcallStub(
          lk.findStatic(MlxRuntime.class, "customVjpCall",
              MethodType.methodType(int.class, MemorySegment.class, long.class, long.class, long.class, MemorySegment.class)),
          FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, JAVA_LONG, JAVA_LONG, ADDRESS), Arena.global());
      RELEASE_STUB = LINKER.upcallStub(
          lk.findStatic(MlxRuntime.class, "release", MethodType.methodType(void.class, MemorySegment.class)),
          FunctionDescriptor.ofVoid(ADDRESS), Arena.global());
    } catch (ReflectiveOperationException e) {
      throw new ExceptionInInitializerError(e);
    }
  }

  private static long register(Object f) {
    long id = NEXT_ID.getAndIncrement();
    REGISTRY.put(id, f);
    return id;
  }

  /** An mlx_closure calling f. Free it with mlx_closure_free; f is released
   *  when MLX no longer needs it. */
  public static long closure(Closure f) {
    return MlxRaw.mlx_closure_new_func_payload(CLOSURE_STUB, MemorySegment.ofAddress(register(f)), RELEASE_STUB);
  }

  /** An mlx_closure_custom calling f (for mlx_custom_vjp). */
  public static long customVjp(CustomVjp f) {
    return MlxRaw.mlx_closure_custom_new_func_payload(CUSTOM_VJP_STUB, MemorySegment.ofAddress(register(f)), RELEASE_STUB);
  }

  /** Number of Java functions MLX holds (for tests of their release). */
  public static int registeredClosures() {
    return REGISTRY.size();
  }
}
