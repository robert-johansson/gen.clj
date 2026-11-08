package gen.llama;

import gen.llama.ffi.Llama;
import gen.llama.ffi.llama_batch;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import static java.lang.foreign.ValueLayout.*;

/** The hand-written part of the llama.cpp binding: loading the library (and
 *  checking its version), llama.cpp's log, and the loops the engine runs over
 *  a batch or a row of logits on every decode. The generated classes in
 *  gen.llama.ffi (jextract, native/jextract-llama.sh) name no library: they
 *  find llama.cpp's symbols in the libraries their class loader has loaded,
 *  so load() must run before any of them is called. */
public final class LlamaRuntime {
  private LlamaRuntime() {}

  /** The llama.cpp release the binding was generated from. Struct layouts
   *  (passed by value) change between releases, so another version is
   *  refused rather than called. */
  public static final String VERSION = "0.9.0";

  // ---- the library ------------------------------------------------------------

  /** The directory holding libllama: the system property gen.llama.lib, the
   *  environment variable GEN_LLAMA_LIB, or native/build/llama/<platform>/lib
   *  under the working directory (where native/build-llama.sh puts it). */
  public static Path libraryDirectory() {
    String dir = System.getProperty("gen.llama.lib");
    if (dir == null) dir = System.getenv("GEN_LLAMA_LIB");
    if (dir != null) return Path.of(dir);
    String os = System.getProperty("os.name").toLowerCase();
    String arch = System.getProperty("os.arch");
    boolean arm = arch.equals("aarch64") || arch.equals("arm64");
    String platform = os.contains("mac") ? "macos-arm64" : "linux-" + (arm ? "aarch64" : "x86_64") + "-cpu";
    return Path.of(System.getProperty("user.dir"), "native", "build", "llama", platform, "lib");
  }

  private static boolean loaded;

  /** Load libllama (once), check its version, route its log (below) and
   *  initialize its backends. Throws UnsatisfiedLinkError when the library
   *  is missing or of another version, IllegalArgumentException when
   *  gen.llama.log is not a level. */
  public static synchronized void load() {
    if (loaded) return;
    Log.threshold = Log.level(System.getProperty("gen.llama.log", "warn"));
    String name = System.getProperty("os.name").toLowerCase().contains("mac") ? "libllama.dylib" : "libllama.so";
    Path lib = libraryDirectory().resolve(name);
    if (!Files.exists(lib)) {
      throw new UnsatisfiedLinkError("llama.cpp not found at " + lib
          + " (build it with native/build-llama.sh, or point -Dgen.llama.lib at its directory)");
    }
    System.load(lib.toAbsolutePath().toString());
    String version = Llama.llama_version().getString(0);
    if (!VERSION.equals(version)) {
      throw new UnsatisfiedLinkError("llama.cpp at " + lib + " is version " + version + ", but the binding is "
          + VERSION + "'s (rebuild with native/build-llama.sh, or regenerate the binding with native/jextract-llama.sh)");
    }
    Llama.llama_log_set(Log.STUB, MemorySegment.NULL);
    Llama.llama_backend_init();
    loaded = true;
  }

  // ---- the log -------------------------------------------------------------------

  /** llama.cpp's and ggml's log, through one callback: messages at or above
   *  the level of the system property gen.llama.log (none, error, warn, info
   *  or debug; default warn) go to System.err. The last error message of each
   *  thread is kept for the exception of the call that logged it. */
  static final class Log {
    // ggml_log_level: NONE 0, DEBUG 1, INFO 2, WARN 3, ERROR 4, CONT 5 (continues the previous message)
    static volatile int threshold = 3;

    static int level(String level) {
      return switch (level) {
        case "none" -> Integer.MAX_VALUE;
        case "error" -> 4;
        case "warn" -> 3;
        case "info" -> 2;
        case "debug" -> 1;
        default -> throw new IllegalArgumentException(
            "gen.llama.log is none, error, warn, info or debug, not " + level);
      };
    }

    static final ThreadLocal<int[]> LEVEL = ThreadLocal.withInitial(() -> new int[] {0});
    static final ThreadLocal<StringBuilder> CAPTURE = new ThreadLocal<>();
    static final ThreadLocal<StringBuilder> ERROR = ThreadLocal.withInitial(StringBuilder::new);
    static final ThreadLocal<String> LAST_ERROR = new ThreadLocal<>();

    /** void (*ggml_log_callback)(enum ggml_log_level level, const char * text, void * user_data) */
    static void log(int level, MemorySegment text, MemorySegment userData) {
      try {
        int[] current = LEVEL.get();
        if (level != 5) current[0] = level;
        int effective = current[0];
        String s = text.reinterpret(Long.MAX_VALUE).getString(0, StandardCharsets.UTF_8);
        StringBuilder capture = CAPTURE.get();
        if (capture != null) capture.append(s);
        if (effective == 4) {
          StringBuilder b = ERROR.get();
          if (level != 5) b.setLength(0);
          b.append(s);
          LAST_ERROR.set(b.toString().strip());
        }
        if (effective >= threshold && effective <= 4) System.err.print("llama.cpp: " + s);
      } catch (Throwable t) {
        // nothing may escape an upcall
      }
    }

    static final MemorySegment STUB;

    static {
      try {
        STUB = Linker.nativeLinker().upcallStub(
            MethodHandles.lookup().findStatic(Log.class, "log",
                MethodType.methodType(void.class, int.class, MemorySegment.class, MemorySegment.class)),
            FunctionDescriptor.ofVoid(JAVA_INT, ADDRESS, ADDRESS), Arena.global());
      } catch (ReflectiveOperationException e) {
        throw new ExceptionInInitializerError(e);
      }
    }
  }

  /** The last error llama.cpp logged on this thread since clearLastError,
   *  or null. */
  public static String lastError() {
    return Log.LAST_ERROR.get();
  }

  public static void clearLastError() {
    Log.LAST_ERROR.remove();
  }

  /** Keep everything llama.cpp logs on this thread, at every level and
   *  whatever gen.llama.log prints, until endCapture. */
  public static void startCapture() {
    Log.CAPTURE.set(new StringBuilder());
  }

  /** What llama.cpp logged on this thread since startCapture, and stop. */
  public static String endCapture() {
    StringBuilder b = Log.CAPTURE.get();
    Log.CAPTURE.remove();
    return b == null ? "" : b.toString();
  }

  // ---- batches -----------------------------------------------------------------------

  /** A llama_batch from llama_batch_init(capacity, 0, 1), with views of its
   *  arrays, which it allocated once, made once: one entry per token, each
   *  token in one sequence. */
  public static final class Batch {
    /** The struct, passed by value to llama_decode and llama_batch_free. */
    public final MemorySegment struct;
    private final int capacity;
    private final MemorySegment token, pos, nSeqId, logits;
    private final MemorySegment[] seqId;

    public Batch(MemorySegment struct, int capacity) {
      this.struct = struct;
      this.capacity = capacity;
      token = llama_batch.token(struct).reinterpret(4L * capacity);
      pos = llama_batch.pos(struct).reinterpret(4L * capacity);
      nSeqId = llama_batch.n_seq_id(struct).reinterpret(4L * capacity);
      logits = llama_batch.logits(struct).reinterpret(capacity);
      MemorySegment seqs = llama_batch.seq_id(struct).reinterpret(8L * capacity);
      seqId = new MemorySegment[capacity];
      for (int i = 0; i < capacity; i++) {
        seqId[i] = seqs.getAtIndex(ADDRESS, i).reinterpret(4);
        nSeqId.setAtIndex(JAVA_INT, i, 1);
      }
    }

    private void set(int i, int tok, int p, int seq, boolean out) {
      token.setAtIndex(JAVA_INT, i, tok);
      pos.setAtIndex(JAVA_INT, i, p);
      seqId[i].set(JAVA_INT, 0, seq);
      logits.set(JAVA_BYTE, i, (byte) (out ? 1 : 0));
    }

    /** One token per sequence: tokens[i] at position pos[i] of sequence
     *  seqs[i], each with its logits output. */
    public void step(int[] tokens, int[] pos, int[] seqs) {
      int n = tokens.length;
      if (n > capacity || pos.length != n || seqs.length != n) {
        throw new IllegalArgumentException("a step of " + n + " tokens for a batch of " + capacity);
      }
      for (int i = 0; i < n; i++) set(i, tokens[i], pos[i], seqs[i], true);
      llama_batch.n_tokens(struct, n);
    }

    /** A run of one sequence: tokens[from .. from+n) at positions pos0,
     *  pos0+1, … of sequence seq, with logits output for every token (all)
     *  or only the last. */
    public void run(int[] tokens, int from, int n, int pos0, int seq, boolean all) {
      if (n < 1 || n > capacity || from < 0 || from + n > tokens.length) {
        throw new IllegalArgumentException("a run of " + n + " tokens from " + from + " for a batch of " + capacity);
      }
      for (int i = 0; i < n; i++) set(i, tokens[from + i], pos0 + i, seq, all || i == n - 1);
      llama_batch.n_tokens(struct, n);
    }
  }

  // ---- recorded tensors -----------------------------------------------------------------

  /** A recorder of a context's intermediate tensors, for checks (experimental):
   *  through llama.cpp's eval callback (llama_context_params.cb_eval, the
   *  ggml_backend_sched_eval_callback stub), the tensors of each graph the
   *  context computes that llama.cpp names base-layer (llm_graph_context's
   *  cb: "ffn_moe_topk-3", the experts layer 3 of a mixture of experts picks)
   *  for one of the bases given, copied to the host (ggml_backend_tensor_get)
   *  as the scheduler computes them. A recorded tensor is F32 or I32, of at
   *  most 2 dimensions, its rows contiguous (a view's rows may be strided, as
   *  ffn_moe_topk's, the first k of an argsort's).
   *
   *  The callback stops the scheduler at each tensor asked for (it computes
   *  the graph in pieces, and synchronizes), so a recording context decodes
   *  more slowly, and may round otherwise than one without (no fusion across
   *  the pieces): a check reads the tensors and the logits of the same
   *  decodes. Nothing may escape an upcall: a failure is kept, the callback
   *  stops the graph, and drain throws it. One thread decodes a context, and
   *  the callback runs on it (the scheduler calls it between the pieces). */
  public static final class Recorder {
    /** One tensor of a decode: its base name, layer, and rows × cols values
     *  (ints for I32, floats for F32), row by row. */
    public static final class Entry {
      public final String base;
      public final int layer;
      public final int rows, cols;
      public final int[] ints;
      public final float[] floats;

      Entry(String base, int layer, int rows, int cols, int[] ints, float[] floats) {
        this.base = base;
        this.layer = layer;
        this.rows = rows;
        this.cols = cols;
        this.ints = ints;
        this.floats = floats;
      }
    }

    // struct ggml_tensor (ggml.h of v0.9.0): enum ggml_type type at 0, then
    // the buffer pointer, int64_t ne[4] at 16 and size_t nb[4] at 48. Each
    // read checks them against ggml's own accessors (count, element size and
    // bytes), so a layout of another release is loud, not misread.
    private static final long TYPE = 0, NE = 16, NB = 48, PREFIX = 80;
    private static final int GGML_TYPE_F32 = 0, GGML_TYPE_I32 = 26;

    private final java.util.Set<String> bases;
    private final java.util.ArrayList<Entry> entries = new java.util.ArrayList<>();
    private Throwable failure;

    /** The callback's stub, for llama_context_params.cb_eval: valid while
     *  arena is open (close it after freeing the context). */
    public final MemorySegment stub;

    public Recorder(java.util.Collection<String> bases, Arena arena) {
      this.bases = java.util.Set.copyOf(bases);
      try {
        stub = Linker.nativeLinker().upcallStub(
            MethodHandles.lookup().findVirtual(Recorder.class, "eval",
                MethodType.methodType(boolean.class, MemorySegment.class, boolean.class, MemorySegment.class))
                .bindTo(this),
            FunctionDescriptor.of(JAVA_BOOLEAN, ADDRESS, JAVA_BOOLEAN, ADDRESS), arena);
      } catch (ReflectiveOperationException e) {
        throw new IllegalStateException(e);
      }
    }

    /** bool (*ggml_backend_sched_eval_callback)(struct ggml_tensor * t, bool ask, void * user_data):
     *  asked (ask), whether t is wanted; then, t computed, record it (and
     *  go on). */
    boolean eval(MemorySegment t, boolean ask, MemorySegment userData) {
      try {
        if (failure != null) return false;
        String name = Llama.ggml_get_name(t).getString(0, StandardCharsets.UTF_8);
        int dash = name.lastIndexOf('-');
        if (dash <= 0 || dash == name.length() - 1 || !bases.contains(name.substring(0, dash))) return false;
        for (int i = dash + 1; i < name.length(); i++) {
          if (!Character.isDigit(name.charAt(i))) return false;
        }
        if (ask) return true;
        entries.add(read(t, name.substring(0, dash), Integer.parseInt(name.substring(dash + 1)), name));
        return true;
      } catch (Throwable e) {
        failure = e;
        return false;
      }
    }

    private static Entry read(MemorySegment tensor, String base, int layer, String name) {
      MemorySegment t = tensor.reinterpret(PREFIX);
      int type = t.get(JAVA_INT, TYPE);
      long[] ne = new long[4], nb = new long[4];
      for (int i = 0; i < 4; i++) {
        ne[i] = t.get(JAVA_LONG, NE + 8L * i);
        nb[i] = t.get(JAVA_LONG, NB + 8L * i);
      }
      if (type != GGML_TYPE_F32 && type != GGML_TYPE_I32) {
        throw new IllegalStateException(name + " is of ggml type " + type + ", not F32 or I32");
      }
      long n = ne[0] * ne[1] * ne[2] * ne[3];
      long bytes = Llama.ggml_nbytes(tensor);
      long span = 4;
      for (int i = 0; i < 4; i++) span += (ne[i] - 1) * nb[i];
      if (n != Llama.ggml_nelements(tensor) || Llama.ggml_element_size(tensor) != 4 || nb[0] != 4
          || (n > 0 && span != bytes)) {
        throw new IllegalStateException("the layout of " + name + " (ne " + java.util.Arrays.toString(ne) + ", nb "
            + java.util.Arrays.toString(nb) + ") is not ggml's (" + Llama.ggml_nelements(tensor) + " elements, "
            + bytes + " bytes): another llama.cpp release's struct ggml_tensor?");
      }
      if (ne[2] != 1 || ne[3] != 1) {
        throw new IllegalStateException(name + " has more than 2 dimensions: " + java.util.Arrays.toString(ne));
      }
      int rows = Math.toIntExact(ne[1]), cols = Math.toIntExact(ne[0]);
      int[] ints = type == GGML_TYPE_I32 ? new int[rows * cols] : null;
      float[] floats = type == GGML_TYPE_F32 ? new float[rows * cols] : null;
      if (n > 0) {
        try (Arena a = Arena.ofConfined()) {
          MemorySegment host = a.allocate(bytes, 8);
          Llama.ggml_backend_tensor_get(tensor, host, 0, bytes);
          for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
              long at = r * nb[1] + 4L * c;
              if (ints != null) ints[r * cols + c] = host.get(JAVA_INT, at);
              else floats[r * cols + c] = host.get(JAVA_FLOAT, at);
            }
          }
        }
      }
      return new Entry(base, layer, rows, cols, ints, floats);
    }

    /** Forget what was recorded (and a failure): before a decode. */
    public void clear() {
      entries.clear();
      failure = null;
    }

    /** What the decodes since clear recorded, in the order computed, and
     *  clear; or throw the callback's failure. */
    public java.util.List<Entry> drain() {
      Throwable f = failure;
      java.util.List<Entry> out = java.util.List.copyOf(entries);
      clear();
      if (f != null) throw new IllegalStateException("the eval callback failed: " + f.getMessage(), f);
      return out;
    }
  }

  // ---- rows of logits -------------------------------------------------------------------

  /** log Σ exp(row), over the n floats of a row of logits, in double: NaN
   *  when a logit is NaN, +Inf when one is +Inf (a failed decode, never a
   *  distribution), -Inf when all are -Inf. */
  public static double logSumExp(MemorySegment row, int n) {
    MemorySegment r = row.reinterpret(4L * n);
    double m = Double.NEGATIVE_INFINITY;
    for (int j = 0; j < n; j++) m = Math.max(m, r.getAtIndex(JAVA_FLOAT, j));
    if (Double.isInfinite(m) || Double.isNaN(m)) return m;
    double s = 0;
    for (int j = 0; j < n; j++) s += Math.exp(r.getAtIndex(JAVA_FLOAT, j) - m);
    return m + Math.log(s);
  }

  /** The log-probabilities of a row of n logits, row - z with z its
   *  logSumExp, as floats. */
  public static float[] logSoftmax(MemorySegment row, int n, double z) {
    MemorySegment r = row.reinterpret(4L * n);
    float[] out = new float[n];
    for (int j = 0; j < n; j++) out[j] = (float) (r.getAtIndex(JAVA_FLOAT, j) - z);
    return out;
  }

  /** The logit of token in a row. */
  public static double logit(MemorySegment row, int n, int token) {
    return row.reinterpret(4L * n).getAtIndex(JAVA_FLOAT, token);
  }
}
