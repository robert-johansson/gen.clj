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
  public static final String VERSION = "0.7.0";

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
    static final ThreadLocal<StringBuilder> ERROR = ThreadLocal.withInitial(StringBuilder::new);
    static final ThreadLocal<String> LAST_ERROR = new ThreadLocal<>();

    /** void (*ggml_log_callback)(enum ggml_log_level level, const char * text, void * user_data) */
    static void log(int level, MemorySegment text, MemorySegment userData) {
      try {
        int[] current = LEVEL.get();
        if (level != 5) current[0] = level;
        int effective = current[0];
        String s = text.reinterpret(Long.MAX_VALUE).getString(0, StandardCharsets.UTF_8);
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
