package gen.mlx;

/** An error MLX reported: a bad shape, an unsupported dtype on a device and
 *  the like. The JVM keeps running (mlx-c's default handler would exit). */
public final class MlxException extends RuntimeException {
  private static final long serialVersionUID = 1L;
  private final String function;

  public MlxException(String function, String message) {
    super(function + ": " + message);
    this.function = function;
  }

  /** The mlx-c function that failed. */
  public String function() {
    return function;
  }
}
