<!-- Draft for ml-explore/mlx (genclj-hf5z). Not posted: sending it is the owner's call, and MLX's contributor rules forbid agent-submitted PRs (docs/research/mlx-fork-audit.md). -->

**Title:** `Full::vjp` multiplies the cotangent by the fill value, so the gradient of `full(shape, x)` with respect to `x` is scaled by `x`

MLX v0.32.2 (the PyPI wheel `mlx_metal-0.32.2`, source at tag v0.32.2, 1f8e74e), macOS 27.0.1 on an M4. It happens on the CPU and on Metal.

### What happens

`full(shape, vals, dtype)` is `full_impl(broadcast_to(vals, shape), dtype)` (`mlx/ops.cpp:358-363`). `full_impl` makes a `Full` primitive whose one input is `astype(vals, dtype)`, already of the output's shape (`mlx/ops.cpp:350-356`). `Full::eval_cpu` copies that input into the output (`mlx/backend/cpu/primitives.cpp:192`). So a fill is the identity on its broadcast input, and its VJP should be the cotangent itself. `broadcast_to`'s VJP then sums it over the broadcast axes.

`Full::vjp` multiplies the cotangent by the primal instead (`mlx/primitives.cpp:2408-2416`):

```cpp
std::vector<array> Full::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  return {multiply(cotangents[0], primals[0], stream())};
}
```

`Full::jvp`, just below it, returns the tangents, which is right. So the forward and reverse derivatives of `full` disagree whenever the fill value is not 1.

### Reproduction

```cpp
// The gradient of a fill: full(shape, x) is x broadcast to shape, so the
// vjp of x is the cotangent summed over the broadcast, whatever x is.
#include <iostream>
#include "mlx/mlx.h"

namespace mx = mlx::core;

int main() {
  mx::set_default_device(mx::Device::cpu);
  auto x = mx::array(2.3f);
  auto ct = mx::array({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});

  auto by_full = mx::vjp([](mx::array a) { return mx::full({2, 2}, a); }, x, ct).second;
  auto by_broadcast = mx::vjp([](mx::array a) { return mx::broadcast_to(a, {2, 2}); }, x, ct).second;
  std::cout << "vjp of full:         " << by_full << std::endl;
  std::cout << "vjp of broadcast_to: " << by_broadcast << std::endl;

  auto by_grad = mx::grad([](mx::array a) { return mx::sum(mx::full({2, 3}, mx::square(a))); })(mx::array(0.5f));
  std::cout << "grad of sum(full([2, 3], x^2)) at 0.5: " << by_grad << " (want 6)" << std::endl;

  auto jvp_full = mx::jvp([](mx::array a) { return mx::full({2, 2}, a); }, x, mx::array(1.0f)).second;
  std::cout << "jvp of full, tangent 1: " << jvp_full << std::endl;
  return 0;
}
```

Built with Apple clang 21 (`clang++ -std=c++20 -I <mlx source> full_vjp.cpp -lmlx`) and linked against the wheel's `libmlx.dylib`, it prints:

```
vjp of full:         array(23, dtype=float32)
vjp of broadcast_to: array(10, dtype=float32)
grad of sum(full([2, 3], x^2)) at 0.5: array(1.5, dtype=float32) (want 6)
jvp of full, tangent 1: array([[1, 1],
       [1, 1]], dtype=float32)
```

### Expected

- The VJP of `full({2, 2}, x)` with cotangent `[[1, 2], [3, 4]]` is the cotangent's sum, 10, as `broadcast_to` gives. MLX gives 23 = 10 × 2.3.
- d/dx Σ full({2, 3}, x²) at x = 0.5 is 6 · 2x = 6. MLX gives 1.5 = 6 × x² × 2x.

The same in Python should show it (not run here: this machine has no MLX Python package):

```python
import mlx.core as mx
mx.grad(lambda x: mx.full((2, 2), x).sum())(mx.array(2.3))   # want 4.0, expect 9.2
```

### Why the test suite passes

The `full` case in `tests/autograd_tests.cpp:399-408` differentiates at `x = ones({2})`, where multiplying by the primal changes nothing:

```cpp
auto full_fn = [](array a) { return full({5, 5, 2}, a); };
auto x = ones({2}, float32);
auto out = vjp(full_fn, x, full({5, 5, 2}, 2.0f)).second;
CHECK(array_equal(out, array({50.0f, 50.0f})).item<bool>());
```

With any other `x`, say `full({2}, 3.0f)`, the expected value is still `{50, 50}` and MLX gives `{150, 150}`.

### Suggested fix

Return the cotangent:

```cpp
  return {cotangents[0]};
```

The primitive's input already has the output's shape and dtype, so no reduction or cast is needed in `Full::vjp`; the `broadcast_to` and `astype` that `full` builds before it carry their own VJPs.

### How we work around it

gen.clj's MLX backend fills with an array value by `astype` and `broadcast_to` and keeps `full` for host scalars (genclj-hf5z, `src/gen/backend/mlx.clj`, the backend's `-full`).
