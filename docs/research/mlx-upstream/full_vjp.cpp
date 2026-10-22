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
