# ext/device/native

- `pjrt_c_api.h`: XLA's PJRT C API header (API 0.76), copied unchanged
  from openxla/xla at commit 0fccb8a6037019b20af2e502ba4b8f5e0f98c8f6,
  the commit jax 0.7.2 pins, so it matches jax-cuda13-plugin 0.7.2's
  `xla_cuda_plugin.so`. Apache License 2.0, The OpenXLA Authors (its
  notice is at its top).
- `pjrt_offsets.c`: prints the struct layouts `gen.device.pjrt` uses, as
  EDN, from that header (see its first lines for the command).
