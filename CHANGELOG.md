# Changelog

## 0.3.0

- This release requires `dendra[solvers]>=0.28.0`. Upgrade Dendra alongside
  Dendra Models to use the supported native and functional execution APIs.
- The [quick start](README.md#quick-start) shows native and compiled execution,
  voltage recording, continuation, and a connected Network driven by NetStim.
  The [thresholding examples](examples/thresholding/README.md) cover native and
  compiled Populations.
- Functional execution supports Yu 2024 cortical current bindings, the fused
  Kumaravelu 2016 Population, and Zhang 2014 dynamic synapses. Zhang synapses
  preserve short-term plasticity across multiple deliveries in a timestep and
  support checkpoint continuation.
- `valid_ids()` lists the 44 packaged L2/3 and 37 packaged L5 pyramidal
  morphologies. The Kumaravelu STN rate has stable gradients at extreme voltages.

The experimental modular Kumaravelu Network uses native execution. The
optional `CHECK_PURITY` diagnostic is currently unsupported for generated fused
Kumaravelu configurations and Yu connection-delay modules.
