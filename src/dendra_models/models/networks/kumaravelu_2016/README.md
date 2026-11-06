# Kumaravelu et al. 2016 CTX-BG-TH network in Dendra

This package is a Dendra module-style implementation of the Kumaravelu, Brocker & Grill CTX-BG-TH network model.

## Layout

```text
kumaravelu_2016/
  __init__.py
  params.py
  net.py
  mechanisms/
    __init__.py
    hh.py
    izhikevich.py
    synapses.py
```

## Usage

```python
from dendra_models.models.networks.kumaravelu_2016 import Kumaravelu2016, default_params

p = default_params()
p["n"] = 10
p["pd"] = 0.0       # 0 healthy, 1 PD
p["corstim"] = 0.0  # used for the MATLAB GPe baseline-current modulation

model = Kumaravelu2016(p, dt=0.01)
```

The public builder returns the fused single-`Population` implementation
described below. The older modular builder remains in `net.py` for development
and mechanism-level inspection; it is not the public `Kumaravelu2016` alias.

The modular builder contains two populations:

- `net.hh`: TH, STN, GPe, GPi, StrD2, StrD1; standard Dendra membrane-voltage integration.
- `net.ctx`: CTX_RS and CTX_FS; `dn.scnv()` integration because the cortical voltage is stored as the mechanism state `v_izh` and exposed by the pure `voltage_values()` hook.

## Key design choices

The isolated `regular_spiking_cortex` and `fast_spiking_interneuron` mechanisms
support `dn.func` with Dendra 0.28. Their pure
voltage hooks apply the existing reset, recovery increment, and hard-forward
spike/crossing flags before ODE integration. Surrogate gradients retain the
original straight-through equations. An isolated population uses `i_stim` for
its applied current and treats unbound synaptic currents as zero; the modular
network supplies declared conductance and reversal inputs. The public fused
builder supports native and functional execution.

1. The network builder labels subranges, inserts renamed mechanisms, and declares the cortical input bindings before constructing `dn.Network`. The remaining imperative spike-router and striatal references are installed after construction.

2. **Cortical Izhikevich currents use explicit inputs.** The RS and FS States
declare conductance and reversal inputs for `i_ie`, `i_thcor`, and `i_ei`.
Bindings read the three event-only alpha synapses in the same population:

```python
ctx.bind_input("ctx_rs.DE.regular_spiking_cortex_states.s_ie", "FS_RS.s")
ctx.bind_input("ctx_rs.DE.regular_spiking_cortex_states.e_ie", "FS_RS.e")
ctx.bind_input("ctx_rs.DE.regular_spiking_cortex_states.s_thcor", "TH_CTX.s")
ctx.bind_input("ctx_rs.DE.regular_spiking_cortex_states.e_thcor", "TH_CTX.e")
ctx.bind_input("ctx_fs.DE.fast_spiking_interneuron_states.s_ei", "RS_FS.s")
ctx.bind_input("ctx_fs.DE.fast_spiking_interneuron_states.e_ei", "RS_FS.e")
```

Each State computes `s * (values["v_izh"] - e)` after the voltage reset and at
its normal position in the mechanism advance order. In this builder the alpha
synapses advance after the cortical States, so the current uses their incoming
conductance states. Saved `i_` fields would introduce a one-step lag. The
bindings follow physical support coordinates and survive population rebuilds
and explicit batching. Unbound inputs default to zero.

Reversal values are now explicit provider parameters, so changing or
differentiating a synapse's `e` affects the cortical current. The original
closure captured reversal as a float when the network was built. The numerical
current convention is unchanged: these three currents retain the original
scale of one and do not multiply the provider's `current_scale` parameter.

An isolated cortical population with these providers supports functional
execution. Whole-Network lowering of the experimental modular builder is
currently unsupported: its cortical spike router assembles disjoint source
supports, while its striatal currents read permuted gate states at current
evaluation rather than State-advance time. These live references do not fit
the existing input-binding contract. Use the public fused builder for
functional execution, or run the modular builder natively.

3. **Recurrent striatal GABA is not event-based.** The MATLAB variables `S1c` and `S8` are voltage-gated continuous states, so they are implemented as `striatal_gaba_gate` mechanisms on the D2/D1 MSN populations. The postsynaptic MSN recurrent-GABA currents then reference permuted copies of those presynaptic gate states:

```python
d2_refs = mh.StrD2_recurrent_gaba.DE["striatal_recurrent_gaba_refs"]
for k, perm in enumerate(realization["str_d2_perms"]):
    d2_refs.setreference(f"s{k}", _index_reference(lambda: mh.StrD2_gate.s, perm))
```

4. **HH presynaptic spike detection is cached once per HH cell.** The builder imports `spikedetect` from `dendra.models.mod`, inserts it once on the full HH population, and all HH-origin synaptic connections use `pre_var="mech.spikedetect.spikes"` with `threshold=None`. The detector threshold defaults to `params["spike_threshold_hh"]`, while `params["spikedetect_hh"]` can tune `tau_gate` and `ste_scale` or explicitly override `threshold`. This avoids repeating the same threshold-crossing calculation separately for every outgoing HH synapse.

5. **Synaptic waveform types are separated.** True double-exponential pathways use a saved-current alias of Dendra's `exp2syn`; alpha-function pathways use the custom `alpha_syn_current`; striatal recurrent GABA uses the continuous gate/current pair above.

6. **Connectivity follows the MATLAB routing.** The builder preserves the ring-shifted pathways (`S21a`, `S31a`, `S51`... style variables), the random cortical/striatal permutations, and the per-postsynaptic random gain vectors.


## Unit handling for the HH population

The MATLAB HH-style equations use point-cell current-density numerics: conductances are written in mS/cm²-like units, voltages in mV, currents in µA/cm²-like units, and `Cm = 1` µF/cm². Dendra's standard single-compartment backward-Euler solver uses SI-scaled capacitance internally and expects HH mechanism currents in mA/cm², with conductances in S/cm².

For that reason the HH intrinsic mechanisms and all synapses inserted onto the `net.hh` population use

```python
params["units"]["hh_current_scale"] = 1.0e-3
```

This converts MATLAB µA/cm² current densities to Dendra mA/cm² current densities while leaving the published MATLAB parameter values visible in `params.py`. The cortical `scnv`/Izhikevich population is not scaled this way; its synaptic currents remain in the original MATLAB/Izhikevich numerical convention because they enter the mechanism-owned voltage equation directly.

You do not need to force the HH compartments to have area 1. The HH intrinsic and synaptic mechanisms are implemented as current-density mechanisms, so geometry/area does not affect the standard voltage update. Area only matters if an external absolute current is supplied through Dendra's `intra` current-injection path; in that case convert the desired density to an absolute current as `I_abs = I_density * area`, using mA and cm² after the same 1e-3 density conversion.

## Current limitations / next steps

- The current builder assumes the same `n` in every nucleus, matching the MATLAB implementation.
- Analytic DBS and cortical pulses are available for ordinary runs. Exact MATLAB
  validation replay additionally imports the sampled `Idbs` and `Iappco`
  arrays through `params_from_matlab_validation(...)`.

## Dendra `v_init` compatibility

The HH population is intentionally constructed with one initial voltage per compartment/cell:

```python
hh_pop = dn.Population(C=len(hh_v_init), v_init=hh_v_init)
```

Dendra accepts scalar or length-`model.nc` `v_init`; the sampled MATLAB-style
initial voltages are broadcast/reshaped at initialization time.


## Connection-weight shape compatibility

The builder keeps scalar weights and delays as Python scalars so Dendra can expand them after applying its connection filtering policy. Heterogeneous per-edge weights are converted to tensors on the postsynaptic population's device/dtype. The local cortical E/I permutation pathways are connected with `allow_multapses=True`, because the four independent MATLAB permutation arrays can repeat the same pre/post pair and those repeats should contribute as separate alpha-synapse events.

## Runtime shape/scaling note

The synapse implementation uses density-style `Synapse` mechanisms for HH-target
pathways and event-only `Synapse` mechanisms for CTX-target pathways.  It does
not use Dendra `PointProcess` for the Kumaravelu conductance-density synapses,
because `PointProcess` represents lumped nA/uS sources and Dendra scales those
by compartment area during current assembly.  The MATLAB model's synaptic
weights are written as point-neuron conductance densities, so the density
mechanism convention avoids unintended area scaling and CTX scnv shape mismatch.

## Fused differentiable fast path

The package also includes a fused backend:

```python
import dendra as dn
from dendra_models.models.networks.kumaravelu_2016 import (
    Kumaravelu2016Fused,
    default_params,
)

p = default_params()
p["n"] = 100
p["pd"] = 0.0
p["corstim"] = 0.0

dn.set_jit_enabled(True)
model = Kumaravelu2016Fused(
    p,
    dt=0.01,
    pick_dbs_freq=1,
    differentiable_spikes=True,
    delay_mode="circular_eager",
)
model.initialize()
model.run(tstop=2000.0, dt=0.01, progressbar=False)
```

The fused builder returns a single `dn.Population`, not a `dn.Network`.  The
population has `C = 8*n` exposed voltage entries and labels:

```text
TH, STN, GPe, GPi, StrD2, StrD1, CTX_RS, CTX_FS
```

All dynamics are implemented in one `VoltageProcess` inserted on this population
and run with `dn.scnv()`.  The mechanism owns the actual voltage states, all HH
and Izhikevich state variables, synaptic conductance/filter variables, delay
queues, routing permutations, and spike indicators.  This removes Dendra
`NetCon`, per-synapse event queues, point-process area scaling, and current-map
bookkeeping from the runtime hot path.

Useful exposed mechanism buffers after `initialize()`:

```python
m = model.mech.kumaravelu
m.v_all       # full exposed voltage vector, shape (..., 8*n)
m.spikes      # synaptic driver events: HH -10 mV crossings; CTX reset spikes
m.syn_spikes  # -10 mV crossings for all groups, including cortical local E/I
m.ap_spikes   # -20 mV crossings, matching the MATLAB find_spike_times threshold
m.v_th, m.v_stn, m.v_gpe, m.v_gpi, m.v_d2, m.v_d1, m.v_rs, m.v_fs
```

For differentiable simulations, call `model.train()` before running so Dendra's
`Population.run()` does not enter `torch.no_grad()`.  For long differentiable
runs, prefer `longrun_checkpointed(...)` and keep callbacks replay-safe.

```python
model = Kumaravelu2016Fused(p, dt=0.01, differentiable_spikes=True)
model.train()
model.initialize()
loss = model.longrun_checkpointed(tstop=2000.0, chunklength=1000, dt=0.01,
                                  callbacks=[...])
loss.backward()
```

Spike events use a straight-through surrogate: the forward value is a hard event,
while the backward path is shaped by a smooth sigmoid onset.  Setting
`differentiable_spikes=False` inserts the same mechanism with `ste_scale=0`.

The fused backend uses the MATLAB current convention directly and bypasses the
standard HH voltage solver.  Therefore the `params["units"]["hh_current_scale"]`
bridge used by the modular HH population is intentionally not applied in the
fused mechanism.

The exact-realization MATLAB comparison uses a Chronux-compatible GPi spectrum
endpoint and a paired statistical design. The fused state-space synaptic
filters and MATLAB's spike-history lookup-table sums agree before synaptic
events but are not bitwise-identical realizations afterward. Small numerical
differences are amplified over long horizons by the recurrent network, so the
comparison reports a pre-event voltage conformance window alongside paired
population rates and GPi 7--35 Hz power rather than requiring bitwise-identical
two-second traces.
