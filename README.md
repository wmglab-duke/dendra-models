<div align="center">
  <img src="https://raw.githubusercontent.com/wmglab-duke/dendra-models/main/docs/banner.png" alt="Dendra Models">
</div>

***

Reference neuronal cell and network models implemented with
[Dendra](https://wmglab-duke.github.io/dendra/).

[Documentation](https://wmglab-duke.github.io/dendra/) ·
[Source](https://github.com/wmglab-duke/dendra-models) ·
[Issue tracker](https://github.com/wmglab-duke/dendra-models/issues) ·
[PyPI](https://pypi.org/project/dendra-models/)

> [!IMPORTANT]
> dendra-models requires Dendra 0.27 or later. Its standard installation
> includes the native CPU solvers used by tree and extracellular models.

## Installation

Dendra Models requires Python 3.11 or later. Install the released package from
PyPI:

```sh
python -m pip install dendra-models
```

This installs Dendra with its `solvers` extra, including
[`dendra-solvers`](https://pypi.org/project/dendra-solvers/). On supported
platforms, pip uses a prebuilt solver wheel. These native solvers are the
default CPU backends for Dendra Models' block and tree models.

For an editable development installation, see
[CONTRIBUTING.md](https://github.com/wmglab-duke/dendra-models/blob/main/CONTRIBUTING.md).

## Quick start

Models are available from `dendra_models.models`:

```python
import dendra as dn
from dendra_models.models import exactMRG

with dn.ctx(DTYPE="float64", DEVICE="cpu", JIT=0):
    model = exactMRG(diameters=[5.7], n_node=3)
    model.eval()
    model.initialize()
```

## Isolated voltage models (development)

The development versions of Dendra and Dendra Models provide pure
`voltage_values` hooks for the Yu 2024 Esser mechanisms and the Kumaravelu
2016 regular-spiking and fast-spiking cortical mechanisms. Use the matching
Dendra development checkout for this API.

For example, an isolated Izhikevich population can use functional execution:

```python
import dendra as dn
from dendra_models.models.networks.kumaravelu_2016.mechanisms.izhikevich import (
    regular_spiking_cortex,
)

with dn.ctx(DTYPE="float64", DEVICE="cpu", JIT=0):
    cells = dn.Population(N=1, C=3, integrator=dn.scnv())
    cells.insert(regular_spiking_cortex, i_stim=10.0)
    cells.initialize()

runner = dn.compile(
    cells,
    dt=0.025,
    steps=80,
    initialization="fresh",
    inference=False,
    trainable=("i_stim",),
    callbacks={"trace": dn.func.Recorder(("v", "regular_spiking_cortex.spikes"))},
)
state, auxiliary = runner()
voltage = auxiliary["callbacks"]["trace"]["v"]
```

Set `i_stim` to drive the mechanism-owned voltage equation; `scnv` skips the
ordinary membrane-voltage solve. Public `v` records the voltage exposed before
the ODE states advance, so it can differ from the mechanism's final voltage
state. Unbound synaptic currents contribute zero in isolated populations.

Esser's `net_receive` spike generation and synaptic event handling retain their
imperative network phase. Its default voltage and threshold initialization
retain their existing detach behavior; use explicit state initialization inputs
when differentiating those initial conditions. Functional execution of connected
networks, the Kumaravelu fused implementation, and live cross-mechanism references
remain outside this isolated-mechanism support.

The cortical Izhikevich States also accept explicit conductance and reversal
inputs from other mechanisms in the same population. `Population.bind_input`
maps each source into the consumer's physical support; the State forms its
current using the voltage after reset. See the
[Kumaravelu cortical input example](src/dendra_models/models/networks/kumaravelu_2016/README.md#key-design-choices).
This requires matching `feat/functional-input-bindings` development checkouts.

## Explicit synaptic delivery (development)

The matching development checkouts support pure delivery
to Dendra's builtin `expsyn` and `exp2syn` mechanisms on compatible companion
cells. For example, attach a regional synapse to a Rattay cable:

```python
import dendra as dn
import torch
from dendra.models.mod import expsyn
from dendra_models.models import Rattay1993

with dn.ctx(DTYPE="float64", DEVICE="cpu", JIT=0):
    fibre = Rattay1993(
        diameters=[1.0], L=40.0, dx=10.0,
        integrator=dn.bwd_euler_ub(method="pcr", imem=False),
    ).double()
    fibre[:, 1:3].insert(expsyn.rename("synapse"), tau=0.7)
    fibre.initialize()

functional, tensors = dn.func.make_functional(fibre, dt=0.01)
prepared = functional.prepare(tensors.parameters, tensors.constants)
payload = torch.full_like(tensors.state["mechanisms"]["synapse"]["g"], 0.001)
state = functional.receive(
    tensors.parameters, prepared, tensors.state,
    target="synapse", payload=payload,
)
state, _ = functional.step(tensors.parameters, prepared, state)
```

The payload uses the synapse's local runtime shape, including explicit batch
axes, and numerical conductance weights in µS. Delivery changes declared
synaptic state without advancing time; call `step` to advance the cell. The
returned state supports differentiation, continuation, and commit through the
usual functional API. `prepare_and_receive` rebuilds parameter-dependent
normalization when preparing and delivering together, including `exp2syn`'s
peak normalization.

This API accepts an explicit payload for one named synapse. Alpha synapses
represent predetermined events and do not need a network event receiver.
Automatic functional execution of an entire Network remains future work.

The matching development checkouts also expose
fixed-delay projections with explicit source observations, detector state, and
queue state. Continue the example by building an ordinary thresholded connection
to the same regional synapse:

```python
source = dn.SingleCompartment(
    N=1, C=2, v_init=-0.2, integrator=dn.scnv(imem=False),
).double()
network = dn.Network(
    {"source": source, "fibre": fibre},
    netcon_delay_backend="dense", netcon_train_backend="dense",
)
network.connect_one_to_one(
    source[:], fibre[:, 1:3], fibre.mech.synapse,
    threshold=-0.1, weight=0.001, delay=0.02,
)
network.train()
network.initialize(0.01)
network.set_synaptic_diff_config(
    diff_weights=True, diff_delays=False, diff_spiking=True,
    diff_scheduled_times=False, train_delay_backend="dense",
)
connection = next(iter(network.synapses.values()))
projection, delivery = dn.func.make_functional_netcon(connection)
functional, tensors = dn.func.make_functional(fibre, dt=0.01)
prepared = functional.prepare(tensors.parameters, tensors.constants)
queue, state = delivery.state, tensors.state
for voltage in (-0.2, -0.1, 0.1, -0.2, 0.0):
    queue, payload = projection.read_due(queue)
    state = functional.receive(
        tensors.parameters, prepared, state, target="synapse", payload=payload,
    )
    source_values = torch.full_like(network.source.v, voltage)
    queue = projection.advance(delivery.weights, queue, source_values)
    state, _ = functional.step(tensors.parameters, prepared, state)
```

Source values have the full initialized source-variable shape, including batch
axes, and use `projection.source_dtype` and `projection.source_device`. Equality
with the threshold counts as above; remaining above does not retrigger. The
boolean `has_spiked` state records this latch. Training with `diff_spiking=True`
preserves hard events in the forward pass and uses the native sigmoid surrogate
for gradients; `diff_spiking=False` gives hard detection. Mixed per-connection
thresholds are supported. A `None`/NaN threshold passes raw numeric gates through,
including fractional values and their gradients; an entirely raw projection
needs no detector latch.

Keep the native order for each projection: read due deliveries, apply the receive
transition, then observe the current source and advance the detector and queue.
Advance populations once all projections have run. This matters when receiving
an event changes the source variable observed by that or a later projection.
`projection.step(weights, queue, source_values)` combines the two projection
phases when the source observation is already independent of receive.

Weights use the connection's expanded edge order. Pending payloads keep
the weights used when they were enqueued, so changing explicit weights affects
new events. Delays use the native rounded timestep rule, with a minimum of one
step even for zero delay. The returned queue supports checkpoint continuation
and `projection.commit_state_(connection, queue)`.

The projection API requires an initialized dense event connection with fixed
delays. Tensor schedules are supported as described below. The ordering of multiple projections remains
explicit in the caller. Detector configuration is fixed when
the projection is created; lower it again after changing training or detector
settings. Checkpoints include both the delay queue and the above-threshold latch.

### Shared NetStim renewal (development)

The matching development checkouts add pure NetStim initialization
and renewal using the same core as native execution. Lower a standard,
initialized, unscheduled `dn.NetStim` and pass each random draw explicitly:

```python
import dendra as dn
import torch
from dendra.models.mod import expsyn
from dendra_models.models import Rattay1993

dt = 0.01
with dn.ctx(DTYPE=torch.float64, JIT=0, REQUIRE_GRAD=1, IMEM=False):
    fibre = Rattay1993(
        diameters=[1.0], L=40.0, dx=10.0,
        integrator=dn.bwd_euler_ub(method="pcr", imem=False),
    ).double()
    fibre[:, 1:3].insert(expsyn.rename("event"), tau=0.7)
    stimulus = dn.NetStim(
        N=2, interval=[0.03, 0.04], noise=0.5, seed=719,
    )
    network = dn.Network(
        {"fibre": fibre}, netstim=stimulus,
        netcon_delay_backend="dense", netcon_train_backend="dense",
    )
    network.connect_one_to_one(
        stimulus[:], fibre[:, 1:3], fibre.mech.event,
        weight=0.001, delay=0.02,
    )
    network.train()
    network.initialize(dt)
    network.set_synaptic_diff_config(
        diff_weights=True, diff_delays=False, diff_spiking=True,
        diff_scheduled_times=False, train_delay_backend="dense",
    )

renewal, source = dn.func.make_functional_netstim(stimulus)
connection = next(iter(network.synapses.values()))
projection, delivery = dn.func.make_functional_netcon(connection)
functional, cable = dn.func.make_functional(fibre, dt=dt)
prepared = functional.prepare(cable.parameters, cable.constants)
source_state, queue, cable_state = source.state, delivery.state, cable.state
for index in range(20):
    # Sample outside compiled/vmapped transitions from the explicit RNG state.
    draw = renewal.sample_step_draw(source_state)
    time = source.parameters["interval"].new_tensor(index * dt)
    source_state, outputs = renewal.step(
        source.parameters, source.constants, source_state, time, draw, bptt=True,
    )
    queue, payload = projection.read_due(queue)
    cable_state = functional.receive(
        cable.parameters, prepared, cable_state, target="event", payload=payload,
    )
    queue = projection.advance(delivery.weights, queue, outputs.spike_gate)
    cable_state, _ = functional.step(cable.parameters, prepared, cable_state)
```

Generate the NetStim output once before all projections in each timestep and
reuse it for every connection from that source. Each projection still follows
`read_due → receive → advance`; populations advance after all projections.
`outputs.spikes` is the hard boolean event mask, while `outputs.spike_gate`
preserves the native hard forward values and sigmoid surrogate gradients.
Use `outputs.spikes.to(projection.source_dtype)` for hard-only connections.
`bptt=True` retains gradients through renewal clocks across timesteps;
`bptt=False` cuts that history as native NetStim does.

The source state contains both renewal clocks and RNG/seeder byte tensors.
Checkpoint it together with queues, population states, and simulation time;
continue sampling from the restored state to reproduce subsequent renewals.
The draw packet includes its post-draw RNG state, so keep it paired with its
sampled exponential values. `sample_step_draw` leaves the source and its input
state untouched. `renewal.commit_state_(stimulus, source_state)` transfers clocks
and RNG state back to a matching native source; it does not change parameters.
`sample_initialization_draw(parameters, state)` followed by
`initial_state(parameters, constants, state, draw)` provides a pure reset with
the native seeded initialization behavior. Tensor schedules are described
below; legacy heap events continue to use native execution.

### Shared tensor schedules (development)

The `feat/functional-tensor-schedules` checkouts support shared pure NetStim and
NetCon schedule evaluation. Declare a fixed event topology through the ordinary
native APIs, then supply explicit time/amplitude packets to the lowered plans.
This continues the previous example, resetting its native Network before
lowering again:

```python
network.initialize(dt)
network.set_synaptic_diff_config(
    diff_weights=True, diff_delays=False, diff_spiking=True,
    diff_scheduled_times=True, sched_width=1.3, train_delay_backend="dense",
)
connection = next(iter(network.synapses.values()))
shared_times = torch.tensor([0.035, 0.065], dtype=torch.float64, requires_grad=True)
shared_amplitudes = torch.tensor([0.4, 0.7], dtype=torch.float64, requires_grad=True)
for owner in (stimulus, connection):
    owner.bind_time_source(shared_times)
    owner.bind_weight_source(shared_amplitudes)
stimulus.schedule_ref([0, 1], times=[0.015, 0.025], weight_idx=[0, 1])
stimulus.schedule_time_ref([0, 1], time_idx=[0, 1], weight=0.3)
connection.schedule_ref(con_indices=[0, 1], times_ms=[0.045, 0.055], weight_idx=[1, 0])
connection.schedule_time_ref(con_indices=[0, 1], time_idx=[0, 1], weight=0.2)

renewal, source = dn.func.make_functional_netstim(stimulus)
projection, delivery = dn.func.make_functional_netcon(connection)
functional, cable = dn.func.make_functional(fibre, dt=dt)
prepared = functional.prepare(cable.parameters, cable.constants)
# Independent snapshots become shared optimizer inputs by explicit rebinding.
connection_schedule = delivery.schedule._replace(
    time_source=source.schedule.time_source,
    amplitude_source=source.schedule.amplitude_source,
)
source_state, queue, cable_state = source.state, delivery.state, cable.state
origin = network.t.detach().clone()
for index in range(20):
    draw = renewal.sample_step_draw(source_state)
    source_state, outputs = renewal.step(
        source.parameters, source.constants, source_state,
        origin + origin.new_tensor(index) * dt,
        draw, bptt=True, schedule=source.schedule,
    )
    queue, payload = projection.read_due(queue)
    cable_state = functional.receive(
        cable.parameters, prepared, cable_state, target="event", payload=payload,
    )
    queue = projection.advance(
        delivery.weights, queue, outputs.spike_gate, schedule=connection_schedule,
    )
    cable_state, _ = functional.step(cable.parameters, prepared, cable_state)
```

`TensorScheduleTensors(times, amplitudes, time_source, amplitude_source)` keeps
floating values explicit; fixed event/reference indices belong to the plan.
Packets from separate owners are independent snapshots, so reuse the same
explicit source tensors when parameter sharing is intended. Include packets
with simulation time, clocks/RNG, queues, and population carry in checkpoints.
`commit_state_` transfers runtime carry while leaving native schedule values
and bindings unchanged; align those values before native continuation.

Hard schedules assign events to the first causal grid step. Differentiable
times use each native triangular-kernel policy, with differentiable amplitudes
and indexed sources. NetStim hard spikes remain boolean causal events; consume
`spike_gate` through `diff_spiking=True` projections to preserve scheduled
amplitudes and gradients. Tensor schedules reevaluate repeated times and miss
skipped hard steps. Legacy heaps consume overdue entries through the shared
pure calendar described below. Topology/binding/policy changes require lowering
again; floating packet values can change without rebuilding the plan.

The integration tests exercise both `expsyn` and `exp2syn` on companion cables,
including batching, hard tracked events, shared-source gradients, compiled
composition, and continuation into a fresh native Network:

```bash
python -m pytest -q tests/test_functional_tensor_schedules.py
```

### Consuming heap schedules (development)

The matching `feat/functional-heap-schedules` checkouts also lower ordinary
unit-amplitude Python schedules declared with `stimulus.schedule(indices, times)`.
Native and functional NetStim use the same pure tensor calendar; no second
NetStim implementation is needed. Heap events retain their consumption policy:
one call consumes all overdue duplicates for each generator that can fire,
coalescing them into one boolean event. A spike cap prevents further consumption.
Repeated or decreasing call times never rewind the cursor. Tensor schedules
retain the distinct stateless policy above.

```python
stimulus.schedule([0, 0, 1], [0.015, 0.015, 0.025])
renewal, source = dn.func.make_functional_netstim(stimulus)
state = source.state
origin = network.t.detach().clone()
for index in range(20):
    time = origin + origin.new_tensor(index) * dt
    state, outputs = renewal.step(
        source.parameters, source.constants, state, time,
        renewal.sample_step_draw(state), bptt=True, schedule=source.schedule,
    )
```

The fixed calendar is in `source.constants["heap_times"]` and
`source.constants["heap_row_offsets"]`. Calendar times use Float64 on CPU/CUDA
and the simulation dtype on MPS. Comparisons use the simulation dtype; native
metadata and checkpoints retain original Python precision across dtype/device
changes.
The flat per-generator `state["heap_cursor"]` travels with clock and RNG carry.
Legacy head-only checkpoint overrides are explicit carry too. Pure
`initial_state` resets the renewal clock and counts, preserving the consumed
cursor and clearing those overrides, as native initialization does.

Save constants, parameter/schedule values, simulation time, source carry,
projection queues, and population carry together for continuation. A compatible
`commit_state_` restores the cursor and RNG into native NetStim. Native runtime
restore also accepts older remaining-heap and head-only checkpoints. Changing
the calendar requires lowering again. Heap event times are not differentiable;
use tensor schedules for trainable event times and amplitudes. Calendars with
NaN entries keep native heap ordering and are rejected by functional lowering.

The companion integration tests compare native and pure trajectories, renewal
and weight gradients, compiled composition, and checkpoint continuation through
both `expsyn` and `exp2syn` on cable populations:

```bash
python -m pytest -q tests/test_functional_heap_schedules.py
```

## Examples

The repository includes focused examples for
[threshold finding](https://github.com/wmglab-duke/dendra-models/tree/main/examples/thresholding),
[kilohertz-frequency stimulation](https://github.com/wmglab-duke/dendra-models/tree/main/examples/khz),
and
[C-fiber simulations](https://github.com/wmglab-duke/dendra-models/tree/main/examples/c-fiber).

## Available models
### Peripheral nerve fibers
#### Unmyelinated

| 1  |`Rattay1993`  |
|----|-----------------|
||Rattay, F., Aberham, M., 1993. Modeling axon membranes for functional electrical stimulation. IEEE Transactions on Biomedical Engineering 40, 1201–1209. https://doi.org/10.1109/10.250575

| 2  |`Schild1994`  |
|----|-----------------|
||Schild, J.H., Clark, J.W., Hay, M., Mendelowitz, D., Andresen, M.C., Kunze, D.L., 1994. A- and C-type rat nodose sensory neurons: model interpretations of dynamic discharge characteristics. Journal of Neurophysiology 71, 2338–2358. https://doi.org/10.1152/jn.1994.71.6.2338

| 3  |`Schild1997`  |
|----|-----------------|
||Schild, J.H., Kunze, D.L., 1997. Experimental and Modeling Study of Na+ Current Heterogeneity in Rat Nodose Neurons and Its Impact on Neuronal Discharge. Journal of Neurophysiology. https://doi.org/10.1152/jn.1997.78.6.3198

| 4  |`Sundt2015`  |
|----|-----------------|
||Sundt, D., Gamper, N., Jaffe, D.B., 2015. Spike propagation through the dorsal root ganglia in an unmyelinated sensory neuron: a modeling study. J Neurophysiol 114, 3140–3153. https://doi.org/10.1152/jn.00226.2015

| 5  |`ThioAutonomic2024`|
|----|-------------------|
||Thio, B.J., Titus, N.D., Pelot, N.A., Grill, W.M., 2024. Reverse-engineered models reveal differential membrane properties of autonomic and cutaneous unmyelinated fibers. PLOS Computational Biology 20, e1012475. https://doi.org/10.1371/journal.pcbi.1012475

| 6  |`ThioCutaneous2024`|
|----|-------------------|
||Thio, B.J., Titus, N.D., Pelot, N.A., Grill, W.M., 2024. Reverse-engineered models reveal differential membrane properties of autonomic and cutaneous unmyelinated fibers. PLOS Computational Biology 20, e1012475. https://doi.org/10.1371/journal.pcbi.1012475

| 7  |`Tigerholm2014`  |
|----|-----------------|
||Tigerholm, J., Petersson, M.E., Obreja, O., Lampert, A., Carr, R., Schmelz, M., Fransén, E., 2014. Modeling activity-dependent changes of axonal spike conduction in primary afferent C-nociceptors. J Neurophysiol 111, 1721–1735. https://doi.org/10.1152/jn.00777.2012


#### Myelinated

##### MRG and surrogate

| 1  |`exactMRG`|
|----|----------|
||McIntyre, C.C., Richardson, A.G., Grill, W.M., 2002. Modeling the Excitability of Mammalian Nerve Fibers:  Influence of Afterpotentials on the Recovery Cycle. Journal of Neurophysiology 87, 995–1006. https://doi.org/10.1152/jn.00353.2001
||McIntyre, C.C., Grill, W.M., Sherman, D.L., Thakor, N.V., 2004. Cellular Effects of Deep Brain Stimulation: Model-Based Analysis of Activation and Inhibition. Journal of Neurophysiology 91, 1457–1469. https://doi.org/10.1152/jn.00989.2003 ($2 \ \mu m$)
||Pelot, N.A., Behrend, C.E., Grill, W.M., 2017. Modeling the response of small myelinated axons in a compound nerve to kilohertz frequency signals. J Neural Eng 14, 046022. https://doi.org/10.1088/1741-2552/aa6a5f ($1 \ \mu m$)



|2   |`bigMRG` (interpolation; diameter ≥ 5.7 µm)|
|----|--------|
||Musselman, E.D., Cariello, J.E., Grill, W.M., Pelot, N.A., 2021. ASCENT (Automated Simulations to Characterize Electrical Nerve Thresholds): A pipeline for sample-specific computational modeling of electrical stimulation of peripheral nerves. PLoS Comput Biol 17, e1009285. https://doi.org/10.1371/journal.pcbi.1009285

|3   |`smolMRG` (thinly myelinated interpolation; diameter 1.011–5.7 µm)|
|----|--------|
||Peña, E., Pelot, N.A., Grill, W.M., 2024. Computational models of compound nerve action potentials: Efficient filter-based methods to quantify effects of tissue conductivities, conduction distance, and nerve fiber parameters. PLOS Computational Biology 20, e1011833. https://doi.org/10.1371/journal.pcbi.1011833

|4   |`SMF` (surrogate; diameter ≥ 5.7 µm)|
|----|------|
||Hussain, M.A., Grill, W.M., Pelot, N.A., 2024. Highly efficient modeling and optimization of neural fiber responses to electrical stimulation. Nat Commun 15, 7597. https://doi.org/10.1038/s41467-024-51709-8


##### Other

| 1  |`FHM` / `SENN`  |
|----|----------------|
||Reilly, J.P., Freeman, V.T., Larkin, W.D., 1985. Sensory Effects of Transient Electrical Stimulation - Evaluation with a Neuroelectric Model. IEEE Trans. Biomed. Eng. BME-32, 1001–1011. https://doi.org/10.1109/TBME.1985.325509

|2   |`Sweeney1987`|
|---|-----|
||Sweeney, J., Mortimer, J., Durand, D., 1987. Modeling of mammalian myelinated nerve for functional neuromuscular stimulation. Presented at the IEEE 9th Annual Conference of the Engineering in Medicine and Biology Society, pp. 1577–1578.


### Cortical neurons
#### Myelinated

The packaged cortical morphologies are modified derivatives of Blue Brain
Project/EPFL cortical neuron morphologies, rather than unmodified copies. Their
axonal arbors were modified and myelinated following the procedures described
in:

- Aberra, A.S., Peterchev, A.V., Grill, W.M., 2018. Biophysically realistic
  neuron models for simulation of cortical stimulation. Journal of Neural
  Engineering 15, 066023. https://doi.org/10.1088/1741-2552/aadbb1
- Aberra, A.S., Wang, B., Grill, W.M., Peterchev, A.V., 2020. Simulation of
  transcranial magnetic stimulation in head model with morphologically-realistic
  cortical neurons. Brain Stimulation 13, 175–189.
  https://doi.org/10.1016/j.brs.2019.10.002

These morphology files are licensed under
[CC BY-NC-SA 4.0](https://github.com/wmglab-duke/dendra-models/blob/main/LICENSES/CC-BY-NC-SA-4.0.txt). See
[THIRD_PARTY_NOTICES.md](https://github.com/wmglab-duke/dendra-models/blob/main/THIRD_PARTY_NOTICES.md) for their source, attribution,
modifications, and file scope.

| |Class|N||
|-|-----|-|-------|
|1|`L23_PC_cADpyr`|44|Layer 2/3 pyramidal cell, continuous adapting (pyramidal) e-type|
|2|`L5_TTPC2_cADpyr`|37|Layer 5 thick-tufted pyramidal cell, continuous adapting (pyramidal) e-type|
|3|`L4_LBC_cACint`|35|Layer 4 large basket interneuron, continuous accommodating e-type|
|4|`L4_LBC_dNAC`|35|Layer 4 large basket interneuron, delayed non-accommodating e-type|
|5|`L4_NBC_cACint`|35|Layer 4 nest basket interneuron, continuous accommodating e-type|
|6|`L4_NBC_dNAC`|35|Layer 4 nest basket interneuron, delayed non-accommodating e-type|
|7|`L4_SBC_bNAC`|35|Layer 4 small basket interneuron, burst non-accommodating e-type|
|8|`L4_SBC_cACint`|35|Layer 4 small basket interneuron, continuous accommodating e-type|


### Networks
#### Brain and cortical

|1 |`Yu2024`|
|---|-----|
||Yu, Gene J., Federico Ranieri, Vincenzo Di Lazzaro, Marc A. Sommer, Angel V. Peterchev, and Warren M. Grill. “Circuits and Mechanisms for TMS-Induced Corticospinal Waves: Connecting Sensitivity Analysis to the Network Graph.” PLOS Computational Biology 20, no. 12 (2024): e1012640. https://doi.org/10.1371/journal.pcbi.1012640.

|2 |`Kumaravelu2016`|
|---|-----|
||Kumaravelu, Karthik, David T. Brocker, and Warren M. Grill. “A BIOPHYSICAL MODEL OF THE CORTEX-BASAL GANGLIA-THALAMUS NETWORK IN THE 6-OHDA LESIONED RAT MODEL OF PARKINSON’S DISEASE.” Journal of Computational Neuroscience 40, no. 2 (2016): 207–29. https://doi.org/10.1007/s10827-016-0593-9.

#### Spine
|1 |`Zhang2014`|
|---|-----|
||Zhang, Tianhe C., John J. Janik, and Warren M. Grill. “Modeling Effects of Spinal Cord Stimulation on Wide-Dynamic Range Dorsal Horn Neurons: Influence of Stimulation Frequency and GABAergic Inhibition.” Journal of Neurophysiology 112, no. 3 (2014): 552–67. https://doi.org/10.1152/jn.00254.2014.

The Zhang 2014 Wind-Up vectors are obtained separately from ModelDB. This is
an explicit one-time download; importing Dendra Models and constructing other
models never accesses the network:

```python
from dendra_models.models.networks.zhang_2014 import download_windup_data

download_windup_data()
```

The downloader uses an immutable upstream revision, verifies SHA-256 hashes,
and stores the vectors in a revision-specific user cache. Pass `data_dir=` to
`load_windup_data()` or `build_windup_network()` to use an existing or modified
vector realization instead. See the
[Wind-Up data notes](https://github.com/wmglab-duke/dendra-models/blob/main/src/dendra_models/models/networks/zhang_2014/WINDUP_DATA.md)
for the cache location and vector semantics.

## License

Dendra Models code and the trained `SMF.pt` model parameters are distributed
under Duke University's custom license for non-commercial research and academic
testing. Commercial use, including industrially sponsored research, requires a
separate agreement with Duke's Office for Translation and Commercialization.
The complete Duke terms are in
[LICENSE.md](https://github.com/wmglab-duke/dendra-models/blob/main/LICENSE.md).

The cortical morphology files under
`src/dendra_models/models/cells/cortical/` are separately licensed adapted
material under
[CC BY-NC-SA 4.0](https://github.com/wmglab-duke/dendra-models/blob/main/LICENSES/CC-BY-NC-SA-4.0.txt). The Duke license
does not replace the Creative Commons terms for those files. File-level details
and required attribution are in
[THIRD_PARTY_NOTICES.md](https://github.com/wmglab-duke/dendra-models/blob/main/THIRD_PARTY_NOTICES.md).

## Contributing and support

See
[CONTRIBUTING.md](https://github.com/wmglab-duke/dendra-models/blob/main/CONTRIBUTING.md)
to set up a development environment and submit a GitHub pull request. Please
report security concerns using the private process in
[SECURITY.md](https://github.com/wmglab-duke/dendra-models/blob/main/SECURITY.md).
