# Threshold finding

`Thresholder` searches for one stimulus amplitude per Population lane. It uses
an activity callback to decide whether each trial activates the model. Supply
either a spatial field and a waveform (`space=` and `time=`), or precomputed
extracellular potentials (`bases=`).

This folder contains ImThera and Livanova cuff fields, fiber diameters, and
NEURON reference thresholds. Run the complete surrogate-model comparison from
this folder after installing Dendra Models:

```sh
cd examples/thresholding
python thresholds.py -f imthera
python thresholds.py -f livanova -v
```

The `-v` option plots computed versus reference thresholds and their error
distribution. The example runs on CPU by default; add `--cuda` when using a CUDA
installation.

## Use precomputed fields

The following smaller example runs the first two ImThera conditions from the
repository root. The model defines the fiber diameters and the callback defines
the activation criterion.

```python
from pathlib import Path

import dendra as dn
import numpy as np
from dendra.models.instruments import Thresholder
from dendra_models.models import SMF


data = Path("examples/thresholding/example_imthera")
diameters = np.load(data / "example_diameters_imthera.npy")
fields = np.memmap(
    data / "example_field_array_imthera.mmap",
    dtype="float32", mode="r", shape=(len(diameters), 1000, 101),
)
# Convert the stored condition-first fields to time-first solver inputs.
bases = np.array(fields[:2]).transpose(1, 0, 2)
dt = 0.005

model = SMF(diameters=diameters[:2], n_node=101)
activity = dn.callbacks.Active(dt=dt)
search = Thresholder(model, active=activity, bases=bases, rtol=0.01)
upper, lower = search.calculate_thresholds(dt=dt)
print(upper, lower)
```

The canonical `bases` layout is `(n_steps, *model.shape)`: time, optional batch
axes, Population lanes, and compartments. Here it is `(1000, 2, 101)`, describing
5 ms at `dt=0.005` ms. Each search amplitude multiplies one lane's entire
spatiotemporal field. Threshold units are therefore the current units used when
the unit-amplitude field was generated (mA for these cuff data).

`upper` is an activating amplitude and `lower` is a non-activating amplitude.
`rtol=0.01` requests `(upper - lower) / upper <= 0.01`; `atol=` optionally adds
an absolute interval tolerance. Failed or ignored searches return `NaN`.
The thresholder ignores Population lanes whose fields contain `NaN`.

## Use a waveform or a compiled Population

For a separable stimulus, use the same callback with
`Thresholder(model, active=activity, space=field, time=waveform, rtol=0.01)` and
pass `tstop=` and `dt=` to `calculate_thresholds`. `field` must broadcast to the
model's Population/compartment shape; `waveform` is a Dendra waveform such as
`dn.mono_rect(...)`.

A Population program returned by `dn.compile` is also accepted. Configure a
named activity callback when compiling and pass its name to `Thresholder`:

```python
model.initialize()
program = dn.compile(
    model, dt=dt, steps=bases.shape[0],
    callbacks={"activation": dn.callbacks.Active(dt=dt)},
    backend="eager",
)
compiled_search = Thresholder(
    program, active="activation", bases=bases, ub=0.2, rtol=0.01,
)
upper, lower = compiled_search.calculate_thresholds()
```

This continues the setup above. The eager backend keeps the example lightweight;
omit `backend=` to use the default optimizing compiler. Each trial starts from
the program's captured initial snapshot. A compiled program needs explicit
`ub=`; it does not infer a search bracket from the source model's diameters.
For `space=`/`time=` stimulation, compile with `extra=dn.VarInput()` so each trial
can supply its scaled field. Compiled Network threshold searches are currently
unsupported.

See the [Thresholder API](https://wmglab-duke.github.io/dendra/api/dendra.models.instruments.html)
and [functional execution guide](https://wmglab-duke.github.io/dendra/advanced/functional_execution.html)
for callback criteria, partitions and conduction-block diagnostics.
