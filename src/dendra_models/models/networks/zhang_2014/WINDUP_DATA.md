# Wind-Up vector realization

The Zhang 2014 translation uses eight connection and external-event vectors
from the Wind-Up realization in ModelDB 168414. The upstream repository does
not state redistribution terms, so these files are not included in Dendra
Models. Install them explicitly after installing the package:

```python
from dendra_models.models.networks.zhang_2014 import download_windup_data

data_dir = download_windup_data()
```

The downloader reads from immutable ModelDB commit
`f0ac25565d31b60fe676738c559d6ab5086c7308` and verifies every file with a
recorded SHA-256 digest. It stores the files in a revision-specific user cache;
set `DENDRA_MODELS_DATA_HOME` to choose a different cache root. Downloads occur
only when `download_windup_data()` is called.

To use an existing or modified realization instead, pass its directory to the
loader or builder:

```python
from dendra_models.models.networks.zhang_2014 import (
    build_windup_network,
    load_windup_data,
)

data = load_windup_data(data_dir="/path/to/WindUp")
network = build_windup_network(data=data)
```

The six parallel connection vectors define one row per NetCon:

- `FromVector.txt`
- `ToVector.txt`
- `SynapseVector.txt`
- `WeightVector.txt`
- `DelayVector.txt`
- `ThresholdVector.txt`

`SpikeStatsVector.txt` declares the number of externally scheduled NetCons, one
entry count per schedule, and the nominal stop time. `SpikeTimesVector.txt`
contains each schedule followed by the original `-1e15` sentinel.

The spike-time values are target delivery times passed to NEURON
`NetCon.event(tdeliver)`, not ordinary source-spike times. See `network.py` for
the Dendra timing bridge.
