<div align="center">
  <img src="docs/banner.png" alt="Dendra Models">
</div>

***

Reference neuronal models implemented in Dendra.

> [!IMPORTANT]
> dendra-models requires Dendra 0.27 or later. Its standard installation
> includes the native CPU solvers used by tree and extracellular models.

## Installation

Clone this repository, navigate to its root, and run:

```sh
python -m pip install .
```

This installs Dendra with its `solvers` extra, including
[`dendra-solvers`](https://pypi.org/project/dendra-solvers/). On supported
platforms, pip uses a prebuilt solver wheel. These native solvers are the
default CPU backends for Dendra Models' cable, tree, and extracellular models.

## Accessing models

Models are available from `dendra_models.models`:

```python
from dendra_models.models import Tigerholm2014
```

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
[CC BY-NC-SA 4.0](LICENSES/CC-BY-NC-SA-4.0.txt). See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for their source, attribution,
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
[Wind-Up data notes](src/dendra_models/models/networks/zhang_2014/WINDUP_DATA.md)
for the cache location and vector semantics.

## License

Dendra Models code and the trained `SMF.pt` model parameters are distributed
under Duke University's custom license for non-commercial research and academic
testing. Commercial use, including industrially sponsored research, requires a
separate agreement with Duke's Office for Translation and Commercialization.
The complete Duke terms are in [LICENSE.md](LICENSE.md).

The cortical morphology files under
`src/dendra_models/models/cells/cortical/` are separately licensed adapted
material under [CC BY-NC-SA 4.0](LICENSES/CC-BY-NC-SA-4.0.txt). The Duke license
does not replace the Creative Commons terms for those files. File-level details
and required attribution are in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Contributing and support

See [CONTRIBUTING.md](CONTRIBUTING.md) to set up a development environment and
submit a GitHub pull request. Please report security concerns using the private
process in [SECURITY.md](SECURITY.md).
