<div align="center">
  <img src="docs/banner.png">
</div>

***

Models implemented in AxonML.

### Installation instructions
---
0. Install [AxonML](https://gitlab.oit.duke.edu/mah148/axonml).
1. Clone this repository.
2. Navigate to the cloned directory.
3. `python -m pip install .`

### Accessing models
---
Models can be access via `axonml_models.models`, e.g., `from axonml_models.models import Tigerholm2014...`

# Available models
## Peripheral nerve fibers
### Unmyelinated

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


### Myelinated

#### MRG & Surrogate

| 1  |`exactMRG`|
|----|----------|
||McIntyre, C.C., Richardson, A.G., Grill, W.M., 2002. Modeling the Excitability of Mammalian Nerve Fibers:  Influence of Afterpotentials on the Recovery Cycle. Journal of Neurophysiology 87, 995–1006. https://doi.org/10.1152/jn.00353.2001
||McIntyre, C.C., Grill, W.M., Sherman, D.L., Thakor, N.V., 2004. Cellular Effects of Deep Brain Stimulation: Model-Based Analysis of Activation and Inhibition. Journal of Neurophysiology 91, 1457–1469. https://doi.org/10.1152/jn.00989.2003 ($2 \ \mu m$)
||Pelot, N.A., Behrend, C.E., Grill, W.M., 2017. Modeling the response of small myelinated axons in a compound nerve to kilohertz frequency signals. J Neural Eng 14, 046022. https://doi.org/10.1088/1741-2552/aa6a5f ($1 \ \mu m$)



|2   |`bigMRG` (intepolation; 5.7+ um diameter)|
|----|--------|
||Musselman, E.D., Cariello, J.E., Grill, W.M., Pelot, N.A., 2021. ASCENT (Automated Simulations to Characterize Electrical Nerve Thresholds): A pipeline for sample-specific computational modeling of electrical stimulation of peripheral nerves. PLoS Comput Biol 17, e1009285. https://doi.org/10.1371/journal.pcbi.1009285

|3   |`smolMRG` (thinly myelinated interpolation; 1.011 - 5.7 um diameter)|
|----|--------|
||Peña, E., Pelot, N.A., Grill, W.M., 2024. Computational models of compound nerve action potentials: Efficient filter-based methods to quantify effects of tissue conductivities, conduction distance, and nerve fiber parameters. PLOS Computational Biology 20, e1011833. https://doi.org/10.1371/journal.pcbi.1011833

|4   |`SMF` (interpolation; 5.7+ um diameter; surrogate)|
|----|------|
||Hussain, M.A., Grill, W.M., Pelot, N.A., 2024. Highly efficient modeling and optimization of neural fiber responses to electrical stimulation. Nat Commun 15, 7597. https://doi.org/10.1038/s41467-024-51709-8


#### Other

| 1  |`FHM` / `SENN`  |
|----|----------------|
||Reilly, J.P., Freeman, V.T., Larkin, W.D., 1985. Sensory Effects of Transient Electrical Stimulation - Evaluation with a Neuroelectric Model. IEEE Trans. Biomed. Eng. BME-32, 1001–1011. https://doi.org/10.1109/TBME.1985.325509

|2   |`Sweeney1987`|
|---|-----|
||Sweeney, J., Mortimer, J., Durand, D., 1987. Modeling of mammalian myelinated nerve for functional neuromuscular stimulation. Presented at the IEEE 9th Annual Conference of the Engineering in Medicine and Biology Society, pp. 1577–1578.


## Cortical neurons
### Myelinated

Myelination scheme from Aberra, A.S., Wang, B., Grill, W.M., Peterchev, A.V., 2020. Simulation of transcranial magnetic stimulation in head model with morphologically-realistic cortical neurons. Brain Stimul 13, 175–189. https://doi.org/10.1016/j.brs.2019.10.002

| |Class|N||
|-|-----|-|-------|
|1|`L23_PC_cADpyr`|44|Layer 2/3 pyramidal cell, continuous adapting (pyramidal) e-type|
|2|`L5_TTPC_cADpyr`|37|Layer 5 thick-tufted pyramidal cell, continuous adapting (pyramidal) e-type|
|3|`L4_LBC_cACint`|35|Layer 4 large basket interneuron, continuous accommodating e-type|
|4|`L4_LBC_dNAC`|35|Layer 4 large basket interneuron, delayed non-accommodating e-type|
|5|`L4_NBC_cACint`|35|Layer 4 nest basket interneuron, continuous accommodating e-type|
|6|`L4_NBC_dNAC`|35|Layer 4 nest basket interneuron, delayed non-accommodating e-type|
|7|`L4_SBC_bNAC`|35|Layer 4 small basket interneuron, burst non-accommodating e-type|
|8|`L4_SBC_cACint`|35|Layer 4 small basket interneuron, continuous accommodating e-type|
