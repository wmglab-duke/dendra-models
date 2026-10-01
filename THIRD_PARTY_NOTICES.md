# Third-party notices

This file identifies material in Dendra Models that is distributed under terms
different from the Duke University license in `LICENSE.md`.

## Blue Brain Project cortical morphology derivatives

**Covered files:** Every `.gml` file under
`src/dendra_models/models/cells/cortical/`.

**Source and attribution:** These files are derived from cortical neuron
morphologies made available by the Blue Brain Project at EPFL through its
Neocortical Microcircuit and Somatosensory Cortex resources:

- Blue Brain Project/EPFL, *Somatosensory Cortex Portal*,
  https://bbp.epfl.ch/sscx-portal/ (accessed 1 October 2026).
- Markram, H., Muller, E., Ramaswamy, S., Reimann, M.W., Abdellah, M.,
  Sanchez, C.A., et al., 2015. Reconstruction and simulation of neocortical
  microcircuitry. *Cell* 163, 456–492.
  https://doi.org/10.1016/j.cell.2015.09.029

The source portal displays the copyright notice “© Blue Brain Project/EPFL
2005–2024.” It identifies its HOC code, Python code, MOD code, and cell
morphologies as licensed under the Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International license.

**Modifications:** The files packaged here are adapted material, not unchanged
copies of the Blue Brain morphologies. Their axonal arbors were modified and
myelinated following the procedures described in:

- Aberra, A.S., Peterchev, A.V., Grill, W.M., 2018. Biophysically realistic
  neuron models for simulation of cortical stimulation. *Journal of Neural
  Engineering* 15, 066023. https://doi.org/10.1088/1741-2552/aadbb1
- Aberra, A.S., Wang, B., Grill, W.M., Peterchev, A.V., 2020. Simulation of
  transcranial magnetic stimulation in head model with morphologically-realistic
  cortical neurons. *Brain Stimulation* 13, 175–189.
  https://doi.org/10.1016/j.brs.2019.10.002

The resulting morphologies were converted to Dendra's GML representation.

**License:** The covered files are distributed under the Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International license
(CC BY-NC-SA 4.0), available in
[`LICENSES/CC-BY-NC-SA-4.0.txt`](LICENSES/CC-BY-NC-SA-4.0.txt) and at
https://creativecommons.org/licenses/by-nc-sa/4.0/. The Duke University license
in `LICENSE.md` does not apply to the covered files.
