"""
Synaptic and coupling mechanisms for the Kumaravelu et al. CTX-BG-TH model.

The MATLAB model writes synaptic conductances as point-neuron conductance
*densities* in the same current-density convention as the intrinsic HH currents.
For that reason the HH-target synapses below intentionally subclass Synapse
only, not PointProcess.  Dendra PointProcess mechanisms are lumped nA/uS
objects that are divided by compartment area by MechanismHandler; that is not
what these MATLAB conductance-density equations need.
"""

import torch

from dendra.models.mechanisms._mechanism import Synapse as Syn, Mechanism as M
from dendra.models.mechanisms._state import State as S
from dendra.models.mechanisms.ops import exp, log


class alpha_state(S):
    S.STATE("s", "z")
    S.RANGE(tau=5.0)
    S.DERIVATIVE("s' = z", "z' = -2.0 * z / tau - s / (tau * tau)")

    def inf(self, v):
        return {"s": torch.zeros_like(v), "z": torch.zeros_like(v)}


class A_state(S):
    S.STATE("A")
    S.RANGE(tau1=0.1)
    S.DERIVATIVE("A' = -A / tau1")

    def inf(self, v):
        return {"A": torch.zeros_like(v)}


class B_state(S):
    S.STATE("B")
    S.RANGE(tau2=10.0)
    S.DERIVATIVE("B' = -B / tau2")

    def inf(self, v):
        return {"B": torch.zeros_like(v)}


# The HH population uses Dendra's standard voltage solver, whose mechanism
# currents are in mA/cm^2. The MATLAB model's current-density convention is
# uA/cm^2, so HH-target synapses use current_scale=1e-3. The Izhikevich/scnv
# cortical population keeps the MATLAB numeric current convention and uses
# event-only conductance states that are read by the VoltageProcess equations.
DENDRA_CURRENT_SCALE = 1.0e-3


def _reshape_like_state(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Best-effort reshape for NetCon delivery tensors.

    NetCon normally delivers a tensor whose shape is exactly the synapse's
    ``shape_f``.  This helper keeps the mechanisms robust to scalar / flattened
    delivery tensors without attempting ambiguous local/full remapping.
    """
    if tuple(x.shape) == tuple(ref.shape):
        return x
    if x.numel() == ref.numel():
        return x.reshape_as(ref)
    if x.numel() == 1:
        return x.expand_as(ref)
    raise RuntimeError(
        f"Cannot reshape delivered weights with shape {tuple(x.shape)} "
        f"to synapse state shape {tuple(ref.shape)}."
    )


class alpha_syn_current(Syn):
    r"""
    Density alpha-function synapse with peak-normalized weights.

    On an event with weight ``w``, the conductance-density state follows

        s(t) = w * (t / tau) * exp(1 - t / tau).

    Pathway delays are supplied as NetCon delays.  This is *not* a Dendra
    PointProcess: weights are interpreted in the target population's
    conductance-density convention, not as lumped uS.
    """

    Syn.STATE(alpha_state)
    Syn.RANGE(e=0.0, current_scale=1.0)
    Syn.ASSIGNED("factor")
    Syn.NONSPECIFIC_CURRENT("i")

    def initial(self, v):
        tau = self.DE["alpha_state"].tau
        self.factor = exp(torch.ones_like(v)) / tau

    def i(self, v):
        return self.current_scale * self.s * (v - self.e)

    def net_receive(self, weights, netcon):
        weights = _reshape_like_state(weights, self.z)
        factor = _reshape_like_state(self.factor, self.z)
        self.z = self.z + weights * factor


class alpha_syn_event(Syn):
    r"""
    Event-driven alpha conductance state with no membrane-current contribution.

    This is used for CTX-target synapses in the scnv/Izhikevich population.
    The Izhikevich VoltageProcess owns the actual voltage state and reads this
    conductance through ``setreference`` to form currents inside its own ODE.
    Since this mechanism declares no NONSPECIFIC_CURRENT, MechanismHandler.i()
    does not include it in the current/conductance bookkeeping pass.
    """

    Syn.STATE(alpha_state)
    Syn.RANGE(e=0.0, current_scale=1.0)
    Syn.ASSIGNED("factor")

    def initial(self, v):
        tau = self.DE["alpha_state"].tau
        self.factor = exp(torch.ones_like(v)) / tau

    def net_receive(self, weights, netcon):
        weights = _reshape_like_state(weights, self.z)
        factor = _reshape_like_state(self.factor, self.z)
        self.z = self.z + weights * factor


class exp2syn_current(Syn):
    r"""
    Density double-exponential synapse with peak-normalized weights.

    This mirrors Dendra's stock exp2syn kinetics, but deliberately does not
    subclass PointProcess.  Weights are conductance densities, matching the
    MATLAB point-neuron equations.
    """

    Syn.STATE(A_state, B_state)
    Syn.RANGE(e=0.0, current_scale=1.0)
    Syn.ASSIGNED("factor")
    Syn.NONSPECIFIC_CURRENT("i")

    def initial(self, v):
        tau1 = self.DE["A_state"].tau1
        tau2 = self.DE["B_state"].tau2
        tp = (tau1 * tau2) / (tau2 - tau1) * log(tau2 / tau1)
        factor = -exp(-tp / tau1) + exp(-tp / tau2)
        self.factor = 1.0 / factor

    def i(self, v):
        return self.current_scale * (self.B - self.A) * (v - self.e)

    def net_receive(self, weights, netcon):
        weights = _reshape_like_state(weights, self.A)
        factor = _reshape_like_state(self.factor, self.A)
        weights = weights * factor
        self.A = self.A + weights
        self.B = self.B + weights


class striatal_gaba_gate_state(S):
    S.STATE("s")
    S.ASSIGNED("drive")
    S.RANGE(tau_i=13.0)
    S.DERIVATIVE("s' = drive * (1.0 - s) - s / tau_i")

    def breakpoint(self, v, states):
        drive = 2.0 * (1.0 + torch.tanh(v / 4.0))
        return {"drive": drive}

    def inf(self, v):
        drive = 2.0 * (1.0 + torch.tanh(v / 4.0))
        return {"s": drive / (drive + 1.0 / self.tau_i)}


class striatal_gaba_gate(M):
    """Continuous voltage-gated MSN recurrent GABA state S1c/S8."""
    M.STATE(striatal_gaba_gate_state)


class striatal_recurrent_gaba_refs(S):
    # Dummy state only to host setreference-bound tensors in a State object.
    S.STATE("x")
    S.RANGE(g=0.1, e=-80.0, current_scale=1.0)
    S.DERIVATIVE("x' = 0.0 * x")

    def inf(self, v):
        return {"x": torch.zeros_like(v)}


class striatal_recurrent_gaba(M):
    r"""
    Postsynaptic MSN-MSN inhibitory current driven by presynaptic S1c/S8 gates.

    Bind ``s0``, ``s1``, ... on ``DE['striatal_recurrent_gaba_refs']`` via
    setreference. The current is ``g * (v - e) * sum_k s_k``.
    """

    M.STATE(striatal_recurrent_gaba_refs)
    M.NONSPECIFIC_CURRENT("i")

    def i(self, v):
        p = self.DE["striatal_recurrent_gaba_refs"]
        total = torch.zeros_like(v)
        for k in range(16):
            ref = getattr(p, f"s{k}", None)
            if ref is not None:
                total = total + ref
        return p.current_scale * p.g * (v - p.e) * total
