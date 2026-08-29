"""Protocol regressions for Zhang calcium dynamics and dynamic synapses."""

from __future__ import annotations

import dendra as dn
import torch

from ..mechanisms import AMPA_DynSyn, CaIntraCellDyn


DTYPE = torch.float64


def test_calcium_solver_state_is_the_shared_ion_write():
    with dn.ctx(JIT=0, DTYPE="float64", DEVICE="cpu"):
        model = dn.SingleCompartment(N=1, C=3, v_init=-65.0)
        model.insert(CaIntraCellDyn, cai_inf=5.0e-5, cai_tau=2.0)
        model.concentrations(cai0=2.0e-4, cao0=2.0)
        model.initialize()

        calcium = model.mech.CaIntraCellDyn
        assert calcium._initial_state_names == ("cai",)
        assert "cai_new" not in calcium._buffers
        torch.testing.assert_close(calcium.cai, model.mech.ca_ion.cai)
        torch.testing.assert_close(
            calcium.cai,
            torch.full_like(calcium.cai, 5.0e-5),
        )

        # Perturb the shared concentration and exercise an accepted public
        # Population step. The State transition and ion commit must stay in
        # lockstep without a private Mechanism._advance synchronization hook.
        calcium.cai.fill_(2.0e-4)
        model.mech.ca_ion.cai.fill_(2.0e-4)
        before = calcium.cai.clone()
        model.step(dt=0.1)

        assert torch.all(calcium.cai < before)
        torch.testing.assert_close(calcium.cai, model.mech.ca_ion.cai)


class _NetConStep:
    def __init__(self, step):
        self.global_step = torch.tensor(step, dtype=torch.long)


def _initialized_ampa():
    shape = (3,)
    voltage = torch.full(shape, -65.0, dtype=DTYPE)
    synapse = AMPA_DynSyn(
        "ampa",
        torch.tensor(37.0, dtype=DTYPE),
        torch.ones(shape, dtype=DTYPE),
        shape,
        shape,
    ).to(dtype=DTYPE)
    synapse._configure_timestep(0.1)
    synapse._init_buffers_s(voltage)
    return synapse


def test_dynamic_synapse_declares_and_initializes_its_state_roles():
    synapse = _initialized_ampa()

    assert synapse._derived_buffers == {"factor"}
    assert synapse._carry == ("P", "Use", "zhang2014_last_stp_step")
    assert synapse._carry_specs["zhang2014_last_stp_step"] == (
        torch.long,
        (1,),
    )
    assert torch.all(synapse.factor > 1.0)
    torch.testing.assert_close(synapse.P, torch.ones_like(synapse.P))
    torch.testing.assert_close(synapse.Use, torch.zeros_like(synapse.Use))
    assert synapse.zhang2014_last_stp_step.dtype is torch.long
    assert synapse.zhang2014_last_stp_step.tolist() == [-1]


def test_dynamic_synapse_recovers_once_per_step_and_preserves_event_updates():
    synapse = _initialized_ampa()
    step = _NetConStep(0)
    weights = torch.tensor([1.0, 0.0, 2.0], dtype=DTYPE)

    synapse.net_receive(weights, step)
    first_a = synapse.A.clone()
    first_p = synapse.P.clone()
    first_use = synapse.Use.clone()
    assert synapse.zhang2014_last_stp_step.tolist() == [0]
    assert first_a[0] > 0 and first_a[1] == 0 and first_a[2] > first_a[0]

    # A second delivery call in the same global step must not apply recovery
    # again; zero weights also leave the event-driven A/B states unchanged.
    synapse.net_receive(torch.zeros_like(weights), step)
    torch.testing.assert_close(synapse.A, first_a)
    torch.testing.assert_close(synapse.P, first_p)
    torch.testing.assert_close(synapse.Use, first_use)

    step.global_step.fill_(1)
    synapse.net_receive(torch.zeros_like(weights), step)
    assert synapse.zhang2014_last_stp_step.tolist() == [1]
    assert torch.all(synapse.P >= first_p)
    assert torch.all(synapse.Use <= first_use)
    torch.testing.assert_close(synapse.A, first_a)

