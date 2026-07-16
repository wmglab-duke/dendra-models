import torch

from dendra.models.mechanisms._mechanism import Mechanism as M, VoltageProcess as V, Synapse as Syn
from dendra.models.mechanisms._state import State as S
from dendra.models.mechanisms.ops import *
from dendra.models.networks.spiking import sigmoid_ste


class esser_states(S):
    S.STATE("v_iaf", "theta", "A", "B")
    S.ASSIGNED("i_total", "i_spike")
    S.DERIVATIVE("v_iaf' = -((i_total / tau_m) + (i_spike / tau_spike))")
    S.DERIVATIVE("theta' = (-(theta - theta_eq) + C*(v_iaf - theta_eq))/tau_theta")
    S.DERIVATIVE("A' = -A / tau1")
    S.DERIVATIVE("B' = -B / tau2")

    S.RANGE(
        gNa_leak=0.14,
        gK_leak=1.0,
        gspike=0.0,
        C = 0.85,
        theta_eq = -53.0,
        tau_theta = 2.0,
        tau_spike = 1.75,
        tau_syn = 2.0,
        tau_m = 15.0,
        ena_iaf = 30.0,
        ek_iaf = -90.0,
        v_iaf0 = -77.5,
        tau1 = 0.1,
        tau2 = 0.2,
        e = 0.0,
        i_stim = 0.0,
    )

    def breakpoint(self, v, states):
        ina_iaf = self.gNa_leak*(v-self.ena_iaf)
        ik_iaf = self.gK_leak*(v-self.ek_iaf)
        ispike = self.gspike*(v-self.ek_iaf)
        g = states["B"] - states["A"]
        i_noise = g * (v - self.e)
        i_total = ina_iaf+ik_iaf-self.i_stim+i_noise+self.i_ampa+self.i_nmda+self.i_gaba_a+self.i_gaba_b
        return {"i_total": i_total, "i_spike": ispike}
    
    def inf(self, v):
        return {
            "v_iaf": self.v_iaf0.detach().clone(),
            "theta": self.theta_eq.detach().clone(),
            "A": torch.zeros_like(v),
            "B": torch.zeros_like(v)
        }


class esser_mech_h(V, Syn):
    V.STATE(esser_states)
    V.RANGE(tspike=2.0)
    V.PARAMETER(tau_gate=0.5, ste_scale=1.0)
    V.BUFFER("factor", "spikes", "h_prev", "time_left")

    def update_v(self, v):
        return self.v_iaf
    
    def initial(self, v):
        tau1 = self.DE["esser_states"].tau1
        tau2 = self.DE["esser_states"].tau2
        tp = (tau1 * tau2) / (tau2 - tau1) * log(tau2 / tau1)
        factor = -exp(-tp / tau1) + exp(-tp / tau2)
        self.factor = 1 / factor
        self.spikes = torch.zeros_like(v)
        self.h_prev = torch.zeros_like(v)   # store previous *gate* (continuous)
        self.time_left = torch.zeros_like(v)

    def net_receive(self, weights, netcon):
        weights = weights * self.factor
        self.A = self.A + weights
        self.B = self.B + weights

        v  = self.v_iaf
        th = self.theta
        ena = self.DE["esser_states"].ena_iaf

        # ----- gating -----
        # vaux = v - th
        tau_on = self.tau_gate.clamp_min(1e-3)
        gate = torch.sigmoid((v - th) / tau_on)
        old_h = self.h_prev
        eligible = (self.time_left <= 0)

        rising = (gate > 0.5) & (old_h <= 0.5) & eligible

        # Soft threshold-crossing detector:
        above_now = torch.sigmoid((gate - 0.5) / tau_on)
        below_prev = torch.sigmoid((0.5 - old_h) / tau_on)
        rise_prob = above_now * below_prev * eligible.to(gate.dtype)

        # self.spike_gate = rise_prob
        self.spikes = rising.to(v.dtype) + self.ste_scale * (rise_prob - rise_prob.detach())

        # Update memory AFTER computing rising/rise_soft
        self.h_prev = gate

        # ----- boxcar / spike conductance timer -----
        time_left = torch.where(rising, self.tspike.expand_as(self.time_left), self.time_left)
        self.time_left = torch.clamp(time_left - self.dt, min=0.0)
        self.DE["esser_states"].gspike = (self.time_left > 0).to(v.dtype)

        # ----- snap toward E_Na at onset (hard path semantics) -----
        self.v_iaf = torch.where(rising, ena.expand_as(v), v)
        self.theta = torch.where(rising, ena.expand_as(th), th)


class esser_mech_s(V, Syn):
    V.STATE(esser_states)
    V.RANGE(tspike=2.0)
    V.PARAMETER(tau_gate=0.5, alpha_peak=1.0)
    V.BUFFER("factor", "spikes", "g_prev", "time_left")

    def update_v(self, v):
        return self.v_iaf
    
    def initial(self, v):
        tau1 = self.DE["esser_states"].tau1
        tau2 = self.DE["esser_states"].tau2
        tp = (tau1 * tau2) / (tau2 - tau1) * log(tau2 / tau1)
        factor = -exp(-tp / tau1) + exp(-tp / tau2)
        self.factor = 1 / factor
        self.spikes = torch.zeros_like(v)
        self.g_prev = torch.zeros_like(v)
        self.time_left = torch.zeros_like(v)

    def net_receive(self, weights, netcon):
        weights = weights * self.factor
        self.A = self.A + weights
        self.B = self.B + weights

        v = self.v_iaf
        th = self.theta

        ena = self.DE["esser_states"].ena_iaf
        gspike = self.DE["esser_states"].gspike
        tau_spike = self.DE["esser_states"].tau_spike

        # spiking
        vaux = v - th
        gate = torch.sigmoid(vaux / self.tau_gate.clamp_min(1e-6))
        rise = torch.relu(gate - self.g_prev)
        self.g_prev = gate

        k = 10.0
        gspike = gspike + self.dt * (-gspike / tau_spike + k * rise / self.dt)
        self.DE["esser_states"].gspike = torch.clamp(gspike, 0.0, 1.0)

        # Optional pull toward peak at onset (keeps it closer to the mod file's snap)
        self.v_iaf = v + self.alpha_peak * rise * (ena - v)

        self.spikes = rise.to(v.dtype)  # 0/1 in forward
