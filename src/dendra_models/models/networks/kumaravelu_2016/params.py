"""Default parameters for the Kumaravelu et al. CTX-BG-TH network."""

from copy import deepcopy


params = {}

# Reproducibility and simulation metadata.
params["seed"] = 1234
params["fp32"] = False
params["pd"] = 0.0
params["corstim"] = 0.0
params["n"] = 10
params["spike_threshold_hh"] = -10.0
params["spikedetect_hh"] = dict(tau_gate=0.5, ste_scale=1.0)

# Unit bridge for the HH population. MATLAB currents are numerically in
# uA/cm^2; the Dendra single-compartment voltage solver expects mA/cm^2.
params["units"] = dict(hh_current_scale=1.0e-3)

# Population layout. The MATLAB implementation uses n cells in every nucleus and
# n regular-spiking plus n fast-spiking cortical cells.
params["cells"] = {
    "hh_celltypes": ["TH", "STN", "GPe", "GPi", "StrD2", "StrD1"],
    "ctx_celltypes": ["CTX_RS", "CTX_FS"],
    "n_by_type": {
        "TH": 10,
        "STN": 10,
        "GPe": 10,
        "GPi": 10,
        "StrD2": 10,
        "StrD1": 10,
        "CTX_RS": 10,
        "CTX_FS": 10,
    },
    "v_init": {
        "TH": (-62.0, 5.0),
        "STN": (-62.0, 5.0),
        "GPe": (-62.0, 5.0),
        "GPi": (-62.0, 5.0),
        "StrD2": (-63.8, 5.0),
        "StrD1": (-63.8, 5.0),
        "CTX_RS": (-65.0, 0.0),
        "CTX_FS": (-65.0, 0.0),
    },
}

# Intrinsic current parameters. These mirror the MATLAB constants.
params["intrinsic"] = {
    "TH": dict(i_stim=1.2),
    "STN": dict(i_stim=0.0),
    "GPe": dict(i_stim=3.0),
    "GPi": dict(i_stim=3.0),
    "StrD2": dict(pd=0.0, i_stim=0.0),
    "StrD1": dict(pd=0.0, i_stim=0.0),
    "CTX_RS": dict(a=0.02, b=0.2, c=-65.0, d=8.0, v0=-65.0, i_stim=0.0),
    "CTX_FS": dict(a=0.1, b=0.2, c=-65.0, d=2.0, v0=-65.0, i_stim=0.0),
}

# Synaptic constants and pathway-specific shapes. ``peak`` is folded into the
# NetCon weights because the Dendra mechanisms are peak-normalized.
params["syn"] = {
    "gpeak": 0.43,
    "gpeak1": 0.3,
    "tau_alpha": 5.0,
    "tau_i_striatum": 13.0,
    "e": {
        "gaba_stn": -85.0,
        "exc": 0.0,
        "gaba_gpe": -85.0,
        "exc_gpi": 0.0,
        "gaba_gpi": -85.0,
        "gaba_th_str": -85.0,
        "gaba_striatum": -80.0,
    },
    "pathways": {
        # alpha-function pathways
        "TH_CTX": dict(kind="alpha", tau=5.0, delay=5.0, e=0.0, peak="gpeak"),
        "STN_GPi": dict(kind="alpha", tau=5.0, delay=1.5, e=0.0, peak="gpeak"),
        "GPe_GPi": dict(kind="alpha", tau=5.0, delay=3.0, e=-85.0, peak="gpeak1"),
        "GPe_GPe": dict(kind="alpha", tau=5.0, delay=1.0, e=-85.0, peak="gpeak1"),
        "GPi_TH": dict(kind="alpha", tau=5.0, delay=5.0, e=-85.0, peak="gpeak1"),
        "D2_GPe": dict(kind="alpha", tau=5.0, delay=5.0, e=-85.0, peak="gpeak1"),
        "D1_GPi": dict(kind="alpha", tau=5.0, delay=4.0, e=-85.0, peak="gpeak1"),
        "CTX_D2": dict(kind="alpha", tau=5.0, delay=5.1, e=0.0, peak="gpeak"),
        "CTX_D1": dict(kind="alpha", tau=5.0, delay=5.1, e=0.0, peak="gpeak"),
        "RS_FS": dict(kind="alpha", tau=5.0, delay=0.0, e=0.0, peak="gpeak"),
        "FS_RS": dict(kind="alpha", tau=5.0, delay=0.0, e=-85.0, peak="gpeak"),
        # double-exponential pathways
        "STN_GPe_AMPA": dict(kind="exp2", tau1=0.4, tau2=2.5, delay=2.0, e=0.0, peak="gpeak"),
        "STN_GPe_NMDA": dict(kind="exp2", tau1=2.0, tau2=67.0, delay=2.0, e=0.0, peak="gpeak"),
        "GPe_STN": dict(kind="exp2", tau1=0.4, tau2=7.7, delay=4.0, e=-85.0, peak="gpeak1"),
        "CTX_STN_AMPA": dict(kind="exp2", tau1=0.5, tau2=2.49, delay=5.9, e=0.0, peak="gpeak"),
        "CTX_STN_NMDA": dict(kind="exp2", tau1=2.0, tau2=90.0, delay=5.9, e=0.0, peak="gpeak"),
    },
}

# Coupling strengths multiplying the peak conductance in each pathway.
params["coupling"] = {
    "ggith": 0.112,
    "ggesn": 0.5,
    "gstrgpe": 0.5,
    "gstrgpi": 0.5,
    "ggigi": 0.5,
    "ggaba": 0.1,
    "gcorindrstr": 0.07,
    "gie": 0.2,
    "gthcor": 0.15,
    "gei": 0.1,
}


def default_params():
    """Return a deep copy of the default parameter dictionary."""
    return deepcopy(params)
