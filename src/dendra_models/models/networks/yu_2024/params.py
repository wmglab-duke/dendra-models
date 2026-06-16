params = {}

# network synapse configuration
params["netcon_delay_backend"] = "dense" # options: "dense", "sparse_calendar"

# globals
params["syn_delay"] = 0.5
params["conduction_velocity"] = 570 # units: microns/ms
params["pscale"] = 1.0
params["seed"] = 1234
params["fp32"] = False
params["forbid_autapses"] = True
params["scale"] = 1 # scale the whole network

# geometry & number

params["diameter_column"] = 500 # units: microns
params["spacing_column"] = 50 # units: microns

params["depth"] = { # um
    "L23": (300, 1350),
    "I23": (300, 1350),
    "L5": (1350, 2250),
    "I5": (1350, 2250),
    "L6": (2250, 3000),
    "I6": (2250, 3000)
}   

params['cells'] = {}

params['cells']['celltypes'] = [
    'L23',
    'L5',
    'L6',
    'I23',
    'I5',
    'I6'
]

params['cells']['n_per_column'] = {
    "L23": 2,
    "L5": 2,
    "L6": 2,
    "I23": 1,
    "I5": 1,
    "I6": 1,
}

# stochastic input

params['netstim'] = {}
params['netstim']['n_per_group'] = {
    "L23": 79,
    "L5": 79,
    "L6": 79,
    "I23": 79,
    "I5": 79,
    "I6": 79
}
params['netstim']['rate'] = 0.25 # Hz

# firing rates
params['FR'] = {}
FR = params['FR']
FR["L23"] = 3
FR["L5"] = 5
FR["L6"] = 5
FR["I23"] = 10
FR["I5"] = 15
FR["I6"] = 15

# synapses
params["syn"] = {}
params["syn"]["AMPA"] = {"tau1": 0.5, "tau2": 2.4, "e": 0}
params["syn"]["NMDA"] = {"tau1": 4, "tau2": 40, "e": 0}
params["syn"]["GABAA"] = {"tau1": 1, "tau2": 7, "e": -70}
params["syn"]["GABAB"] = {"tau1": 60, "tau2": 200, "e": -90}

params["syn"]["gpeak"] = {
    "AMPA": 0.1,
    "NMDA": 0.1,
    "GABAA": 0.33,
    "GABAB": 0.132/4
}


# -- connectivity --
params["con"] = {}
params["con"]["p"] = {}
params["con"]["p"]["L23"] = {
    "L23": 0.05,
    "L5": 1,
    "L6": 1,
    "I23": 0.05,
    "I5": 1,
    "I6": 1,
}

params["con"]["p"]["L5"] = {
    "L23": 1,
    "L5": 0.025,
    "L6": 1,
    "I23": 1,
    "I5": 0.025,
    "I6": 1,
}

params["con"]["p"]["L6"] = {
    "L5": 1, 
    "L6": 0.025, 
    "I5": 1, 
    "I6": 0.025
}

params["con"]["p"]["I23"] = {
    "L23": 0.2, 
    "L5": 0.25, 
    "L6": 0.25, 
    "I23": 0.2
}

params["con"]["p"]["I5"] = {
    "L5": 0.2, 
    "I5": 0.2
}

params["con"]["p"]["I6"] = {
    "L6": 0.2, 
    "I6": 0.2
}

params["con"]["p"]["L23input"] = {"L23": 0.125}
params["con"]["p"]["L5input"] = {"L5": 0.125}
params["con"]["p"]["L6input"] = {"L6": 0.125}
params["con"]["p"]["I23input"] = {"I23": 0.125}
params["con"]["p"]["I5input"] = {"I5": 0.125}
params["con"]["p"]["I6input"] = {"I6": 0.125}

params["con"]["strength"] = {}
params["con"]["strength"]["L23"] = {
    "L23": {"AMPA": 2, "NMDA": 2},
    "L5": {"AMPA": 2, "NMDA": 2},
    "L6": {"AMPA": 0.5, "NMDA": 0.5},
    "I23": {"AMPA": 2, "NMDA": 2},
    "I5": {"AMPA": 2, "NMDA": 2},
    "I6": {"AMPA": 0.5, "NMDA": 0.5},
}

params["con"]["strength"]["L5"] = {
    "L23": {"AMPA": 1, "NMDA": 1},
    "L5": {"AMPA": 2, "NMDA": 2},
    "L6": {"AMPA": 1, "NMDA": 1},
    "I23": {"AMPA": 1, "NMDA": 1},
    "I5": {"AMPA": 2, "NMDA": 2},
    "I6": {"AMPA": 1, "NMDA": 1},
}

params["con"]["strength"]["L6"] = {
    "L5": {"AMPA": 0.25, "NMDA": 0.25},
    "L6": {"AMPA": 2, "NMDA": 2},
    "I5": {"AMPA": 0.25, "NMDA": 0.25},
    "I6": {"AMPA": 2, "NMDA": 2},
}

params["con"]["strength"]["I23"] = {
    "L23": {"GABAA": 2, "GABAB": 1},
    "L5": {"GABAA": 1, "GABAB": 0.5},
    "L6": {"GABAA": 1, "GABAB": 0.5},
    "I23": {"GABAA": 2, "GABAB": 1},
}

params["con"]["strength"]["I5"] = {
    "L5": {"GABAA": 1, "GABAB": 1},
    "I5": {"GABAA": 1, "GABAB": 1},
}

params["con"]["strength"]["I6"] = {
    "L6": {"GABAA": 1, "GABAB": 1},
    "I6": {"GABAA": 1, "GABAB": 1},
}

params["con"]["strength"]["L23input"] = {"L23": {"AMPA": 1, "NMDA": 1}}
params["con"]["strength"]["L5input"] = {"L5": {"AMPA": 1, "NMDA": 1}}
params["con"]["strength"]["L6input"] = {"L6": {"AMPA": 1, "NMDA": 1}}
params["con"]["strength"]["I23input"] = {"I23": {"AMPA": 1, "NMDA": 1}}
params["con"]["strength"]["I5input"] = {"I5": {"AMPA": 1, "NMDA": 1}}
params["con"]["strength"]["I6input"] = {"I6": {"AMPA": 1, "NMDA": 1}}

params["con"]["sigma"] = {}
params["con"]["sigma"]["L23"] = {
    "L23": 12 * 50 / 3,
    "L5": 2 * 50 / 3,
    "L6": 2 * 50 / 3,
    "I23": 12 * 50 / 3,
    "I5": 2 * 50 / 3,
    "I6": 2 * 50 / 3,
}

params["con"]["sigma"]["L5"] = {
    "L23": 2 * 50 / 3,
    "L5": 12 * 50 / 3,
    "L6": 2 * 50 / 3,
    "I23": 2 * 50 / 3,
    "I5": 12 * 50 / 3,
    "I6": 2 * 50 / 3,
}

params["con"]["sigma"]["L6"] = {
    "L5": 2 * 50 / 3, 
    "L6": 9 * 50 / 3, 
    "I5": 2 * 50 / 3, 
    "I6": 9 * 50 / 3
}

params["con"]["sigma"]["I23"] = {
    "L23": 7 * 50 / 3, 
    "L5": 2 * 50 / 3, 
    "L6": 2 * 50 / 3, 
    "I23": 7 * 50 / 3
}

params["con"]["sigma"]["I5"] = {
    "L5": 7 * 50 / 3, 
    "I5": 7 * 50 / 3
}

params["con"]["sigma"]["I6"] = {
    "L6": 7 * 50 / 3, 
    "I6": 7 * 50 / 3
}

params["con"]["sigma"]["L23input"] = {"L23": 8 * 50 / 3}
params["con"]["sigma"]["L5input"] = {"L5": 8 * 50 / 3}
params["con"]["sigma"]["L6input"] = {"L6": 8 * 50 / 3}
params["con"]["sigma"]["I23input"] = {"I23": 8 * 50 / 3}
params["con"]["sigma"]["I5input"] = {"I5": 8 * 50 / 3}
params["con"]["sigma"]["I6input"] = {"I6": 8 * 50 / 3}

params["con"]["delay_mean"] = {}
params["con"]["delay_mean"]["L23"] = {
    "L23": 1.0,
    "L5": 2.0,
    "L6": 3.0,
    "I23": 1.0,
    "I5": 2.0,
    "I6": 3.0,
}

params["con"]["delay_mean"]["L5"] = {
    "L23": 2.0,
    "L5": 1.0,
    "L6": 2.0,
    "I23": 2.0,
    "I5": 1.0,
    "I6": 2.0,
}

params["con"]["delay_mean"]["L6"] = {"L5": 2.0, "L6": 1.0, "I5": 2.0, "I6": 1.0}

params["con"]["delay_mean"]["I23"] = {"L23": 1.0, "L5": 2.0, "L6": 3.0, "I23": 1.0}

params["con"]["delay_mean"]["I5"] = {"L5": 1.0, "I5": 1.0}

params["con"]["delay_mean"]["I6"] = {"L6": 1.0, "I6": 1.0}

params["con"]["delay_mean"]["L23input"] = {"L23": 1.67}
params["con"]["delay_mean"]["L5input"] = {"L5": 1.67}
params["con"]["delay_mean"]["L6input"] = {"L6": 1.67}
params["con"]["delay_mean"]["I23input"] = {"I23": 1.67}
params["con"]["delay_mean"]["I5input"] = {"I5": 1.67}
params["con"]["delay_mean"]["I6input"] = {"I6": 1.67}

params["con"]["delay_std"] = {}
params["con"]["delay_std"]["L23"] = {
    "L23": 0.1,
    "L5": 0.61,
    "L6": 1.68,
    "I23": 0.1,
    "I5": 0.61,
    "I6": 1.68,
}

params["con"]["delay_std"]["L5"] = {
    "L23": 0.61,
    "L5": 0.1,
    "L6": 1.57,
    "I23": 0.61,
    "I5": 0.1,
    "I6": 1.57,
}

params["con"]["delay_std"]["L6"] = {"L5": 1.57, "L6": 0.1, "I5": 1.57, "I6": 0.1}

params["con"]["delay_std"]["I23"] = {"L23": 0.1, "L5": 0.61, "L6": 1.68, "I23": 0.1}

params["con"]["delay_std"]["I5"] = {"L5": 0.1, "I5": 0.1}

params["con"]["delay_std"]["I6"] = {"L6": 0.1, "I6": 0.1}

params["con"]["delay_std"]["L23input"] = {"L23": 0.4}
params["con"]["delay_std"]["L5input"] = {"L5": 0.4}
params["con"]["delay_std"]["L6input"] = {"L6": 0.4}
params["con"]["delay_std"]["I23input"] = {"I23": 0.4}
params["con"]["delay_std"]["I5input"] = {"I5": 0.4}
params["con"]["delay_std"]["I6input"] = {"I6": 0.4}

params["con"]["syn_types"] = {}
params["con"]["syn_types"]["L23"] = {
    "L23": ["AMPA", "NMDA"],
    "L5": ["AMPA", "NMDA"],
    "L6": ["AMPA", "NMDA"],
    "I23": ["AMPA", "NMDA"],
    "I5": ["AMPA", "NMDA"],
    "I6": ["AMPA", "NMDA"],
}

params["con"]["syn_types"]["L5"] = {
    "L23": ["AMPA", "NMDA"],
    "L5": ["AMPA", "NMDA"],
    "L6": ["AMPA", "NMDA"],
    "I23": ["AMPA", "NMDA"],
    "I5": ["AMPA", "NMDA"],
    "I6": ["AMPA", "NMDA"],
}

params["con"]["syn_types"]["L6"] = {
    "L5": ["AMPA", "NMDA"],
    "L6": ["AMPA", "NMDA"],
    "I5": ["AMPA", "NMDA"],
    "I6": ["AMPA", "NMDA"],
}

params["con"]["syn_types"]["I23"] = {
    "L23": ["GABAA", "GABAB"],
    "L5": ["GABAA", "GABAB"],
    "L6": ["GABAA", "GABAB"],
    "I23": ["GABAA", "GABAB"],
}

params["con"]["syn_types"]["I5"] = {"L5": ["GABAA", "GABAB"], "I5": ["GABAA", "GABAB"]}

params["con"]["syn_types"]["I6"] = {"L6": ["GABAA", "GABAB"], "I6": ["GABAA", "GABAB"]}

params["con"]["syn_types"]["L23input"] = {"L23": ["AMPA", "NMDA"]}
params["con"]["syn_types"]["L5input"] = {"L5": ["AMPA", "NMDA"]}
params["con"]["syn_types"]["L6input"] = {"L6": ["AMPA", "NMDA"]}
params["con"]["syn_types"]["I23input"] = {"I23": ["AMPA", "NMDA"]}
params["con"]["syn_types"]["I5input"] = {"I5": ["AMPA", "NMDA"]}
params["con"]["syn_types"]["I6input"] = {"I6": ["AMPA", "NMDA"]}
