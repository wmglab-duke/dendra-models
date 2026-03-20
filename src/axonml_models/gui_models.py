try:
    from axonml.gui.models import register_model_node, model_param
    from axonml.units import mm

    from axonml_models.models import (
        Sundt2015,
        Tigerholm2014,
        Rattay1993,
        Schild1994,
        Schild1997,
        ThioCutaneous2024,
        SENN,
        Sweeney1987,
        bigMRG,
        smolMRG,
        exactMRG,
        SMF,
    )

    @register_model_node(
        key="thio_cutaneous_2024",
        label="Thio Cutaneous 2024",
        description="Cutaneous afferent from Thio et al., 2024.",
        parameters=[
            model_param("diameters", label="diameters", unit="µm", kind="number_array", default=[1.0], allow_tensor_input=True),
            model_param("L_mm", label="length", kind="number", default=5.0, unit="mm"),
            model_param("dx", label="dx", kind="number", default=10.0, unit="µm"),
            model_param("celsius", label="temperature", kind="number", default=37.0, unit="°C"),
            model_param("v_init", label="initial voltage", kind="number", default=-58.5, unit="mV"),
        ],
        category="C-fiber",
    )
    def build_thio_cutaneous_2024(
        diameters,
        L_mm=5.0,
        dx=10.0,
        celsius=37.0,
        v_init=-58.5,
        integrator=None,
    ):
        return ThioCutaneous2024(
            diameters=diameters,
            L=L_mm * mm,
            dx=dx,
            celsius=celsius,
            v_init=v_init,
            integrator=integrator,
        )


    @register_model_node(
        key="sundt_2015",
        label="Sundt 2015",
        description="Model of an unmyelinated sensory neuron from Sundt et al., 2015.",
        parameters=[
            model_param("diameters", label="diameters", kind="number_array", default=[1.0], allow_tensor_input=True),
            model_param("L_mm", label="length", kind="number", default=5.0, unit="mm"),
            model_param("dx", label="dx", kind="number", default=10.0, unit="µm"),
            model_param("celsius", label="temperature", kind="number", default=37.0, unit="°C"),
            model_param("v_init", label="initial voltage", kind="number", default=-60.0, unit="mV"),
        ],
        category="C-fiber",
    )
    def build_sundt_2015(
        diameters,
        L_mm=5.0,
        dx=10.0,
        celsius=37.0,
        v_init=-60.0,
        integrator=None,
    ):
        return Sundt2015(
            diameters=diameters,
            L=L_mm * mm,
            dx=dx,
            celsius=celsius,
            v_init=v_init,
            integrator=integrator,
        )


    @register_model_node(
        key="tigerholm_2014",
        label="Tigerholm 2014",
        description="Model of an unmyelinated nociceptive neuron from Tigerholm et al., 2014.",
        parameters=[
            model_param("diameters", label="diameters", kind="number_array", default=[1.0], allow_tensor_input=True),
            model_param("L_mm", label="length", kind="number", default=5.0, unit="mm"),
            model_param("dx", label="dx", kind="number", default=10.0, unit="µm"),
            model_param("celsius", label="temperature", kind="number", default=37.0, unit="°C"),
            model_param("v_init", label="initial voltage", kind="number", default=-55.0, unit="mV"),
        ],
        category="C-fiber",
    )
    def build_tigerholm_2014(
        diameters,
        L_mm=5.0,
        dx=10.0,
        celsius=37.0,
        v_init=-55.0,
        integrator=None,
    ):
        return Tigerholm2014(
            diameters=diameters,
            L=L_mm * mm,
            dx=dx,
            celsius=celsius,
            v_init=v_init,
            integrator=integrator,
        )


    @register_model_node(
        key="rattay_1993",
        label="Rattay 1993",
        description="Model of an unmyelinated neuron from Rattay, 1993.",
        parameters=[
            model_param("diameters", label="diameters", kind="number_array", default=[1.0], allow_tensor_input=True),
            model_param("L_mm", label="length", kind="number", default=5.0, unit="mm"),
            model_param("dx", label="dx", kind="number", default=10.0, unit="µm"),
            model_param("celsius", label="temperature", kind="number", default=37.0, unit="°C"),
            model_param("v_init", label="initial voltage", kind="number", default=-70.0, unit="mV"),
        ],
        category="C-fiber",
    )
    def build_rattay_1993(
        diameters,
        L_mm=5.0,
        dx=10.0,
        celsius=37.0,
        v_init=-70.0,
        integrator=None,
    ):
        return Rattay1993(
            diameters=diameters,
            L=L_mm * mm,
            dx=dx,
            celsius=celsius,
            v_init=v_init,
            integrator=integrator,
        )


    @register_model_node(
        key="schild_1997",
        label="Schild 1997",
        description="Model of an unmyelinated neuron from Schild, 1997.",
        parameters=[
            model_param("diameters", label="diameters", kind="number_array", default=[1.0], allow_tensor_input=True),
            model_param("L_mm", label="length", kind="number", default=5.0, unit="mm"),
            model_param("dx", label="dx", kind="number", default=10.0, unit="µm"),
            model_param("celsius", label="temperature", kind="number", default=37.0, unit="°C"),
            model_param("v_init", label="initial voltage", kind="number", default=-68.5, unit="mV"),
        ],
        category="C-fiber",
    )
    def build_schild_1997(  
        diameters,
        L_mm=5.0,
        dx=10.0,
        celsius=37.0,
        v_init=-68.5,
        integrator=None,
    ):
        return Schild1997(
            diameters=diameters,
            L=L_mm * mm,
            dx=dx,
            celsius=celsius,
            v_init=v_init,
            integrator=integrator,
        )


    @register_model_node(
        key="schild_1994",
        label="Schild 1994",
        description="Model of an unmyelinated neuron from Schild, 1994.",
        parameters=[
            model_param("diameters", label="diameters", kind="number_array", default=[1.0], allow_tensor_input=True),
            model_param("L_mm", label="length", kind="number", default=5.0, unit="mm"),
            model_param("dx", label="dx", kind="number", default=10.0, unit="µm"),
            model_param("celsius", label="temperature", kind="number", default=37.0, unit="°C"),
            model_param("v_init", label="initial voltage", kind="number", default=-46.5, unit="mV"),
        ],
        category="C-fiber",
    )
    def build_schild_1994(
        diameters,
        L_mm=5.0,
        dx=10.0,
        celsius=37.0,
        v_init=-46.5,
        integrator=None,
    ):
        return Schild1994(
            diameters=diameters,
            L=L_mm * mm,
            dx=dx,
            celsius=celsius,
            v_init=v_init,
            integrator=integrator,
        )


    # myelinated models

    @register_model_node(
        key="senn",
        label="SENN",
        description="Spatially Extended Node Model (SENN)",
        parameters=[
            model_param("diameters", label="diameters", kind="number_array", default=[10.0], unit="µm", allow_tensor_input=True),
            model_param("n_node", label="number of nodes", kind="integer", default=101),
            model_param("node_length", label="node length", kind="number", default=2.5, unit="µm"),
            model_param("celsius", label="temperature", kind="number", default=20.0, unit="°C"),
            model_param("v_init", label="initial voltage", kind="number", default=-70.0, unit="mV"),
        ],
        category="Myelinated",
    )
    def build_senn(
        diameters,
        n_node=101,
        node_length=2.5,
        celsius=20.0,
        v_init=-70.0,
        integrator=None,
    ):
        return SENN(
            diameters=diameters,
            n_node=n_node,
            node_length=node_length,
            celsius=celsius,
            v_init=v_init,
            integrator=integrator,
        )


    @register_model_node(
        key="sweeney_1987",
        label="Sweeney 1987",
        description="Model of a myelinated neuron from Sweeney et al., 1987.",
        parameters=[
            model_param("diameters", label="diameters", kind="number_array", default=[10.0], unit="µm", allow_tensor_input=True),
            model_param("n_node", label="number of nodes", kind="integer", default=101),
            model_param("node_length", label="node length", kind="number", default=1.5, unit="µm"),
            model_param("celsius", label="temperature", kind="number", default=37.0, unit="°C"),
            model_param("v_init", label="initial voltage", kind="number", default=-80.0, unit="mV"),
        ],
        category="Myelinated",
    )
    def build_sweeney_1987(
        diameters,
        n_node=101,
        node_length=1.5,
        celsius=37.0,
        v_init=-80.0,
        integrator=None,
    ):
        return Sweeney1987(
            diameters=diameters,
            n_node=n_node,
            node_length=node_length,
            celsius=celsius,
            v_init=v_init,
            integrator=integrator,
        )


    @register_model_node(
        key="big_mrg",
        label="MRG (large diameter)",
        description="Large diameter myelinated axon (>= 5.7 µm).",
        parameters=[
            model_param("diameters", label="diameters", kind="number_array", default=[10.0], unit="µm", allow_tensor_input=True),
            model_param("n_node", label="number of nodes", kind="integer", default=101),
            model_param("celsius", label="temperature", kind="number", default=37.0, unit="°C"),
            model_param("v_init", label="initial voltage", kind="number", default=-80.0, unit="mV"),
        ],
        category="Myelinated",
    )
    def build_big_mrg(
        diameters,
        n_node=101,
        celsius=37.0,
        v_init=-80.0,
        integrator=None,
    ):
        return bigMRG(
            diameters=diameters,
            n_node=n_node,
            celsius=celsius,
            v_init=v_init,
            integrator=integrator,
        )


    @register_model_node(
        key="smol_mrg",
        label="MRG (small diameter)",
        description="Small diameter myelinated axon (< 5.7 µm, > 1.011 µm).",
        parameters=[
            model_param("diameters", label="diameters", kind="number_array", default=[1.0], unit="µm", allow_tensor_input=True),
            model_param("n_node", label="number of nodes", kind="integer", default=101),
            model_param("celsius", label="temperature", kind="number", default=37.0, unit="°C"),
            model_param("v_init", label="initial voltage", kind="number", default=-80.0, unit="mV"),
        ],
        category="Myelinated",
    )
    def build_smol_mrg(
        diameters,
        n_node=101,
        celsius=37.0,
        v_init=-80.0,
        integrator=None,
    ):
        return smolMRG(
            diameters=diameters,
            n_node=n_node,
            celsius=celsius,
            v_init=v_init,
            integrator=integrator,
        )
    

    @register_model_node(
        key="exact_mrg",
        label="MRG (exact diameter)",
        description="Exact diameter myelinated axon.",
        parameters=[
            model_param("diameters", label="diameters", kind="number_array", default=[5.7], unit="µm", allow_tensor_input=True),
            model_param("n_node", label="number of nodes", kind="integer", default=101),
            model_param("celsius", label="temperature", kind="number", default=37.0, unit="°C"),
            model_param("v_init", label="initial voltage", kind="number", default=-80.0, unit="mV"),
        ],
        category="Myelinated",
    )
    def build_exact_mrg(
        diameters,
        n_node=101,
        celsius=37.0,
        v_init=-80.0,
        integrator=None,
    ):
        return exactMRG(
            diameters=diameters,
            n_node=n_node,
            celsius=celsius,
            v_init=v_init,
            integrator=integrator,
        )


    @register_model_node(
        key="SMF",
        label="SMF",
        description="Surrogate myelinated fiber (SMF) model from Hussain et al., 2024 (> 5 µm).",
        parameters=[
            model_param("diameters", label="diameters", kind="number_array", default=[10.0], unit="µm", allow_tensor_input=True),
            model_param("n_node", label="number of nodes", kind="integer", default=101),
            model_param("celsius", label="temperature", kind="number", default=37.0, unit="°C"),
            model_param("v_init", label="initial voltage", kind="number", default=-80.0, unit="mV"),
        ],
        category="Surrogate",
    )
    def build_smf(
        diameters,
        n_node=101,
        celsius=37.0,
        v_init=-80.0,
    ):
        return SMF(
            diameters=diameters,
            n_node=n_node,
            celsius=celsius,
            v_init=v_init,
        )
    
except ImportError:
    pass