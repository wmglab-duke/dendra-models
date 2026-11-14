# Chapman JB, Johnson EA, Kootsey JM. (1983)
# Electrical and Biochemical Properties of an Enzyme Model of the Sodium Pump
# J. Membrane Biol. 74, 139-153


import torch
from dendra.models.mechanisms import Mechanism as M
from torch.nn.functional import softplus


class _LegacyPumpParameters:
    """Load scalar pump fields from before their RANGE declarations."""

    _legacy_unbounded_parameters = ()
    _legacy_positive_parameters = ()

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        for name in (
            *self._legacy_unbounded_parameters,
            *self._legacy_positive_parameters,
        ):
            value_key = prefix + name
            value = state_dict.get(value_key)
            old_scalar_value = torch.is_tensor(value) and value.ndim == 0
            target = self._buffers.get(name)
            if old_scalar_value and target is not None and target.ndim != 0:
                # Preserve the saved effective value separately from the raw
                # leaf, which the optimizer may have updated since preparation.
                state_dict[value_key] = value.expand_as(target).clone()

            parameter_name = f"{name}_param"
            parameter = self._parameters.get(parameter_name)
            if parameter is None:
                continue

            parameter_key = prefix + parameter_name
            if name in self._legacy_positive_parameters:
                rho_key = parameter_key + ".rho"
                rho = state_dict.get(rho_key)
                if (
                    parameter_key in state_dict
                    or not torch.is_tensor(rho)
                    or rho.ndim != 0
                ):
                    continue
                # The old positive scalar leaf stores its softplus coordinate.
                state_dict[parameter_key] = softplus(state_dict.pop(rho_key))
            elif not old_scalar_value:
                continue

            raw = state_dict.get(parameter_key)
            if torch.is_tensor(raw) and raw.ndim == 0 and parameter.ndim != 0:
                state_dict[parameter_key] = raw.expand_as(parameter).clone()

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class nakpump(_LegacyPumpParameters, M):
    _legacy_unbounded_parameters = ("smalla",)
    _legacy_positive_parameters = ("b1",)

    M.RANGE(smalla=0.0, b1=1.0)

    M.USEION("na", read=["nai"], write=["ina"])
    M.USEION("k", read=["ko"], write=["ik"])
    M.EXPLICIT("ina", "ik")

    def _pump_current(self):
        return (
            self.smalla
            / ((1.0 + self.b1 / self.ko) ** 2)
            * (
                1.62 / (1.0 + (6.7 / (self.nai + 8.0)) ** 3)
                + 1.0 / (1.0 + (67.6 / (self.nai + 8.0)) ** 3)
            )
        )

    def ina(self, v):
        return -1.5 * self._pump_current()

    def ik(self, v):
        return self._pump_current()
