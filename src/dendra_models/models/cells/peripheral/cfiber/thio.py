import math

from dendra.models.core import Unmyelinated
from dendra.units import mm

from ..mech import (
    nav7,
    newnav8,
    nav9,
    bk,
    cav12,
    cav22,
    caextscale,
    caintscale,
    km,
    hcn,
    kv21,
    ka34,
    ka14,
    sk,
    nacx,
    naoi,
    koi,
    nakpumpSchild,
    extrapump,
    leak,
    get_mechanism,
)
from ._balance import register_thio_balance


class ThioAutonomic2024(Unmyelinated):
    Unmyelinated.RANGE(cm=1.326291192, rhoa=23.117539)

    def __init__(
        self,
        diameters=[1.0],
        L=5.0 * mm,
        dx=10,
        celsius=37.0,
        v_init=-58.5,
        integrator=None,
    ):
        super().__init__(diameters, L, dx, celsius, v_init, integrator)
        R = 8314  # molar gas constant
        F = 96485.3329  # Faraday's constant

        ko = 5.4
        ki = 145.0
        ek = ((R * (celsius + 273.15)) / F) * math.log(ko / ki)
        ki_real = 144.9
        ko_real = 5.6

        nao = 154.0
        nai = 8.9
        ena = ((R * (celsius + 273.15)) / F) * math.log(nao / nai)
        nai_real = 11.4

        self.ion_style("na", 1, 1)
        self.ion_style("k", 1, 1)

        self.insert(nav7, gbar=0.036813)
        self.insert(newnav8, gbar=0.075747)
        self.insert(nav9, gbar=0.000376)
        self.insert(bk, gbar=0.000156)
        self.insert(cav12, gbar=0.000004)
        self.insert(cav22, gbar=0.009546)
        self.insert(km, gbar=0.002864)
        self.insert(caextscale, lseg=(1e-4) * dx, fhspace=0.03)
        self.insert(caintscale, lseg=(1e-4) * dx)
        self.insert(hcn, gbar=0.002789)
        self.insert(kv21, gbar=0.005337)
        self.insert(ka34, gbar=0.000289)
        self.insert(ka14, gbar=0.000024)
        self.insert(sk, gbar=0.000006)
        self.insert(nacx, gbar=0.000210)
        self.insert(nakpumpSchild, gbar_INaKmax22=0.056316)
        self.insert(naoi)
        self.insert(koi)
        self.insert(leak)
        self.insert(extrapump)
        register_thio_balance(self)

        self.equilibria(ena=ena, ek=ek)
        self.concentrations(
            cao0=2.0,
            cai0=0.000117,
            ko0=ko_real,
            ki0=ki_real,
            nao0=nao,
            nai0=nai_real,
        )


class ThioCutaneous2024(Unmyelinated):
    Unmyelinated.RANGE(cm=1.326291192, rhoa=27.513088)

    def __init__(
        self,
        diameters=[1.0],
        L=5.0 * mm,
        dx=10,
        celsius=37.0,
        v_init=-58.5,
        integrator=None,
    ):
        super().__init__(diameters, L, dx, celsius, v_init, integrator)
        R = 8314  # molar gas constant
        F = 96485.3329  # Faraday's constant

        ko = 5.4
        ki = 145.0
        ek = ((R * (celsius + 273.15)) / F) * math.log(ko / ki)
        ki_real = 144.9
        ko_real = 5.6

        nao = 154.0
        nai = 8.9
        ena = ((R * (celsius + 273.15)) / F) * math.log(nao / nai)
        nai_real = 11.4

        self.ion_style("na", 1, 1)
        self.ion_style("k", 1, 1)

        self.insert(nav7, gbar=0.035663)
        self.insert(newnav8, gbar=0.115643)
        self.insert(nav9, gbar=0.000504)
        self.insert(bk, gbar=0.002016)
        self.insert(cav12, gbar=0.000188)
        self.insert(cav22, gbar=0.000361)
        self.insert(km, gbar=0.000003)
        self.insert(caextscale, lseg=(1e-4) * dx, fhspace=0.03)
        self.insert(caintscale, lseg=(1e-4) * dx)
        self.insert(hcn, gbar=0.000106)
        self.insert(kv21, gbar=0.327196)
        self.insert(ka34, gbar=0.001786)
        self.insert(ka14, gbar=0.000044)
        self.insert(sk, gbar=0.000755)
        self.insert(nacx, gbar=0.009242)
        self.insert(nakpumpSchild, gbar_INaKmax22=0.000456)
        self.insert(naoi)
        self.insert(koi)
        self.insert(leak)
        self.insert(extrapump)
        register_thio_balance(self)

        self.equilibria(ena=ena, ek=ek)
        self.concentrations(
            cao0=2.0,
            cai0=0.000117,
            ko0=ko_real,
            ki0=ki_real,
            nao0=nao,
            nai0=nai_real,
        )


class ThioCutaneousAugmented2024(Unmyelinated):
    Unmyelinated.RANGE(cm=1.326291192, rhoa=27.513088)

    def __init__(
        self,
        diameters=[1.0],
        L=5.0 * mm,
        dx=10,
        celsius=37.0,
        v_init=-58.5,
        integrator=None,
    ):
        super().__init__(diameters, L, dx, celsius, v_init, integrator)

        R = 8314  # molar gas constant
        F = 96485.3329  # Faraday's constant

        ko = 5.4
        ki = 145.0
        ek = ((R * (celsius + 273.15)) / F) * math.log(ko / ki)
        ki_real = 144.9
        ko_real = 5.6

        nao = 154.0
        nai = 8.9
        ena = ((R * (celsius + 273.15)) / F) * math.log(nao / nai)
        nai_real = 11.4

        self.ion_style("na", 1, 1)
        self.ion_style("k", 1, 1)

        self.insert(get_mechanism("nav7_augmented"), gbar=0.035663)
        self.insert(get_mechanism("newnav8_augmented"), gbar=0.115643)
        self.insert(get_mechanism("nav9_augmented"), gbar=0.000504)
        self.insert(get_mechanism("bk_augmented"), gbar=0.002016)
        self.insert(get_mechanism("cav12_augmented"), gbar=0.000188)
        self.insert(get_mechanism("cav22_augmented"), gbar=0.000361)
        self.insert(get_mechanism("km_augmented"), gbar=0.000003)
        self.insert(
            get_mechanism("caextscale_augmented"), lseg=(1e-4) * dx, fhspace=0.03
        )
        self.insert(get_mechanism("caintscale_augmented"), lseg=(1e-4) * dx)
        self.insert(get_mechanism("hcn_augmented"), gbar=0.000106)
        self.insert(get_mechanism("kv21_augmented"), gbar=0.327196)
        self.insert(get_mechanism("ka34_augmented"), gbar=0.001786)
        self.insert(get_mechanism("ka14_augmented"), gbar=0.000044)
        self.insert(get_mechanism("sk_augmented"), gbar=0.000755)
        self.insert(get_mechanism("nacx_augmented"), gbar=0.009242)
        self.insert(get_mechanism("nakpumpSchild_augmented"), gbar_INaKmax22=0.000456)
        self.insert(get_mechanism("naoi_augmented"))
        self.insert(get_mechanism("koi_augmented"))
        self.insert(leak)
        self.insert(get_mechanism("extrapump"))
        register_thio_balance(self)

        self.equilibria(ena=ena, ek=ek)
        self.concentrations(
            cao0=2.0,
            cai0=0.000117,
            ko0=ko_real,
            ki0=ki_real,
            nao0=nao,
            nai0=nai_real,
        )


class ThioCutaneousReduced(Unmyelinated):
    Unmyelinated.RANGE(cm=1.326291192, rhoa=27.513088)

    def __init__(
        self,
        diameters=[1.0],
        L=5.0 * mm,
        dx=10,
        celsius=37.0,
        v_init=-58.5,
        integrator=None,
    ):
        super().__init__(diameters, L, dx, celsius, v_init, integrator)
        R = 8314  # molar gas constant
        F = 96485.3329  # Faraday's constant

        ko = 5.4
        ki = 145.0
        ek = ((R * (celsius + 273.15)) / F) * math.log(ko / ki)
        ki_real = 144.9
        ko_real = 5.6

        nao = 154.0
        nai = 8.9
        ena = ((R * (celsius + 273.15)) / F) * math.log(nao / nai)
        nai_real = 11.4

        self.ion_style("na", 1, 1)
        self.ion_style("k", 1, 1)

        self.insert(nav7, gbar=0.035663)
        self.insert(newnav8, gbar=0.115643)
        self.insert(kv21, gbar=0.327196)
        self.insert(naoi)
        self.insert(koi)
        self.insert(leak)
        self.insert(extrapump)
        register_thio_balance(self)

        self.equilibria(ena=ena, ek=ek)
        self.concentrations(
            cao0=2.0,
            cai0=0.000117,
            ko0=ko_real,
            ki0=ki_real,
            nao0=nao,
            nai0=nai_real,
        )
