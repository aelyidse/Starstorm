"""Stealth technology simulation module for Starstorm SDK.

This module provides simulation capabilities for various stealth technologies
including metamaterial-based electromagnetic cloaking, thermal signature
management, and multispectral camouflage.
"""

from .metamaterial_simulation import MetamaterialSimulator, MetamaterialLayer, MetamaterialProperties
from .metamaterial_control import MetamaterialControlSystem
from .metamaterial_thermal import MetamaterialThermalManagement
from .aerogel_thermal import AerogelThermalInsulation
from .nanotube_structure import CarbonNanotubeStructure
from .electromagnetic_cloaking import ElectromagneticCloakingSimulator
