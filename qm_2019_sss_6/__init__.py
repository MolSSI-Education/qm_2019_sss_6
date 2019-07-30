"""
qm_2019_sss_6
QM Group 6 MolSSI
"""

# Add imports here
from .NobleGasModel import *
from .fock_fast import *
from .mp2 import MP2
from .scf import scf


# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
