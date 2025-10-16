# coding: utf-8

"""
Helpful utils.
"""

from __future__ import annotations

__all__ = ["IF_NANO_V9", "IF_NANO_V10"]

from typing import Any
from functools import wraps

import law

from columnflow.types import Any
from columnflow.columnar_util import ArrayFunction, deferred_column
from columnflow.util import maybe_import

np = maybe_import("numpy")
ak = maybe_import("awkward")
coffea = maybe_import("coffea")

_logger = law.logger.get_logger(__name__)


def build_2e2mu(muons_plus, muons_minus, electrons_plus, electrons_minus):
    mu1, mu2, e1, e2 = ak.unzip(
        ak.cartesian([muons_plus, muons_minus, electrons_plus, electrons_minus])
    )
    z_mu = mu1 + mu2
    z_e = e1 + e2
    is_mu_closer = abs(z_mu.mass - 91.1876) < abs(z_e.mass - 91.1876)
    z1 = ak.where(is_mu_closer, z_mu, z_e)
    z2 = ak.where(is_mu_closer, z_e, z_mu)
    zz = z1 + z2
    return ak.zip({"z1": z1, "z2": z2, "zz": zz}, depth_limit=1)


def build_4sf(leptons_plus, leptons_minus):
    lp1, lp2 = ak.unzip(ak.combinations(leptons_plus, 2))
    lm1, lm2 = ak.unzip(ak.combinations(leptons_minus, 2))

    lp11, lm11 = ak.unzip(ak.cartesian([lp1, lm1]))
    lp12, lm12 = ak.unzip(ak.cartesian([lp1, lm2]))
    lp21, lm21 = ak.unzip(ak.cartesian([lp2, lm1]))
    lp22, lm22 = ak.unzip(ak.cartesian([lp2, lm2]))

    z11 = lp11 + lm11
    z12 = lp12 + lm12
    z22 = lp22 + lm22
    z21 = lp21 + lm21

    cond1 = abs(z11.mass - 91.1876) < abs(z12.mass - 91.1876)
    z1cand = ak.where(cond1, z11, z12)
    z2cand = ak.where(cond1, z22, z21)

    cond2 = abs(z1cand.mass - 91.1876) < abs(z21.mass - 91.1876)
    z1cand = ak.where(cond2, z1cand, z21)
    z2cand = ak.where(cond2, z2cand, z12)

    cond3 = abs(z1cand.mass - 91.1876) < abs(z22.mass - 91.1876)
    z1 = ak.where(cond3, z1cand, z22)
    z2 = ak.where(cond3, z2cand, z11)

    zz = z1 + z2

    return ak.zip({"z1": z1, "z2": z2, "zz": zz}, depth_limit=1)


# TODO #
# Task 2.
# Add ZZ selection following the official HZZ one
# Task 3.
# Define ZZ, Z1, Z2 observables and plot them
# Hint: Use the above to create Z1, Z2, ZZ candidates for 2e2mu
# Now write the one for the 4e and 4mu channels.
# The skeleton is:
# def build_4sf(leptons_plus, leptons_minus):
#    lepplus1, lepplus2 = ak.unzip(ak.combinations(leptons_plus, 2))
#    lepminus1, lepminus2 = ak.unzip(ak.combinations(leptons_minus, 2))
#    ...
#    ... DO ALL THE PAIRS ...
#    ... BUILD ALL Zs ...
#    ... FIND Z1: closest to true Z ...
#    ... FIND Z2 ...
#    ... DON'T FORGET TO LOOK AT ALL POSSIBILITIES ...
#    ...
#    return ak.zip({"z1": z1, "z2": z2, "zz": zz}, depth_limit=1)


def masked_sorted_indices(
    mask: ak.Array, sort_var: ak.Array, ascending: bool = False
) -> ak.Array:
    """
    Helper function to obtain the correct indices of an object mask
    """
    indices = ak.argsort(sort_var, axis=-1, ascending=ascending)
    return indices[mask[indices]]


def call_once_on_config(func=None, *, include_hash=False):
    """
    Parametrized decorator to ensure that function *func* is only called once for the config *config*.
    Can be used with or without parentheses.
    """
    if func is None:
        # If func is None, it means the decorator was called with arguments.
        def wrapper(f):
            return call_once_on_config(f, include_hash=include_hash)

        return wrapper

    @wraps(func)
    def inner(config, *args, **kwargs):
        tag = f"{func.__name__}_called"
        if include_hash:
            tag += f"_{func.__hash__()}"

        if config.has_tag(tag):
            return

        config.add_tag(tag)
        return func(config, *args, **kwargs)

    return inner


@deferred_column
def IF_NANO_V9(self, func: ArrayFunction) -> Any | set[Any]:
    return self.get() if func.config_inst.campaign.x.version == 9 else None


@deferred_column
def IF_NANO_V10(self, func: ArrayFunction) -> Any | set[Any]:
    return self.get() if func.config_inst.campaign.x.version >= 10 else None
