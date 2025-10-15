# coding: utf-8

"""
Column production methods related to higher-level features.
"""

import functools

from columnflow.production import Producer, producer
from columnflow.production.categories import category_ids
from columnflow.production.normalization import normalization_weights
from columnflow.production.util import attach_coffea_behavior
from columnflow.production.cms.seeds import deterministic_seeds
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column

# TODO #
# Task 1.1
# Add electrons and muons SFs
# Start by uncommenting these lines and
# add electron_weights, muon_weights below to uses and produces:
from columnflow.production.cms.electron import electron_weights
from columnflow.production.cms.muon import muon_weights


from h4l.production.invariant_mass import four_lep_invariant_mass

ak = maybe_import("awkward")
coffea = maybe_import("coffea")
np = maybe_import("numpy")
maybe_import("coffea.nanoevents.methods.nanoaod")

set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)


@producer(
    uses={
        attach_coffea_behavior,
        deterministic_seeds,
        category_ids,
        normalization_weights,
        four_lep_invariant_mass,
        "process_id",
        electron_weights,
        muon_weights,
    },
    produces={
        attach_coffea_behavior,
        deterministic_seeds,
        category_ids,
        normalization_weights,
        four_lep_invariant_mass,
        "process_id",
        electron_weights,
        muon_weights,
    },
)
def default(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # Build categories
    events = self[category_ids](events, **kwargs)

    # deterministic seeds
    events = self[deterministic_seeds](events, **kwargs)

    if self.dataset_inst.is_mc:
        # normalization weights
        events = self[normalization_weights](events, **kwargs)

        # TODO #
        # Task 1.1
        # Add electrons and muons SFs
        # Hint: the json in columnflow contain maps from pT > 15 GeV
        # Hint: call
        ele_pt_mask = events.Electron.pt > 15
        ele_high_pt = events.Electron[ele_pt_mask]
        events_high_pt = ak.with_field(events, ele_high_pt, "Electron")
        muon_pt_mask = events.Muon.pt > 15
        muon_high_pt = events.Muon[muon_pt_mask]
        events_muon_high_pt = ak.with_field(events, muon_high_pt, "Muon")
        events_high_pt = self[electron_weights](events_high_pt, **kwargs)
        events_muon_high_pt = self[muon_weights](events_muon_high_pt, **kwargs)
        # but first you need to define events_high_pt and events_muon_high_pt
        # as those events where only leptons with pT > 15 GeV are saved
        # Hint: use the ak.with_field() method
        # Hint: the following gives you a skeleton on top of which to build
        # high_pt_mask = ... # define mask for electrons
        # events_high_pt = ak.with_field(...)
        # events_high_pt = self[electron_weights](events_high_pt, **kwargs)
        for postfix in ["", "_up", "_down"]:
            colname = f"{electron_weights.weight_name}{postfix}"
            if hasattr(events_high_pt, colname):
                sf_ele = getattr(events_high_pt, colname)
                events = set_ak_column(events, colname, sf_ele, value_type=np.float32)
        for postfix in ["", "_up", "_down"]:
            colname = f"{muon_weights.weight_name}{postfix}"
            if hasattr(events_muon_high_pt, colname):
                sf_muon = getattr(events_muon_high_pt, colname)
                events = set_ak_column(events, colname, sf_muon, value_type=np.float32)

        # Now repeat for muons

    events = self[four_lep_invariant_mass](events, **kwargs)

    return events
