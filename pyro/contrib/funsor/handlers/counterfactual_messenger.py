# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Implementations of an expression-level intervention primitive and higher-level handler interface,
and the requisite effect handlers for giving different counterfactual semantics to interventions.

The pyro.contrib.funsor.handlers.counterfactual module roughly follows this design:
    - Single primitive do(f, cf) for interventions on values
    - pyro.condition-style interface handler for automating application of the primitive to named sample sites
    - Semi-automated reparametrization handler for transforming endogenous noise to exogenous noise
    - Soft conditioning handler/reparametrizer for appropximately conditioning deterministic sites.
    - Multiple handlers that give different semantics to the intervention primitive do():
        - Single-world counterfactual behavior (default): do(f, cf) returns cf only
        - Factual-only behavior: do(f, cf) returns f only
        - "Diagonal" twin-world behavior: do(f, cf) concatenates f, cf along a single special dimension
        - Full twin-world behavior: do(f, cf) stacks f, cf along a fresh dimension. The resulting values
            contain every possible combination of factual and counterfactual worlds, but they will also
            grow in size exponentially in the number of interacting interventions in the program.
        - One-step counterfactual twin-world behavior: do(f, cf) indexes only the counterfactual values of f, cf
            and stacks the results along a fresh dimension. The resulting values are almost fully counterfactual.
        - One-step factual twin-world behavior: do(f, cf) indexes only the factual values of f, cf
            and stacks the results along a fresh dimension. The resulting values are almost fully factual.
"""
from typing import Optional, Union

import funsor
import torch

import pyro.infer.reparam
import pyro.poutine.reparam_messenger

from pyro.contrib.funsor.handlers.named_messenger import NamedMessenger
from pyro.contrib.funsor.handlers.primitives import to_data, to_funsor
from pyro.contrib.funsor.handlers.runtime import _DIM_STACK

funsor.set_backend("torch")


class CounterfactualMessenger(NamedMessenger):
    """
    Base handler class for experiments in counterfactual semantics.
    Reads off counterfactual dimensions and adds them to the plate stack at each site
    """
    @property
    def cf_name(self):
        return "__CF"

    def _pyro_sample(self, msg):
        name_to_dim = _DIM_STACK.names_from_batch_shape(msg["fn"].batch_shape)
        cf_plates = tuple(sorted([
            CondIndepStackFrame(name=name, dim=dim, size=2, vectorized=True)
            for name,self.cf_name dim in name_to_dim.items()
            # TODO track this inside child classes instead of using a heuristic
            if "__CF" in name
        ], key=lambda f: f.dim))
        for f in cf_plates:
            if f not in msg["cond_indep_stack"]:
                msg["cond_indep_stack"] += (f,)


class FactualWorldMessenger(CounterfactualMessenger):
    """
    Diagnostic utility: ignore interventions and return factual values from interventions.
    """
    def _pyro_do(self, msg):
        if msg["done"]:
            return
        obs, act, event_dim = msg["args"][0], msg["args"][1], msg["kwargs"]["event_dim"]
        msg["value"] = obs
        msg["done"] = True


class TwinWorldMessenger(CounterfactualMessenger):
    """
    Creates and evaluates twin worlds using broadcasting.
    Enforces sparsity by concatenating rather than stacking values.
    """
    def _pyro_do(self, msg):
        if msg["done"]:
            return
        obs, act, event_dim = msg["args"][0], msg["args"][1], msg["kwargs"]["event_dim"]
        output = funsor.domains.Array['real', obs.shape[len(obs.shape) - event_dim:]]
        obs_funsor, act_funsor = to_funsor(obs, output), to_funsor(act, output)
        if self.cf_name in obs_funsor.inputs and self.cf_name in act_funsor.inputs:
            raise ValueError("Not compatible with DiagonalWorldMessenger")
        elif self.cf_name in obs_funsor.inputs or self.cf_name in act_funsor.inputs:
            # concatenate along the worlds dimension
            funsor_value = funsor.terms.Cat(self.cf_name, (obs_funsor, act_funsor))
        else:
            # stack purely factual values along the worlds dimension
            funsor_value = funsor.terms.Stack(self.cf_name, (obs_funsor, act_funsor))
        msg["value"] = to_data(funsor_value)
        msg["done"] = True


class MultiWorldMessenger(CounterfactualMessenger):
    """
    Create and evaluate twin worlds using broadcasting.
    General, but expensive: new world introduced at every intervention
    """
    def __enter__(self):
        if self._ref_count == 0:
            self._counter = 0
        return super().__enter__()

    @property
    def cf_name(self):
        return f"__CF_{self._counter}"

    def _pyro_do(self, msg):
        if msg["done"]:
            return
        obs, act, event_dim = msg["args"][0], msg["args"][1], msg["kwargs"]["event_dim"]
        value = torch.stack([obs, act], -1)
        output = funsor.domains.Array['real', value.shape[len(value.shape) - event_dim:]]
        funsor_value = to_funsor(value, output, dim_to_name={event_dim - len(value.shape): self.cf_name})
        # read off counterfactual variables and convert them to plates
        msg["value"] = to_data(funsor_value)
        msg["done"] = True
        self._counter += 1


class OneStepFactualWorldMessenger(MultiWorldMessenger):
    """
    Create and evaluate twin worlds using broadcasting.
    Intermediate between TwinWorld and MultiWorld semantics:
    only propagates factual histories through interventions.
    """
    def _pyro_do(self, msg):
        if msg["done"]:
            return
        obs, act, event_dim = msg["args"][0], msg["args"][1], msg["kwargs"]["event_dim"]

        output = funsor.domains.Array['real', obs.shape[len(obs.shape) - event_dim:]]
        obs_funsor, act_funsor = to_funsor(obs, output), to_funsor(act, output)
        
        # slice both values along world axes to extract only factual history
        # TODO replace heuristic name checks
        world_subs = {name: 0 for name in obs_funsor.inputs if "__CF" in name}
        world_subs.update({name: 0 for name in act_funsor.inputs if "__CF" in name})
        msg["args"] = (to_data(obs_funsor)(**world_subs), to_data(act_funsor)(**world_subs))
        return super()._pyro_do(msg)


class OneStepCounterfactualWorldMessenger(MultiWorldMessenger):
    """
    Create and evaluate twin worlds using broadcasting.
    Intermediate between TwinWorld and MultiWorld semantics:
    only propagates counterfactual histories through interventions.
    """
    def _pyro_do(self, msg):
        if msg["done"]:
            return
        obs, act, event_dim = msg["args"][0], msg["args"][1], msg["kwargs"]["event_dim"]

        output = funsor.domains.Array['real', obs.shape[len(obs.shape) - event_dim:]]
        obs_funsor, act_funsor = to_funsor(obs, output), to_funsor(act, output)

        # slice both values along world axes to extract only counterfactual history
        # TODO replace heuristic name checks
        world_subs = {name: 1 for name in obs_funsor.inputs if "__CF" in name}
        world_subs.update({name: 1 for name in act_funsor.inputs if "__CF" in name})
        msg["args"] = (to_data(obs_funsor)(**world_subs), to_data(act_funsor)(**world_subs))
        return super()._pyro_do(msg)


class ExogenizeMessenger(pyro.poutine.reparam_messenger.ReparamMessenger):
    """
    Uses :class:`~pyro.infer.reparam.Reparam`s to transform sites whose parameters
    depend on an intervened value into deterministic functions of exogenous noise, ensuring
    that noise values are shared across factual and counterfactual worlds.
    """
    def __init__(self):
        # TODO add optional validation functionality that checks whether sites were successfully "exogenized"
        super().__init__(self, config=self._config)

    @staticmethod
    def _is_cf(
            v: Union[torch.tensor, pyro.distributions.Distribution],
            output: Optional[funsor.domains.Domain] = funsor.Real,
        ) -> bool:
        return any("__CF" in name for name in to_funsor(v, output=output).inputs)

    def _config(self, msg: dict) -> Optional[pyro.infer.reparam.Reparam]:
        # check if the site is downstream of an intervention
        if not self._is_cf(msg["fn"]):
            return None

        if hasattr(msg["fn"], "loc") and hasattr(msg["fn"], "scale"):
            return pyro.infer.reparam.LocScaleReparam()
        else:
            raise NotImplementedError(
                f"{msg['fn']} could not be reparameterized with exogenous noise. "
                "TODO handle other reparametrizable cases."
            )
