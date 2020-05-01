# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.nn.functional import pad

import pyro
import pyro.distributions as dist
from pyro.ops.indexing import Index

from .compartmental import CompartmentalModel
from .distributions import infection_dist


class SimpleSIRModel(CompartmentalModel):
    """
    Susceptible-Infected-Recovered model.

    To customize this model we recommend forking and editing this class.

    This is a stochastic discrete-time discrete-state model with three
    compartments: "S" for susceptible, "I" for infected, and "R" for
    recovered individuals (the recovered individuals are implicit: ``R =
    population - S - I``) with transitions ``S -> I -> R``.

    :param int population: Total ``population = S + I + R``.
    :param float recovery_time: Mean recovery time (duration in state
        ``I``). Must be greater than 1.
    :param iterable data: Time series of new observed infections. Each time
        step is Binomial distributed between 0 and the number of ``S -> I``
        transitions. This allows false negative but no false positives.
    """

    def __init__(self, population, recovery_time, data):
        compartments = ("S", "I")  # R is implicit.
        duration = len(data)
        super().__init__(compartments, duration, population)

        assert isinstance(recovery_time, float)
        assert recovery_time > 1
        self.recovery_time = recovery_time

        self.data = data

    series = ("S2I", "I2R", "obs")
    full_mass = [("R0", "rho")]

    def global_model(self):
        tau = self.recovery_time
        R0 = pyro.sample("R0", dist.LogNormal(0., 1.))
        rho = pyro.sample("rho", dist.Uniform(0, 1))
        return R0, tau, rho

    def initialize(self, params):
        # Start with a single infection.
        return {"S": self.population - 1, "I": 1}

    def transition_fwd(self, params, state, t):
        R0, tau, rho = params

        # Sample flows between compartments.
        S2I = pyro.sample("S2I_{}".format(t),
                          infection_dist(individual_rate=R0 / tau,
                                         num_susceptible=state["S"],
                                         num_infectious=state["I"],
                                         population=self.population))
        I2R = pyro.sample("I2R_{}".format(t),
                          dist.Binomial(state["I"], 1 / tau))

        # Update compartments with flows.
        state["S"] = state["S"] - S2I
        state["I"] = state["I"] + S2I - I2R

        # Condition on observations.
        pyro.sample("obs_{}".format(t),
                    dist.ExtendedBinomial(S2I, rho),
                    obs=self.data[t] if t < self.duration else None)

    def transition_bwd(self, params, prev, curr, t):
        R0, tau, rho = params

        # Reverse the flow computation.
        S2I = prev["S"] - curr["S"]
        I2R = prev["I"] - curr["I"] + S2I

        # Condition on flows between compartments.
        pyro.sample("S2I_{}".format(t),
                    infection_dist(individual_rate=R0 / tau,
                                   num_susceptible=prev["S"],
                                   num_infectious=prev["I"],
                                   population=self.population),
                    obs=S2I)
        pyro.sample("I2R_{}".format(t),
                    dist.ExtendedBinomial(prev["I"], 1 / tau),
                    obs=I2R)

        # Condition on observations.
        pyro.sample("obs_{}".format(t),
                    dist.ExtendedBinomial(S2I, rho),
                    obs=self.data[t])


class OverdispersedSIRModel(CompartmentalModel):
    """
    Overdispersed Susceptible-Infected-Recovered model.

    To customize this model we recommend forking and editing this class.

    This is a stochastic discrete-time discrete-state model with three
    compartments: "S" for susceptible, "I" for infected, and "R" for
    recovered individuals (the recovered individuals are implicit: ``R =
    population - S - I``) with transitions ``S -> I -> R``.

    This model accounts for superspreading (overdispersed individual
    reproductive number) by assuming each infected individual infects
    BetaBinomial-many susceptible individuals, where the BetaBinomial
    distribution acts as an overdispersed Binomial distribution, adapting the
    more standard NegativeBinomial distribution that acts as an overdispersed
    Poisson distribution [1,2] to the setting of finite populations. To
    preserve Markov structure, we follow [2] and assume all infections by a
    single individual occur on the single time step where that individual makes
    an ``I -> R`` transition. That is, whereas the :class:`SimpleSIRModel`
    assumes infected individuals infect `Binomial(S,R/tau)`-many susceptible
    individuals during each infected time step (over `tau`-many steps on
    average), this model assumes they infect `BetaBinomial(k,...,S)`-many
    susceptible individuals but only on the final time step before recovering.

    **References**

    [1] J. O. Lloyd-Smith, S. J. Schreiber, P. E. Kopp, W. M. Getz (2005)
        "Superspreading and the effect of individual variation on disease
        emergence"
        https://www.nature.com/articles/nature04153.pdf
    [2] Lucy M. Li, Nicholas C. Grassly, Christophe Fraser (2017)
        "Quantifying Transmission Heterogeneity Using Both Pathogen Phylogenies
        and Incidence Time Series"
        https://academic.oup.com/mbe/article/34/11/2982/3952784

    :param int population: Total ``population = S + I + R``.
    :param float recovery_time: Mean recovery time (duration in state
        ``I``). Must be greater than 1.
    :param iterable data: Time series of new observed infections. Each time
        step is Binomial distributed between 0 and the number of ``S -> I``
        transitions. This allows false negative but no false positives.
    """

    def __init__(self, population, recovery_time, data):
        compartments = ("S", "I")  # R is implicit.
        duration = len(data)
        super().__init__(compartments, duration, population)

        assert isinstance(recovery_time, float)
        assert recovery_time > 1
        self.recovery_time = recovery_time

        self.data = data

    series = ("S2I", "I2R", "obs")
    full_mass = [("R0", "rho", "k")]

    def global_model(self):
        tau = self.recovery_time
        R0 = pyro.sample("R0", dist.LogNormal(0., 1.))
        k = pyro.sample("k", dist.Exponential(1.))
        rho = pyro.sample("rho", dist.Uniform(0, 1))
        return R0, k, tau, rho

    def initialize(self, params):
        # Start with a single infection.
        return {"S": self.population - 1, "I": 1}

    def transition_fwd(self, params, state, t):
        R0, k, tau, rho = params

        # Sample flows between compartments.
        I2R = pyro.sample("I2R_{}".format(t),
                          dist.Binomial(state["I"], 1 / tau))
        S2I = pyro.sample("S2I_{}".format(t),
                          infection_dist(individual_rate=R0,
                                         num_susceptible=state["S"],
                                         num_infectious=state["I"],
                                         population=self.population,
                                         concentration=k))

        # Update compartments with flows.
        state["S"] = state["S"] - S2I
        state["I"] = state["I"] + S2I - I2R

        # Condition on observations.
        pyro.sample("obs_{}".format(t),
                    dist.ExtendedBinomial(S2I, rho),
                    obs=self.data[t] if t < self.duration else None)

    def transition_bwd(self, params, prev, curr, t):
        R0, k, tau, rho = params

        # Reverse the flow computation.
        S2I = prev["S"] - curr["S"]
        I2R = prev["I"] - curr["I"] + S2I

        # Condition on flows between compartments.
        pyro.sample("S2I_{}".format(t),
                    infection_dist(individual_rate=R0,
                                   num_susceptible=prev["S"],
                                   num_infectious=prev["I"],
                                   population=self.population,
                                   concentration=k),
                    obs=S2I)
        pyro.sample("I2R_{}".format(t),
                    dist.ExtendedBinomial(prev["I"], 1 / tau),
                    obs=I2R)

        # Condition on observations.
        pyro.sample("obs_{}".format(t),
                    dist.ExtendedBinomial(S2I, rho),
                    obs=self.data[t])


class SparseSIRModel(CompartmentalModel):
    """
    Susceptible-Infected-Recovered model with sparsely observed infections.

    To customize this model we recommend forking and editing this class.

    This is a stochastic discrete-time discrete-state model with four
    compartments: "S" for susceptible, "I" for infected, and "R" for
    recovered individuals (the recovered individuals are implicit: ``R =
    population - S - I``) with transitions ``S -> I -> R``.

    This model allows observations of **cumulative** infections at uneven time
    intervals. To preserve Markov structure (and hence tractable inference)
    this model adds an auxiliary compartment ``O`` denoting the fully-observed
    cumulative number of observations at each time point. At observed times
    (when ``mask[t] == True``) ``O`` must exactly match the provided data;
    between observed times ``O`` stochastically imputes the provided data.

    :param int population: Total ``population = S + I + R``.
    :param float recovery_time: Mean recovery time (duration in state
        ``I``). Must be greater than 1.
    :param iterable data: Time series of **cumulative** observed infections.
        Whenever ``mask[t] == True``, ``data[t]`` corresponds to an
        observation; otherwise ``data[t]`` can be arbitrary, e.g. NAN.
    :param iterable mask: Boolean time series denoting whether an observation
        is made at each time step. Should satisfy ``len(mask) == len(data)``.
    """

    def __init__(self, population, recovery_time, data, mask):
        assert len(data) == len(mask)
        duration = len(data)
        compartments = ("S", "I", "O")  # O is auxiliary, R is implicit.
        super().__init__(compartments, duration, population)

        assert isinstance(recovery_time, float)
        assert recovery_time > 1
        self.recovery_time = recovery_time

        self.data = data
        self.mask = mask

    series = ("S2I", "I2R", "S2O", "obs")
    full_mass = [("R0", "rho")]

    def global_model(self):
        tau = self.recovery_time
        R0 = pyro.sample("R0", dist.LogNormal(0., 1.))
        rho = pyro.sample("rho", dist.Uniform(0, 1))
        return R0, tau, rho

    def initialize(self, params):
        # Start with a single infection.
        return {"S": self.population - 1, "I": 1, "O": 0}

    def transition_fwd(self, params, state, t):
        R0, tau, rho = params

        # Sample flows between compartments.
        S2I = pyro.sample("S2I_{}".format(t),
                          infection_dist(individual_rate=R0 / tau,
                                         num_susceptible=state["S"],
                                         num_infectious=state["I"],
                                         population=self.population))
        I2R = pyro.sample("I2R_{}".format(t),
                          dist.Binomial(state["I"], 1 / tau))
        S2O = pyro.sample("S2O_{}".format(t),
                          dist.ExtendedBinomial(S2I, rho))

        # Update compartments with flows.
        state["S"] = state["S"] - S2I
        state["I"] = state["I"] + S2I - I2R
        state["O"] = state["O"] + S2O

        # Condition on cumulative observations.
        mask_t = self.mask[t] if t < self.duration else False
        data_t = self.data[t] if t < self.duration else None
        pyro.sample("obs_{}".format(t),
                    dist.Delta(state["O"]).mask(mask_t),
                    obs=data_t)

    def transition_bwd(self, params, prev, curr, t):
        R0, tau, rho = params

        # Reverse the flow computation.
        S2I = prev["S"] - curr["S"]
        I2R = prev["I"] - curr["I"] + S2I
        S2O = curr["O"] - prev["O"]

        # Condition on flows between compartments.
        pyro.sample("S2I_{}".format(t),
                    infection_dist(individual_rate=R0 / tau,
                                   num_susceptible=prev["S"],
                                   num_infectious=prev["I"],
                                   population=self.population),
                    obs=S2I)
        pyro.sample("I2R_{}".format(t),
                    dist.ExtendedBinomial(prev["I"], 1 / tau),
                    obs=I2R)
        pyro.sample("S2O_{}".format(t),
                    dist.ExtendedBinomial(S2I, rho),
                    obs=S2O)

        # Condition on cumulative observations.
        pyro.sample("obs_{}".format(t),
                    dist.Delta(curr["O"]).mask(self.mask[t]),
                    obs=self.data[t])


class UnknownStartSIRModel(CompartmentalModel):
    """
    Susceptible-Infected-Recovered model with unknown date of first infection.

    To customize this model we recommend forking and editing this class.

    This is a stochastic discrete-time discrete-state model with three
    compartments: "S" for susceptible, "I" for infected, and "R" for
    recovered individuals (the recovered individuals are implicit: ``R =
    population - S - I``) with transitions ``S -> I -> R``.

    This model demonstrates:

    1.  How to incorporate spontaneous infections from external sources;
    2.  How to incorporate time-varying piecwise ``rho`` by supporting
        forecasting in :meth:`transition_fwd` and using the
        :class:`~pyro.ops.index.ing.Index` helper in :meth:`transition_bwd`.
    3.  How to override the :meth:`predict` method to compute extra
        statistics.

    :param int population: Total ``population = S + I + R``.
    :param float recovery_time: Mean recovery time (duration in state
        ``I``). Must be greater than 1.
    :param int pre_window: Number of time steps before beginning ``data``
        where the initial infection may have occurred. Must be positive.
    :param iterable data: Time series of new observed infections. Each time
        step is Binomial distributed between 0 and the number of ``S -> I``
        transitions. This allows false negative but no false positives.
    """

    def __init__(self, population, recovery_time, pre_window, data):
        compartments = ("S", "I")  # R is implicit.
        duration = pre_window + len(data)
        super().__init__(compartments, duration, population)

        assert isinstance(recovery_time, float)
        assert recovery_time > 1
        self.recovery_time = recovery_time

        assert isinstance(pre_window, int) and pre_window > 0
        self.pre_window = pre_window
        self.post_window = len(data)

        # Expect a single external infection during the pre_window.
        self.external_rate = 1 / pre_window

        # Prepend data with zeros.
        if isinstance(data, list):
            data = [0.] * self.pre_window + data
        else:
            data = pad(data, (self.pre_window, 0), value=0.)
        self.data = data

    series = ("S2I", "I2R", "obs")
    full_mass = [("R0", "rho0", "rho1")]

    def global_model(self):
        tau = self.recovery_time
        R0 = pyro.sample("R0", dist.LogNormal(0., 1.))

        # Assume two different response rates: rho0 before any observations
        # were made (in pre_window), followed by a higher response rate rho1
        # after observations were made (in post_window).
        rho0 = pyro.sample("rho0", dist.Uniform(0, 1))
        rho1 = pyro.sample("rho1", dist.Uniform(0, 1))
        rho = torch.cat([
            rho0.unsqueeze(-1).expand(rho0.shape + (self.pre_window,)),
            rho1.unsqueeze(-1).expand(rho1.shape + (self.post_window,)),
        ], dim=-1)

        # Model external infections as an infectious pseudo-individual added
        # to num_infectious when sampling S2I below.
        X = self.external_rate * tau / R0

        return R0, X, tau, rho

    def initialize(self, params):
        # Start with no internal infections.
        return {"S": self.population, "I": 0}

    def transition_fwd(self, params, state, t):
        R0, X, tau, rho = params

        # Sample flows between compartments.
        S2I = pyro.sample("S2I_{}".format(t),
                          infection_dist(individual_rate=R0 / tau,
                                         num_susceptible=state["S"],
                                         num_infectious=state["I"] + X,
                                         population=self.population))
        I2R = pyro.sample("I2R_{}".format(t),
                          dist.Binomial(state["I"], 1 / tau))

        # Update compartments with flows.
        state["S"] = state["S"] - S2I
        state["I"] = state["I"] + S2I - I2R

        # In .transition_fwd() t will always be an integer but may lie outside
        # of [0,self.duration) when forecasting.
        rho_t = rho[..., t] if t < self.duration else rho[..., -1]
        data_t = self.data[t] if t < self.duration else None

        # Condition on observations.
        pyro.sample("obs_{}".format(t),
                    dist.ExtendedBinomial(S2I, rho_t),
                    obs=data_t)

    def transition_bwd(self, params, prev, curr, t):
        R0, X, tau, rho = params

        # Reverse the flow computation.
        S2I = prev["S"] - curr["S"]
        I2R = prev["I"] - curr["I"] + S2I

        # Condition on flows between compartments.
        pyro.sample("S2I_{}".format(t),
                    infection_dist(individual_rate=R0 / tau,
                                   num_susceptible=prev["S"],
                                   num_infectious=prev["I"] + X,
                                   population=self.population),
                    obs=S2I)
        pyro.sample("I2R_{}".format(t),
                    dist.ExtendedBinomial(prev["I"], 1 / tau),
                    obs=I2R)

        # In .transition_bwd() t may be either an integer in [0,self.duration)
        # or may be a nonstandard slicing object like (Ellipsis, None, None);
        # we use the Index(-)[-] helper to support indexing with nonstandard t.
        rho_t = Index(rho)[..., t]

        # Condition on observations.
        pyro.sample("obs_{}".format(t),
                    dist.ExtendedBinomial(S2I, rho_t),
                    obs=self.data[t])

    def predict(self, forecast=0):
        """
        Augments
        :meth:`~pyro.contrib.epidemiology.compartmental.Compartmental.predict`
        with samples of ``first_infection``.

        :param int forecast: The number of time steps to forecast forward.
        :returns: A dictionary mapping sample site name (or compartment name)
            to a tensor whose first dimension corresponds to sample batching.
        :rtype: dict
        """
        samples = super().predict(forecast)
        samples["first_infection"] = samples["I"].cumsum(-1).eq(0).sum(-1)
        return samples
