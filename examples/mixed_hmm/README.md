# Hierarchical mixed-effect hidden Markov models

Note: This is a cleaned-up version of the seal experiments in [Bingham et al 2019] that is a simplified variant of some of the analysis in the [momentuHMM harbour seal example](https://github.com/bmcclintock/momentuHMM/blob/master/vignettes/harbourSealExample.R) [McClintock et al 2018].

Recent advances in sensor technology have made it possible to capture the movements of multiple wild animals within a single population at high spatiotemporal resolution over long periods of time [McClintock et al 2013, Towner et al 2016]. Discrete state-space models, where the latent state is thought of as corresponding to a behavior state such as "foraging" or "resting", have become popular computational tools for analyzing these new datasets thanks to their interpretability and tractability.

This example applies several different hierarchical discrete state-space models to location data recorded from a colony of harbour seals on foraging excursions in the North Sea [McClintock et al 2013].
The raw data are irregularly sampled time series (roughly 5-15 minutes between samples) of GPS coordinates and diving activity for each individual in the colony (10 male and 7 female) over the course of a single day recorded by lightweight tracking devices physically attached to each animal by researchers. They have been preprocessed using the momentuHMM example code into smoothed, temporally regular series of step sizes, turn angles, and diving activity for each individual.

The models are special cases of a time-inhomogeneous discrete state space model
whose state transition distribution is specified by a hierarchical generalized linear mixed model (GLMM).
At each timestep `t`, for each individual trajectory `b` in each group `a`, we have

```
logit(p(x[t,a,b] = state i | x[t-1,a,b] = state j)) =
    (epsilon_G[a] + epsilon_I[a,b] + Z_I[a,b].T @ beta1 + Z_G[a].T @ beta2 + Z_T[t,a,b].T @ beta3)[i,j]
```

where `a,b` correspond to plate indices, `epsilon_G` and `epsilon_I` are independent random variables for each group and individual within each group respectively, `Z`s are covariates, and `beta`s are parameter vectors.

The random variables `epsilon` may be either discrete or continuous.
If continuous, they are normally distributed.
If discrete, they are sampled from a set of three possible values shared across the innermost plate of a particular variable.
That is, for each individual trajectory `b` in each group `a`, we sample single random effect values for an entire trajectory:

```
iota_G[a] ~ Categorical(pi_G)
epsilon_G[a] = Theta_G[iota_G[a]]
iota_I[a,b] ~ Categorical(pi_I[a])
epsilon_I[a,b] = Theta_I[a][iota_I[a,b]]
```

Here `pi_G`, `Theta_G`, `pi_I`, and `Theta_I` are all learnable real-valued parameter vectors and `epsilon` values are batches of vectors the size of state transition matrices.

Observations `y[t,a,b]` are represented as sequences of real-valued step lengths and turn angles, modelled by zero-inflated Gamma and von Mises likelihoods respectively.
The seal models also include a third observed variable indicating the amount of diving activity between successive locations, which we model with a zero-inflated Beta distribution following [McClintock et al 2018].

We grouped animals by sex and implemented versions of this model with (i) no random effects (as a baseline), and with random effects present at the (ii) group, (iii) individual, or (iv) group+individual levels. Unlike the models in [Towner et al 2016], we do not consider fixed effects on any of the parameters.

# References
* [Obermeyer et al 2019] Obermeyer, F.\*, Bingham, E.\*, Jankowiak, M.\*, Chiu, J., Pradhan, N., Rush, A., and Goodman, N.  Tensor Variable Elimination for Plated Factor Graphs, 2019
* [McClintock et al 2013] McClintock, B. T., Russell, D. J., Matthiopoulos, J., and King, R.  Combining individual animal movement and ancillary biotelemetry data to investigate population-level activity budgets. Ecology, 94(4):838–849, 2013
* [McClintock et al 2018] McClintock, B. T. and Michelot,T. momentuhmm: R package for generalized hidden markov models of animal movement. Methods in Ecology and  Evolution,  9(6): 1518–1530, 2018. doi: 10.1111/2041-210X.12995
* [Towner et al 2016] Towner, A. V., Leos-Barajas, V., Langrock, R., Schick, R. S., Smale, M. J., Kaschke, T., Jewell, O. J., and Papastamatiou, Y. P.  Sex-specific and individual preferences for hunting strategies in white sharks. Functional Ecology, 30(8):1397–1407, 2016.
