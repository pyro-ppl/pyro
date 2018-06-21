Analysis of the eight schools data (chapter 5 of [Gelman et al 2013]) using MCMC (NUTS) and SVI.

The starting model is the Stan model:
```
data {
  int<lower=0> J; // number of schools
  real y[J]; // estimated treatment effects
  real<lower=0> sigma[J]; // s.e. of effect estimates
}
parameters {
  real mu;
  real<lower=0> tau;
  real eta[J];
}
transformed parameters {
  real theta[J];
  for (j in 1:J)
    theta[j] <- mu + tau * eta[j];
}
model {
  eta ~ normal(0, 1);
  y ~ normal(theta, sigma);
}
```

# References
* [Gelman et al 2013] Gelman A., Carlin J.B., Stern H.S., Dunson D.B., Vehtari A., Rubin D.B. "Bayesian Data Analysis, Third Edition". CRC Press 2013.