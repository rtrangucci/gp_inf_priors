// Inverse gamma prior on the lengthscale allows
// a little bit of mass below the covariate separation,
// delta_min = 2 and features an extremely heavy tail.

data {
  int<lower=1> N;
  real x[N];
  vector[N] y;
}

parameters {
  real<lower=0> rho;
  real<lower=0> alpha;
  real<lower=0> sigma;
  vector[N] f_tilde;
}

transformed parameters {
  vector[N] f;
  {
    matrix[N, N] cov =   cov_exp_quad(x, alpha, rho)
                       + diag_matrix(rep_vector(1e-10, N));
    matrix[N, N] L_cov = cholesky_decompose(cov);
    f = L_cov * f_tilde;
  }
}

model {
  rho ~ inv_gamma(3, 7);
  alpha ~ normal(0, 2);
  sigma ~ normal(0, 1);
  f_tilde ~ normal(0, 1);

  y ~ normal(f, sigma);
}
