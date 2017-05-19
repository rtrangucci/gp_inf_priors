// Gamma prior on the lengthscale  allows significant
// mass below the covariate separation, delta_min = 2

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
  rho ~ gamma(7, 1.6);
  alpha ~ normal(0, 2);
  sigma ~ normal(0, 1);
  f_tilde ~ normal(0, 1);

  y ~ normal(f, sigma);
}
