package bootstrap

// Model is just a state space probability model. Note that this is a superset
// of independent probability models, to define one of those just set the
// transition sampler to the identity function.
type Model interface {
	Prior() (x []float64)                  // Samples from the prior latent distribution.
	Transition(x []float64) (xn []float64) // Samples from the conditional transition density.
	Likelihood(y, x []float64) (p float64) // Computes conditional model likelihood.
}
