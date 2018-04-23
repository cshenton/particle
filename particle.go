package particle

// Prior is a function which samples from the prior latent distribution.
type Prior func() (x []float64)

// Sampler samples from the transition density given the current latent state.
type Sampler func(x []float64) (xn []float64)

// Likelihood returns the likelihood of observation y given latent state x.
type Likelihood func(x, y []float64) (p float64)
