# particle

A toy particle filter written in Go.

This is a small test to see what the minimal interfaces one would need for a particle
filter in Go. I've gone with:

```go
type Model interface {
	Prior() (x []float64)                  // Samples from the prior latent distribution.
	Transition(x []float64) (xn []float64) // Samples from the conditional transition density.
	Likelihood(y, x []float64) (p float64) // Computes conditional model likelihood.
}
```

Obviously, within a full PPL, we'd have conditional sampler and likelihood for both the
observation and transition models, which we'd compile from a single model definition. The
point here is just to define the simplest interface against which we can build a bootstrap
filter.

See `main.go` for a quick example doing posterior inference on a normal mean.