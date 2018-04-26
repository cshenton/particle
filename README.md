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

## Some thoughts

Let's assume we're limited to running this on the CPU, what are our options to speed things up?

- Simple parallelisation of sample, likelihood, and resample steps
- Asynchronous implementation to avoid worker blocking on weight normalisation

These will enable us to increase data throughput by efficient utilising multiple cpu cores,
and are even amenable to a distributed implementation. The anytime implementation effectively
ties an worker to a particle subset of particles.

So in the maximum throughput setting, we could essentially have a cpu core per particle,
and the throughput would be the cost of a single particle sample plus communication overheads.

On the other end of the spectrum, a single CPU core is responsible for a large number of particles.
Potentially it's such a large number of particles that it would be daft to hold it in memory.
In such a case we could just use an embedded DB to load off main memory.

If implemented properly this wouldn't damage throughput, we could simply have an excess of worker
processes so that a particle can always be processed while another worker is doing IO to disk.