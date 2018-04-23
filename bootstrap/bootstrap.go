package particle

import (
	"math/rand"
	"sort"
)

// Prior is a function which samples from the prior latent distribution.
type Prior func() (x []float64)

// Sampler samples from the transition density given the current latent state.
type Sampler func(x []float64) (xn []float64)

// Likelihood returns the likelihood of observation y given latent state x.
type Likelihood func(y, x []float64) (p float64)

// Bootstrap is a bootstrap filter.
type Bootstrap struct {
	particles  [][]float64
	likelihood Likelihood
	sampler    Sampler
}

// New returns a new bootstrap filter with n particles, and using the provided prior,
// sampler, and likelihood.
func New(n int, p Prior, s Sampler, l Likelihood) (b *Bootstrap) {
	part := make([][]float64, n)

	for i := range part {
		part[i] = p()
	}

	b = &Bootstrap{
		particles:  part,
		likelihood: l,
		sampler:    s,
	}

	return b
}

// Update updates the particle distribution of the Bootstrap filter given the next observation.
func (b *Bootstrap) Update(y []float64) {
	temp := make([][]float64, len(b.particles))
	prob := make([]float64, len(b.particles))
	total := 0.0
	for i := range temp {
		temp[i] = b.sampler(b.particles[i]) // Sample new particle
		p := b.likelihood(y, temp[i])       // Compute next prob
		total += p                          // Update Total
		prob[i] = total                     // Set prob weight to current total
	}
	for i := range prob {
		prob[i] /= total
	}

	for i := range b.particles {
		p := rand.Float64()               // Sample quantile
		j := sort.SearchFloat64s(prob, p) // Get weighted sample index
		b.particles[i] = temp[j]          // Overwrite the current particle
	}
}

// Mean returns the average particle value of the bootstrap filter.
func (b *Bootstrap) Mean() (xm []float64) {
	xm = make([]float64, len(b.particles[0]))
	for i := range b.particles {
		for j := range xm {
			xm[j] += b.particles[i][j]
		}
	}
	for i := range xm {
		xm[i] /= float64(len(b.particles))
	}

	return xm
}
