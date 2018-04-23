// This is a simple example using the bootstrap filter to do inference against the
// unknown mean of samples from a normal distribution with known standard deviation.
package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/cshenton/particle/bootstrap"
)

const (
	priorSd      = 100.0 // Standard deviation of prior on mean
	numParticles = 1000  // Number of particles
	numData      = 100   // Number of data points
	trueSd       = 10.0  // The known, true standard deviation of the distribution
	trueMean     = 45.32 // The unknown, true mean of the distribution
)

var pre = 1 / math.Sqrt(2*math.Pi) // Pre-computed standard normal prefix

func main() {
	// Make our filter
	p := func() []float64 { return []float64{rand.NormFloat64() * priorSd} }
	s := func(x []float64) []float64 { return x }
	l := func(y, x []float64) float64 {
		z := (y[0] - x[0]) / trueSd
		p := pre * math.Exp(-math.Pow(z, 2)/2)
		return p
	}
	b := bootstrap.New(numParticles, p, s, l)

	// Generate some data to do inference against.
	y := make([]float64, numData)
	for i := range y {
		y[i] = trueMean + rand.NormFloat64()*trueSd
	}

	// Do inference with the bootstrap filter
	for i := range y {
		b.Update(y[i : i+1])
	}

	fmt.Println("Inferred Mean:", b.Mean())
}
