// This is a simple example using the bootstrap filter to do inference against the
// unknown mean of samples from a normal distribution with known standard deviation.
// Of course it's totally silly to use a particle filter for this, since we'll suffer
// from particle degeneracy.
package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/cshenton/particle/bootstrap"
)

const (
	numParticles = 1e5  // Number of particles
	numData      = 100  // Number of data points
	trueSd       = 5.0  // The known, true standard deviation of the distribution
	trueMean     = 6.78 // The unknown, true mean of the distribution
)

func main() {
	// Create model, bootstrap filter
	m := bootstrap.NewNormalModel(trueSd)
	b := bootstrap.New(numParticles, m)

	// Generate some data to do inference against.
	y := make([]float64, numData)
	for i := range y {
		y[i] = trueMean + rand.NormFloat64()*trueSd
	}

	// Perform and time inference
	t := time.Now()
	for i := range y {
		b.Update(y[i : i+1])
	}

	fmt.Printf("%v updates with %v particles each in %v\n", numData, numParticles, time.Now().Sub(t))
	fmt.Printf("True Mean %v, Inferred Mean %v\n", trueMean, b.Mean()[0])
}
