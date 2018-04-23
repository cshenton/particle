package bootstrap

import (
	"math"
	"math/rand"
)

// Pre-computed standard normal prefix
var (
	priorSd = 1000.0
	pre     = 1 / math.Sqrt(2*math.Pi)
)

// NormalModel models a normal distribution with known sd and unknown mean.
type NormalModel struct {
	Sd float64
}

// NewNormalModel creates a new NormalModel with the specified, known standard deviation.
func NewNormalModel(sd float64) (n *NormalModel) {
	return &NormalModel{Sd: sd}
}

// Prior returns a draw from the N(0, 100) prior over the state (mean).
func (n *NormalModel) Prior() []float64 { return []float64{rand.NormFloat64() * priorSd} }

// Transition is the identity function, since this isn't a state space model
func (n *NormalModel) Transition(x []float64) []float64 { return x }

// Likelihood returns the normal likelihood of y assuming the mean is x.
func (n *NormalModel) Likelihood(y, x []float64) (p float64) {
	z := (y[0] - x[0]) / n.Sd
	p = pre * math.Exp(-math.Pow(z, 2)/2)
	return p
}
