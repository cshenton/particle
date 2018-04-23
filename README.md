# particle

A toy particle filter written in Go

## Components

Let's restrict ourselves to a few key things:

- a continuous latent variable `x_t` with fixed dimension.
- a prior that returns initial samples `x_0`
- a transition sampler that takes values `x_t` and returns samples `x_t+1`
- an observation likeihood that takes a latent value `x_t` and an observation `y_t` and returns a probability.

Given these simple ingredients we have define a bootstrap filter.
