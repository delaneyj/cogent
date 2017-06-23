package cogent

import (
	"errors"
	"math"
	"math/rand"
)

type activationFunc func(x float64) float64

var activations = map[string]activationFunc{
	"relu": func(x float64) float64 {
		if x <= 0 {
			return 0
		}
		return x
	},
	"sigmoid": func(x float64) float64 {
		return 1 / (1 + math.Exp(-x))
	},
}

//Neuron x
type Neuron struct {
	Weights    []float64 `json:"weights"`
	Bias       float64   `json:"bias"`
	Output     float64   `json:"output"`
	Activation string    `json:"activation"`
}

func (n *Neuron) activate(inputs []float64) error {
	if len(inputs) != len(n.Weights) {
		return errors.New("wrong number of inputs")
	}

	var sum float64
	for i, x := range inputs {
		w := n.Weights[i]
		sum += x * w
	}
	sum += n.Bias
	n.Output = activations[n.Activation](sum)

	return nil
}

func newNeuron(inputCount int, activationName *string) Neuron {
	activation := "relu"
	if activationName != nil {
		activation = *activationName
	}

	weights := make([]float64, inputCount)
	for i := range weights {
		weights[i] = rand.NormFloat64()
	}

	return Neuron{
		Activation: activation,
		Bias:       rand.NormFloat64(),
		Weights:    weights,
	}
}
