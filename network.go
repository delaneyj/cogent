package cogent

import (
	"github.com/pkg/errors"
)

//Layer x
type Layer []Neuron

//Network x
type Network struct {
	Layers    []Layer
	ErrorName string
}

var errorFunctions = map[string]func(actual, expected []float64) float64{
	"meanSquared": func(actual, expected []float64) float64 {
		var err float64
		for i, a := range actual {
			e := expected[i]
			diff := e - a
			err += diff * diff
		}
		return err / float64(len(actual))
	},
}

func newNetwork(activationName string, neuronCounts ...int) *Network {
	layersCount := len(neuronCounts) - 1

	layers := make([]Layer, layersCount)
	for i := range layers {
		inputCount := neuronCounts[i]
		neuronCount := neuronCounts[i+1]

		l := make(Layer, neuronCount)
		for i := range l {
			l[i] = newNeuron(inputCount, &activationName)
		}

		layers[i] = l
	}

	return &Network{
		Layers:    layers,
		ErrorName: "meanSquared",
	}
}

//Activate x
func (n *Network) Activate(inputs []float64) ([]float64, error) {
	if len(inputs) != len(n.Layers[0][0].Weights) {
		return nil, errors.New("Inputs don't match network")
	}

	var outputs []float64
	for _, layer := range n.Layers {
		outputs = []float64{}
		for _, n := range layer {
			n.activate(inputs)
			outputs = append(outputs, n.Output)
		}
		inputs = outputs
	}

	return outputs, nil
}

//WeightsAndBias x
func (n *Network) WeightsAndBias() []float64 {
	allWeights := []float64{}
	for _, l := range n.Layers {
		for _, n := range l {
			allWeights = append(allWeights, n.Weights...)
			allWeights = append(allWeights, n.Bias)
		}
	}

	return allWeights
}

//UpdateWeightsAndBias x
func (n *Network) UpdateWeightsAndBias(updated []float64) {
	index := 0
	for _, layer := range n.Layers {
		for _, n := range layer {
			for i := range n.Weights {
				n.Weights[i] = updated[i]
				index++
			}
			n.Bias = updated[index]
			index++
		}
	}
}

//Error x
func (n *Network) Error(expected []float64) (float64, error) {
	layerCount := len(n.Layers)
	lastLayer := n.Layers[layerCount-1]

	outputCount := len(lastLayer)
	if len(expected) != outputCount {
		return 0, errors.New("expect count mismatch")
	}

	actuals := make([]float64, outputCount)
	for i, n := range lastLayer {
		actuals[i] = n.Output
	}

	errFunc := errorFunctions[n.ErrorName]
	val := errFunc(expected, actuals)
	return val, nil
}
