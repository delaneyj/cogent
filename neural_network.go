package cogent

import (
	"log"
	"math"
	"math/rand"
	"runtime"
)

var (
	//DefaultTrainingConfig x
	DefaultTrainingConfig = TrainingConfiguration{
		InertiaWeight:     0.729,
		CognitiveWeight:   1.49445,
		SocialWeight:      1.49445,
		GlobalWeight:      0.3645,
		MaxIterations:     700,
		TargetAccuracy:    0.000001,
		WeightRange:       100,
		WeightDecayRate:   0.01,
		ProbablityOfDeath: 0.005,
	}
)

func (l *LayerData) reset(weightRange float64) {
	for i := range l.WeightsAndBiases {
		l.WeightsAndBiases[i] = (2*weightRange)*rand.Float64() - weightRange

		lo := -0.1 * weightRange
		hi := 0.1 * weightRange
		l.Velocities[i] = (hi-lo)*rand.Float64() + lo
	}
}

func (nn *NeuralNetworkData) weightsAndBiasesCount() int {
	count := 0
	for _, l := range nn.Layers {
		count += len(l.WeightsAndBiases)
	}
	return count
}

func (nn *NeuralNetworkData) weights() []float64 {
	weights := make([]float64, nn.weightsAndBiasesCount())
	offset := 0
	for _, l := range nn.Layers {
		copy(weights[offset:], l.WeightsAndBiases)
		offset += len(l.WeightsAndBiases)
	}
	return weights
}

func (nn *NeuralNetworkData) setWeights(weights []float64) {
	offset := 0
	for _, l := range nn.Layers {
		copy(l.WeightsAndBiases, weights[offset:])
		offset += len(l.WeightsAndBiases)
	}
}

func (nn *NeuralNetworkData) velocities() []float64 {
	velocities := make([]float64, nn.weightsAndBiasesCount())
	offset := 0
	for _, l := range nn.Layers {
		copy(velocities[offset:], l.Velocities)
		offset += len(l.Velocities)
	}
	return velocities
}

func (nn *NeuralNetworkData) setVelocities(velocities []float64) {
	offset := 0
	for _, l := range nn.Layers {
		copy(l.Velocities, velocities[offset:])
		offset += len(l.Velocities)
	}
}

func (nn *NeuralNetworkData) reset(weightRange float64) {
	for _, l := range nn.Layers {
		l.reset(weightRange)
	}
	nn.CurrentLoss = math.MaxFloat64
	nn.Best.Loss = math.MaxFloat64
}

//Activate feeds forward through the network
func (nn *NeuralNetworkData) Activate(intialInputs ...float64) []float64 {
	inputs := append(intialInputs, 1) // add bias
	var outputs []float64
	for _, l := range nn.Layers {
		nc := int(l.NodeCount)
		outputs = make([]float64, nc)

		delta := len(l.WeightsAndBiases) / nc
		offset := 0
		for n := 0; n < nc; n++ {
			for i, w := range l.WeightsAndBiases[offset : offset+delta] {
				input := inputs[i]
				outputs[n] += input * w
			}
			offset += delta
		}

		activationFunc := activations[l.Activation]
		outputs = activationFunc(outputs)
		inputs = append(outputs, 1)
	}
	return outputs
}

//ClassificationAccuracy percentage correct using winner-takes all
func (nn *NeuralNetworkData) ClassificationAccuracy(testData []*Data) float64 {
	correctCount := 0
	maxIndex := func(s []float64) int {
		// index of largest value
		bigIndex := 0
		biggestVal := s[0]
		for i, x := range s {
			if x > biggestVal {
				biggestVal = x
				bigIndex = i
			}
		}
		return bigIndex
	}

	for _, d := range testData {
		expectedOutput := d.Outputs
		actualOuputs := nn.Activate(d.Inputs...)
		i := maxIndex(actualOuputs)

		if expectedOutput[i] == 1 {
			correctCount++
		}
	}
	return float64(correctCount) / float64(len(testData))
}

func checkErr(err error) {
	if err != nil {
		log.Print(err)
		runtime.Breakpoint()
	}
}
