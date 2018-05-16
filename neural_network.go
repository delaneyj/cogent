package cogent

import (
	"log"
	"math"
	"math/rand"
)

const (
	inertiaWeight   = 0.729
	cognitiveWeight = 1.49445
	socialWeight    = 1.49445
	globalWeight    = 0.3645

	maxIterations     = 700
	targetAccuracy    = 0.0001
	weightRange       = 10
	weightDecayRate   = 0.01
	probablityOfDeath = 0.005
)

//LayerConfig x
type LayerConfig struct {
	NodeCount      int
	ActivationName ActivationType
}

//NeuralNetworkConfiguration x
type NeuralNetworkConfiguration struct {
	LossType     LossType
	InputCount   int
	LayerConfigs []LayerConfig
}

//NewNeuralNetworkConfiguration x
func NewNeuralNetworkConfiguration(inputCount int, lc ...LayerConfig) NeuralNetworkConfiguration {
	nnc := NeuralNetworkConfiguration{
		LossType:     CrossLoss,
		InputCount:   inputCount,
		LayerConfigs: lc,
	}
	return nnc
}

type layer struct {
	NodeCount        int
	WeightsAndBiases []float64
	Velocities       []float64
	ActivationName   ActivationType `json:"activationName"`
}

type position struct {
	WeightsAndBiases []float64
	FitnessLoss      float64
}

type neuralNetwork struct {
	LossType    LossType
	Layers      []*layer `json:"layers"`
	WeightCount int      `json:"weightCount"`

	CurrentFitnessLoss float64  `json:"CurrentFitnessLoss"`
	Best               position `json:"best"`
	lossFn             lossFn
}

func newNeuralNetwork(config *NeuralNetworkConfiguration) *neuralNetwork {
	fn := lossFns[config.LossType]
	if fn == nil {
		log.Fatalf("Invalid loss type '%s'", config.LossType)
	}
	nn := neuralNetwork{
		Layers:             make([]*layer, len(config.LayerConfigs)),
		CurrentFitnessLoss: math.MaxFloat64,
		Best:               position{FitnessLoss: math.MaxFloat64},
		LossType:           config.LossType,
		lossFn:             fn,
	}

	previousLayerCount := config.InputCount
	for i, layerConfig := range config.LayerConfigs {
		wbCount := (previousLayerCount + 1) * layerConfig.NodeCount
		l := layer{
			NodeCount:        layerConfig.NodeCount,
			WeightsAndBiases: make([]float64, wbCount),
			Velocities:       make([]float64, wbCount),
			// Nodes:          nodes,
			ActivationName: layerConfig.ActivationName,
		}
		l.reset()
		nn.Layers[i] = &l
		previousLayerCount = layerConfig.NodeCount
		nn.WeightCount += wbCount
	}

	return &nn
}

func (l *layer) reset() {
	for i := range l.WeightsAndBiases {
		l.WeightsAndBiases[i] = (2*weightRange)*rand.Float64() - weightRange

		lo := -0.1 * weightRange
		hi := 0.1 * weightRange
		l.Velocities[i] = (hi-lo)*rand.Float64() + lo
	}
}

func (nn *neuralNetwork) weights() []float64 {
	weights := make([]float64, nn.WeightCount)
	offset := 0
	for _, l := range nn.Layers {
		copy(weights[offset:], l.WeightsAndBiases)
		offset += len(l.WeightsAndBiases)
	}
	return weights
}

func (nn *neuralNetwork) setWeights(weights []float64) {
	offset := 0
	for _, l := range nn.Layers {
		copy(l.WeightsAndBiases, weights[offset:])
		offset += len(l.WeightsAndBiases)
	}
}

func (nn *neuralNetwork) velocities() []float64 {
	velocities := make([]float64, nn.WeightCount)
	offset := 0
	for _, l := range nn.Layers {
		copy(velocities[offset:], l.Velocities)
		offset += len(l.Velocities)
	}
	return velocities
}

func (nn *neuralNetwork) setVelocities(velocities []float64) {
	offset := 0
	for _, l := range nn.Layers {
		copy(l.Velocities, velocities[offset:])
		offset += len(l.Velocities)
	}
}

func (nn *neuralNetwork) train(trainingData []Data, bestSwarmWeights, bestGlobalWeights []float64) float64 {
	flatArrayIndex := 0

	// Compute new velocity.  Depends on old velocity, best position of parrticle, and best position of any particle
	for _, l := range nn.Layers {
		for i, currentLocalWeight := range l.WeightsAndBiases {
			bestGlobalPosition := bestGlobalWeights[flatArrayIndex]
			bestSwarmPosition := bestSwarmWeights[flatArrayIndex]
			bestLocalPosition := nn.Best.WeightsAndBiases[flatArrayIndex]

			currentLocalVelocity := l.Velocities[i]

			oldVelocityFactor := inertiaWeight * currentLocalVelocity

			localRandomness := rand.Float64()
			bestLocationDelta := bestLocalPosition - currentLocalWeight
			localPositionFactor := cognitiveWeight * localRandomness * bestLocationDelta

			swarmRandomness := rand.Float64()
			bestSwarmlDelta := bestSwarmPosition - currentLocalWeight
			swarmPositionFactor := socialWeight * swarmRandomness * bestSwarmlDelta

			globalRandomness := rand.Float64()
			bestGlobalDelta := bestGlobalPosition - currentLocalWeight
			globalPositionFactor := globalWeight * globalRandomness * bestGlobalDelta

			revisedVelocity := oldVelocityFactor + localPositionFactor + swarmPositionFactor + globalPositionFactor
			l.Velocities[i] = revisedVelocity

			flatArrayIndex++
		}
	}

	flatArrayIndex = 0
	for _, l := range nn.Layers {
		for i, w := range l.WeightsAndBiases {
			v := l.Velocities[i]
			revisedPosition := w + v
			clamped := math.Max(-weightRange, math.Min(weightRange, revisedPosition)) // restriction
			decayed := clamped * (1 + weightDecayRate)                                // decay (large weights tend to overfit)

			l.WeightsAndBiases[i] = decayed
			flatArrayIndex++
		}
	}

	nn.checkAndSetloss(trainingData)

	deathChance := rand.Float64()
	if deathChance < probablityOfDeath {
		nn.reset()
		nn.checkAndSetloss(trainingData)
	}

	return nn.CurrentFitnessLoss
}

func (nn *neuralNetwork) checkAndSetloss(data []Data) float64 {
	loss := nn.calculateMeanLoss(data)

	nn.CurrentFitnessLoss = loss
	if loss < nn.Best.FitnessLoss {
		nn.Best.FitnessLoss = loss
		nn.Best.WeightsAndBiases = nn.weights()
	}
	return loss
}

func (nn *neuralNetwork) reset() {
	for _, l := range nn.Layers {
		l.reset()
	}
	nn.CurrentFitnessLoss = math.MaxFloat64
	nn.Best.FitnessLoss = math.MaxFloat64
}

func (nn *neuralNetwork) activate(intialInputs ...float64) []float64 {
	inputs := append(intialInputs, 1) // add bias

	var outputs []float64
	for _, l := range nn.Layers {
		outputs = make([]float64, l.NodeCount)

		delta := len(l.WeightsAndBiases) / l.NodeCount
		offset := 0
		for n := 0; n < l.NodeCount; n++ {
			for i, w := range l.WeightsAndBiases[offset : offset+delta] {
				input := inputs[i]
				outputs[n] += input * w
			}
			offset += delta
		}

		activationFunc := activations[l.ActivationName]
		outputs = activationFunc(outputs)
		inputs = append(outputs, 1)
	}

	return outputs
}

func (nn *neuralNetwork) classificationAccuracy(testData []Data) float64 {
	correctCount := 0 // percentage correct using winner-takes all

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
		actualOuputs := nn.activate(d.Inputs...)
		i := maxIndex(actualOuputs)

		if expectedOutput[i] == 1 {
			correctCount++
		}
	}
	return float64(correctCount) / float64(len(testData))
}

func (nn *neuralNetwork) calculateMeanLoss(data []Data) float64 {
	sum := 0.0
	for _, d := range data {
		actualOuputs := nn.activate(d.Inputs...)
		err := nn.lossFn(d.Outputs, actualOuputs)
		sum += err
	}
	loss := sum / float64(len(data))
	return loss
}
