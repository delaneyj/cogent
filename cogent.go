package cogent

import (
	"fmt"
	"log"
	"math"
	"math/rand"
)

const (
	maxIterations     = 700
	targetAccuracy    = 0.0001
	weightRange       = 10
	weightDecayRate   = 0.01
	inertiaWeight     = 0.729
	cognitiveWeight   = 1.49445
	socialWeight      = 1.49445
	globalWeight      = inertiaWeight / 2
	probablityOfDeath = 0.005
)

type connection struct {
}

type node struct {
	weights, velocities []float64
	bias                float64
}

func newNode(inputCount int) *node {
	inputWithBiasCount := inputCount + 1
	n := &node{
		weights:    make([]float64, inputWithBiasCount),
		velocities: make([]float64, inputWithBiasCount),
	}
	n.reset()

	return n
}

func (n *node) reset() {
	for i := range n.weights {
		n.weights[i] = (2*weightRange)*rand.Float64() - weightRange

		lo := -0.1 * weightRange
		hi := 0.1 * weightRange
		n.velocities[i] = (hi-lo)*rand.Float64() + lo
	}
}

type layer struct {
	Nodes          []*node        `json:"nodes"`
	ActivationName ActivationType `json:"activationName"`
}

type neuralNetwork struct {
	EntropyType EntropyType
	Layers      []*layer `json:"layers"`
	WeightCount int      `json:"weightCount"`

	CurrentFitnessError float64 `json:"currentFitnessError"`
	BestFitnessEntropy  float64 `json:"BestFitnessEntropy"`
	bestWeights         []float64
	entropyFn           entropyFn
}

func newNeuralNetwork(config *NeuralNetworkConfiguration) *neuralNetwork {
	fn := entropyFns[config.EntropyType]
	if fn == nil {
		log.Fatalf("Invalid entropy type '%s'", config.EntropyType)
	}
	nn := neuralNetwork{
		Layers:              make([]*layer, len(config.LayerConfigs)),
		CurrentFitnessError: math.MaxFloat64,
		BestFitnessEntropy:  math.MaxFloat64,
		EntropyType:         config.EntropyType,
		entropyFn:           fn,
	}
	weightCount := 0

	previousLayerCount := config.InputCount
	for i, layerConfig := range config.LayerConfigs {
		nodes := make([]*node, layerConfig.NodeCount)
		for j := 0; j < layerConfig.NodeCount; j++ {
			nodes[j] = newNode(previousLayerCount)
			weightCount += previousLayerCount + 1
		}

		l := layer{
			Nodes:          nodes,
			ActivationName: layerConfig.ActivationName,
		}
		nn.Layers[i] = &l
		previousLayerCount = layerConfig.NodeCount
	}
	nn.WeightCount = weightCount

	return &nn
}

func (nn *neuralNetwork) weights() []float64 {
	weights := make([]float64, 0, nn.WeightCount)

	for _, l := range nn.Layers {
		for _, n := range l.Nodes {
			weights = append(weights, n.weights...)
		}
	}

	if len(weights) != nn.WeightCount {
		log.Fatal("Oh noes")
	}
	return weights
}

func (nn *neuralNetwork) setWeights(weights []float64) {
	offset := 0
	for _, l := range nn.Layers {
		for _, n := range l.Nodes {
			weightCount := len(n.weights)
			max := weightCount + offset
			copy(n.weights, weights[offset:max])
			offset = max
		}
	}
}

func (nn *neuralNetwork) velocities() []float64 {
	velocities := make([]float64, 0, nn.WeightCount)

	for _, l := range nn.Layers {
		for _, n := range l.Nodes {
			velocities = append(velocities, n.velocities...)
		}
	}

	if len(velocities) != nn.WeightCount {
		log.Fatal("Oh noes")
	}
	return velocities
}

func (nn *neuralNetwork) setVelocities(velocities []float64) {
	offset := 0
	for _, l := range nn.Layers {
		for _, n := range l.Nodes {
			velocityCount := len(n.velocities)
			max := velocityCount + offset

			copy(n.velocities, velocities[offset:max])
			offset = max
		}
	}
}

func (nn *neuralNetwork) String() string {
	weights := make([]float64, nn.WeightCount)
	velocities := make([]float64, nn.WeightCount)

	index := 0
	for _, l := range nn.Layers {
		for _, n := range l.Nodes {
			for i, weight := range n.weights {
				weights[index] = weight
				velocities[index] = n.velocities[i]
				index++
			}
		}
	}

	x := struct {
		weights    []float64
		velocities []float64
		mse        float64
	}{
		weights, velocities, nn.CurrentFitnessError,
	}
	return fmt.Sprint(x)
}

func (nn *neuralNetwork) train(trainingData []Data, bestSwarmWeights, bestGlobalWeights []float64) float64 {
	flatArrayIndex := 0

	// 1. compute new velocity.  Depends on old velocity, best position of parrticle, and best position of any particle
	for _, l := range nn.Layers {
		for _, n := range l.Nodes {
			for i, currentLocalWeight := range n.weights {
				bestGlobalWeight := bestGlobalWeights[flatArrayIndex]
				bestSwarmWeight := bestSwarmWeights[flatArrayIndex]
				bestLocalWeight := nn.bestWeights[flatArrayIndex]

				currentLocalVelocity := n.velocities[i]

				oldVelocityFactor := inertiaWeight * currentLocalVelocity

				localRandomness := rand.Float64()
				bestLocationDelta := bestLocalWeight - currentLocalWeight
				localPositionFactor := cognitiveWeight * localRandomness * bestLocationDelta

				swarmRandomness := rand.Float64()
				bestSwarmlDelta := bestSwarmWeight - currentLocalWeight
				swarmPositionFactor := socialWeight * swarmRandomness * bestSwarmlDelta

				globalRandomness := rand.Float64()
				bestGlobalDelta := bestGlobalWeight - currentLocalWeight
				globalPositionFactor := globalWeight * globalRandomness * bestGlobalDelta

				revisedVelocity := oldVelocityFactor + localPositionFactor + swarmPositionFactor + globalPositionFactor
				n.velocities[i] = revisedVelocity

				flatArrayIndex++
			}
		}
	}

	// 2. use new velocity to compute new position but keep in range
	flatArrayIndex = 0
	for _, l := range nn.Layers {
		for _, n := range l.Nodes {
			for i, w := range n.weights {
				v := n.velocities[i]
				revisedPosition := w + v

				// 2a. optional: apply weight restriction
				clamped := math.Max(-weightRange, math.Min(weightRange, revisedPosition))

				// 2b. optional: apply weight decay (large weights tend to overfit)
				decayed := clamped * (1 + weightDecayRate)

				n.weights[i] = decayed
				flatArrayIndex++
			}
		}
	}

	// 3. use new position to compute new error
	nn.checkAndSetEntropy(trainingData)

	// 4. optional: does curr particle die?
	deathChance := rand.Float64()
	if deathChance < probablityOfDeath {
		nn.reset()
		nn.checkAndSetEntropy(trainingData)
	}

	return nn.CurrentFitnessError
}

func (nn *neuralNetwork) checkAndSetEntropy(data []Data) float64 {
	entropy := nn.calculateMeanEntropy(data)

	nn.CurrentFitnessError = entropy
	if entropy < nn.BestFitnessEntropy {
		nn.BestFitnessEntropy = entropy
		nn.bestWeights = nn.weights()
	}
	return entropy
}

func (nn *neuralNetwork) reset() {
	for _, l := range nn.Layers {
		for _, n := range l.Nodes {
			n.reset()
		}
	}
	nn.CurrentFitnessError = math.MaxFloat64
	nn.BestFitnessEntropy = math.MaxFloat64
}

func (nn *neuralNetwork) activate(intialInputs ...float64) []float64 {
	inputs := append(intialInputs, 1) //bias

	var outputs []float64
	for _, l := range nn.Layers {
		outputs = make([]float64, len(l.Nodes))

		for j, n := range l.Nodes {
			sum := 0.0
			for j, weight := range n.weights {
				input := inputs[j]
				sum += input * weight
			}

			outputs[j] = sum
		}

		activationFunc := activations[l.ActivationName]
		outputs = activationFunc(outputs)
		inputs = append(outputs, 1)
	}

	return outputs
}

func (nn *neuralNetwork) classificationAccuracy(testData []Data) float64 {
	// percentage correct using winner-takes all
	correctCount := 0

	maxIndex := func(s []float64) int { // helper for Accuracy(){
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

func (nn *neuralNetwork) calculateMeanEntropy(data []Data) float64 {
	sum := 0.0
	for _, d := range data {
		actualOuputs := nn.activate(d.Inputs...)
		err := nn.entropyFn(d.Outputs, actualOuputs)
		sum += err
	}
	entropy := sum / float64(len(data))
	return entropy
}

//Data x
type Data struct {
	Inputs  []float64 `json:"inputs"`
	Outputs []float64 `json:"outputs"`
}

type swarm struct {
	particles            []*neuralNetwork
	bestFitnessEntropy   float64
	bestFitnessPositions []float64
}

//MultiSwarm x
type MultiSwarm struct {
	swarms               []*swarm
	bestFitnessEntropy   float64
	bestFitnessPositions []float64
}

//LayerConfig x
type LayerConfig struct {
	NodeCount      int
	ActivationName ActivationType
}

//NeuralNetworkConfiguration x
type NeuralNetworkConfiguration struct {
	EntropyType  EntropyType
	InputCount   int
	LayerConfigs []LayerConfig
}

//MultiSwarmConfiguration x
type MultiSwarmConfiguration struct {
	NeuralNetworkConfiguration
	SwarmCount    int
	ParticleCount int
}

//NewMultiSwarm x
func NewMultiSwarm(config *MultiSwarmConfiguration) *MultiSwarm {
	if config.SwarmCount <= 0 {
		log.Fatal("No swarm count in config")
	}

	ms := MultiSwarm{
		swarms:             make([]*swarm, config.SwarmCount),
		bestFitnessEntropy: math.MaxFloat64,
	}
	for i := range ms.swarms {
		s := swarm{
			particles:          make([]*neuralNetwork, config.ParticleCount),
			bestFitnessEntropy: math.MaxFloat64,
		}

		for i := 0; i < config.ParticleCount; i++ {
			s.particles[i] = newNeuralNetwork(&config.NeuralNetworkConfiguration)
		}
		ms.swarms[i] = &s
	}

	return &ms
}

//Train x
func (ms *MultiSwarm) Train(trainingData []Data) int {
	tries := 0

	//set entropy for given training set
	for _, s := range ms.swarms {
		for _, p := range s.particles {
			fitnessError := p.checkAndSetEntropy(trainingData)
			if fitnessError < s.bestFitnessEntropy {
				w := p.weights()
				s.bestFitnessEntropy = fitnessError
				s.bestFitnessPositions = w

				if fitnessError < ms.bestFitnessEntropy {
					ms.bestFitnessEntropy = fitnessError
					ms.bestFitnessPositions = w
				}
			}
		}
	}

	if ms.bestFitnessEntropy == math.MaxFloat64 {
		log.Fatal("oh noes")
	}

	for _, s := range ms.swarms {
		// process particles in random order
		sequence := make([]int, len(s.particles))
		for i := range sequence {
			sequence[i] = i
		}

		// log.Printf("Best global error %f.", s.bestFitnessEntropy)
		for i := 0; i < maxIterations; i++ {
			if s.bestFitnessEntropy <= targetAccuracy {
				return tries
			}

			shuffle(sequence) // move particles in random sequence

			for pi := range s.particles {
				index := sequence[pi]
				currentParticle := s.particles[index]
				mse := currentParticle.train(trainingData, s.bestFitnessPositions, ms.bestFitnessPositions)
				tries++

				if mse < s.bestFitnessEntropy {
					w := currentParticle.weights()
					// log.Printf("New global best found fitness error %9f => %9f on try %d.", s.bestFitnessEntropy, mse, tries)
					s.bestFitnessEntropy = mse
					s.bestFitnessPositions = w

					if mse < ms.bestFitnessEntropy {
						ms.bestFitnessEntropy = mse
						ms.bestFitnessPositions = w
					}
				}
			}
		}
	}

	return tries
}

func shuffle(sequence []int) {
	l := len(sequence)
	for i, s := range sequence {
		ri := rand.Intn(l-i) + i
		tmp := sequence[ri]
		sequence[ri] = s
		sequence[i] = tmp
	}
}

//ClassificationAccuracy x
func (ms *MultiSwarm) ClassificationAccuracy(testData []Data) float64 {
	var swarmBest, globalBest *neuralNetwork
	for _, s := range ms.swarms {
		for _, p := range s.particles {
			if swarmBest == nil || p.BestFitnessEntropy < swarmBest.BestFitnessEntropy {
				swarmBest = p

				if globalBest == nil || p.BestFitnessEntropy < globalBest.BestFitnessEntropy {
					globalBest = p
				}
			}
		}
		swarmBest.setWeights(s.bestFitnessPositions)
	}
	globalBest.setWeights(ms.bestFitnessPositions)
	return globalBest.classificationAccuracy(testData)
}

func (ms *MultiSwarm) String() string {
	var best *neuralNetwork
	for _, s := range ms.swarms {
		for _, p := range s.particles {
			if best == nil || p.BestFitnessEntropy < best.BestFitnessEntropy {
				best = p
			}
		}
	}
	return best.String()
}
