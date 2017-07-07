package main

import (
	"fmt"
	"log"
	"math"
)

const (
	weightRange       = 10
	inertiaWeight     = 0.729
	cognitiveWeight   = 1.49445
	socialWeight      = 1.49445
	probablityOfDeath = 0.005
)

type connection struct {
}

type dataPair struct {
	Weight   float64 `json:"weight"`
	Velocity float64 `json:"velocity"`
}

type node struct {
	Current []*dataPair `json:"current"`
	Best    []*dataPair `json:"best"`
}

func newNode(inputCount int) *node {
	inputWithBiasCount := inputCount + 1
	n := &node{
		Current: make([]*dataPair, inputWithBiasCount),
		Best:    make([]*dataPair, inputWithBiasCount),
	}
	n.reset()

	return n
}

func (n *node) reset() {
	for i := range n.Current {
		w := (2*weightRange)*r.float64() - weightRange

		lo := -0.1 * weightRange
		hi := 0.1 * weightRange
		v := (hi-lo)*r.float64() + lo

		// log.Println(w, v)
		n.Current[i] = &dataPair{
			Weight:   w,
			Velocity: v,
		}
		n.Best[i] = &dataPair{
			Weight:   w,
			Velocity: v,
		}
	}
}

type layer struct {
	Nodes []*node `json:"nodes"`
}

type neuralNetwork struct {
	Layers              []*layer `json:"layers"`
	CurrentFitnessError float64  `json:"currentFitnessError"`
	BestFitnessError    float64  `json:"bestFitnessError"`
	WeightCount         int      `json:"weightCount"`
}

func newNeuralNetwork(layerCounts ...int) *neuralNetwork {
	if len(layerCounts) < 3 {
		log.Fatal("Have to have at least one hidden layer")
	}
	nn := neuralNetwork{
		Layers:              make([]*layer, len(layerCounts)-1),
		CurrentFitnessError: math.MaxFloat64,
		BestFitnessError:    math.MaxFloat64,
	}
	weightCount := 0

	previousLayerCount := layerCounts[0]
	for i, layerCount := range layerCounts[1:] {
		nodes := make([]*node, layerCount)
		for j := 0; j < layerCount; j++ {
			nodes[j] = newNode(previousLayerCount)
			weightCount += previousLayerCount + 1
		}

		l := layer{
			Nodes: nodes,
		}
		nn.Layers[i] = &l
		previousLayerCount = layerCount
	}
	nn.WeightCount = weightCount

	return &nn
}

func (nn *neuralNetwork) weights() []float64 {
	weights := make([]float64, nn.WeightCount)
	i := 0
	for _, l := range nn.Layers {
		for _, n := range l.Nodes {
			for _, c := range n.Current {
				weights[i] = c.Weight
				i++
			}
		}
	}
	return weights
}

func (nn *neuralNetwork) setWeights(weights []float64) {
	i := 0
	for _, l := range nn.Layers {
		for _, n := range l.Nodes {
			for _, c := range n.Current {
				c.Weight = weights[i]
				i++
			}
		}
	}
}

func (nn *neuralNetwork) String() string {
	weights := make([]float64, nn.WeightCount)
	velocities := make([]float64, nn.WeightCount)

	i := 0
	for _, l := range nn.Layers {
		for _, n := range l.Nodes {
			for _, c := range n.Current {
				weights[i] = c.Weight
				velocities[i] = c.Velocity
				i++
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

func (nn *neuralNetwork) train(trainingData []Data, globalBestWeights []float64) float64 {
	bestIndex := 0
	// 1. compute new velocity.  Depends on old velocity, best position of parrticle, and best position of any particle
	for _, l := range nn.Layers {
		for _, n := range l.Nodes {
			for k, c := range n.Current {
				bestGlobalWeight := globalBestWeights[bestIndex]
				bestIndex++
				bestLocal := n.Best[k]

				oldVelocityFactor := inertiaWeight * c.Velocity

				localRandomness := r.float64()
				bestLocationDelta := bestLocal.Weight - c.Weight
				localPositionFactor := cognitiveWeight * localRandomness * bestLocationDelta

				globalRandomness := r.float64()
				bestGlobalDelta := bestGlobalWeight - c.Weight
				globalPositionFactor := socialWeight * globalRandomness * bestGlobalDelta

				revisedVelocity := oldVelocityFactor + localPositionFactor + globalPositionFactor
				c.Velocity = revisedVelocity
			}
		}
	}

	// 2. use new velocity to compute new position but keep in range
	for _, l := range nn.Layers {
		for _, n := range l.Nodes {
			for _, c := range n.Current {
				revisedPosition := c.Weight + c.Velocity
				clamped := math.Max(-weightRange, math.Min(weightRange, revisedPosition))
				c.Weight = clamped
			}
		}
	}

	// 2b. optional: apply weight decay (large weights tend to overfit)

	// 3. use new position to compute new error
	nn.checkAndSetMSE(trainingData)

	// 4. optional: does curr particle die?
	deathChance := r.float64()
	if deathChance < probablityOfDeath {
		// new position, leave velocity, update error
		nn.reset()
		nn.checkAndSetMSE(trainingData)
	}

	return nn.CurrentFitnessError
}

func (nn *neuralNetwork) checkAndSetMSE(data []Data) float64 {
	mse := nn.meanSquaredError(data)
	nn.CurrentFitnessError = mse
	if mse < nn.BestFitnessError {
		nn.BestFitnessError = mse
		for _, l := range nn.Layers {
			for _, n := range l.Nodes {
				for i, c := range n.Current {
					b := n.Best[i]
					b.Weight = c.Weight
					b.Velocity = c.Velocity
				}
			}
		}
	}
	return mse
}

func (nn *neuralNetwork) reset() {
	for _, l := range nn.Layers {
		for _, n := range l.Nodes {
			n.reset()
		}
	}
	nn.CurrentFitnessError = math.MaxFloat64
	nn.BestFitnessError = math.MaxFloat64
}

func (nn *neuralNetwork) activate(inputs ...float64) []float64 {
	previousInputs := append(inputs, 1) //bias

	var outputs []float64

	lastLayerIndex := len(nn.Layers) - 1
	for i, l := range nn.Layers {
		outputs = make([]float64, len(l.Nodes))

		for _, n := range l.Nodes {
			sum := 0.0
			for j, c := range n.Current {
				input := previousInputs[j]
				weight := c.Weight
				sum += input * weight
			}

			if i != lastLayerIndex {
				sum = hyperTan(sum)
			}
			outputs[i] = sum
		}

		previousInputs = append(outputs, 1)
	}

	classified := softmax(outputs)
	// log.Println(outputs, classified)
	return classified
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

func hyperTan(x float64) float64 {
	switch {
	case x < -20:
		return -1
	case x > 20:
		return 1
	default:
		return math.Tanh(x)
	}
}

func softmax(outputs []float64) []float64 {
	// does all output nodes at once so scale doesn't have to be re-computed each time
	// determine max output sum
	max := outputs[0]
	for _, x := range outputs {
		if x > max {
			max = x
		}
	}

	// determine scaling factor -- sum of exp(each val - max)
	scale := 0.0
	for _, x := range outputs {
		scale += math.Exp(x - max)
	}

	result := make([]float64, len(outputs))
	for i, x := range outputs {
		result[i] = math.Exp(x-max) / scale
	}

	return result // now scaled so that xi sum to 1.0
}

func (nn *neuralNetwork) meanSquaredError(data []Data) float64 {
	sumSquaredError := 0.0
	for _, d := range data {
		actualOuputs := nn.activate(d.Inputs...)
		for i, actualOutput := range actualOuputs {
			expectedOutput := d.Outputs[i]
			delta := actualOutput - expectedOutput
			sumSquaredError += delta * delta
		}
	}
	mse := sumSquaredError / float64(len(data))
	return mse
}

//Data x
type Data struct {
	Inputs  []float64 `json:"inputs"`
	Outputs []float64 `json:"outputs"`
}

//Swarm x
type Swarm struct {
	particles                  []*neuralNetwork
	globalBestFitnessError     float64
	globalBestFitnessPositions []float64
}

//NewSwarm particles, inputsCount, hiddenLayerCounts ..., outputsCount
func NewSwarm(particleCount int, layerCounts ...int) *Swarm {
	s := Swarm{
		particles:              make([]*neuralNetwork, particleCount),
		globalBestFitnessError: math.MaxFloat64,
	}

	for i := 0; i < particleCount; i++ {
		s.particles[i] = newNeuralNetwork(layerCounts...)
	}

	return &s
}

//Train x
func (s *Swarm) Train(maxIterations int, targetAccuracy float64, trainingData []Data) int {
	iterations := 0

	//set mse for given training set
	for _, p := range s.particles {
		mse := p.checkAndSetMSE(trainingData)

		if mse < s.globalBestFitnessError {
			s.globalBestFitnessError = mse
			s.globalBestFitnessPositions = p.weights()
		}
	}
	// log.Println(s.globalBest)

	// process particles in random order
	sequence := make([]int, len(s.particles))
	for i := range sequence {
		sequence[i] = i
	}

	log.Println(s.globalBestFitnessError)
	for i := 0; i < maxIterations; i++ {
		if s.globalBestFitnessError <= targetAccuracy {
			return iterations
		}

		shuffle(sequence) // move particles in random sequence

		for pi := range s.particles {
			index := sequence[pi]
			currentParticle := s.particles[index]

			// log.Printf("%+v", currentParticle)

			mse := currentParticle.train(trainingData, s.globalBestFitnessPositions)
			// log.Println(i, index, iterations, mse)
			iterations++

			if mse < s.globalBestFitnessError {
				log.Printf("<%d:%d> %9f => %9f", i, pi, s.globalBestFitnessError, mse)
				s.globalBestFitnessError = mse
				s.globalBestFitnessPositions = currentParticle.weights()
			}
		}
	}

	return iterations
}

func shuffle(sequence []int) {
	l := len(sequence)
	for i, s := range sequence {
		ri := r.nextRange(i, l)
		tmp := sequence[ri]
		sequence[ri] = s
		sequence[i] = tmp
	}
}

//ClassificationAccuracy x
func (s *Swarm) ClassificationAccuracy(testData []Data) float64 {
	var best *neuralNetwork
	for _, p := range s.particles {
		if best == nil || p.BestFitnessError < best.BestFitnessError {
			best = p
		}
	}
	best.setWeights(s.globalBestFitnessPositions)
	return best.classificationAccuracy(testData)
}
