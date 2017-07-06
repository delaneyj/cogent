package cogent

import (
	"encoding/json"
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

	previousLayerCount := layerCounts[0]
	for i, layerCount := range layerCounts[1:] {
		nodes := make([]*node, layerCount)
		for j := 0; j < layerCount; j++ {
			nodes[j] = newNode(previousLayerCount)
		}

		l := layer{
			Nodes: nodes,
		}
		nn.Layers[i] = &l
		previousLayerCount = layerCount
	}

	return &nn
}

func (nn *neuralNetwork) train(trainingData []Data, bestOfSwarm *neuralNetwork) float64 {
	// 1. compute new velocity.  Depends on old velocity, best position of parrticle, and best position of any particle
	for i, l := range nn.Layers {
		bestGlobalLayer := bestOfSwarm.Layers[i]
		for j, n := range l.Nodes {
			bestNodes := bestGlobalLayer.Nodes[j]

			for k, c := range n.Current {
				bestGlobal := bestNodes.Best[k]
				bestLocal := n.Best[k]

				oldVelocityFactor := inertiaWeight * c.Velocity

				localRandomness := r.float64()
				bestLocationDelta := bestLocal.Weight - c.Weight
				localPositionFactor := cognitiveWeight * localRandomness * bestLocationDelta

				globalRandomness := r.float64()
				bestGlobalDelta := bestGlobal.Weight - c.Weight
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
	mse := nn.meanSquaredError(trainingData)
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

	// 4. optional: does curr particle die?
	deathChance := r.float64()
	if deathChance < probablityOfDeath {
		// new position, leave velocity, update error
		nn.reset()
	}

	return nn.CurrentFitnessError
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

	for i, l := range nn.Layers {
		outputs = make([]float64, len(l.Nodes))

		for _, n := range l.Nodes {
			sum := 0.0
			for j, c := range n.Current {
				sum += c.Weight * previousInputs[j]
			}
			outputs[i] = hyperTan(sum)
		}

		previousInputs = append(outputs, 1)
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
	return sumSquaredError / float64(len(data))
}

//Data x
type Data struct {
	Inputs  []float64 `json:"inputs"`
	Outputs []float64 `json:"outputs"`
}

//Swarm x
type Swarm struct {
	particles  []*neuralNetwork
	globalBest *neuralNetwork
}

//NewSwarm particles, inputsCount, hiddenLayerCounts ..., outputsCount
func NewSwarm(particleCount int, layerCounts ...int) *Swarm {
	s := Swarm{
		particles: make([]*neuralNetwork, particleCount),
	}

	for i := 0; i < particleCount; i++ {
		s.particles[i] = newNeuralNetwork(layerCounts...)
	}

	s.globalBest = s.particles[0]

	return &s
}

//JSON x
func (s *Swarm) JSON() string {
	bytes, _ := json.MarshalIndent(s.globalBest, ``, `  `)
	return string(bytes)
}

//Train x
func (s *Swarm) Train(maxIterations int, targetAccuracy float64, trainingData []Data) int {
	for i := 0; i < maxIterations; i++ {
		for j, p := range s.particles {
			mse := p.train(trainingData, s.globalBest)
			if mse < s.globalBest.BestFitnessError {
				log.Printf("<%d:%d> %6f => %6f", i, j, s.globalBest.BestFitnessError, mse)
				s.globalBest = p

				if mse <= targetAccuracy {
					return i
				}
			}
		}
	}

	return maxIterations
}

//ClassificationAccuracy x
func (s *Swarm) ClassificationAccuracy(testData []Data) float64 {
	return s.globalBest.classificationAccuracy(testData)
}
