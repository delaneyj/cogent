package cogent

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
	// n.reset()

	return n
}

type layer struct {
	Nodes []*node `json:"nodes"`
}

type neuralNetwork struct {
	Layers      []*layer `json:"layers"`
	WeightCount int      `json:"weightCount"`

	CurrentFitnessError float64 `json:"currentFitnessError"`
	BestFitnessError    float64 `json:"bestFitnessError"`
	bestWeights         []float64
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

func (nn *neuralNetwork) train(trainingData []Data, globalBestWeights []float64) float64 {
	flatArrayIndex := 0

	// 1. compute new velocity.  Depends on old velocity, best position of parrticle, and best position of any particle
	newVelocity := make([]float64, nn.WeightCount)
	for _, l := range nn.Layers {
		for _, n := range l.Nodes {
			for i, currentLocalWeight := range n.weights {
				bestGlobalWeight := globalBestWeights[flatArrayIndex]
				bestLocalWeight := nn.bestWeights[flatArrayIndex]

				currentLocalVelocity := n.velocities[i]

				oldVelocityFactor := inertiaWeight * currentLocalVelocity

				localRandomness := r.float64()
				bestLocationDelta := bestLocalWeight - currentLocalWeight
				localPositionFactor := cognitiveWeight * localRandomness * bestLocationDelta

				globalRandomness := r.float64()
				bestGlobalDelta := bestGlobalWeight - currentLocalWeight
				globalPositionFactor := socialWeight * globalRandomness * bestGlobalDelta

				revisedVelocity := oldVelocityFactor + localPositionFactor + globalPositionFactor
				newVelocity[flatArrayIndex] = revisedVelocity

				flatArrayIndex++
			}
		}
	}
	nn.setVelocities(newVelocity)

	// 2. use new velocity to compute new position but keep in range
	flatArrayIndex = 0
	newPosition := make([]float64, nn.WeightCount)
	for _, l := range nn.Layers {
		for _, n := range l.Nodes {
			for i, w := range n.weights {
				v := n.velocities[i]
				revisedPosition := w + v
				clamped := math.Max(-weightRange, math.Min(weightRange, revisedPosition))
				newPosition[flatArrayIndex] = clamped
				flatArrayIndex++
			}
		}
	}
	nn.setWeights(newPosition)

	// 2b. optional: apply weight decay (large weights tend to overfit)

	// 3. use new position to compute new error
	nn.checkAndSetMSE(trainingData, newPosition)

	// 4. optional: does curr particle die?
	deathChance := r.float64()
	if deathChance < probablityOfDeath {
		// new position, leave velocity, update error
		afterDeathPosition := make([]float64, nn.WeightCount)
		for i := range afterDeathPosition {
			afterDeathPosition[i] = (2*weightRange)*r.float64() - weightRange
		}
		nn.setWeights(afterDeathPosition)
		nn.checkAndSetMSE(trainingData, newPosition)
	}

	return nn.CurrentFitnessError
}

func (nn *neuralNetwork) checkAndSetMSE(data []Data, weights []float64) float64 {
	mse := nn.meanSquaredError(data, weights)
	nn.CurrentFitnessError = mse
	if mse < nn.BestFitnessError {
		nn.BestFitnessError = mse
		nn.bestWeights = nn.weights()
	}
	return mse
}

// func (nn *neuralNetwork) reset() {
// 	for _, l := range nn.Layers {
// 		for _, n := range l.Nodes {
// 			n.reset()
// 		}
// 	}
// 	nn.CurrentFitnessError = math.MaxFloat64
// 	nn.BestFitnessError = math.MaxFloat64
// }

func (nn *neuralNetwork) activate(intialInputs ...float64) []float64 {
	inputs := append(intialInputs, 1) //bias

	var outputs []float64

	lastLayerIndex := len(nn.Layers) - 1
	for i, l := range nn.Layers {
		outputs = make([]float64, len(l.Nodes))

		for j, n := range l.Nodes {
			sum := 0.0
			for j, weight := range n.weights {
				input := inputs[j]
				sum += input * weight
			}

			if i != lastLayerIndex {
				sum = hyperTan(sum)
			}
			outputs[j] = sum
		}

		inputs = append(outputs, 1)
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

func (nn *neuralNetwork) meanSquaredError(data []Data, weights []float64) float64 {
	// assumes that centroids and widths have been set!
	nn.setWeights(weights) // copy the weights to evaluate in

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
	tries := 0

	weightCount := s.particles[0].WeightCount

	//set mse for given training set
	for _, p := range s.particles {
		randomPosition := make([]float64, weightCount)
		randomVelocity := make([]float64, weightCount)

		for j := range randomPosition {
			//double lo = minX;
			//double hi = maxX;
			//randomPosition[j] = (hi - lo) * rnd.NextDouble() + lo;
			randomPosition[j] = (2*weightRange)*r.float64() - weightRange

			lo := -0.1 * weightRange
			hi := 0.1 * weightRange
			randomVelocity[j] = (hi-lo)*r.float64() + lo
		}

		p.setVelocities(randomVelocity)
		mse := p.checkAndSetMSE(trainingData, randomPosition)

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

	log.Printf("Best global error %f.", s.globalBestFitnessError)
	for i := 0; i < maxIterations; i++ {
		if s.globalBestFitnessError <= targetAccuracy {
			return tries
		}

		shuffle(sequence) // move particles in random sequence

		for pi := range s.particles {
			index := sequence[pi]
			currentParticle := s.particles[index]

			// log.Printf("%+v", currentParticle)

			mse := currentParticle.train(trainingData, s.globalBestFitnessPositions)
			// log.Println(i, index, iterations, mse)
			tries++

			if mse < s.globalBestFitnessError {
				log.Printf("New global best found mean square error %9f => %9f on try %d.", s.globalBestFitnessError, mse, tries)
				s.globalBestFitnessError = mse
				s.globalBestFitnessPositions = currentParticle.weights()
			}
		}
	}

	return tries
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
