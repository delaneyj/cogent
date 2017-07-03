package pso

import (
	"log"
	"math/rand"
	"os"

	"fmt"

	"math"

	"github.com/olekukonko/tablewriter"
)

//ShowVector x
func ShowVector(vector []float64, valsPerRow, decimals int) {
	table := tablewriter.NewWriter(os.Stdout)

	s := []string{}
	for i, x := range vector {
		if i == 0 {
			s = append(s, fmt.Sprint(i))
		}

		if i%valsPerRow == 0 {
			table.Append(s)
			s = []string{fmt.Sprint(i)}
		}

		s = append(s, fmt.Sprintf("%9.4f", x))
	}
	table.Render()
}

//ShowMatrix x
func ShowMatrix(labels []string, matrix [][]float64) {
	table := tablewriter.NewWriter(os.Stdout)
	headers := append([]string{"ID"}, labels...)
	table.SetHeader(headers)

	for i, row := range matrix {
		s := []string{fmt.Sprint(i)}
		for _, cell := range row {
			s = append(s, fmt.Sprint(cell))
		}
		table.Append(s)
	}
	table.Render()
}

//NeuralNetwork x
type NeuralNetwork struct {
	numInput, numHidden, numOutput                     int
	inputs                                             []float64
	hiddenInputWeights, hiddenOutputWeights            [][]float64
	hiddenBiases, hiddenOutputs, outputBiases, outputs []float64
}

//NewNeuralNetwork x
func NewNeuralNetwork(numInput, numHidden, numOutput int) *NeuralNetwork {
	nn := NeuralNetwork{
		numInput:            numInput,
		numHidden:           numHidden,
		numOutput:           numOutput,
		inputs:              make([]float64, numInput),
		hiddenInputWeights:  makeMatrix(numInput, numHidden),
		hiddenBiases:        make([]float64, numHidden),
		hiddenOutputs:       make([]float64, numHidden),
		hiddenOutputWeights: makeMatrix(numHidden, numOutput),
		outputBiases:        make([]float64, numOutput),
		outputs:             make([]float64, numOutput),
	}
	return &nn
}

func makeMatrix(rows, cols int) [][]float64 {
	results := make([][]float64, rows)
	for r := range results {
		results[r] = make([]float64, cols)
	}
	return results
}

//SetWeights x
func (nn *NeuralNetwork) SetWeights(weights []float64) {
	// copy weights and biases in weights[] array to i-h weights, i-h biases, h-o weights, h-o biases
	numWeights := (nn.numInput * nn.numHidden) + (nn.numHidden * nn.numOutput) + nn.numHidden + nn.numOutput
	if len(weights) != numWeights {
		log.Fatal("Bad weights array length: ")
	}

	k := 0 // points into weights param

	for i := 0; i < nn.numInput; i++ {
		for j := 0; j < nn.numHidden; j++ {
			nn.hiddenInputWeights[i][j] = weights[k]
			k++
		}
	}

	for i := 0; i < nn.numHidden; i++ {
		nn.hiddenBiases[i] = weights[k]
		k++
	}

	for i := 0; i < nn.numHidden; i++ {
		for j := 0; j < nn.numOutput; j++ {
			nn.hiddenOutputWeights[i][j] = weights[k]
			k++
		}
	}

	for i := 0; i < nn.numOutput; i++ {
		nn.outputBiases[i] = weights[k]
		k++
	}
}

func (nn *NeuralNetwork) getWeights() []float64 {
	// returns the current set of wweights, presumably after training
	numWeights := (nn.numInput * nn.numHidden) + (nn.numHidden * nn.numOutput) + nn.numHidden + nn.numOutput
	result := make([]float64, numWeights)
	k := 0
	for _, x := range nn.hiddenInputWeights {
		for _, y := range x {
			result[k] = y
			k++
		}
	}

	for _, b := range nn.hiddenBiases {
		result[k] = b
		k++
	}

	for _, x := range nn.hiddenOutputWeights {
		for _, y := range x {
			result[k] = y
			k++
		}
	}

	for _, b := range nn.outputBiases {
		result[k] = b
		k++
	}
	return result
}

func (nn *NeuralNetwork) computeOutputs(xValues []float64) []float64 {
	if len(xValues) != nn.numInput {
		log.Fatal("Bad xValues array length")
	}

	hSums := make([]float64, nn.numHidden) // hidden nodes sums scratch array
	oSums := make([]float64, nn.numOutput) // output nodes sums

	copy(nn.inputs, xValues)

	for j := 0; j < nn.numHidden; j++ { // compute i-h sum of weights * inputs
		for i := 0; i < nn.numInput; i++ {
			hSums[j] += nn.inputs[i] * nn.hiddenInputWeights[i][j] // note +=
		}
	}

	for i := 0; i < nn.numHidden; i++ { // add biases to input-to-hidden sums
		hSums[i] += nn.hiddenBiases[i]
	}

	for i := 0; i < nn.numHidden; i++ { // apply activation
		nn.hiddenOutputs[i] = hyperTanFunction(hSums[i]) // hard-coded
	}

	for j := 0; j < nn.numOutput; j++ { // compute h-o sum of weights * hOutputs
		for i := 0; i < nn.numHidden; i++ {
			oSums[j] += nn.hiddenOutputs[i] * nn.hiddenOutputWeights[i][j]
		}
	}

	for i := 0; i < nn.numOutput; i++ { // add biases to input-to-hidden sums
		oSums[i] += nn.outputBiases[i]
	}

	softOut := softmax(oSums) // softmax activation does all outputs at once for efficiency\
	copy(nn.outputs, softOut)

	retResult := make([]float64, nn.numOutput) // could define a GetOutputs method instead
	copy(retResult, nn.outputs)

	return retResult
}

func hyperTanFunction(x float64) float64 {
	switch {
	case x < -20:
		return -1
	case x > 20:
		return 1
	default:
		return math.Tanh(x)
	}
}

func softmax(oSums []float64) []float64 {
	// does all output nodes at once so scale doesn't have to be re-computed each time
	// determine max output sum
	max := oSums[0]
	for _, x := range oSums {
		if x > max {
			max = x
		}
	}

	// determine scaling factor -- sum of exp(each val - max)
	scale := 0.0
	for _, x := range oSums {
		scale += math.Exp(x - max)
	}

	result := make([]float64, len(oSums))
	for i, x := range oSums {
		result[i] = math.Exp(x-max) / scale
	}

	return result // now scaled so that xi sum to 1.0
}

//Train x
func (nn *NeuralNetwork) Train(trainData [][]float64, numParticles, maxEpochs int, exitError, probDeath float64) ([]float64, int) {
	// PSO version training. best weights stored into NN and returned
	// particle position == NN weights
	weightCount := (nn.numInput * nn.numHidden) + (nn.numHidden * nn.numOutput) + nn.numHidden + nn.numOutput

	// use PSO to seek best weights
	tries := 0
	epoch := 0
	weightRange := 10.0 // for each weight. assumes data has been normalized about 0
	w := 0.729          // inertia weight
	c1 := 1.49445       // cognitive/local weight
	c2 := 1.49445       // social/global weight

	swarm := make([]*particle, numParticles)
	// log.Println("best solution found by any particle in the swarm. implicit initialization to all 0.0")
	bestGlobalPosition := make([]float64, weightCount)
	bestGlobalError := math.MaxFloat64 // smaller values better

	//double minV = -0.01 * maxX;  // velocities
	//double maxV = 0.01 * maxX;

	// log.Println("swarm initialization")
	// log.Println(" initialize each Particle in the swarm with random positions and velocities")
	for i := range swarm {
		randomPosition := make([]float64, weightCount)
		for j := range randomPosition {
			//double lo = minX;
			//double hi = maxX;
			//randomPosition[j] = (hi - lo) * rnd.NextDouble() + lo;
			randomPosition[j] = (2*weightRange)*rand.Float64() - weightRange
		}

		// log.Println("randomPosition is a set of weights; sent to NN")
		fitnessError := nn.meanSquaredError(trainData, randomPosition)
		randomVelocity := make([]float64, weightCount)

		for j := range randomVelocity {
			//double lo = -1.0 * Math.Abs(maxX - minX);
			//double hi = Math.Abs(maxX - minX);
			//randomVelocity[j] = (hi - lo) * rnd.NextDouble() + lo;
			lo := -0.1 * weightRange
			hi := 0.1 * weightRange
			randomVelocity[j] = (hi-lo)*rand.Float64() + lo
		}
		swarm[i] = newParticle(randomPosition, randomVelocity, randomPosition, fitnessError, fitnessError) // last two are best-position and best-error

		// log.Println("does current Particle have global best position/solution?")
		if swarm[i].FitnessError < bestGlobalError {
			bestGlobalError = swarm[i].FitnessError
			copy(bestGlobalPosition, swarm[i].Position)
		}
	}
	// log.Println("end of initialization")

	// log.Println("Entering main PSO weight estimation processing loop")

	// log.Println("main PSO algorithm")
	sequence := make([]int, numParticles) // process particles in random order
	for i := range sequence {
		sequence[i] = i
	}

	for epoch < maxEpochs {
		if bestGlobalError < exitError {
			break // early exit (MSE error)
		}

		newVelocity := make([]float64, weightCount) // step 1
		newPosition := make([]float64, weightCount) // step 2
		var newError float64                        // step 3

		shuffle(sequence) // move particles in random sequence

		for pi := range swarm {
			tries++

			i := sequence[pi]
			currP := swarm[i] // for coding convenience

			// 1. compute new velocity
			for j := range currP.Velocity { // each x value of the velocity
				r1 := rand.Float64()
				r2 := rand.Float64()

				// velocity depends on old velocity, best position of parrticle, and
				// best position of any particle
				newVelocity[j] = (w * currP.Velocity[j]) +
					(c1 * r1 * (currP.BestPosition[j] - currP.Position[j])) +
					(c2 * r2 * (bestGlobalPosition[j] - currP.Position[j]))
			}

			copy(currP.Velocity, newVelocity)

			// 2. use new velocity to compute new position
			for j := range currP.Position {
				newPosition[j] = currP.Position[j] + newVelocity[j] // compute new position
				if newPosition[j] < -weightRange {                  // keep in range
					newPosition[j] = -weightRange
				} else if newPosition[j] > weightRange {
					newPosition[j] = weightRange
				}
			}

			copy(currP.Position, newPosition)

			// 2b. optional: apply weight decay (large weights tend to overfit)

			// 3. use new position to compute new error
			//newError = MeanCrossEntropy(trainData, newPosition); // makes next check a bit cleaner
			newError = nn.meanSquaredError(trainData, newPosition)
			currP.FitnessError = newError

			if newError < currP.BestFitnessError { // new particle best?
				copy(currP.BestPosition, newPosition)
				currP.BestFitnessError = newError
			}

			if newError < bestGlobalError { // new global best?
				copy(bestGlobalPosition, newPosition)
				bestGlobalError = newError
			}

			// 4. optional: does curr particle die?
			die := rand.Float64()
			if die < probDeath {
				// new position, leave velocity, update error
				for j := range currP.Position {
					currP.Position[j] = (2*weightRange)*rand.Float64() - weightRange
				}
				currP.FitnessError = nn.meanSquaredError(trainData, currP.Position)
				copy(currP.BestPosition, currP.Position)
				currP.BestFitnessError = currP.FitnessError

				if currP.FitnessError < bestGlobalError { // global best by chance?
					bestGlobalError = currP.FitnessError
					copy(bestGlobalPosition, currP.Position)
				}
			}
		}

		epoch++
	}

	nn.SetWeights(bestGlobalPosition) // best position is a set of weights
	retResult := make([]float64, weightCount)
	copy(retResult, bestGlobalPosition)
	return retResult, tries
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

func (nn *NeuralNetwork) meanSquaredError(trainData [][]float64, weights []float64) float64 {
	// assumes that centroids and widths have been set!
	nn.SetWeights(weights) // copy the weights to evaluate in

	xValues := make([]float64, nn.numInput)  // inputs
	tValues := make([]float64, nn.numOutput) // targets
	sumSquaredError := 0.0

	for _, t := range trainData { // walk through each training data item
		// following assumes data has all x-values first, followed by y-values!
		copy(xValues, t)                      // extract inputs
		copy(tValues, t[nn.numInput:])        // extract targets
		yValues := nn.computeOutputs(xValues) // compute the outputs using centroids, widths, weights, bias values
		for j := range yValues {
			sumSquaredError += ((yValues[j] - tValues[j]) * (yValues[j] - tValues[j]))
		}
	}
	return sumSquaredError / float64(len(trainData))
}

//Accuracy x
func (nn *NeuralNetwork) Accuracy(testData [][]float64) float64 {
	// percentage correct using winner-takes all
	correctCount := 0
	xValues := make([]float64, nn.numInput)  // inputs
	tValues := make([]float64, nn.numOutput) // targets

	for _, t := range testData {
		copy(xValues, t) // parse test data into x-values and t-values
		copy(tValues, t[nn.numInput:])

		yValues := nn.computeOutputs(xValues)
		maxIndex := maxIndex(yValues) // which cell in yValues has largest value?

		if tValues[maxIndex] == 1.0 {
			correctCount++
		}
	}
	return float64(correctCount) / float64(len(testData))
}

func maxIndex(vector []float64) int { // helper for Accuracy(){
	// index of largest value
	bigIndex := 0
	biggestVal := vector[0]
	for i, x := range vector {
		if x > biggestVal {
			biggestVal = x
			bigIndex = i
		}
	}
	return bigIndex
}

type particle struct {
	Position     []float64 // equivalent to NN weights
	FitnessError float64   // measure of fitness
	Velocity     []float64

	BestPosition     []float64 // best position found so far by this Particle
	BestFitnessError float64

	//public double age; // optional used to determine death-birth
}

func newParticle(position, velocity, bestPosition []float64, fitnessError, bestFitnessError float64) *particle {
	p := particle{
		Position:         make([]float64, len(position)),
		FitnessError:     fitnessError,
		Velocity:         make([]float64, len(velocity)),
		BestPosition:     make([]float64, len(bestPosition)),
		BestFitnessError: bestFitnessError,
		//this.age = 0;
	}

	copy(p.Position, position)
	copy(p.Velocity, velocity)
	copy(p.BestPosition, bestPosition)

	return &p
}
