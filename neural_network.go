package cogent

import (
	"log"
	"math"
	"math/rand"
	"runtime"
	"sync"

	t "gorgonia.org/tensor"
)

var (
	//DefaultTrainingConfig x
	DefaultTrainingConfig = TrainingConfiguration{
		InertialWeight:        0.729,
		CognitiveWeight:       1.49445,
		SocialWeight:          1.49445,
		GlobalWeight:          0.3645,
		MaxIterations:         700,
		TargetAccuracy:        0.000001,
		WeightRange:           100,
		WeightDecayRate:       0.01,
		ProbablityOfDeath:     0.005,
		RidgeRegressionWeight: 0.1,
	}
	//Float x
	Float = t.Float64
)

//NeuralNetworkConfiguration x
type NeuralNetworkConfiguration struct {
	Loss         LossMode
	InputCount   int
	BucketSize   int
	LayerConfigs []LayerConfig
}

//NeuralNetwork x
type NeuralNetwork struct {
	Loss        LossMode
	Layers      []LayerData
	CurrentLoss float64
	Best        Position
}

//LayerConfig x
type LayerConfig struct {
	NodeCount  int
	Activation ActivationMode
}

//LayerData x
type LayerData struct {
	NodeCount        int
	WeightsAndBiases t.Tensor
	Velocities       t.Tensor
	Activation       ActivationMode
}

func (l *LayerData) reset(weightRange float64) {
	runtime.Breakpoint()
	data := l.WeightsAndBiases.Data().([]float64)
	for i := range data {
		lo := -0.1 * weightRange
		hi := 0.1 * weightRange
		data[i] = (hi-lo)*rand.Float64() + lo
	}
	// l.Weights = t.Random(Float)
	// for i := range l.WeightsAndBiases {
	// 	l.WeightsAndBiases[i] = (2*weightRange)*rand.Float64() - weightRange

	// 	lo := -0.1 * weightRange
	// 	hi := 0.1 * weightRange
	// 	l.Velocities[i] = (hi-lo)*rand.Float64() + lo
	// }
}

func (nn *NeuralNetwork) weightsAndBiasesCount() int {
	count := 0
	for _, l := range nn.Layers {
		runtime.Breakpoint()
		count += l.WeightsAndBiases.DataSize()
		// count += len(l.WeightsAndBiases)
	}
	return count
}

func (nn *NeuralNetwork) weights() []float64 {
	weights := make([]float64, nn.weightsAndBiasesCount())
	log.Fatal("oh noes")
	// offset := 0
	// for _, l := range nn.Layers {
	// 	copy(weights[offset:], l.WeightsAndBiases)
	// 	offset += len(l.WeightsAndBiases)
	// }
	return weights
}

func (nn *NeuralNetwork) setWeights(weights []float64) {
	log.Fatal("oh noes")
	// offset := 0
	// for _, l := range nn.Layers {
	// 	copy(l.WeightsAndBiases, weights[offset:])
	// 	offset += len(l.WeightsAndBiases)
	// }
}

func (nn *NeuralNetwork) velocities() []float64 {
	velocities := make([]float64, nn.weightsAndBiasesCount())
	log.Fatal("oh noes")
	// offset := 0
	// for _, l := range nn.Layers {
	// 	copy(velocities[offset:], l.Velocities)
	// 	offset += len(l.Velocities)
	// }
	return velocities
}

func (nn *NeuralNetwork) setVelocities(velocities []float64) {
	log.Fatal("oh noes")
	// offset := 0
	// for _, l := range nn.Layers {
	// copy(l.Velocities, velocities[offset:])
	// offset += len(l.Velocities)
	// }
}

func (nn *NeuralNetwork) reset(weightRange float64) {
	for _, l := range nn.Layers {
		l.reset(weightRange)
	}
	nn.CurrentLoss = math.MaxFloat64
	nn.Best.Loss = math.MaxFloat64
}

//Activate feeds forward through the network
func (nn *NeuralNetwork) Activate(intialInputs ...float64) []float64 {
	// inputs := append(intialInputs, 1) // add bias
	var outputs []float64
	log.Fatal("oh noes")
	// for _, l := range nn.Layers {
	// 	nc := int(l.NodeCount)
	// 	outputs = make([]float64, nc)

	// 	delta := len(l.WeightsAndBiases) / nc
	// 	offset := 0
	// 	for n := 0; n < nc; n++ {
	// 		for i, w := range l.WeightsAndBiases[offset : offset+delta] {
	// 			input := inputs[i]
	// 			outputs[n] += input * w
	// 		}
	// 		offset += delta
	// 	}

	// 	activationFunc := activations[l.Activation]
	// 	outputs = activationFunc(outputs)
	// 	inputs = append(outputs, 1)
	// }
	return outputs
}

//ClassificationAccuracy percentage correct using winner-takes all
func (nn *NeuralNetwork) ClassificationAccuracy(testData []*Data, shouldSplit bool) float64 {
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

	correctCount := 0.0
	splitIndex := len(testData[0].Outputs) / 2
	wg := &sync.WaitGroup{}
	wg.Add(len(testData))
	mu := &sync.Mutex{}

	for _, d := range testData {
		go func(d *Data) {
			expectedOutput := d.Outputs
			actualOuputs := nn.Activate(d.Inputs...)

			if shouldSplit {
				correctness := func(e, a []float64) float64 {
					eIndex := maxIndex(e)
					aIndex := maxIndex(a)
					delta := math.Abs(float64(eIndex - aIndex))
					x := 0.5 * math.Pow(0.5, delta)
					return x
				}

				lC := correctness(expectedOutput[:splitIndex], actualOuputs[:splitIndex])
				rC := correctness(expectedOutput[splitIndex:], actualOuputs[splitIndex:])
				mu.Lock()
				correctCount += (lC + rC)
				mu.Unlock()
			} else {
				expectedOutput := d.Outputs
				actualOuputs := nn.Activate(d.Inputs...)
				i := maxIndex(actualOuputs)

				if expectedOutput[i] == 1 {
					mu.Lock()
					correctCount++
					mu.Unlock()
				}
			}

			wg.Done()
		}(d)
	}

	wg.Wait()

	return float64(correctCount) / float64(len(testData))
}

func checkErr(err error) {
	if err != nil {
		log.Print(err)
		runtime.Breakpoint()
	}
}
