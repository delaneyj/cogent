package cogent

import (
	"log"
	"math"
	"math/rand"
	"runtime"

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
		KFolds:                10,
	}
	//Float x
	Float = t.Float64
)

//NeuralNetworkConfiguration x
type NeuralNetworkConfiguration struct {
	Loss         LossMode
	InputCount   int
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
	WeightsAndBiases *t.Dense
	Velocities       *t.Dense
	Activation       ActivationMode
}

func (l *LayerData) reset(weightRange float64) {
	rnd := func(x t.Tensor, scaler float64) {
		data := x.Data().([]float64)
		s := x.Shape()
		rowCount := s[0]
		colCount := s[1]

		for i := range data {
			if i%colCount == rowCount {
				data[i] = 1
				continue
			}

			lo := -scaler * weightRange
			hi := scaler * weightRange
			data[i] = (hi-lo)*rand.Float64() + lo
		}
		log.Printf("%+v", x)
	}

	rnd(l.WeightsAndBiases, 1)
	rnd(l.Velocities, 0.1)
}

func (nn *NeuralNetwork) weightsAndBiasesCount() int {
	count := 0
	for _, l := range nn.Layers {
		count += l.WeightsAndBiases.DataSize()
		// count += len(l.WeightsAndBiases)
	}
	return count
}

func (nn *NeuralNetwork) weights() []float64 {
	weights := make([]float64, nn.weightsAndBiasesCount())
	offset := 0
	for _, l := range nn.Layers {
		layerWBs := l.WeightsAndBiases.Data().([]float64)
		copy(weights[offset:], layerWBs)
		offset += len(layerWBs)
	}
	return weights
}

func (nn *NeuralNetwork) setWeights(weights []float64) {
	offset := 0
	for _, l := range nn.Layers {
		layerWBs := l.WeightsAndBiases.Data().([]float64)
		copy(layerWBs, weights[offset:])
		offset += len(layerWBs)
	}
}

func (nn *NeuralNetwork) velocities() []float64 {
	velocities := make([]float64, nn.weightsAndBiasesCount())
	offset := 0
	for _, l := range nn.Layers {
		v := l.Velocities.Data().([]float64)
		copy(velocities[offset:], v)
		offset += len(v)
	}
	return velocities
}

func (nn *NeuralNetwork) setVelocities(velocities []float64) {
	offset := 0
	for _, l := range nn.Layers {
		v := l.Velocities.Data().([]float64)
		copy(v, velocities[offset:])
		offset += len(v)
	}
}

func (nn *NeuralNetwork) reset(weightRange float64) {
	for _, l := range nn.Layers {
		l.reset(weightRange)
	}
	nn.CurrentLoss = math.MaxFloat64
	nn.Best.Loss = math.MaxFloat64
}

func cloneAndExpandColumn(x *t.Dense) *t.Dense {
	s := x.Shape()
	rowCount := s[0]
	colCount := s[1]
	var y t.Dense
	err := x.CopyTo(&y)
	checkErr(err)
	y.Reshape(rowCount, colCount+1)

	data := y.Data().([]float64)
	for i := colCount; i < rowCount*colCount; i += colCount {
		data[i] = 1
	}
	return &y
}

//Activate feeds forward through the network
func (nn *NeuralNetwork) Activate(initialInputs *t.Dense) *t.Dense {
	inputs := cloneAndExpandColumn(initialInputs)

	var activated *t.Dense
	for _, l := range nn.Layers {
		outputs := must(inputs.Mul(l.WeightsAndBiases))
		activationFunc := activations[l.Activation]
		activated = activationFunc(outputs)
		inputs = cloneAndExpandColumn(activated)
	}
	return activated
}

//ClassificationAccuracy percentage correct using winner-takes all
func (nn *NeuralNetwork) ClassificationAccuracy(testData *Dataset) float64 {
	// maxIndex := func(s []float64) int {
	// 	// index of largest value
	// 	bigIndex := 0
	// 	biggestVal := s[0]
	// 	for i, x := range s {
	// 		if x > biggestVal {
	// 			biggestVal = x
	// 			bigIndex = i
	// 		}
	// 	}
	// 	return bigIndex
	// }

	correctCount := 0.0
	// splitIndex := len(testData.Outputs) / 2
	// wg := &sync.WaitGroup{}
	// wg.Add(len(testData))
	// mu := &sync.Mutex{}

	// lastLayer := nn.Layers[len(nn.Layers)-1]
	// shouldSplit := lastLayer.Activation == SplitSoftmax

	// for _, d := range testData {
	// 	go func(d *Data) {
	// 		expectedOutput := d.Outputs
	// 		actualOuputs := nn.Activate(d.Inputs)

	// 		if shouldSplit {
	// 			correctness := func(e, a []float64) float64 {
	// 				eIndex := maxIndex(e)
	// 				aIndex := maxIndex(a)
	// 				delta := math.Abs(float64(eIndex - aIndex))
	// 				x := 0.5 * math.Pow(0.5, delta)
	// 				return x
	// 			}

	// 			lC := correctness(expectedOutput[:splitIndex], actualOuputs[:splitIndex])
	// 			rC := correctness(expectedOutput[splitIndex:], actualOuputs[splitIndex:])
	// 			mu.Lock()
	// 			correctCount += (lC + rC)
	// 			mu.Unlock()
	// 		} else {
	// 			expectedOutput := d.Outputs
	// 			actualOuputs := nn.Activate(d.Inputs...)
	// 			i := maxIndex(actualOuputs)

	// 			if expectedOutput[i] == 1 {
	// 				mu.Lock()
	// 				correctCount++
	// 				mu.Unlock()
	// 			}
	// 		}

	// 		wg.Done()
	// 	}(d)
	// }

	// wg.Wait()

	rowCount := testData.Outputs.Shape()[0]
	return float64(correctCount) / float64(rowCount)
}

func checkErr(err error) {
	if err != nil {
		log.Print(err)
		runtime.Breakpoint()
	}
}
