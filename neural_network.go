package cogent

import (
	"bytes"
	"encoding/gob"
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
		MaxIterations:         500,
		TargetAccuracy:        1,
		WeightRange:           10,
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
	WeightsAndBiases t.Dense
	Velocities       t.Dense
	Activation       ActivationMode
}

func (l *LayerData) reset(weightRange float64) {
	rnd := func(x t.Tensor, scaler float64) {
		data := x.Data().([]float64)

		for i := range data {
			lo := -scaler * weightRange
			hi := scaler * weightRange
			data[i] = (hi-lo)*rand.Float64() + lo
		}
		// log.Printf("%+v", x)
	}

	rnd(&l.WeightsAndBiases, 1)
	rnd(&l.Velocities, 0.1)
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

func cloneAndExpandColumn(initialT *t.Dense) *t.Dense {
	s := initialT.Shape()
	initialRowCount := s[0]
	initialColCount := s[1]
	expandedColCount := initialColCount + 1
	expandedT := t.New(
		t.Of(Float),
		t.WithShape(initialRowCount, expandedColCount),
	)

	initial := initialT.Data().([]float64)
	expanded := expandedT.Data().([]float64)

	intialIndex := 0
	for i := range expanded {
		if i%expandedColCount == expandedColCount-1 {
			expanded[i] = 1
			continue
		}
		expanded[i] = initial[intialIndex]
		intialIndex++
	}

	// log.Printf("was initially\n%+v. But now is\n%+v", initialT, expandedT)
	return expandedT
}

//Activate feeds forward through the network
func (nn *NeuralNetwork) Activate(initialInputs *t.Dense) *t.Dense {
	inputs := cloneAndExpandColumn(initialInputs)

	var activated *t.Dense
	for _, l := range nn.Layers {
		// log.Printf("<Activate Layer %d>\nInput\n%+v\nLayer\n%+v", i, inputs, l.WeightsAndBiases)
		outputs := must(inputs.MatMul(&l.WeightsAndBiases))
		activationFunc := activations[l.Activation]
		activated = activationFunc(outputs)
		// log.Printf("Outputs\n%+v\nActivated\n%+v", outputs, activated)
		inputs = cloneAndExpandColumn(activated)
	}
	return activated
}

//ClassificationAccuracy percentage correct using winner-takes all
func (nn *NeuralNetwork) ClassificationAccuracy(testData *Dataset) float64 {
	rowCount := testData.rowCount()
	colCount := testData.outputColCount()

	correctCount := 0.0
	splitIndex := colCount / 2
	shouldSplit := nn.Layers[len(nn.Layers)-1].Activation == SplitSoftmax

	eT := testData.Outputs
	expectedOutput := eT.Data().([]float64)
	aT := nn.Activate(testData.Inputs)
	actualOuputs := aT.Data().([]float64)
	log.Printf("%+v", aT)

	for i := 0; i < rowCount; i++ {
		start := i * colCount
		end := start + colCount

		expected := expectedOutput[start:end]
		actual := actualOuputs[start:end]
		if shouldSplit {
			correctness := func(e, a []float64) float64 {
				eIndex := argmax(e)
				aIndex := argmax(a)
				delta := math.Abs(float64(eIndex - aIndex))
				x := 0.5 * math.Pow(0.5, delta)
				return x
			}

			lC := correctness(expected[:splitIndex], actual[:splitIndex])
			rC := correctness(expected[splitIndex:], actual[splitIndex:])
			correctCount += (lC + rC)
		} else {
			eI := argmax(expected)
			aI := argmax(actual)

			if eI == aI {
				correctCount++
			}
		}
	}

	ratio := correctCount / float64(rowCount)
	return ratio
}

//Marshal x
func (nn *NeuralNetwork) Marshal() []byte {
	var buf bytes.Buffer
	e := gob.NewEncoder(&buf)
	err := e.Encode(nn)
	if err != nil {
		log.Fatal(err)
	}
	return buf.Bytes()
}

//Unmarshal x
func (nn *NeuralNetwork) Unmarshal(buf []byte) {
	r := bytes.NewReader(buf)
	d := gob.NewDecoder(r)
	err := d.Decode(nn)
	if err != nil {
		log.Fatal(err)
	}
}

func checkErr(err error) {
	if err != nil {
		log.Print(err)
		runtime.Breakpoint()
	}
}

func argmax(a []float64) int {
	maxVal := -math.MaxFloat64
	maxInt := -1

	for i := range a {
		if a[i] > maxVal {
			maxVal = a[i]
			maxInt = i
		}
	}
	return maxInt
}
