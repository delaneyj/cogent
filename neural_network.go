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
		ProbablityOfDeath:     0.005,
		RidgeRegressionWeight: 0.1,
		KFolds:                10,
		StoreGlobalBest:       false,
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
	Activation       ActivationMode
}

func fillTensorWithRandom(r *rand.Rand, x *t.Dense, scaler, weightRange float64) {
	data := x.Data().([]float64)

	for i := range data {
		lo := -scaler * weightRange
		hi := scaler * weightRange
		data[i] = (hi-lo)*rand.Float64() + lo
	}
	// log.Printf("%+v", x)
}

func (l *LayerData) reset(r *rand.Rand, lti *layerTrainingInfo, weightRange float64) {
	fillTensorWithRandom(r, l.WeightsAndBiases, 1, weightRange)
	fillTensorWithRandom(r, lti.Velocities, 0.1, weightRange)
}

//Clone x
func (l LayerData) Clone() LayerData {
	return LayerData{
		NodeCount:        l.NodeCount,
		WeightsAndBiases: l.WeightsAndBiases.Clone().(*t.Dense),
		Activation:       l.Activation,
	}
}

func (nn *NeuralNetwork) weightsAndBiasesCount() int {
	count := 0
	for _, l := range nn.Layers {
		count += l.WeightsAndBiases.DataSize()
	}
	return count
}

func (nn *NeuralNetwork) reset(r *rand.Rand, ltis []*layerTrainingInfo, weightRange float64) {
	for i, l := range nn.Layers {
		l.reset(r, ltis[i], weightRange)
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

func resetBiasColumn(tt *t.Dense) {
	colCount := tt.Shape()[1]
	data := tt.Data().([]float64)
	for i := colCount - 1; i < len(data); i += colCount {
		data[i] = 1
	}
}

//Activate feeds forward through the network
func (nn *NeuralNetwork) Activate(initialInputs *t.Dense) *t.Dense {
	inputs := initialInputs

	lastLayerIndex := len(nn.Layers) - 1
	var activated *t.Dense
	for i, l := range nn.Layers {
		// log.Printf("<Activate Layer %d>\nInput\n%+v\nLayer\n%+v", i, inputs, l.WeightsAndBiases)
		outputs := must(inputs.MatMul(l.WeightsAndBiases))
		activationFunc := activations[l.Activation]
		activated = activationFunc(outputs)
		// log.Printf("Outputs\n%+v\nActivated\n%+v", outputs, activated)

		if i != lastLayerIndex {
			resetBiasColumn(activated)
			inputs = activated
		}
	}
	return activated
}

//ClassificationAccuracy percentage correct using winner-takes all
func (nn *NeuralNetwork) ClassificationAccuracy(testData *Dataset) float64 {
	rowCount := testData.RowCount()
	colCount := testData.OutputColCount()

	correctCount := 0.0
	splitIndex := colCount / 2
	shouldSplit := nn.Layers[len(nn.Layers)-1].Activation == SplitSoftmax

	expected := testData.Outputs
	expectedBacking := expected.Data().([]float64)
	actual := nn.Activate(testData.Inputs)
	actualBacking := actual.Data().([]float64)
	// log.Printf("Expected\n%+v\nActual\n%+v", expected, actual)

	for i := 0; i < rowCount; i++ {
		start := i * colCount
		end := start + colCount

		expected := expectedBacking[start:end]
		actual := actualBacking[start:end]
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
