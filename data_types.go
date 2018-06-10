package cogent

import t "gorgonia.org/tensor"

//Dataset x
type Dataset struct {
	Inputs  *t.Dense
	Outputs *t.Dense
}

//OutputColCount x
func (d *Dataset) OutputColCount() int {
	return d.Outputs.Shape()[1]
}

//RowCount x
func (d *Dataset) RowCount() int {
	return d.Outputs.Shape()[0]
}

//CloneAndAddBiasColumn x
func (d *Dataset) CloneAndAddBiasColumn() *Dataset {
	cloned := &Dataset{
		Inputs:  cloneAndExpandColumn(d.Inputs),
		Outputs: d.Outputs.Clone().(*t.Dense),
	}
	return cloned
}

//ActivationMode x
type ActivationMode int

//ActivationModes
const (
	Identity ActivationMode = iota
	BinaryStep
	Sigmoid
	HyperbolicTangent
	ArcTan
	Softsign
	ISRU
	ReLU
	LeakyReLU
	ELU
	SELU
	SoftPlus
	BentIdentity
	Sinusoid
	Sinc
	Gaussian
	Softmax
	Maxout
	SplitSoftmax
)

//LossMode x
type LossMode int

//LossModes
const (
	SquaredLoss LossMode = iota
	CrossLoss
	HingeLoss
	ExponentialLoss
	HellingerDistanceLoss
	KullbackLeiblerDivergenceLoss
	GeneralizedKullbackLeiblerDivergenceLoss
	ItakuraSaitoDistanceLoss
)

//Position x
type Position struct {
	Layers []LayerData
	Loss   float64
}

func nnToPosition(loss float64, nn *NeuralNetwork) Position {
	layers := make([]LayerData, len(nn.Layers))
	for i := range nn.Layers {
		layers[i] = nn.Layers[i].Clone()
	}
	p := Position{
		Loss:   loss,
		Layers: layers,
	}
	return p
}

//TrainingConfiguration x
type TrainingConfiguration struct {
	InertialWeight        float64
	CognitiveWeight       float64
	SocialWeight          float64
	GlobalWeight          float64
	MaxIterations         int
	TargetAccuracy        float64
	WeightRange           float64
	ProbablityOfDeath     float64
	RidgeRegressionWeight float64
	KFolds                int
	StoreGlobalBest       bool
}

//MultiSwarmConfiguration x
type MultiSwarmConfiguration struct {
	NeuralNetworkConfiguration NeuralNetworkConfiguration
	SwarmCount                 int
	ParticleCount              int
}

//Data x
type Data []struct {
	Inputs  []float64
	Outputs []float64
}

//DataToTensorDataset x
func DataToTensorDataset(data Data) *Dataset {
	rows := len(data)
	iColCount := len(data[0].Inputs)
	oColCount := len(data[0].Outputs)
	dataset := Dataset{
		Inputs: t.New(
			t.Of(Float),
			t.WithShape(rows, iColCount),
		),
		Outputs: t.New(
			t.Of(Float),
			t.WithShape(rows, oColCount),
		),
	}
	inputsBacking := dataset.Inputs.Data().([]float64)
	outputsBacking := dataset.Outputs.Data().([]float64)

	i, o := 0, 0
	for _, x := range data {
		copy(inputsBacking[i:], x.Inputs)
		i += len(x.Inputs)

		copy(outputsBacking[o:], x.Outputs)
		o += len(x.Outputs)
	}

	return dataset.CloneAndAddBiasColumn()
}
