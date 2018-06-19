package cogent

import t "gorgonia.org/tensor"

//DataBucket x
type DataBucket struct {
	Inputs  *t.Dense
	Outputs *t.Dense
}

//OutputColCount x
func (d *DataBucket) OutputColCount() int {
	return d.Outputs.Shape()[1]
}

//RowCount x
func (d *DataBucket) RowCount() int {
	return d.Outputs.Shape()[0]
}

//CloneAndAddBiasColumn x
func (d *DataBucket) CloneAndAddBiasColumn() *DataBucket {
	cloned := &DataBucket{
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
	Loss   float32
}

func nnToPosition(loss float32, nn *NeuralNetwork) Position {
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
	InertialWeight        float32
	CognitiveWeight       float32
	SocialWeight          float32
	GlobalWeight          float32
	MaxIterations         int
	TargetAccuracy        float32
	WeightRange           float32
	ProbablityOfDeath     float32
	RidgeRegressionWeight float32
	StoreGlobalBest       bool
}

//MultiSwarmConfiguration x
type MultiSwarmConfiguration struct {
	NeuralNetworkConfiguration NeuralNetworkConfiguration
	SwarmCount                 int
	ParticleCount              int
}

//DataRow x
type DataRow struct {
	Inputs  []float32
	Outputs []float32
}

//Data x
type Data []DataRow

//DataToTensorDataBucket x
func DataToTensorDataBucket(data Data, shouldAddBiasColum bool) *DataBucket {
	rows := len(data)
	iColCount := len(data[0].Inputs)
	oColCount := len(data[0].Outputs)
	bucket := DataBucket{
		Inputs: t.New(
			t.Of(Float),
			t.WithShape(rows, iColCount),
		),
		Outputs: t.New(
			t.Of(Float),
			t.WithShape(rows, oColCount),
		),
	}
	inputsBacking := bucket.Inputs.Data().([]float32)
	outputsBacking := bucket.Outputs.Data().([]float32)

	i, o := 0, 0
	for _, x := range data {
		copy(inputsBacking[i:], x.Inputs)
		i += len(x.Inputs)

		copy(outputsBacking[o:], x.Outputs)
		o += len(x.Outputs)
	}

	if shouldAddBiasColum {
		return bucket.CloneAndAddBiasColumn()
	}
	return &bucket
}
