package cogent

import t "gorgonia.org/tensor"

//Dataset x
type Dataset struct {
	Inputs  *t.Dense
	Outputs *t.Dense
}

func (d *Dataset) outputColCount() int {
	return d.Outputs.Shape()[1]
}

func (d *Dataset) rowCount() int {
	return d.Outputs.Shape()[0]
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
