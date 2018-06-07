package cogent

import t "gorgonia.org/tensor"

//Dataset x
type Dataset struct {
	Inputs  *t.Dense
	Outputs *t.Dense
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
	Squared LossMode = iota
	Cross
	Hinge
	Exponential
	HellingerDistance
	KullbackLeiblerDivergence
	GeneralizedKullbackLeiblerDivergence
	ItakuraSaitoDistance
)

//Position x
type Position struct {
	Layers []LayerData
	Loss   float64
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
	WeightDecayRate       float64
	ProbablityOfDeath     float64
	RidgeRegressionWeight float64
	KFolds                int
}

//MultiSwarmConfiguration x
type MultiSwarmConfiguration struct {
	NeuralNetworkConfiguration NeuralNetworkConfiguration
	SwarmCount                 int
	ParticleCount              int
}
