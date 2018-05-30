package cogent

//Data x
type Data struct {
	Inputs  []float64
	Outputs []float64
}

//Dataset x
type Dataset []*Data

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

//LayerConfig x
type LayerConfig struct {
	NodeCount  int
	Activation ActivationMode
}

//LayerData x
type LayerData struct {
	NodeCount        int
	WeightsAndBiases []float64
	Velocities       []float64
	Activation       ActivationMode
}

//Position x
type Position struct {
	WeightsAndBiases []float64
	Loss             float64
}

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

//TrainingConfiguration x
type TrainingConfiguration struct {
	InertialWeight    float64
	CognitiveWeight   float64
	SocialWeight      float64
	GlobalWeight      float64
	MaxIterations     int
	TargetAccuracy    float64
	WeightRange       float64
	WeightDecayRate   float64
	ProbablityOfDeath float64
}

//MultiSwarmConfiguration x
type MultiSwarmConfiguration struct {
	NeuralNetworkConfiguration NeuralNetworkConfiguration
	SwarmCount                 int
	ParticleCount              int
}
