package cogent

import (
	"log"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func basicMathConfig() MultiSwarmConfiguration {
	deep := 1

	lc := LayerConfig{
		NodeCount:  3,
		Activation: ReLU,
	}

	sm := LayerConfig{
		NodeCount:  2,
		Activation: Softmax,
	}

	msc := MultiSwarmConfiguration{
		SwarmCount: 5,
		NeuralNetworkConfiguration: NeuralNetworkConfiguration{
			Loss:         Cross,
			InputCount:   2,
			LayerConfigs: make([]LayerConfig, deep+1),
		},
		ParticleCount: 10,
	}

	i := 0
	for ; i < deep; i++ {
		msc.NeuralNetworkConfiguration.LayerConfigs[i] = lc
	}
	msc.NeuralNetworkConfiguration.LayerConfigs[i] = sm

	return msc
}

func Test_XOR(t *testing.T) {
	t.Parallel()
	start := time.Now()
	examples := Dataset{
		{
			Inputs:  []float64{0, 0},
			Outputs: []float64{0, 1},
		},
		{
			Inputs:  []float64{0, 1},
			Outputs: []float64{1, 0},
		},
		{
			Inputs:  []float64{1, 0},
			Outputs: []float64{1, 0},
		},
		{
			Inputs:  []float64{1, 1},
			Outputs: []float64{0, 1},
		},
	}

	s := NewMultiSwarm(basicMathConfig(), DefaultTrainingConfig)
	s.Train(examples)
	accuracy := s.ClassificationAccuracy(examples...)
	assert.Equal(t, 1.0, accuracy)
	log.Print("XOR ", time.Since(start))
}

func Test_AND(t *testing.T) {
	t.Parallel()
	examples := Dataset{
		{
			Inputs:  []float64{0, 0},
			Outputs: []float64{0, 1},
		},
		{
			Inputs:  []float64{0, 1},
			Outputs: []float64{0, 1},
		},
		{
			Inputs:  []float64{1, 0},
			Outputs: []float64{0, 1},
		},
		{
			Inputs:  []float64{1, 1},
			Outputs: []float64{1, 0},
		},
	}

	s := NewMultiSwarm(basicMathConfig(), DefaultTrainingConfig)
	s.Train(examples)
	accuracy := s.ClassificationAccuracy(examples...)
	assert.Equal(t, 1.0, accuracy)
}

func Test_NOT(t *testing.T) {
	t.Parallel()
	examples := Dataset{
		{
			Inputs:  []float64{0, 0},
			Outputs: []float64{1, 0},
		},
		{
			Inputs:  []float64{0, 1},
			Outputs: []float64{0, 1},
		},
		{
			Inputs:  []float64{1, 0},
			Outputs: []float64{0, 1},
		},
		{
			Inputs:  []float64{1, 1},
			Outputs: []float64{0, 1},
		},
	}

	s := NewMultiSwarm(basicMathConfig(), DefaultTrainingConfig)
	s.Train(examples)
	accuracy := s.ClassificationAccuracy(examples...)
	assert.Equal(t, 1.0, accuracy)
}

func Test_OR(t *testing.T) {
	t.Parallel()
	examples := Dataset{
		{
			Inputs:  []float64{0, 0},
			Outputs: []float64{0, 1},
		},
		{
			Inputs:  []float64{0, 1},
			Outputs: []float64{1, 0},
		},
		{
			Inputs:  []float64{1, 0},
			Outputs: []float64{1, 0},
		},
		{
			Inputs:  []float64{1, 1},
			Outputs: []float64{1, 0},
		},
	}

	s := NewMultiSwarm(basicMathConfig(), DefaultTrainingConfig)
	s.Train(examples)
	accuracy := s.ClassificationAccuracy(examples...)
	assert.Equal(t, 1.0, accuracy)
}

func Test_NAND(t *testing.T) {
	t.Parallel()
	examples := Dataset{
		{
			[]float64{0, 0},
			[]float64{1, 0},
		},
		{
			[]float64{0, 1},
			[]float64{1, 0},
		},
		{
			[]float64{1, 0},
			[]float64{1, 0},
		},
		{
			[]float64{1, 1},
			[]float64{0, 1},
		},
	}
	s := NewMultiSwarm(basicMathConfig(), DefaultTrainingConfig)
	s.Train(examples)
	accuracy := s.ClassificationAccuracy(examples...)
	assert.Equal(t, 1.0, accuracy)
}

func Test_NOR(t *testing.T) {
	t.Parallel()
	examples := Dataset{
		{
			Inputs:  []float64{0, 0},
			Outputs: []float64{1, 0},
		},
		{
			Inputs:  []float64{0, 1},
			Outputs: []float64{0, 1},
		},
		{
			Inputs:  []float64{1, 0},
			Outputs: []float64{0, 1},
		},
		{
			Inputs:  []float64{1, 1},
			Outputs: []float64{0, 1},
		},
	}
	s := NewMultiSwarm(basicMathConfig(), DefaultTrainingConfig)

	s.Train(examples)
	accuracy := s.ClassificationAccuracy(examples...)
	assert.Equal(t, 1.0, accuracy)
}

func Test_XNOR(t *testing.T) {
	t.Parallel()
	examples := Dataset{
		{
			Inputs:  []float64{0, 0},
			Outputs: []float64{1, 0},
		},
		{
			Inputs:  []float64{0, 1},
			Outputs: []float64{0, 1},
		},
		{
			Inputs:  []float64{1, 0},
			Outputs: []float64{0, 1},
		},
		{
			Inputs:  []float64{1, 1},
			Outputs: []float64{1, 0},
		},
	}
	s := NewMultiSwarm(basicMathConfig(), DefaultTrainingConfig)
	s.Train(examples)
	accuracy := s.ClassificationAccuracy(examples...)
	assert.Equal(t, 1.0, accuracy)
}

func Test_Flowers(t *testing.T) {
	data := Dataset{
		{[]float64{6.3, 2.9, 5.6, 1.8}, []float64{1, 0, 0}},
		{[]float64{6.9, 3.1, 4.9, 1.5}, []float64{0, 1, 0}},
		{[]float64{4.6, 3.4, 1.4, 0.3}, []float64{0, 0, 1}},
		{[]float64{7.2, 3.6, 6.1, 2.5}, []float64{1, 0, 0}},
		{[]float64{4.7, 3.2, 1.3, 0.2}, []float64{0, 0, 1}},
		{[]float64{4.9, 3, 1.4, 0.2}, []float64{0, 0, 1}},
		{[]float64{7.6, 3, 6.6, 2.1}, []float64{1, 0, 0}},
		{[]float64{4.9, 2.4, 3.3, 1}, []float64{0, 1, 0}},
		{[]float64{5.4, 3.9, 1.7, 0.4}, []float64{0, 0, 1}},
		{[]float64{4.9, 3.1, 1.5, 0.1}, []float64{0, 0, 1}},
		{[]float64{5, 3.6, 1.4, 0.2}, []float64{0, 0, 1}},
		{[]float64{6.4, 3.2, 4.5, 1.5}, []float64{0, 1, 0}},
		{[]float64{4.4, 2.9, 1.4, 0.2}, []float64{0, 0, 1}},
		{[]float64{5.8, 2.7, 5.1, 1.9}, []float64{1, 0, 0}},
		{[]float64{6.3, 3.3, 6, 2.5}, []float64{1, 0, 0}},
		{[]float64{5.2, 2.7, 3.9, 1.4}, []float64{0, 1, 0}},
		{[]float64{7, 3.2, 4.7, 1.4}, []float64{0, 1, 0}},
		{[]float64{6.5, 2.8, 4.6, 1.5}, []float64{0, 1, 0}},
		{[]float64{4.9, 2.5, 4.5, 1.7}, []float64{1, 0, 0}},
		{[]float64{5.7, 2.8, 4.5, 1.3}, []float64{0, 1, 0}},
		{[]float64{5, 3.4, 1.5, 0.2}, []float64{0, 0, 1}},
		{[]float64{6.5, 3, 5.8, 2.2}, []float64{1, 0, 0}},
		{[]float64{5.5, 2.3, 4, 1.3}, []float64{0, 1, 0}},
		{[]float64{6.7, 2.5, 5.8, 1.8}, []float64{1, 0, 0}},
		{[]float64{4.6, 3.1, 1.5, 0.2}, []float64{0, 0, 1}},
		{[]float64{7.1, 3, 5.9, 2.1}, []float64{1, 0, 0}},
		{[]float64{5.1, 3.5, 1.4, 0.2}, []float64{0, 0, 1}},
		{[]float64{6.3, 3.3, 4.7, 1.6}, []float64{0, 1, 0}},
		{[]float64{6.6, 2.9, 4.6, 1.3}, []float64{0, 1, 0}},
		{[]float64{7.3, 2.9, 6.3, 1.8}, []float64{1, 0, 0}},
	}

	config := MultiSwarmConfiguration{
		SwarmCount: 5,
		NeuralNetworkConfiguration: NeuralNetworkConfiguration{
			Loss:       Cross,
			InputCount: 4,
			LayerConfigs: []LayerConfig{
				{
					NodeCount:  10,
					Activation: ReLU,
				},
				{
					NodeCount:  10,
					Activation: ReLU,
				},
				{
					NodeCount:  10,
					Activation: ReLU,
				},
				{
					NodeCount:  10,
					Activation: ReLU,
				},
				{
					NodeCount:  10,
					Activation: ReLU,
				},
				{
					NodeCount:  3,
					Activation: Softmax,
				},
			},
		},
		ParticleCount: 8,
	}
	s := NewMultiSwarm(config, DefaultTrainingConfig)
	s.Train(data)
	accuracy := s.ClassificationAccuracy(data...)
	assert.Equal(t, 1.0, accuracy)
}

// func Test_Rastrigin(t *testing.T) {
// 	const tau = math.Pi * 2
// 	rand.Seed(1)
// 	samples := 100
// 	positiveRange := 5.12
// 	trainData := make([]Data, samples)
// 	for i := range trainData {
// 		x := (2 * positiveRange * rand.Float64()) - positiveRange
// 		y := (2 * positiveRange * rand.Float64()) - positiveRange
// 		z := y*y + x*x - 10*(math.Cos(tau*x)+math.Cos(tau*y)) + 20
// 		data := Data{[]float64{x, y}, []float64{z}}
// 		trainData[i] = data
// 	}
// 	//Put the real global minima for sure
// 	trainData[0] = Data{Inputs: []float64{0, 0}, Outputs: []float64{0}}

// 	config := MultiSwarmConfiguration{
// 		SwarmCount: 5,
// 		NeuralNetworkConfiguration: NeuralNetworkConfiguration{
// 			LossType:   ItakuraSaitoDistanceLoss,
// 			InputCount: 2,
// 			LayerConfigs: []LayerConfig{
// 				LayerConfig{
// 					NodeCount:      10,
// 					ActivationName: Sigmoid,
// 				},
// 				LayerConfig{
// 					NodeCount:      10,
// 					ActivationName: Sigmoid,
// 				},
// 				LayerConfig{
// 					NodeCount:      10,
// 					ActivationName: HyperbolicTangent,
// 				},
// 				LayerConfig{
// 					NodeCount:      10,
// 					ActivationName: HyperbolicTangent,
// 				},
// 				LayerConfig{
// 					NodeCount:      10,
// 					ActivationName: HyperbolicTangent,
// 				},
// 				LayerConfig{
// 					NodeCount:      5,
// 					ActivationName: LeakyReLU,
// 				},
// 				LayerConfig{
// 					NodeCount:      3,
// 					ActivationName: LeakyReLU,
// 				},
// 				LayerConfig{
// 					NodeCount:      1,
// 					ActivationName: LeakyReLU,
// 				},
// 			},
// 		},
// 		ParticleCount: 1,
// 	}
// 	s := NewMultiSwarm(&config)
// 	tries := s.Train(trainData)

// 	actual := s.Predict(0, 0)
// 	assert.Equal(t, -24.344032236124725, actual[0])
// 	assert.Equal(t, 3500, tries)
// }

// func Test_Error(t *testing.T) {
// 	rand.Seed(1)
// 	examples := [][]Data{
// 		{
// 			{[]float64{0.25, 0.75}, []float64{0.5, 0.5}},
// 			{[]float64{0.75, 0.25}, []float64{0.5, 0.5}},
// 			{[]float64{-0.5, 0.5}, []float64{0.5, 0.5}},
// 			{[]float64{-0.5, 0.5}, []float64{0.5, 0.5}},
// 			{[]float64{0.5, 0.5}, []float64{0.5, 0.5}},
// 		},
// 		{
// 			{[]float64{0.45, 0.55}, []float64{0.5, 0.5}},
// 			{[]float64{0.55, 0.45}, []float64{0.5, 0.5}},
// 			{[]float64{0.525, 0.525}, []float64{0.5, 0.5}},
// 			{[]float64{0.475, 0.475}, []float64{0.5, 0.5}},
// 			{[]float64{0.5, 0.5}, []float64{0.5, 0.5}},
// 		},
// 	}
// 	for _, tt := range []struct {
// 		lt       LossType
// 		expected []float64
// 	}{
// 		{SquaredLoss, []float64{
// 			0.5, 0.5,
// 		}},
// 		{CrossLoss, []float64{
// 			6.656104149082171, 9.299352568611798,
// 		}},
// 		{ExponentialLoss, []float64{
// 			1.6485192252482843, 1.6487212484141207,
// 		}},
// 		{HellingerDistanceLoss, []float64{
// 			0.48645931737868475, 0.45235979244565366,
// 		}},
// 		{KullbackLeiblerDivergenceLoss, []float64{
// 			24.321514050007988, 18.123348049136933,
// 		}},
// 		{GeneralizedKullbackLeiblerDivergenceLoss, []float64{
// 			53.42095928253063, 65.44891067419549,
// 		}},
// 		{ItakuraSaitoDistanceLoss, []float64{
// 			1.5654031833143122e+06, 1487.9011553625498,
// 		}},
// 	} {
// 		nn := newNeuralNetwork(&NeuralNetworkConfiguration{
// 			LossType:   tt.lt,
// 			InputCount: 2,
// 			LayerConfigs: []LayerConfig{
// 				{5, ReLU},
// 				{2, Softmax},
// 			},
// 		})
// 		for i, e := range examples {
// 			expected := tt.expected[i]
// 			actual := nn.calculateMeanLoss(e)
// 			assert.Equal(t, expected, actual, string(tt.lt))
// 		}
// 	}
// }
