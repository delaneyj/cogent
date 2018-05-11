package cogent

import (
	"math/rand"

	"testing"

	"github.com/stretchr/testify/assert"
)

func basicMathConfig() *SwarmConfiguration {
	return &SwarmConfiguration{
		NeuralNetworkConfiguration: NeuralNetworkConfiguration{
			InputCount: 2,
			LayerConfigs: []LayerConfig{
				LayerConfig{
					NodeCount:      3,
					ActivationName: HyperbolicTangent,
				},
				LayerConfig{
					NodeCount:      2,
					ActivationName: Softmax,
				},
			},
		},
		ParticleCount: 8,
	}
}

func Test_XOR(t *testing.T) {
	rand.Seed(1)
	trainData := []Data{
		Data{
			Inputs:  []float64{0, 0},
			Outputs: []float64{0, 1},
		},
		Data{
			Inputs:  []float64{0, 1},
			Outputs: []float64{1, 0},
		},
		Data{
			Inputs:  []float64{1, 0},
			Outputs: []float64{1, 0},
		},
		Data{
			Inputs:  []float64{1, 1},
			Outputs: []float64{0, 1},
		},
	}

	s := NewSwarm(basicMathConfig())
	tries := s.Train(trainData)
	accuracy := s.ClassificationAccuracy(trainData)
	assert.Equal(t, accuracy, 1.0)
	assert.Equal(t, tries, 3096)
}

func Test_AND(t *testing.T) {
	rand.Seed(0)
	trainData := []Data{
		Data{
			Inputs:  []float64{0, 0},
			Outputs: []float64{0, 1},
		},
		Data{
			Inputs:  []float64{0, 1},
			Outputs: []float64{0, 1},
		},
		Data{
			Inputs:  []float64{1, 0},
			Outputs: []float64{0, 1},
		},
		Data{
			Inputs:  []float64{1, 1},
			Outputs: []float64{1, 0},
		},
	}

	s := NewSwarm(basicMathConfig())
	tries := s.Train(trainData)
	accuracy := s.ClassificationAccuracy(trainData)
	assert.Equal(t, accuracy, 1.0, "and")
	assert.Equal(t, tries, 160, "and")
}

func Test_NOT(t *testing.T) {
	rand.Seed(1)
	trainData := []Data{
		Data{
			Inputs:  []float64{0, 0},
			Outputs: []float64{1, 0},
		},
		Data{
			Inputs:  []float64{0, 1},
			Outputs: []float64{0, 1},
		},
		Data{
			Inputs:  []float64{1, 0},
			Outputs: []float64{0, 1},
		},
		Data{
			Inputs:  []float64{1, 1},
			Outputs: []float64{0, 1},
		},
	}

	s := NewSwarm(basicMathConfig())
	tries := s.Train(trainData)
	accuracy := s.ClassificationAccuracy(trainData)
	assert.Equal(t, accuracy, 1.0, "not")
	assert.Equal(t, tries, 56, "not")
}

func Test_OR(t *testing.T) {
	rand.Seed(1)
	trainData := []Data{
		Data{
			Inputs:  []float64{0, 0},
			Outputs: []float64{0, 1},
		},
		Data{
			Inputs:  []float64{0, 1},
			Outputs: []float64{1, 0},
		},
		Data{
			Inputs:  []float64{1, 0},
			Outputs: []float64{1, 0},
		},
		Data{
			Inputs:  []float64{1, 1},
			Outputs: []float64{1, 0},
		},
	}

	s := NewSwarm(basicMathConfig())
	tries := s.Train(trainData)
	accuracy := s.ClassificationAccuracy(trainData)
	assert.Equal(t, accuracy, 1.0)
	assert.Equal(t, tries, 16)
}

func Test_NAND(t *testing.T) {
	rand.Seed(1)
	trainData := []Data{
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
	s := NewSwarm(basicMathConfig())
	tries := s.Train(trainData)
	accuracy := s.ClassificationAccuracy(trainData)
	assert.Equal(t, accuracy, 1.0, "NAND")
	assert.Equal(t, tries, 64, "NAND")
}

func Test_NOR(t *testing.T) {
	rand.Seed(1)
	trainData := []Data{
		Data{
			Inputs:  []float64{0, 0},
			Outputs: []float64{1, 0},
		},
		Data{
			Inputs:  []float64{0, 1},
			Outputs: []float64{0, 1},
		},
		Data{
			Inputs:  []float64{1, 0},
			Outputs: []float64{0, 1},
		},
		Data{
			Inputs:  []float64{1, 1},
			Outputs: []float64{0, 1},
		},
	}
	s := NewSwarm(basicMathConfig())
	tries := s.Train(trainData)
	accuracy := s.ClassificationAccuracy(trainData)
	assert.Equal(t, accuracy, 1.0, "NOR")
	assert.Equal(t, tries, 56, "NOR")
}

func Test_XNOR(t *testing.T) {
	rand.Seed(1)
	trainData := []Data{
		Data{
			Inputs:  []float64{0, 0},
			Outputs: []float64{1, 0},
		},
		Data{
			Inputs:  []float64{0, 1},
			Outputs: []float64{0, 1},
		},
		Data{
			Inputs:  []float64{1, 0},
			Outputs: []float64{0, 1},
		},
		Data{
			Inputs:  []float64{1, 1},
			Outputs: []float64{1, 0},
		},
	}
	s := NewSwarm(basicMathConfig())
	tries := s.Train(trainData)
	accuracy := s.ClassificationAccuracy(trainData)
	assert.Equal(t, accuracy, 1.0, "XNOR")
	assert.Equal(t, tries, 232, "XNOR")
}

func Test_Flowers(t *testing.T) {
	rand.Seed(1)
	trainData := []Data{
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
	}

	testData := []Data{
		{[]float64{4.6, 3.1, 1.5, 0.2}, []float64{0, 0, 1}},
		{[]float64{7.1, 3, 5.9, 2.1}, []float64{1, 0, 0}},
		{[]float64{5.1, 3.5, 1.4, 0.2}, []float64{0, 0, 1}},
		{[]float64{6.3, 3.3, 4.7, 1.6}, []float64{0, 1, 0}},
		{[]float64{6.6, 2.9, 4.6, 1.3}, []float64{0, 1, 0}},
		{[]float64{7.3, 2.9, 6.3, 1.8}, []float64{1, 0, 0}},
	}

	config := SwarmConfiguration{
		NeuralNetworkConfiguration: NeuralNetworkConfiguration{
			InputCount: 4,
			LayerConfigs: []LayerConfig{
				LayerConfig{
					NodeCount:      10,
					ActivationName: HyperbolicTangent,
				},
				LayerConfig{
					NodeCount:      3,
					ActivationName: Softmax,
				},
			},
		},
		ParticleCount: 8,
	}
	s := NewSwarm(&config)
	tries := s.Train(trainData)

	accuracy := s.ClassificationAccuracy(testData)
	assert.Equal(t, accuracy, 1.0)
	assert.Equal(t, tries, 560)
}

func Test_Error(t *testing.T) {
	rand.Seed(1)
	nn := newNeuralNetwork(&NeuralNetworkConfiguration{
		InputCount: 2,
		LayerConfigs: []LayerConfig{
			{5, ReLU},
			{2, Softmax},
		},
	})
	examples := []Data{
		{[]float64{0.25, 0.75}, []float64{0.5, 0, 5}},
	}
	for _, tt := range []struct {
		fn       func(data []Data) float64
		expected float64
	}{
		{
			fn:       nn.meanSquaredError,
			expected: 1.25,
		},
		{
			fn:       nn.meanCrossEntropyError,
			expected: 26.479421371046712,
		},
	} {
		actual := tt.fn(examples)
		assert.Equal(t, tt.expected, actual)
	}
}

func Test_internals(t *testing.T) {
	rand.Seed(1)
	nn := newNeuralNetwork(&NeuralNetworkConfiguration{
		InputCount: 2,
		LayerConfigs: []LayerConfig{
			{2, ReLU},
			{2, Softmax},
		},
	})

	v := []float64{
		0.8810181760900249,
		-0.12457162562603963,
		0.37364614573421884,
		-0.6869614905344175,
		-0.39817627882942586,
		0.6272799219801937,
		-0.238685621400628,
		-0.062220310195153616,
		-0.4137962853263685,
		-0.5628938948144715,
		-0.278257166286188,
		0.7249828748957727,
	}
	assert.NotPanics(t, func() {
		nn.setVelocities(v)
	})
	assert.Equal(t, v, nn.velocities())
	assert.Equal(t, `{[2.093205759592392 3.291201064369808 -1.5072500585746855 -8.687259615650476 -8.06060962171031 0.30425257004130835 -5.714722548352501 -3.638836513393403 -4.339316976391096 3.5816935184043253 -5.936262467053543 1.4134655214204521] [0.8810181760900249 -0.12457162562603963 0.37364614573421884 -0.6869614905344175 -0.39817627882942586 0.6272799219801937 -0.238685621400628 -0.062220310195153616 -0.4137962853263685 -0.5628938948144715 -0.278257166286188 0.7249828748957727] 1.7976931348623157e+308}`, nn.String())
}
