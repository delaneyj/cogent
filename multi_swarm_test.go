package cogent

import (
	"log"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	t "gorgonia.org/tensor"
)

func basicMathConfig(data Data) MultiSwarmConfiguration {
	msc := MultiSwarmConfiguration{
		NeuralNetworkConfiguration: NeuralNetworkConfiguration{
			Loss:       Cross,
			InputCount: len(data[0].Inputs),
			LayerConfigs: []LayerConfig{
				LayerConfig{
					NodeCount:  6,
					Activation: ReLU,
				},
				LayerConfig{
					NodeCount:  len(data[0].Outputs),
					Activation: Softmax,
				},
			},
		},
		ParticleCount: 4,
		SwarmCount:    2,
	}

	return msc
}

func basicMathTest(tt *testing.T, data Data) {
	// tt.Parallel()
	dataset := DataToTensorDataset(data)
	s := NewMultiSwarm(basicMathConfig(data), DefaultTrainingConfig)
	s.Train(dataset, false)
	accuracy := s.ClassificationAccuracy(dataset)
	assert.Equal(tt, 1.0, accuracy)
}

func Test_XOR(tt *testing.T) {
	basicMathTest(tt, Data{
		{Inputs: []float64{0, 0}, Outputs: []float64{0, 1}},
		{Inputs: []float64{0, 1}, Outputs: []float64{1, 0}},
		{Inputs: []float64{1, 0}, Outputs: []float64{1, 0}},
		{Inputs: []float64{1, 1}, Outputs: []float64{0, 1}},
	})
}

func Test_AND(tt *testing.T) {
	basicMathTest(tt, Data{
		{Inputs: []float64{0, 0}, Outputs: []float64{0, 1}},
		{Inputs: []float64{0, 1}, Outputs: []float64{0, 1}},
		{Inputs: []float64{1, 0}, Outputs: []float64{0, 1}},
		{Inputs: []float64{1, 1}, Outputs: []float64{1, 0}},
	})
}

func Test_NOT(tt *testing.T) {
	basicMathTest(tt, Data{
		{Inputs: []float64{0, 0}, Outputs: []float64{1, 0}},
		{Inputs: []float64{0, 1}, Outputs: []float64{0, 1}},
		{Inputs: []float64{1, 0}, Outputs: []float64{0, 1}},
		{Inputs: []float64{1, 1}, Outputs: []float64{0, 1}},
	})
}

func Test_OR(tt *testing.T) {
	basicMathTest(tt, Data{
		{Inputs: []float64{0, 0}, Outputs: []float64{0, 1}},
		{Inputs: []float64{0, 1}, Outputs: []float64{1, 0}},
		{Inputs: []float64{1, 0}, Outputs: []float64{1, 0}},
		{Inputs: []float64{1, 1}, Outputs: []float64{0, 0}},
	})
}

func Test_NAND(tt *testing.T) {
	basicMathTest(tt, Data{
		{Inputs: []float64{0, 0}, Outputs: []float64{1, 0}},
		{Inputs: []float64{0, 1}, Outputs: []float64{1, 0}},
		{Inputs: []float64{1, 0}, Outputs: []float64{1, 0}},
		{Inputs: []float64{1, 1}, Outputs: []float64{0, 1}},
	})
}

func Test_NOR(tt *testing.T) {
	basicMathTest(tt, Data{
		{Inputs: []float64{0, 0}, Outputs: []float64{1, 0}},
		{Inputs: []float64{0, 1}, Outputs: []float64{0, 1}},
		{Inputs: []float64{1, 0}, Outputs: []float64{0, 1}},
		{Inputs: []float64{1, 1}, Outputs: []float64{0, 1}},
	})
}

func Test_XNOR(tt *testing.T) {
	basicMathTest(tt, Data{
		{Inputs: []float64{0, 0}, Outputs: []float64{1, 0}},
		{Inputs: []float64{0, 1}, Outputs: []float64{0, 1}},
		{Inputs: []float64{1, 0}, Outputs: []float64{0, 1}},
		{Inputs: []float64{1, 1}, Outputs: []float64{1, 0}},
	})
}

func Test_Flowers(tt *testing.T) {
	log.SetFlags(log.LstdFlags | log.Lshortfile)
	data := Data{
		{Inputs: []float64{6.3, 2.9, 5.6, 1.8}, Outputs: []float64{1, 0, 0}},
		{Inputs: []float64{6.9, 3.1, 4.9, 1.5}, Outputs: []float64{0, 1, 0}},
		{Inputs: []float64{4.6, 3.4, 1.4, 0.3}, Outputs: []float64{0, 0, 1}},
		{Inputs: []float64{7.2, 3.6, 6.1, 2.5}, Outputs: []float64{1, 0, 0}},
		{Inputs: []float64{4.7, 3.2, 1.3, 0.2}, Outputs: []float64{0, 0, 1}},
		{Inputs: []float64{4.9, 3, 1.4, 0.2}, Outputs: []float64{0, 0, 1}},
		{Inputs: []float64{7.6, 3, 6.6, 2.1}, Outputs: []float64{1, 0, 0}},
		{Inputs: []float64{4.9, 2.4, 3.3, 1}, Outputs: []float64{0, 1, 0}},
		{Inputs: []float64{5.4, 3.9, 1.7, 0.4}, Outputs: []float64{0, 0, 1}},
		{Inputs: []float64{4.9, 3.1, 1.5, 0.1}, Outputs: []float64{0, 0, 1}},
		{Inputs: []float64{5, 3.6, 1.4, 0.2}, Outputs: []float64{0, 0, 1}},
		{Inputs: []float64{6.4, 3.2, 4.5, 1.5}, Outputs: []float64{0, 1, 0}},
		{Inputs: []float64{4.4, 2.9, 1.4, 0.2}, Outputs: []float64{0, 0, 1}},
		{Inputs: []float64{5.8, 2.7, 5.1, 1.9}, Outputs: []float64{1, 0, 0}},
		{Inputs: []float64{6.3, 3.3, 6, 2.5}, Outputs: []float64{1, 0, 0}},
		{Inputs: []float64{5.2, 2.7, 3.9, 1.4}, Outputs: []float64{0, 1, 0}},
		{Inputs: []float64{7, 3.2, 4.7, 1.4}, Outputs: []float64{0, 1, 0}},
		{Inputs: []float64{6.5, 2.8, 4.6, 1.5}, Outputs: []float64{0, 1, 0}},
		{Inputs: []float64{4.9, 2.5, 4.5, 1.7}, Outputs: []float64{1, 0, 0}},
		{Inputs: []float64{5.7, 2.8, 4.5, 1.3}, Outputs: []float64{0, 1, 0}},
		{Inputs: []float64{5, 3.4, 1.5, 0.2}, Outputs: []float64{0, 0, 1}},
		{Inputs: []float64{6.5, 3, 5.8, 2.2}, Outputs: []float64{1, 0, 0}},
		{Inputs: []float64{5.5, 2.3, 4, 1.3}, Outputs: []float64{0, 1, 0}},
		{Inputs: []float64{6.7, 2.5, 5.8, 1.8}, Outputs: []float64{1, 0, 0}},
		{Inputs: []float64{4.6, 3.1, 1.5, 0.2}, Outputs: []float64{0, 0, 1}},
		{Inputs: []float64{7.1, 3, 5.9, 2.1}, Outputs: []float64{1, 0, 0}},
		{Inputs: []float64{5.1, 3.5, 1.4, 0.2}, Outputs: []float64{0, 0, 1}},
		{Inputs: []float64{6.3, 3.3, 4.7, 1.6}, Outputs: []float64{0, 1, 0}},
		{Inputs: []float64{6.6, 2.9, 4.6, 1.3}, Outputs: []float64{0, 1, 0}},
		{Inputs: []float64{7.3, 2.9, 6.3, 1.8}, Outputs: []float64{1, 0, 0}},
	}

	config := MultiSwarmConfiguration{
		NeuralNetworkConfiguration: NeuralNetworkConfiguration{
			Loss:       Cross,
			InputCount: len(data[0].Inputs),
			LayerConfigs: []LayerConfig{
				{
					NodeCount:  16,
					Activation: LeakyReLU,
				},
				{
					NodeCount:  len(data[0].Outputs),
					Activation: Softmax,
				},
			},
		},
		ParticleCount: 4,
		SwarmCount:    2,
	}
	tc := DefaultTrainingConfig
	tc.MaxIterations = 2000
	tc.WeightRange = 10

	s := NewMultiSwarm(config, tc)

	start := time.Now()
	dataset := DataToTensorDataset(data)
	s.Train(dataset, true)
	accuracy := s.ClassificationAccuracy(dataset)
	assert.Equal(tt, 1.0, accuracy)
	log.Print(time.Since(start))
}

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

type Data []struct {
	Inputs  []float64
	Outputs []float64
}

func DataToTensorDataset(data Data) *Dataset {
	rows := len(data)
	iColCount := len(data[0].Inputs)
	oColCount := len(data[0].Outputs)
	dataset := &Dataset{
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

	return dataset
}
