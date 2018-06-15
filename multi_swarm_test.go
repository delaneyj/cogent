package cogent

import (
	"log"
	"math/rand"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func basicMathConfig(data Data) MultiSwarmConfiguration {
	msc := MultiSwarmConfiguration{
		NeuralNetworkConfiguration: NeuralNetworkConfiguration{
			Loss:       CrossLoss,
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
	bucket := DataToTensorDataBucket(data, true)
	buckets := DataBucketToBuckets(4, bucket)
	s := NewMultiSwarm(basicMathConfig(data), DefaultTrainingConfig)
	s.Train(buckets, false)
	accuracy := s.ClassificationAccuracy(buckets)
	assert.Equal(tt, 1.0, accuracy)
}

func Test_XOR(tt *testing.T) {
	basicMathTest(tt, Data{
		{Inputs: []float32{0, 0}, Outputs: []float32{0, 1}},
		{Inputs: []float32{0, 1}, Outputs: []float32{1, 0}},
		{Inputs: []float32{1, 0}, Outputs: []float32{1, 0}},
		{Inputs: []float32{1, 1}, Outputs: []float32{0, 1}},
	})
}

func Test_AND(tt *testing.T) {
	basicMathTest(tt, Data{
		{Inputs: []float32{0, 0}, Outputs: []float32{0, 1}},
		{Inputs: []float32{0, 1}, Outputs: []float32{0, 1}},
		{Inputs: []float32{1, 0}, Outputs: []float32{0, 1}},
		{Inputs: []float32{1, 1}, Outputs: []float32{1, 0}},
	})
}

func Test_NOT(tt *testing.T) {
	basicMathTest(tt, Data{
		{Inputs: []float32{0, 0}, Outputs: []float32{1, 0}},
		{Inputs: []float32{0, 1}, Outputs: []float32{0, 1}},
		{Inputs: []float32{1, 0}, Outputs: []float32{0, 1}},
		{Inputs: []float32{1, 1}, Outputs: []float32{0, 1}},
	})
}

func Test_OR(tt *testing.T) {
	basicMathTest(tt, Data{
		{Inputs: []float32{0, 0}, Outputs: []float32{0, 1}},
		{Inputs: []float32{0, 1}, Outputs: []float32{1, 0}},
		{Inputs: []float32{1, 0}, Outputs: []float32{1, 0}},
		{Inputs: []float32{1, 1}, Outputs: []float32{0, 0}},
	})
}

func Test_NAND(tt *testing.T) {
	basicMathTest(tt, Data{
		{Inputs: []float32{0, 0}, Outputs: []float32{1, 0}},
		{Inputs: []float32{0, 1}, Outputs: []float32{1, 0}},
		{Inputs: []float32{1, 0}, Outputs: []float32{1, 0}},
		{Inputs: []float32{1, 1}, Outputs: []float32{0, 1}},
	})
}

func Test_NOR(tt *testing.T) {
	basicMathTest(tt, Data{
		{Inputs: []float32{0, 0}, Outputs: []float32{1, 0}},
		{Inputs: []float32{0, 1}, Outputs: []float32{0, 1}},
		{Inputs: []float32{1, 0}, Outputs: []float32{0, 1}},
		{Inputs: []float32{1, 1}, Outputs: []float32{0, 1}},
	})
}

func Test_XNOR(tt *testing.T) {
	basicMathTest(tt, Data{
		{Inputs: []float32{0, 0}, Outputs: []float32{1, 0}},
		{Inputs: []float32{0, 1}, Outputs: []float32{0, 1}},
		{Inputs: []float32{1, 0}, Outputs: []float32{0, 1}},
		{Inputs: []float32{1, 1}, Outputs: []float32{1, 0}},
	})
}

func Test_Flowers(tt *testing.T) {
	log.SetFlags(log.LstdFlags | log.Lshortfile)

	var buckets DataBuckets
	var inputCount, outputCount int
	{
		data := Data{
			{Inputs: []float32{6.3, 2.9, 5.6, 1.8}, Outputs: []float32{1, 0, 0}},
			{Inputs: []float32{6.9, 3.1, 4.9, 1.5}, Outputs: []float32{0, 1, 0}},
			{Inputs: []float32{4.6, 3.4, 1.4, 0.3}, Outputs: []float32{0, 0, 1}},
			{Inputs: []float32{7.2, 3.6, 6.1, 2.5}, Outputs: []float32{1, 0, 0}},
			{Inputs: []float32{4.7, 3.2, 1.3, 0.2}, Outputs: []float32{0, 0, 1}},
			{Inputs: []float32{4.9, 3, 1.4, 0.2}, Outputs: []float32{0, 0, 1}},
			{Inputs: []float32{7.6, 3, 6.6, 2.1}, Outputs: []float32{1, 0, 0}},
			{Inputs: []float32{4.9, 2.4, 3.3, 1}, Outputs: []float32{0, 1, 0}},
			{Inputs: []float32{5.4, 3.9, 1.7, 0.4}, Outputs: []float32{0, 0, 1}},
			{Inputs: []float32{4.9, 3.1, 1.5, 0.1}, Outputs: []float32{0, 0, 1}},
			{Inputs: []float32{5, 3.6, 1.4, 0.2}, Outputs: []float32{0, 0, 1}},
			{Inputs: []float32{6.4, 3.2, 4.5, 1.5}, Outputs: []float32{0, 1, 0}},
			{Inputs: []float32{4.4, 2.9, 1.4, 0.2}, Outputs: []float32{0, 0, 1}},
			{Inputs: []float32{5.8, 2.7, 5.1, 1.9}, Outputs: []float32{1, 0, 0}},
			{Inputs: []float32{6.3, 3.3, 6, 2.5}, Outputs: []float32{1, 0, 0}},
			{Inputs: []float32{5.2, 2.7, 3.9, 1.4}, Outputs: []float32{0, 1, 0}},
			{Inputs: []float32{7, 3.2, 4.7, 1.4}, Outputs: []float32{0, 1, 0}},
			{Inputs: []float32{6.5, 2.8, 4.6, 1.5}, Outputs: []float32{0, 1, 0}},
			{Inputs: []float32{4.9, 2.5, 4.5, 1.7}, Outputs: []float32{1, 0, 0}},
			{Inputs: []float32{5.7, 2.8, 4.5, 1.3}, Outputs: []float32{0, 1, 0}},
			{Inputs: []float32{5, 3.4, 1.5, 0.2}, Outputs: []float32{0, 0, 1}},
			{Inputs: []float32{6.5, 3, 5.8, 2.2}, Outputs: []float32{1, 0, 0}},
			{Inputs: []float32{5.5, 2.3, 4, 1.3}, Outputs: []float32{0, 1, 0}},
			{Inputs: []float32{6.7, 2.5, 5.8, 1.8}, Outputs: []float32{1, 0, 0}},
			{Inputs: []float32{4.6, 3.1, 1.5, 0.2}, Outputs: []float32{0, 0, 1}},
			{Inputs: []float32{7.1, 3, 5.9, 2.1}, Outputs: []float32{1, 0, 0}},
			{Inputs: []float32{5.1, 3.5, 1.4, 0.2}, Outputs: []float32{0, 0, 1}},
			{Inputs: []float32{6.3, 3.3, 4.7, 1.6}, Outputs: []float32{0, 1, 0}},
			{Inputs: []float32{6.6, 2.9, 4.6, 1.3}, Outputs: []float32{0, 1, 0}},
			{Inputs: []float32{7.3, 2.9, 6.3, 1.8}, Outputs: []float32{1, 0, 0}},
		}
		inputCount = len(data[0].Inputs)
		outputCount = len(data[0].Outputs)
		bucket := DataToTensorDataBucket(data, true)
		buckets = DataBucketToBuckets(10, bucket)
	}

	config := MultiSwarmConfiguration{
		NeuralNetworkConfiguration: NeuralNetworkConfiguration{
			Loss:       CrossLoss,
			InputCount: inputCount,
			LayerConfigs: []LayerConfig{
				{
					NodeCount:  16,
					Activation: LeakyReLU,
				},
				{
					NodeCount:  outputCount,
					Activation: Softmax,
				},
			},
		},
		ParticleCount: 8,
		SwarmCount:    8,
	}
	tc := DefaultTrainingConfig
	tc.RidgeRegressionWeight = 0.05
	tc.MaxIterations = 5000
	tc.WeightRange = 10
	tc.ProbablityOfDeath = 0.0001

	s := NewMultiSwarm(config, tc)

	start := time.Now()
	s.Train(buckets, true)
	accuracy := s.ClassificationAccuracy(buckets)
	assert.Equal(tt, 1.0, accuracy)
	log.Print(time.Since(start))
}

func Test_Error(t *testing.T) {
	log.SetFlags(log.LstdFlags | log.Lshortfile)
	rand.Seed(1)
	type sample struct {
		predictedRow [][]float32
		expectedRow  [][]float32
	}

	samples := []sample{
		{
			predictedRow: [][]float32{
				{0.25, 0.75},
				{0.75, 0.25},
				{-0.5, 0.5},
				{-0.5, 0.5},
				{0.5, 0.5},
			},
			expectedRow: [][]float32{
				{0.5, 0.5},
				{0.5, 0.5},
				{0.5, 0.5},
				{0.5, 0.5},
				{0.5, 0.5},
				{0.5, 0.5},
			},
		},
		{
			predictedRow: [][]float32{
				{0.45, 0.55},
				{0.55, 0.45},
				{0.525, 0.525},
				{0.475, 0.475},
				{0.5, 0.5},
			},
			expectedRow: [][]float32{
				{0.5, 0.5},
				{0.5, 0.5},
				{0.5, 0.5},
				{0.5, 0.5},
				{0.5, 0.5},
				{0.5, 0.5},
			},
		},
	}
	for _, tt := range []struct {
		lm   LossMode
		loss []float32
	}{
		{SquaredLoss, []float32{
			0.225, 0.0012500000000000007,
		}},
		{CrossLoss, []float32{
			0.6120541589383125, 0.6956578737742694,
		}},
		{ExponentialLoss, []float32{
			1.2523227161918644, 1.0012507815756226,
		}},
		{HellingerDistanceLoss, []float32{
			0.15075123182539685, 0.011195253820219403,
		}},
		{KullbackLeiblerDivergenceLoss, []float32{
			0.057536414490356166, 0.0025106932143239766,
		}},
		{GeneralizedKullbackLeiblerDivergenceLoss, []float32{
			0.05753641449035616, 0.002510693214324,
		}},
		{ItakuraSaitoDistanceLoss, []float32{
			0.7476321198163529, 0.020387987815337284,
		}},
	} {
		lf := LossFns[tt.lm]

		for i, sample := range samples {
			expectedLoss := tt.loss[i]
			actualLoss := lf(sample.expectedRow, sample.predictedRow)

			assert.Equal(t, expectedLoss, actualLoss, string(tt.lm))
		}
	}
}
