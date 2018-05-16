package cogent

import (
	"log"
	"math"
	"math/rand"
)

//Data x
type Data struct {
	Inputs  []float64 `json:"inputs"`
	Outputs []float64 `json:"outputs"`
}

type swarm struct {
	particles []*neuralNetwork
	Best      position
}

//MultiSwarm x
type MultiSwarm struct {
	swarms []*swarm
	best   position
}

//MultiSwarmConfiguration x
type MultiSwarmConfiguration struct {
	NeuralNetworkConfiguration
	SwarmCount    int
	ParticleCount int
}

//NewMultiSwarm x
func NewMultiSwarm(config *MultiSwarmConfiguration) *MultiSwarm {
	if config.SwarmCount <= 0 {
		panic("No swarm count in config")
	}

	ms := MultiSwarm{
		swarms: make([]*swarm, config.SwarmCount),
		best: position{
			FitnessLoss: math.MaxFloat64,
		},
	}
	for i := range ms.swarms {
		s := swarm{
			particles: make([]*neuralNetwork, config.ParticleCount),
			Best: position{
				FitnessLoss: math.MaxFloat64,
			},
		}

		for i := 0; i < config.ParticleCount; i++ {
			s.particles[i] = newNeuralNetwork(&config.NeuralNetworkConfiguration)
		}
		ms.swarms[i] = &s
	}

	return &ms
}

//Train x
func (ms *MultiSwarm) Train(trainingData []Data) int {
	tries := 0

	//set loss for given training set
	for _, s := range ms.swarms {
		for _, p := range s.particles {
			fitnessError := p.checkAndSetloss(trainingData)
			if fitnessError < s.Best.FitnessLoss {
				w := p.weights()
				s.Best.FitnessLoss = fitnessError
				s.Best.WeightsAndBiases = w

				if fitnessError < ms.best.FitnessLoss {
					ms.best.FitnessLoss = fitnessError
					ms.best.WeightsAndBiases = w
				}
			}
		}
	}

	for i := 0; i < maxIterations; i++ {
		for si, s := range ms.swarms {
			// process particles in random order
			sequence := make([]int, len(s.particles))
			for i := range sequence {
				sequence[i] = i
			}

			// log.Printf("Best global error %f.", s.Best.FitnessLoss)
			if s.Best.FitnessLoss <= targetAccuracy {
				return tries
			}

			shuffle(sequence) // move particles in random sequence

			for pi := range s.particles {
				index := sequence[pi]
				currentParticle := s.particles[index]
				mse := currentParticle.train(trainingData, s.Best.WeightsAndBiases, ms.best.WeightsAndBiases)
				tries++

				if mse < s.Best.FitnessLoss {
					w := currentParticle.weights()

					s.Best.FitnessLoss = mse
					s.Best.WeightsAndBiases = w

					if mse < ms.best.FitnessLoss {
						msg := "<%d:%d:%d> New global best found fitness error %9f => %9f on try %d."
						log.Printf(msg, i, si, pi, ms.best.FitnessLoss, mse, tries)

						ms.best.FitnessLoss = mse
						ms.best.WeightsAndBiases = w
					}
				}
			}
		}
	}

	return tries
}

func shuffle(sequence []int) {
	l := len(sequence)
	for i, s := range sequence {
		ri := rand.Intn(l-i) + i
		tmp := sequence[ri]
		sequence[ri] = s
		sequence[i] = tmp
	}
}

//ClassificationAccuracy x
func (ms *MultiSwarm) ClassificationAccuracy(testData []Data) float64 {
	var swarmBest, globalBest *neuralNetwork
	for _, s := range ms.swarms {
		for _, p := range s.particles {
			if swarmBest == nil || p.Best.FitnessLoss < swarmBest.Best.FitnessLoss {
				swarmBest = p

				if globalBest == nil || p.Best.FitnessLoss < globalBest.Best.FitnessLoss {
					globalBest = p
				}
			}
		}
		swarmBest.setWeights(s.Best.WeightsAndBiases)
	}
	globalBest.setWeights(ms.best.WeightsAndBiases)
	return globalBest.classificationAccuracy(testData)
}

//Predict x
func (ms *MultiSwarm) Predict(inputs ...float64) []float64 {
	var globalBest *neuralNetwork
	for _, s := range ms.swarms {
		for _, p := range s.particles {
			if globalBest == nil || p.Best.FitnessLoss < globalBest.Best.FitnessLoss {
				globalBest = p
			}
		}
	}
	return globalBest.activate(inputs...)
}
