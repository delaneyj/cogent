package cogent

import (
	"math/rand"
	"sync"

	"github.com/dgraph-io/badger"
	"github.com/google/uuid"
)

//MultiSwarm x
type MultiSwarm struct {
	particleCount int
	db            *badger.DB
	swarms        []*swarm
}

type swarm struct {
	id        []byte
	particles []*particle
}

//NewMultiSwarm x
func NewMultiSwarm(config *MultiSwarmConfiguration, trainingConfig *TrainingConfiguration) *MultiSwarm {
	opts := badger.DefaultOptions
	opts.Dir = "badger"
	opts.ValueDir = "badger"
	db, err := badger.Open(opts)
	checkErr(err)

	if config.SwarmCount <= 0 {
		panic("No swarm count in config")
	}

	nnConfigBytes, err := config.NeuralNetworkConfiguration.Marshal()
	trainingConfigBytes, err := trainingConfig.Marshal()

	checkErr(db.Update(func(txn *badger.Txn) error {
		checkErr(txn.Set(neuralNetworkConfigPath, nnConfigBytes))
		checkErr(txn.Set(trainingConfigPath, trainingConfigBytes))
		return nil
	}))

	ms := MultiSwarm{
		db:            db,
		swarms:        make([]*swarm, config.SwarmCount),
		particleCount: int(config.SwarmCount * config.ParticleCount),
	}
	for i := range ms.swarms {
		uuid := uuid.New()
		swarmID, err := uuid.MarshalBinary()
		checkErr(err)

		s := &swarm{
			id:        swarmID,
			particles: make([]*particle, config.ParticleCount),
		}

		for i := 0; i < int(config.ParticleCount); i++ {
			p := newParticle(swarmID, db)
			s.particles[i] = p
		}
		ms.swarms[i] = s
	}

	return &ms
}

//Train x
func (ms *MultiSwarm) Train(trainingData []Data, maxIterations int, targetAccuracy float64) int {
	tries := 0
	for i := 0; i < maxIterations; i++ {
		wg := &sync.WaitGroup{}
		wg.Add(ms.particleCount)

		for _, s := range ms.swarms {
			for _, p := range s.particles {
				go p.train(wg)
			}
		}
		wg.Wait()
	}
	return tries
}

// //Train x
// func (ms *MultiSwarm) Train(trainingData []Data, maxIterations int, targetAccuracy float64) int {
// 	tries := 0

// 	//set loss for given training set
// 	for _, s := range ms.Swarms {
// 		for _, p := range s.Particles {
// 			fitnessError := p.checkAndSetloss(trainingData)
// 			if fitnessError < s.Best.Loss {
// 				w := p.weights()
// 				s.Best.Loss = fitnessError
// 				s.Best.WeightsAndBiases = w

// 				if fitnessError < ms.Best.Loss {
// 					ms.Best.Loss = fitnessError
// 					ms.Best.WeightsAndBiases = w
// 				}
// 			}
// 		}
// 	}

// 	for i := 0; i < maxIterations; i++ {
// 		for si, s := range ms.Swarms {
// 			// process Particles in random order
// 			sequence := make([]int, len(s.Particles))
// 			for i := range sequence {
// 				sequence[i] = i
// 			}

// 			// log.Printf("Best global error %f.", s.Best.Loss)
// 			if s.Best.Loss <= targetAccuracy {
// 				return tries
// 			}

// 			shuffle(sequence) // move Particles in random sequence

// 			for pi := range s.Particles {
// 				index := sequence[pi]
// 				currentParticle := s.Particles[index]
// 				mse := currentParticle.train(trainingData, s.Best.WeightsAndBiases, ms.Best.WeightsAndBiases)
// 				tries++

// 				if mse < s.Best.Loss {
// 					w := currentParticle.weights()

// 					s.Best.Loss = mse
// 					s.Best.WeightsAndBiases = w

// 					if mse < ms.Best.Loss {
// 						msg := "<%d:%d:%d> New global best found fitness error %9f => %9f on try %d."
// 						log.Printf(msg, i, si, pi, ms.Best.Loss, mse, tries)

// 						ms.Best.Loss = mse
// 						ms.Best.WeightsAndBiases = w
// 					}
// 				}
// 			}
// 		}
// 	}

// 	return tries
// }

func shuffle(sequence []int) {
	l := len(sequence)
	for i, s := range sequence {
		ri := rand.Intn(l-i) + i
		tmp := sequence[ri]
		sequence[ri] = s
		sequence[i] = tmp
	}
}

// //ClassificationAccuracy x
// func (msr *MultiSwarmRuntime) ClassificationAccuracy(testData []Data) float64 {
// 	var swarmBest, globalBest *NeuralNetwork
// 	for _, s := range msr.ms.Swarms {
// 		for _, p := range s.Particles {
// 			if swarmBest == nil || p.Best.Loss < swarmBest.Best.Loss {
// 				swarmBest = p

// 				if globalBest == nil || p.Best.Loss < globalBest.Best.Loss {
// 					globalBest = p
// 				}
// 			}
// 		}
// 		swarmBest.setWeights(s.Best.WeightsAndBiases)
// 	}
// 	globalBest.setWeights(ms.Best.WeightsAndBiases)
// 	return globalBest.classificationAccuracy(testData)
// }

// //Predict x
// func (ms *MultiSwarm) Predict(inputs ...float64) []float64 {
// 	var globalBest *NeuralNetwork
// 	for _, s := range ms.Swarms {
// 		for _, p := range s.Particles {
// 			if globalBest == nil || p.Best.Loss < globalBest.Best.Loss {
// 				globalBest = p
// 			}
// 		}
// 	}
// 	return globalBest.activate(inputs...)
// }
