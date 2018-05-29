package cogent

import (
	"fmt"
	"log"
	"math"
	"runtime"
	"sync"
)

const (
	globalKey      = "global"
	swarmKeyFormat = "swarm_%d"
)

//MultiSwarm x
type MultiSwarm struct {
	particleCount  int
	blackboard     *sync.Map
	swarms         []*swarm
	trainingConfig TrainingConfiguration
	dataset        Dataset
}

type swarm struct {
	id        int
	particles []*particle
}

//NewMultiSwarm x
func NewMultiSwarm(config MultiSwarmConfiguration, trainingConfig TrainingConfiguration) *MultiSwarm {
	if config.SwarmCount <= 0 {
		panic("No swarm count in config")
	}

	if trainingConfig.MaxIterations <= 0 {
		panic("No iterations in training config")
	}

	// bb := &blackboard{
	// 	mutex:          keymutex.New(997),
	// 	best:           map[string]Position{},
	// 	nnConfig:       *config.NeuralNetworkConfiguration,
	// 	trainingConfig: trainingConfig,
	// }

	bb := &sync.Map{}
	tmpParticle := newParticle(-1, -1, bb, trainingConfig.WeightRange, config.NeuralNetworkConfiguration)

	bb.Store(globalKey, Position{
		Loss:             math.MaxFloat64,
		WeightsAndBiases: tmpParticle.nn.weights(),
	})

	ms := MultiSwarm{
		swarms:         make([]*swarm, config.SwarmCount),
		particleCount:  int(config.SwarmCount * config.ParticleCount),
		blackboard:     bb,
		trainingConfig: trainingConfig,
	}
	for swarmID := range ms.swarms {
		s := &swarm{
			id:        swarmID,
			particles: make([]*particle, config.ParticleCount),
		}

		for particleID := 0; particleID < int(config.ParticleCount); particleID++ {
			s.particles[particleID] = newParticle(swarmID, particleID, bb, trainingConfig.WeightRange, config.NeuralNetworkConfiguration)
		}
		ms.swarms[swarmID] = s

		tmpParticle.nn.reset(trainingConfig.WeightRange)
		swarmKey := fmt.Sprintf(swarmKeyFormat, s.id)

		bb.Store(swarmKey, Position{
			Loss:             math.MaxFloat64,
			WeightsAndBiases: tmpParticle.nn.weights(),
		})
	}

	res, ok := bb.Load(globalKey)
	if !ok {
		runtime.Breakpoint()
	}
	wbCount := len(res.(Position).WeightsAndBiases)
	log.Printf("using %d weights and biases", wbCount)

	return &ms
}

//Train x
func (ms *MultiSwarm) Train(dataset Dataset) {
	ms.dataset = dataset

	pti := particleTrainingInfo{
		Dataset:         dataset,
		MaxIterations:   ms.trainingConfig.MaxIterations,
		MaxAccuracy:     ms.trainingConfig.TargetAccuracy,
		InertialWeight:  ms.trainingConfig.InertialWeight,
		CognitiveWeight: ms.trainingConfig.CognitiveWeight,
		SocialWeight:    ms.trainingConfig.SocialWeight,
		GlobalWeight:    ms.trainingConfig.GlobalWeight,
		WeightRange:     ms.trainingConfig.WeightRange,
		WeightDecayRate: ms.trainingConfig.WeightDecayRate,
		DeathRate:       ms.trainingConfig.ProbablityOfDeath,
	}

	wg := &sync.WaitGroup{}
	wg.Add(ms.particleCount)
	for _, s := range ms.swarms {
		for _, p := range s.particles {
			go p.train(pti, wg)
		}
	}
	wg.Wait()
}

//Best x
func (ms *MultiSwarm) Best() *NeuralNetwork {
	var best *NeuralNetwork
	for _, s := range ms.swarms {
		for _, p := range s.particles {
			if best == nil || p.nn.Best.Loss < best.Best.Loss {
				best = p.nn
			}
		}
	}
	return best
}

//ClassificationAccuracy x
func (ms *MultiSwarm) ClassificationAccuracy(testData ...*Data) float64 {
	return ms.Best().ClassificationAccuracy(testData)
}

//Predict x
func (ms *MultiSwarm) Predict(inputs ...float64) []float64 {
	return ms.Best().Activate(inputs...)
}
