package cogent

import (
	"fmt"
	"log"
	"math"
	"sync"
)

const (
	globalKey      = "global"
	swarmKeyFormat = "swarm_%d"
)

//MultiSwarm x
type MultiSwarm struct {
	particleCount int
	blackboard    *blackboard
	swarms        []*swarm
}

type swarm struct {
	id        int
	particles []*particle
}

type blackboard struct {
	mutex          *sync.RWMutex
	best           map[string]Position
	nnConfig       NeuralNetworkConfiguration
	trainingConfig TrainingConfiguration
	trainingData   TrainingData
}

//NewMultiSwarm x
func NewMultiSwarm(config MultiSwarmConfiguration, trainingConfig TrainingConfiguration) *MultiSwarm {
	if config.SwarmCount <= 0 {
		panic("No swarm count in config")
	}

	if trainingConfig.MaxIterations <= 0 {
		panic("No iterations in training config")
	}

	bb := &blackboard{
		mutex:          &sync.RWMutex{},
		best:           map[string]Position{},
		nnConfig:       *config.NeuralNetworkConfiguration,
		trainingConfig: trainingConfig,
	}

	tmpParticle := newParticle(-1, -1, bb)

	bb.mutex.Lock()
	bb.best[globalKey] = Position{
		Loss:             math.MaxFloat64,
		WeightsAndBiases: tmpParticle.data.weights(),
	}
	bb.mutex.Unlock()

	ms := MultiSwarm{
		swarms:        make([]*swarm, config.SwarmCount),
		particleCount: int(config.SwarmCount * config.ParticleCount),
		blackboard:    bb,
	}
	for swarmID := range ms.swarms {
		s := &swarm{
			id:        swarmID,
			particles: make([]*particle, config.ParticleCount),
		}

		for particleID := 0; particleID < int(config.ParticleCount); particleID++ {
			s.particles[particleID] = newParticle(swarmID, particleID, bb)
		}
		ms.swarms[swarmID] = s

		tmpParticle.data.reset(trainingConfig.WeightRange)
		swarmKey := fmt.Sprintf(swarmKeyFormat, s.id)
		bb.mutex.Lock()
		bb.best[swarmKey] = Position{
			Loss:             math.MaxFloat64,
			WeightsAndBiases: tmpParticle.data.weights(),
		}
		bb.mutex.Unlock()
	}

	log.Printf("using %d weights and biases", len(bb.best[globalKey].WeightsAndBiases))

	return &ms
}

//Train x
func (ms *MultiSwarm) Train(training *TrainingData) {
	ms.blackboard.mutex.Lock()
	ms.blackboard.trainingData = *training
	ms.blackboard.mutex.Unlock()

	wg := &sync.WaitGroup{}
	wg.Add(ms.particleCount)
	for _, s := range ms.swarms {
		for _, p := range s.particles {
			go p.train(wg)
		}
	}
	wg.Wait()
}

//Best x
func (ms *MultiSwarm) Best() *NeuralNetworkData {
	var best *NeuralNetworkData
	for _, s := range ms.swarms {
		for _, p := range s.particles {
			if best == nil || p.data.Best.Loss < best.Best.Loss {
				best = p.data
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
