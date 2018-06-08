package cogent

import (
	"fmt"
	"log"
	"math"
	"sync"

	t "gorgonia.org/tensor"
)

const (
	globalKey       = "global"
	bestGlobalNNKey = "globalNN"
	swarmKeyFormat  = "swarm_%d"
)

//MultiSwarm x
type MultiSwarm struct {
	particleCount  int
	blackboard     *sync.Map
	swarms         []*swarm
	trainingConfig TrainingConfiguration
	dataset        *Dataset
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

	bb := &sync.Map{}
	tmpParticle := newParticle(-1, -1, bb, trainingConfig.WeightRange, config.NeuralNetworkConfiguration)

	bb.Store(globalKey, Position{
		Loss:   math.MaxFloat64,
		Layers: tmpParticle.nn.Layers,
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
			Loss:   math.MaxFloat64,
			Layers: tmpParticle.nn.Layers,
		})
	}

	wbCount := ms.swarms[0].particles[0].nn.weightsAndBiasesCount()
	log.Printf("using %d weights and biases", wbCount)

	return &ms
}

//Train x
func (ms *MultiSwarm) Train(dataset *Dataset) {
	ms.dataset = dataset

	pti := particleTrainingInfo{
		Dataset:               dataset,
		MaxIterations:         ms.trainingConfig.MaxIterations,
		TargetAccuracy:        ms.trainingConfig.TargetAccuracy,
		InertialWeight:        ms.trainingConfig.InertialWeight,
		CognitiveWeight:       ms.trainingConfig.CognitiveWeight,
		SocialWeight:          ms.trainingConfig.SocialWeight,
		GlobalWeight:          ms.trainingConfig.GlobalWeight,
		WeightRange:           ms.trainingConfig.WeightRange,
		WeightDecayRate:       ms.trainingConfig.WeightDecayRate,
		DeathRate:             ms.trainingConfig.ProbablityOfDeath,
		RidgeRegressionWeight: ms.trainingConfig.RidgeRegressionWeight,
		KFolds:                ms.trainingConfig.KFolds,
		StoreGlobalBest:       ms.trainingConfig.StoreGlobalBest,
	}

loop:
	for i := 0; i < ms.trainingConfig.MaxIterations; i++ {
		// start := time.Now()
		ttSets := kfoldTestTrainSets(pti.KFolds, pti.Dataset)
		// wg := &sync.WaitGroup{}
		// wg.Add(ms.particleCount)
		chs := make([]chan float64, ms.particleCount)
		for i := range chs {
			chs[i] = make(chan float64)
		}

		chIndex := 0
		for i, s := range ms.swarms {
			for _, p := range s.particles {
				ch := chs[chIndex]
				go p.train(i, pti, ttSets, ch)
				chIndex++
			}
		}

		for _, ch := range chs {
			testAcc := <-ch
			// wg.Done()
			if testAcc >= ms.trainingConfig.TargetAccuracy {
				break loop
			}
		}
		// wg.Wait()
		// log.Printf("iteration %d took %s.", i, time.Since(start))
	}
}

//Best x
func (ms *MultiSwarm) Best() *NeuralNetwork {
	res, ok := ms.blackboard.Load(bestGlobalNNKey)
	checkOk(ok)
	bestGlobal := res.(NeuralNetwork)
	return &bestGlobal
}

//ClassificationAccuracy x
func (ms *MultiSwarm) ClassificationAccuracy(testData *Dataset) float64 {
	b := ms.Best()
	acc := b.ClassificationAccuracy(testData)
	return acc
}

//Predict x
func (ms *MultiSwarm) Predict(inputs *t.Dense) *t.Dense {
	return ms.Best().Activate(inputs)
}
