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

	predictor *NeuralNetwork
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
		TargetAccuracy:        ms.trainingConfig.TargetAccuracy,
		InertialWeight:        ms.trainingConfig.InertialWeight,
		CognitiveWeight:       ms.trainingConfig.CognitiveWeight,
		SocialWeight:          ms.trainingConfig.SocialWeight,
		GlobalWeight:          ms.trainingConfig.GlobalWeight,
		WeightRange:           ms.trainingConfig.WeightRange,
		DeathRate:             ms.trainingConfig.ProbablityOfDeath,
		RidgeRegressionWeight: ms.trainingConfig.RidgeRegressionWeight,
		KFolds:                ms.trainingConfig.KFolds,
		StoreGlobalBest:       ms.trainingConfig.StoreGlobalBest,
	}

	for i := 0; i < ms.trainingConfig.MaxIterations; i++ {
		// start := time.Now()
		ttSets := kfoldTestTrainSets(pti.KFolds, pti.Dataset)

		wg := &sync.WaitGroup{}
		wg.Add(ms.particleCount)
		for i, s := range ms.swarms {
			for _, p := range s.particles {
				go p.train(wg, i, pti, ttSets)
			}
		}
		wg.Wait()

		var nn *NeuralNetwork
		for _, s := range ms.swarms {
			for _, p := range s.particles {
				if nn == nil || nn.Best.Loss < nn.Best.Loss {
					nn = p.nn
				}
			}
		}
		bestAcc := nn.ClassificationAccuracy(pti.Dataset)
		// log.Printf("iteration %d took %s. t:%0.2f", i, time.Since(start), 100*bestAcc)

		if bestAcc >= pti.TargetAccuracy {
			ms.predictor = nn
			break
		}
	}
}

//Best x
func (ms *MultiSwarm) predictNN() *NeuralNetwork {
	if ms.predictor != nil {
		return ms.predictor
	}

	res, ok := ms.blackboard.Load(bestGlobalNNKey)
	checkOk(ok)
	bestGlobal := res.(NeuralNetwork)
	return &bestGlobal
}

//ClassificationAccuracy x
func (ms *MultiSwarm) ClassificationAccuracy(testData *Dataset) float64 {
	ms.predictNN()
	acc := ms.predictNN().ClassificationAccuracy(testData)
	return acc
}

//Predict x
func (ms *MultiSwarm) Predict(inputs *t.Dense) *t.Dense {
	return ms.predictNN().Activate(inputs)
}
