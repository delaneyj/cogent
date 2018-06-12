package cogent

import (
	"fmt"
	"log"
	"math"
	"sync"
	"time"

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
	buckets        DataBuckets

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

		tmpParticle.nn.reset(tmpParticle.r, tmpParticle.layersTrainingInfo, trainingConfig.WeightRange)
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
func (ms *MultiSwarm) Train(buckets DataBuckets, shouldMultithread bool) {
	// log.Printf("%+v %+v", ms.dataset.Inputs, ms.dataset.Outputs)

	pti := particleTrainingInfo{
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

	iterations, avgTime := 0, time.Duration(0)
	for ; iterations < ms.trainingConfig.MaxIterations; iterations++ {
		// log.Printf("iteration %d started.", iterations)
		start := time.Now()
		wg := &sync.WaitGroup{}
		wg.Add(ms.particleCount)
		for _, s := range ms.swarms {
			for _, p := range s.particles {
				if shouldMultithread {
					go p.train(wg, iterations, pti, buckets)
				} else {
					p.train(wg, iterations, pti, buckets)
				}
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
		bestAcc := nn.ClassificationAccuracy(buckets, -1)
		log.Printf("iteration %d took %s.", iterations, time.Since(start))
		avgTime += time.Since(start)

		if bestAcc >= pti.TargetAccuracy {
			ms.predictor = nn
			break
		}
	}

	log.Printf("Did %d iterations taking on average %s.", iterations, avgTime/time.Duration(iterations+1))
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
func (ms *MultiSwarm) ClassificationAccuracy(buckets DataBuckets) float64 {
	ms.predictNN()
	acc := ms.predictNN().ClassificationAccuracy(buckets, -1)
	return acc
}

//Predict x
func (ms *MultiSwarm) Predict(inputs *t.Dense) *t.Dense {
	return ms.predictNN().Activate(inputs)
}
