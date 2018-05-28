package cogent

import (
	"encoding/gob"
	fmt "fmt"
	"log"
	math "math"
	"math/rand"
	"os"
	"runtime"
	"sync"
)

type particle struct {
	id         int
	fn         lossFn
	nn         *NeuralNetwork
	blackboard *sync.Map
	swarmID    int
}

//NewNeuralNetworkConfiguration x
func NewNeuralNetworkConfiguration(inputCount int, lc ...LayerConfig) *NeuralNetworkConfiguration {
	nnc := NeuralNetworkConfiguration{
		Loss:         Cross,
		InputCount:   inputCount,
		LayerConfigs: lc,
	}
	return &nnc
}

func newParticle(swarmID, particleID int, blackboard *sync.Map, weightRange float64, nnConfig NeuralNetworkConfiguration) *particle {
	// var nnConfig NeuralNetworkConfiguration
	// var trainingConfig TrainingConfiguration

	fn := lossFns[nnConfig.Loss]
	if fn == nil {
		log.Fatalf("Invalid loss type '%s'", nnConfig.Loss)
	}
	nn := NeuralNetwork{
		Layers:      make([]LayerData, len(nnConfig.LayerConfigs)),
		CurrentLoss: math.MaxFloat64,
		Loss:        nnConfig.Loss,
	}

	previousLayerCount := nnConfig.InputCount
	for i, layerConfig := range nnConfig.LayerConfigs {
		wbCount := (previousLayerCount + 1) * layerConfig.NodeCount
		l := LayerData{
			NodeCount:        layerConfig.NodeCount,
			WeightsAndBiases: make([]float64, wbCount),
			Velocities:       make([]float64, wbCount),
			// Nodes:          nodes,
			Activation: layerConfig.Activation,
		}
		l.reset(weightRange)
		nn.Layers[i] = l
		previousLayerCount = layerConfig.NodeCount
	}
	nn.Best = Position{
		Loss:             math.MaxFloat64,
		WeightsAndBiases: nn.weights(),
	}

	return &particle{
		swarmID:    swarmID,
		id:         particleID,
		fn:         fn,
		nn:         &nn,
		blackboard: blackboard,
	}
}

type particleTrainingInfo struct {
	Dataset         Dataset
	MaxAccuracy     float64
	InertialWeight  float64
	CognitiveWeight float64
	SocialWeight    float64
	GlobalWeight    float64
	WeightRange     float64
	WeightDecayRate float64
	DeathRate       float64
}

func (p *particle) train(pti particleTrainingInfo) {
	res, ok := p.blackboard.Load(globalKey)
	checkOk(ok)
	bestGlobal := res.(Position)

	bestSwarmKey := fmt.Sprintf(swarmKeyFormat, p.swarmID)
	res, ok = p.blackboard.Load(bestSwarmKey)
	checkOk(ok)
	bestSwarm := res.(Position)

	if bestGlobal.Loss <= pti.MaxAccuracy {
		return
	}

	flatArrayIndex := 0
	// Compute new velocity.  Depends on old velocity, best position of parrticle, and best position of any particle
	for _, l := range p.nn.Layers {
		for i, currentLocalWeight := range l.WeightsAndBiases {
			bestGlobalPosition := bestGlobal.WeightsAndBiases[flatArrayIndex]
			bestSwarmPosition := bestSwarm.WeightsAndBiases[flatArrayIndex]
			bestLocalPosition := p.nn.Best.WeightsAndBiases[flatArrayIndex]

			currentLocalVelocity := l.Velocities[i]

			oldVelocityFactor := pti.InertialWeight * currentLocalVelocity

			localRandomness := rand.Float64()
			bestLocationDelta := bestLocalPosition - currentLocalWeight
			localPositionFactor := pti.CognitiveWeight * localRandomness * bestLocationDelta

			swarmRandomness := rand.Float64()
			bestSwarmlDelta := bestSwarmPosition - currentLocalWeight
			swarmPositionFactor := pti.SocialWeight * swarmRandomness * bestSwarmlDelta

			globalRandomness := rand.Float64()
			bestGlobalDelta := bestGlobalPosition - currentLocalWeight
			globalPositionFactor := pti.GlobalWeight * globalRandomness * bestGlobalDelta

			revisedVelocity := oldVelocityFactor + localPositionFactor + swarmPositionFactor + globalPositionFactor
			l.Velocities[i] = revisedVelocity

			flatArrayIndex++
		}
	}

	flatArrayIndex = 0
	for _, l := range p.nn.Layers {
		for i, w := range l.WeightsAndBiases {
			v := l.Velocities[i]
			revisedPosition := w + v
			wr := pti.WeightRange
			clamped := math.Max(-wr, math.Min(wr, revisedPosition)) // restriction
			decayed := clamped * (1 + pti.WeightDecayRate)          // decay (large weights tend to overfit)

			l.WeightsAndBiases[i] = decayed
			flatArrayIndex++
		}
	}

	p.checkAndSetLoss(pti.Dataset)

	deathChance := rand.Float64()
	if deathChance < pti.DeathRate {
		p.nn.reset(pti.WeightRange)
		p.checkAndSetLoss(pti.Dataset)
	}
}

type testTrainSet struct {
	train, test Dataset
}

func kfoldTestTrainSets(dataset Dataset) []testTrainSet {
	k := 10
	datasetCount := len(dataset)
	if datasetCount <= k {
		k = len(dataset) - 1
	}

	buckets := make([]Dataset, k)
	for i := range buckets {
		buckets[i] = Dataset{}
	}

	for _, d := range dataset {
		bucketIndex := rand.Int() % k
		buckets[bucketIndex] = append(buckets[bucketIndex], d)
	}

	tt := make([]testTrainSet, k)
	for i, b := range buckets {
		for j, t := range tt {
			if i == j {
				t.test = append(t.test, b...)
			} else {
				t.train = append(t.train, b...)
			}
		}
	}
	return tt
}

func (p *particle) checkAndSetLoss(dataset Dataset) float64 {
	testTrainSet := kfoldTestTrainSets(dataset)

	sumLosses := 0.0
	for _, ttSet := range testTrainSet {
		sumLosses += p.calculateMeanLoss(ttSet.train)
	}
	avgLoss := sumLosses / float64(len(testTrainSet))

	p.nn.CurrentLoss = avgLoss
	if avgLoss < p.nn.Best.Loss {
		blf := "max"
		if p.nn.Best.Loss != math.MaxFloat64 {
			blf = fmt.Sprintf("%0.16f", p.nn.Best.Loss)
		}
		log.Printf("Local best <%d:%d> from %s->%f", p.swarmID, p.id, blf, avgLoss)
		updatedBest := Position{
			Loss:             avgLoss,
			WeightsAndBiases: p.nn.weights(),
		}

		p.nn.Best = updatedBest

		bestSwarmKey := fmt.Sprintf(swarmKeyFormat, p.swarmID)
		res, ok := p.blackboard.Load(bestSwarmKey)
		checkOk(ok)
		bestSwarm := res.(Position)

		if avgLoss < bestSwarm.Loss {
			log.Printf("Swarm best <%d:%d> from %f->%f", p.swarmID, p.id, bestSwarm.Loss, avgLoss)
			p.blackboard.Store(bestSwarmKey, updatedBest)

			res, ok = p.blackboard.Load(globalKey)
			checkOk(ok)
			bestGlobal := res.(Position)
			if avgLoss < bestGlobal.Loss {
				p.blackboard.Store(globalKey, updatedBest)

				for _, ttSet := range testTrainSet {
					trainAccuracy := p.nn.ClassificationAccuracy(ttSet.train)
					testAccuracy := p.nn.ClassificationAccuracy(ttSet.test)

					trap := trainAccuracy * 100
					ttap := testAccuracy * 100
					log.Printf("Global best <%d:%d> from %3.16f->%3.16f  (R%0.16f/T%0.16f)", p.swarmID, p.id, bestGlobal.Loss, avgLoss, trap, ttap)

					filename := fmt.Sprintf("L%0.12f_R%0.24f_T%3.24f.net", avgLoss, trainAccuracy, testAccuracy)
					f, err := os.Create(filename)
					checkErr(err)
					e := gob.NewEncoder(f)
					err = e.Encode(p.nn)
					checkErr(err)
					err = f.Close()
					checkErr(err)
				}
			}
		}
	}

	return avgLoss
}

func (p *particle) calculateMeanLoss(dataset Dataset) float64 {
	sum := 0.0
	for _, d := range dataset {
		actualOuputs := p.nn.Activate(d.Inputs...)
		err := p.fn(d.Outputs, actualOuputs)
		sum += err
	}
	loss := sum / float64(len(dataset))
	return loss
}

func checkOk(ok bool) {
	if !ok {
		runtime.Breakpoint()
	}
}
