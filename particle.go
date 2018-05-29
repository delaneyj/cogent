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
		log.Fatalf("Invalid loss type '%d'", nnConfig.Loss)
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
	MaxIterations   int
	MaxAccuracy     float64
	InertialWeight  float64
	CognitiveWeight float64
	SocialWeight    float64
	GlobalWeight    float64
	WeightRange     float64
	WeightDecayRate float64
	DeathRate       float64
}

func (p *particle) train(pti particleTrainingInfo, wg *sync.WaitGroup) {
	bestKFoldLoss := math.MaxFloat64

	for iteration := 0; iteration < pti.MaxIterations; iteration++ {
		kfoldLossAvg := 0.0
		ttSets := kfoldTestTrainSets(pti.Dataset)
		for kfold, ttSet := range ttSets {
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

			loss := p.checkAndSetLoss(iteration, kfold, ttSet)

			deathChance := rand.Float64()
			if deathChance < pti.DeathRate {
				p.nn.reset(pti.WeightRange)
				loss = p.checkAndSetLoss(iteration, kfold, ttSet)
			}

			kfoldLossAvg += loss
		}

		kfoldLossAvg /= float64(len(ttSets))
		testAcc := p.nn.ClassificationAccuracy(pti.Dataset)
		log.Printf("%d <%d:%d>  %0.16f (T%0.16f)", iteration, p.swarmID, p.id, kfoldLossAvg, testAcc)

		if kfoldLossAvg < bestKFoldLoss {
			bestKFoldLoss = kfoldLossAvg
			filename := fmt.Sprintf("%d%d%d_L%0.12f_T%3.9f.net", iteration, p.swarmID, p.id, kfoldLossAvg, 100*testAcc)
			f, err := os.Create(filename)
			checkErr(err)
			e := gob.NewEncoder(f)
			err = e.Encode(p.nn)
			checkErr(err)
			err = f.Close()
			checkErr(err)
		}
	}
	wg.Done()
}

type testTrainSet struct {
	train, test Dataset
}

func kfoldTestTrainSets(dataset Dataset) []*testTrainSet {
	k := 10
	datasetCount := len(dataset)
	if datasetCount < k {
		k = datasetCount
	}

	buckets := make([]Dataset, k)
	for i := range buckets {
		buckets[i] = Dataset{}
	}

	shuffledIndexes := make([]int, datasetCount)
	for i := range shuffledIndexes {
		shuffledIndexes[i] = i
	}
	for i := datasetCount - 1; i > 0; i-- {
		j := rand.Intn(i + 1)
		shuffledIndexes[i], shuffledIndexes[j] = shuffledIndexes[j], shuffledIndexes[i]
	}

	for i := 0; i < datasetCount; i++ {
		ri := shuffledIndexes[i]
		d := dataset[ri]
		bi := i % k
		buckets[bi] = append(buckets[bi], d)
	}

	tt := make([]*testTrainSet, k)
	for i := range tt {
		tt[i] = &testTrainSet{
			train: Dataset{},
			test:  Dataset{},
		}
	}

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

func (p *particle) checkAndSetLoss(iteration, kfold int, ttSet *testTrainSet) float64 {
	loss := p.calculateMeanLoss(ttSet.train)
	p.nn.CurrentLoss = loss
	if loss < p.nn.Best.Loss {
		blf := "max"
		if p.nn.Best.Loss != math.MaxFloat64 {
			blf = fmt.Sprintf("%0.16f", p.nn.Best.Loss)
		}
		log.Printf("<%d/%d> Local best <%d:%d> from %s->%f", iteration, kfold, p.swarmID, p.id, blf, loss)
		updatedBest := Position{
			Loss:             loss,
			WeightsAndBiases: p.nn.weights(),
		}

		p.nn.Best = updatedBest

		bestSwarmKey := fmt.Sprintf(swarmKeyFormat, p.swarmID)
		res, ok := p.blackboard.Load(bestSwarmKey)
		checkOk(ok)
		bestSwarm := res.(Position)

		if loss < bestSwarm.Loss {
			log.Printf("<%d/%d> Swarm best <%d:%d> from %f->%f", iteration, kfold, p.swarmID, p.id, bestSwarm.Loss, loss)
			p.blackboard.Store(bestSwarmKey, updatedBest)

			res, ok = p.blackboard.Load(globalKey)
			checkOk(ok)
			bestGlobal := res.(Position)
			if loss < bestGlobal.Loss {
				p.blackboard.Store(globalKey, updatedBest)

				trainAcc := p.nn.ClassificationAccuracy(ttSet.train)
				testAcc := p.nn.ClassificationAccuracy(ttSet.test)

				gbMsg := "<%d/%d> Global best <%d:%d> from %3.16f->%3.16f  (R%0.16f/T%0.16f)"
				log.Printf(gbMsg, iteration, kfold, p.swarmID, p.id, bestGlobal.Loss, loss, trainAcc, testAcc)
			}
		}
	}

	return loss
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
