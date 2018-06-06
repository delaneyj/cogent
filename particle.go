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
	"time"
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

	fn := LossFns[nnConfig.Loss]
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
	Dataset               Dataset
	MaxIterations         int
	MaxAccuracy           float64
	InertialWeight        float64
	CognitiveWeight       float64
	SocialWeight          float64
	GlobalWeight          float64
	WeightRange           float64
	WeightDecayRate       float64
	DeathRate             float64
	RidgeRegressionWeight float64
}

func (p *particle) train(pti particleTrainingInfo, ttSets []*testTrainSet, wg *sync.WaitGroup) {

	start := time.Now()

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

	mu := &sync.Mutex{}
	kfoldLossAvg := 0.0

	ttSetsWG := &sync.WaitGroup{}
	ttSetsWG.Add(len(ttSets))
	for _, ttSet := range ttSets {
		go func() {
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

			loss := p.calculateMeanLoss(ttSet.train, pti.RidgeRegressionWeight)

			mu.Lock()
			kfoldLossAvg += loss
			mu.Unlock()
			ttSetsWG.Done()
		}()
	}
	ttSetsWG.Wait()
	kfoldLossAvg /= float64(len(ttSets))
	log.Printf("<%d:%d> took %s.", p.swarmID, p.id, time.Since(start))

	wasGlobalBest := p.setBest(kfoldLossAvg, pti.RidgeRegressionWeight)
	if wasGlobalBest {
		// rmse := p.rmse(pti.Dataset)
		testAcc := p.nn.ClassificationAccuracy(pti.Dataset, true)
		log.Printf("<%d:%d> accuracy:%f loss:%f", p.swarmID, p.id, testAcc, kfoldLossAvg)

		filename := fmt.Sprintf("KFX_%0.8f_TACC%0.16f.nn", kfoldLossAvg, testAcc)
		log.Printf(filename)

		if kfoldLossAvg < math.MaxFloat64/2 {
			f, err := os.Create(filename)
			checkErr(err)
			e := gob.NewEncoder(f)
			err = e.Encode(p.nn)
			checkErr(err)
			err = f.Close()
			checkErr(err)
		}
	}

	deathChance := rand.Float64()
	if deathChance < pti.DeathRate {
		log.Printf("<%d:%d> died!", p.swarmID, p.id)
		p.nn.reset(pti.WeightRange)
		index := rand.Intn(len(ttSets))

		loss := p.calculateMeanLoss(ttSets[index].train, pti.RidgeRegressionWeight)
		p.setBest(loss, pti.RidgeRegressionWeight)
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

func (p *particle) setBest(loss float64, ridgeRegressionWeight float64) bool {
	p.nn.CurrentLoss = loss
	wasGlobalBest := false
	if loss < p.nn.Best.Loss {
		blf := "max"
		if p.nn.Best.Loss != math.MaxFloat64 {
			blf = fmt.Sprintf("%0.16f", p.nn.Best.Loss)
		}
		log.Printf("Local best <%d:%d> from %s->%f", p.swarmID, p.id, blf, loss)
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
			blf = "max"
			if bestSwarm.Loss != math.MaxFloat64 {
				blf = fmt.Sprintf("%0.16f", bestSwarm.Loss)
			}
			log.Printf("Swarm best <%d:%d> from %s->%f", p.swarmID, p.id, blf, loss)
			p.blackboard.Store(bestSwarmKey, updatedBest)

			res, ok = p.blackboard.Load(globalKey)
			checkOk(ok)
			bestGlobal := res.(Position)
			if loss < bestGlobal.Loss {
				blf = "max"
				if bestGlobal.Loss != math.MaxFloat64 {
					blf = fmt.Sprintf("%0.16f", bestGlobal.Loss)
				}
				log.Printf("Global best <%d:%d> from %s->%f", p.swarmID, p.id, blf, loss)
				p.blackboard.Store(globalKey, updatedBest)
				wasGlobalBest = true
			}
		}
	}

	return wasGlobalBest
}

func (p *particle) rmse(dataset Dataset) float64 {
	wg := &sync.WaitGroup{}
	mu := &sync.Mutex{}
	rmse := 0.0

	wg.Add(len(dataset))
	for _, d := range dataset {
		go func() {
			expected := d.Outputs
			actual := p.nn.Activate(d.Inputs...)
			for j, a := range actual {
				e := expected[j]
				diff := a - e
				mu.Lock()
				rmse += diff * diff
				mu.Unlock()
			}
			wg.Done()
		}()
	}
	wg.Wait()
	return math.Sqrt(rmse / float64(len(dataset)))
}

func (p *particle) calculateMeanLoss(dataset Dataset, ridgeRegressionWeight float64) float64 {
	sum := 0.0
	for _, d := range dataset {
		actualOuputs := p.nn.Activate(d.Inputs...)
		err := p.fn(d.Outputs, actualOuputs)
		if math.IsNaN(err) {
			runtime.Breakpoint()
		}
		sum += err
	}
	loss := sum / float64(len(dataset))

	l2Regularization := 0.0
	for _, w := range p.nn.weights() {
		l2Regularization += w * w
	}
	l2Regularization /= float64(p.nn.weightsAndBiasesCount())
	l2Regularization *= ridgeRegressionWeight
	// log.Printf("<%02d:%02d>  LF:%f L2:%f", p.swarmID, p.id, loss, l2Regularization)
	return loss + l2Regularization
}

func checkOk(ok bool) {
	if !ok {
		runtime.Breakpoint()
	}
}
