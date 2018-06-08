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

	t "gorgonia.org/tensor"
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

	rowCount := nnConfig.InputCount + 1
	colCount := 0
	lastLayerIndex := len(nnConfig.LayerConfigs) - 1

	for i, layerConfig := range nnConfig.LayerConfigs {
		colCount = layerConfig.NodeCount
		if i != lastLayerIndex {
			colCount++
		}

		l := LayerData{
			NodeCount: layerConfig.NodeCount,
			WeightsAndBiases: t.New(
				t.Of(Float),
				t.WithShape(rowCount, colCount),
			),
			Velocities: t.New(
				t.Of(t.Float64),
				t.WithShape(rowCount, colCount),
			),
			Activation: layerConfig.Activation,
		}
		l.reset(weightRange)
		nn.Layers[i] = l
		rowCount = colCount
	}
	nn.Best = Position{
		Loss:   math.MaxFloat64,
		Layers: nn.Layers,
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
	Dataset               *Dataset
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
	KFolds                int
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
		func(ttSet *testTrainSet) {
			// Compute new velocity.  Depends on old velocity, best position of parrticle, and best position of any particle
			for i, l := range p.nn.Layers {
				currentLocal := l.WeightsAndBiases

				bestLocal := p.nn.Best.Layers[i].WeightsAndBiases
				currentLocalVelocity := l.Velocities

				// oldVelocityFactor := pti.InertialWeight * currentLocalVelocity
				oldVelocityFactor := must(currentLocalVelocity.MulScalar(pti.InertialWeight, true))

				// 	localRandomness := rand.Float64()
				localRandomness := t.NewDense(
					Float,
					l.WeightsAndBiases.Shape(),
					t.WithBacking(t.Random(Float, l.WeightsAndBiases.DataSize())),
				)
				// 	bestLocationDelta := bestLocalPosition - currentLocalWeight
				bestLocalDelta := must(bestLocal.Sub(currentLocal))
				// 	localPositionFactor := pti.CognitiveWeight * localRandomness * bestLocationDelta
				localPositionFactor := must(must(localRandomness.MulScalar(pti.CognitiveWeight, true)).Mul(bestLocalDelta))

				// 	swarmRandomness := rand.Float64()
				bestSwarm := bestSwarm.Layers[i].WeightsAndBiases
				swarmRandomness := t.NewDense(
					Float,
					l.WeightsAndBiases.Shape(),
					t.WithBacking(t.Random(Float, l.WeightsAndBiases.DataSize())),
				)
				// 	bestSwarmlDelta := bestSwarmPosition - currentLocalWeight
				bestSwarmlDelta := must(bestSwarm.Sub(currentLocal))
				// 	swarmPositionFactor := pti.SocialWeight * swarmRandomness * bestSwarmlDelta
				swarmPositionFactor := must(must(swarmRandomness.MulScalar(pti.SocialWeight, true)).Mul(bestSwarmlDelta))

				// 	globalRandomness := rand.Float64()
				bestGlobal := bestGlobal.Layers[i].WeightsAndBiases
				globalRandomness := t.NewDense(
					Float,
					l.WeightsAndBiases.Shape(),
					t.WithBacking(t.Random(Float, l.WeightsAndBiases.DataSize())),
				)
				// 	bestGlobalDelta := bestGlobalPosition - currentLocalWeight
				bestGlobalDelta := must(bestGlobal.Sub(currentLocal))
				// 	globalPositionFactor := pti.GlobalWeight * globalRandomness * bestGlobalDelta
				globalPositionFactor := must(must(globalRandomness.MulScalar(pti.GlobalWeight, true)).Mul(bestGlobalDelta))

				// 	revisedVelocity := oldVelocityFactor + localPositionFactor + swarmPositionFactor + globalPositionFactor
				revisedVelocity := must(must(must(oldVelocityFactor.Add(localPositionFactor)).Add(swarmPositionFactor)).Add(globalPositionFactor))

				// 	l.Velocities[i] = revisedVelocity
				l.Velocities = revisedVelocity
				// 	i++
				// }
			}

			wr := pti.WeightRange
			for _, l := range p.nn.Layers {
				revisedPosition := must(l.WeightsAndBiases.Add(l.Velocities))
				data := revisedPosition.Data().([]float64)
				for i, w := range data {
					clamped := math.Max(-wr, math.Min(wr, w))      // restriction
					decayed := clamped * (1 + pti.WeightDecayRate) // decay (large weights tend to overfit)
					data[i] = decayed
				}
			}

			loss := p.calculateMeanLoss(ttSet.train, pti.RidgeRegressionWeight)

			mu.Lock()
			kfoldLossAvg += loss
			mu.Unlock()
			ttSetsWG.Done()
		}(ttSet)
	}
	ttSetsWG.Wait()
	kfoldLossAvg /= float64(len(ttSets))
	log.Printf("<%d:%d> took %s.", p.swarmID, p.id, time.Since(start))

	wasGlobalBest := p.setBest(kfoldLossAvg, pti.RidgeRegressionWeight)
	if wasGlobalBest {
		// rmse := p.rmse(pti.Dataset)
		testAcc := p.nn.ClassificationAccuracy(pti.Dataset)
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

func must(d *t.Dense, err error) *t.Dense {
	if err != nil {
		log.Fatal(err)
	}
	return d
}

type testTrainSet struct {
	train, test *Dataset
}

type testTrainSets []*testTrainSet

func kfoldTestTrainSets(k int, dataset *Dataset) testTrainSets {
	rowCount := dataset.rowCount()
	if rowCount < k {
		k = rowCount
	}

	inputs := dataset.Inputs.Data().([]float64)
	iColCount := dataset.Inputs.Shape()[1]
	outputs := dataset.Outputs.Data().([]float64)
	oColCount := dataset.Outputs.Shape()[1]

	rand.Shuffle(rowCount, func(i, j int) {
		var x, y, tmp []float64

		iStartI := iColCount * i
		iEndI := iStartI + iColCount
		iStartJ := iColCount * j
		iEndJ := iStartJ + iColCount
		x = inputs[iStartI:iEndI]
		y = inputs[iStartJ:iEndJ]
		tmp = append([]float64{}, x...)
		copy(x, y)
		copy(y, tmp)

		oStartI := oColCount * i
		oEndI := oStartI + oColCount
		oStartJ := oColCount * j
		oEndJ := oStartJ + oColCount
		x = outputs[oStartI:oEndI]
		y = outputs[oStartJ:oEndJ]
		tmp = append([]float64{}, x...)
		copy(x, y)
		copy(y, tmp)
	})

	type bucket struct {
		inputs  []float64
		outputs []float64
	}
	bucketRowCount := rowCount / k
	buckets := make([]bucket, k)
	for i := range buckets {
		iOffset := i * iColCount
		oOffset := i * oColCount
		buckets[i] = bucket{
			inputs:  inputs[iOffset : iOffset+iColCount],
			outputs: outputs[oOffset : oOffset+oColCount],
		}
	}

	tt := make(testTrainSets, k)
	for i := 0; i < k; i++ {
		testBucket := buckets[i]
		test := Dataset{
			Inputs: t.New(
				t.Of(Float),
				t.WithShape(bucketRowCount, iColCount),
				t.WithBacking(testBucket.inputs),
			),
			Outputs: t.New(
				t.Of(Float),
				t.WithShape(bucketRowCount, oColCount),
				t.WithBacking(testBucket.outputs),
			),
		}

		trainingRowCount := 0
		trainBucket := bucket{}
		for j, b := range buckets {
			if j == i {
				continue
			}

			trainBucket.inputs = append(trainBucket.inputs, b.inputs...)
			trainBucket.outputs = append(trainBucket.outputs, b.outputs...)
			trainingRowCount++
		}
		train := Dataset{
			Inputs: t.New(
				t.Of(Float),
				t.WithShape(trainingRowCount, iColCount),
				t.WithBacking(trainBucket.inputs),
			),
			Outputs: t.New(
				t.Of(Float),
				t.WithShape(trainingRowCount, oColCount),
				t.WithBacking(trainBucket.outputs),
			),
		}

		tt[i] = &testTrainSet{
			train: &train,
			test:  &test,
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
			Loss:   loss,
			Layers: p.nn.Layers,
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

	log.Fatal("oh noes")
	rmse := 0.0
	// wg := &sync.WaitGroup{}
	// mu := &sync.Mutex{}

	// wg.Add(len(dataset))
	// for _, d := range dataset {
	// 	go func(d *Data) {
	// 		expected := d.Outputs
	// 		actual := p.nn.Activate(d.Inputs...)
	// 		for j, a := range actual {
	// 			e := expected[j]
	// 			diff := a - e
	// 			mu.Lock()
	// 			rmse += diff * diff
	// 			mu.Unlock()
	// 		}
	// 		wg.Done()
	// 	}(d)
	// }
	// wg.Wait()
	// return math.Sqrt(rmse / float64(len(dataset)))
	return rmse
}

func (p *particle) calculateMeanLoss(dataset *Dataset, ridgeRegressionWeight float64) float64 {
	log.Fatal("oh noes")
	// sum := 0.0
	// for _, d := range dataset {
	// 	actualOuputs := p.nn.Activate(d.Inputs...)
	// 	err := p.fn(d.Outputs, actualOuputs)
	// 	if math.IsNaN(err) {
	// 		runtime.Breakpoint()
	// 	}
	// 	sum += err
	// }
	// loss := sum / float64(len(dataset))

	// l2Regularization := 0.0
	// for _, w := range p.nn.weights() {
	// 	l2Regularization += w * w
	// }
	// l2Regularization /= float64(p.nn.weightsAndBiasesCount())
	// l2Regularization *= ridgeRegressionWeight
	// // log.Printf("<%02d:%02d>  LF:%f L2:%f", p.swarmID, p.id, loss, l2Regularization)
	// return loss + l2Regularization
	return 0
}

func checkOk(ok bool) {
	if !ok {
		runtime.Breakpoint()
	}
}
