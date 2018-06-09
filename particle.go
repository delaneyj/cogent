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
	// lastLayerIndex := len(nnConfig.LayerConfigs) - 1

	for i, layerConfig := range nnConfig.LayerConfigs {
		colCount = layerConfig.NodeCount
		// if i != lastLayerIndex {
		// 	colCount++
		// }

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

		// log.Printf("Weights Tensor: %+v", l.WeightsAndBiases)
		nn.Layers[i] = l
		rowCount = colCount + 1
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
	TargetAccuracy        float64
	InertialWeight        float64
	CognitiveWeight       float64
	SocialWeight          float64
	GlobalWeight          float64
	WeightRange           float64
	WeightDecayRate       float64
	DeathRate             float64
	RidgeRegressionWeight float64
	KFolds                int
	StoreGlobalBest       bool
}

func (p *particle) train(wg *sync.WaitGroup, iteration int, pti particleTrainingInfo, ttSets []*testTrainSet) {
	// start := time.Now()
	res, ok := p.blackboard.Load(globalKey)
	checkOk(ok)
	bestGlobal := res.(Position)

	bestSwarmKey := fmt.Sprintf(swarmKeyFormat, p.swarmID)
	res, ok = p.blackboard.Load(bestSwarmKey)
	checkOk(ok)
	bestSwarm := res.(Position)

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
				oldVelocityFactor := must(currentLocalVelocity.MulScalar(pti.InertialWeight, true))

				localRandomness := t.NewDense(
					Float,
					l.WeightsAndBiases.Shape(),
					t.WithBacking(t.Random(Float, l.WeightsAndBiases.DataSize())),
				)
				bestLocalDelta := must(bestLocal.Sub(currentLocal))
				localPositionFactor := must(must(localRandomness.MulScalar(pti.CognitiveWeight, true)).Mul(bestLocalDelta))

				bestSwarm := bestSwarm.Layers[i].WeightsAndBiases
				swarmRandomness := t.NewDense(
					Float,
					l.WeightsAndBiases.Shape(),
					t.WithBacking(t.Random(Float, l.WeightsAndBiases.DataSize())),
				)
				bestSwarmlDelta := must(bestSwarm.Sub(currentLocal))
				swarmPositionFactor := must(must(swarmRandomness.MulScalar(pti.SocialWeight, true)).Mul(bestSwarmlDelta))

				bestGlobal := bestGlobal.Layers[i].WeightsAndBiases
				globalRandomness := t.NewDense(
					Float,
					l.WeightsAndBiases.Shape(),
					t.WithBacking(t.Random(Float, l.WeightsAndBiases.DataSize())),
				)
				bestGlobalDelta := must(bestGlobal.Sub(currentLocal))
				globalPositionFactor := must(must(globalRandomness.MulScalar(pti.GlobalWeight, true)).Mul(bestGlobalDelta))

				revisedVelocity := must(
					must(
						must(
							must(
								oldVelocityFactor.Add(localPositionFactor),
							).Add(swarmPositionFactor),
						).Add(globalPositionFactor),
					).AddScalar(1-pti.WeightDecayRate, true),
				)
				// log.Printf("Velocities were\n%+v\nNow\n%+v", l.Velocities, revisedVelocity)
				revisedVelocity.CopyTo(l.Velocities)
			}

			wr := pti.WeightRange
			for _, l := range p.nn.Layers {
				revisedPosition := must(l.WeightsAndBiases.Add(l.Velocities))
				data := revisedPosition.Data().([]float64)
				for i, w := range data {
					clamped := math.Max(-wr, math.Min(wr, w)) // restriction
					data[i] = clamped
				}

				// log.Printf("Weight were\n%+v\nNow\n%+v", l.WeightsAndBiases, revisedPosition)
				revisedPosition.CopyTo(l.WeightsAndBiases)
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
	// log.Printf("<%d:%d> took %s. %f", p.swarmID, p.id, time.Since(start), kfoldLossAvg)

	wasGlobalBest := p.setBest(iteration, kfoldLossAvg, pti.RidgeRegressionWeight)
	if wasGlobalBest {
		rmse := p.rmse(pti.Dataset)
		testAcc := p.nn.ClassificationAccuracy(pti.Dataset)
		filename := fmt.Sprintf("RMSE_%0.8f_KFX_%0.8f_TACC%0.2f.nn", rmse, kfoldLossAvg, 100*testAcc)
		log.Printf("Iteration:%d <%d:%d> New global best found", iteration, p.swarmID, p.id)
		log.Print(filename)
		if pti.StoreGlobalBest {
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
		// log.Printf("<%d:%d:%d> died!", iteration, p.swarmID, p.id)
		p.nn.reset(pti.WeightRange)
		index := rand.Intn(len(ttSets))

		loss := p.calculateMeanLoss(ttSets[index].train, pti.RidgeRegressionWeight)
		p.setBest(iteration, loss, pti.RidgeRegressionWeight)
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
	iDelta := iColCount * bucketRowCount
	oDelta := oColCount * bucketRowCount

	for i := range buckets {
		iOffset := i * iDelta
		oOffset := i * oDelta
		buckets[i] = bucket{
			inputs:  inputs[iOffset : iOffset+iDelta],
			outputs: outputs[oOffset : oOffset+oDelta],
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
			trainingRowCount += len(b.inputs) / iColCount
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

		// log.Printf("ti%+v to%+v", test.Inputs, test.Outputs)

		tt[i] = &testTrainSet{
			train: &train,
			test:  &test,
		}
	}

	return tt
}

func (p *particle) setBest(iteration int, loss float64, ridgeRegressionWeight float64) bool {
	p.nn.CurrentLoss = loss
	wasGlobalBest := false
	localBestLoss := p.nn.Best.Loss
	if loss < localBestLoss {
		// blf := "max"
		// if p.nn.Best.Loss != math.MaxFloat64 {
		// 	blf = fmt.Sprintf("%0.16f", p.nn.Best.Loss)
		// }
		// log.Printf("<%d> Local best <%d:%d> from %s->%f", iteration, p.swarmID, p.id, blf, loss)
		updatedBest := nnToPosition(loss, p.nn)
		p.nn.Best = updatedBest

		bestSwarmKey := fmt.Sprintf(swarmKeyFormat, p.swarmID)
		res, ok := p.blackboard.Load(bestSwarmKey)
		checkOk(ok)
		bestSwarm := res.(Position)

		if loss < bestSwarm.Loss {
			// blf = "max"
			// if bestSwarm.Loss != math.MaxFloat64 {
			// 	blf = fmt.Sprintf("%0.16f", bestSwarm.Loss)
			// }
			// log.Printf("<%d> Swarm best <%d:%d> from %s->%f", iteration, p.swarmID, p.id, blf, loss)
			p.blackboard.Store(bestSwarmKey, updatedBest)

			res, ok = p.blackboard.Load(globalKey)
			checkOk(ok)
			bestGlobal := res.(Position)
			if loss < bestGlobal.Loss {
				// blf := "max"
				// if bestGlobal.Loss != math.MaxFloat64 {
				// 	blf = fmt.Sprintf("%0.16f", bestGlobal.Loss)
				// }
				// log.Printf("<%d> Global best <%d:%d> from %s->%f", iteration, p.swarmID, p.id, blf, loss)
				p.blackboard.Store(globalKey, updatedBest)
				wasGlobalBest = true
			}
		}
	}

	if wasGlobalBest {
		p.blackboard.Store(bestGlobalNNKey, *p.nn)
	}

	return wasGlobalBest
}

func (p *particle) rmse(dataset *Dataset) float64 {
	expected := dataset.Outputs
	actual := p.nn.Activate(dataset.Inputs)

	// log.Printf("In rmse \nExpected:%+v\nActual:%+v", expected, actual)
	diff := must(actual.Sub(expected))
	backing := diff.Data().([]float64)

	rmse, count := 0.0, 0.0
	for _, x := range backing {
		rmse += x * x
		count++
	}
	rmse = math.Sqrt(rmse / count)
	return rmse
}

func (p *particle) calculateMeanLoss(dataset *Dataset, ridgeRegressionWeight float64) float64 {
	expected := dataset.Outputs
	actual := p.nn.Activate(dataset.Inputs)
	// log.Printf("calculateMeanLoss \nExpected:%+v\nActual:%+v", expected, actual)
	loss := p.fn(expected, actual)

	l2Regularization := 0.0
	count := 0.0
	for _, w := range p.nn.weights() {
		l2Regularization += w * w
		count++
	}
	l2Regularization /= count
	l2Regularization *= ridgeRegressionWeight

	// log.Printf("<%02d:%02d>  LF:%f L2:%f", p.swarmID, p.id, loss, l2Regularization)
	return loss + l2Regularization
}

func checkOk(ok bool) {
	if !ok {
		runtime.Breakpoint()
	}
}
