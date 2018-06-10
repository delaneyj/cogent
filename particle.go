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

type layerTrainingInfo struct {
	Velocities *t.Dense
	Jitter     *t.Dense
}

type particle struct {
	id                 int
	fn                 lossFn
	nn                 *NeuralNetwork
	blackboard         *sync.Map
	swarmID            int
	r                  *rand.Rand
	layersTrainingInfo []*layerTrainingInfo
}

//NewNeuralNetworkConfiguration x
func NewNeuralNetworkConfiguration(inputCount int, lc ...LayerConfig) *NeuralNetworkConfiguration {
	nnc := NeuralNetworkConfiguration{
		Loss:         CrossLoss,
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

	seed := int64(uint(swarmID) << uint(particleID))
	r := rand.New(rand.NewSource(seed))

	ltis := make([]*layerTrainingInfo, len(nnConfig.LayerConfigs))
	for i, layerConfig := range nnConfig.LayerConfigs {
		colCount = layerConfig.NodeCount
		if i != lastLayerIndex {
			colCount++
		}

		lti := &layerTrainingInfo{
			Velocities: t.New(
				t.Of(t.Float64),
				t.WithShape(rowCount, colCount),
			),
			Jitter: t.New(
				t.Of(Float),
				t.WithShape(rowCount, colCount),
			),
		}
		ltis[i] = lti
		l := LayerData{
			NodeCount: layerConfig.NodeCount,
			WeightsAndBiases: t.New(
				t.Of(Float),
				t.WithShape(rowCount, colCount),
			),
			Activation: layerConfig.Activation,
		}

		l.reset(r, lti, weightRange)

		// log.Printf("Weights Tensor: %+v", l.WeightsAndBiases)
		nn.Layers[i] = l
		rowCount = colCount
	}
	nn.Best = Position{
		Loss:   math.MaxFloat64,
		Layers: nn.Layers,
	}

	return &particle{
		swarmID:            swarmID,
		id:                 particleID,
		fn:                 fn,
		nn:                 &nn,
		blackboard:         blackboard,
		r:                  r,
		layersTrainingInfo: ltis,
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

type updateData struct {
	p                                  *particle
	bestSwarm, bestGlobal              *Position
	inertialWeight, cognitiveWeight    float64
	socialWeight, globalWeight         float64
	weightRange, ridgeRegressionWeight float64
	ttSet                              *testTrainSet
	lossCh                             chan float64
}

func updatePositionsAndVelocities(ud updateData) float64 {
	p := ud.p
	bestSwarm := ud.bestSwarm
	bestGlobal := ud.bestGlobal
	for i, l := range p.nn.Layers {
		currentLocal := l.WeightsAndBiases
		lti := p.layersTrainingInfo[i]

		bestLocal := p.nn.Best.Layers[i].WeightsAndBiases
		currentLocalVelocity := lti.Velocities
		oldVelocityFactor := must(currentLocalVelocity.MulScalar(ud.inertialWeight, true))

		bestLocalDelta := must(bestLocal.Sub(currentLocal))
		localPositionFactor := must(must(lti.Jitter.MulScalar(ud.cognitiveWeight, true)).Mul(bestLocalDelta))

		bestSwarm := bestSwarm.Layers[i].WeightsAndBiases
		bestSwarmlDelta := must(bestSwarm.Sub(currentLocal))
		swarmPositionFactor := must(must(lti.Jitter.MulScalar(ud.socialWeight, true)).Mul(bestSwarmlDelta))

		bestGlobal := bestGlobal.Layers[i].WeightsAndBiases
		bestGlobalDelta := must(bestGlobal.Sub(currentLocal))
		globalPositionFactor := must(must(lti.Jitter.MulScalar(ud.globalWeight, true)).Mul(bestGlobalDelta))

		revisedVelocity := must(
			must(
				must(
					oldVelocityFactor.Add(localPositionFactor),
				).Add(swarmPositionFactor),
			).Add(globalPositionFactor),
		)
		// log.Printf("Layer:%d velocities were\n%+v\nNow\n%+v", i, l.Velocities, revisedVelocity)
		revisedVelocity.CopyTo(lti.Velocities)
	}

	for i, l := range p.nn.Layers {
		lti := p.layersTrainingInfo[i]
		revisedPosition := must(l.WeightsAndBiases.Add(lti.Velocities))
		data := revisedPosition.Data().([]float64)
		for i, w := range data {
			clamped := math.Max(-ud.weightRange, math.Min(ud.weightRange, w)) // restriction
			data[i] = clamped
		}

		// log.Printf("Layer:%d weights were\n%+v\nNow\n%+v", i, l.WeightsAndBiases, revisedPosition)
		revisedPosition.CopyTo(l.WeightsAndBiases)
	}

	loss := p.calculateMeanLoss(ud.ttSet.train, ud.ridgeRegressionWeight)
	return loss
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

	for i := range p.nn.Layers {
		lti := p.layersTrainingInfo[i]
		fillTensorWithRandom(p.r, lti.Jitter, 1, pti.WeightRange)
	}

	kfoldLossAvg := 0.0
	for _, ttSet := range ttSets {
		loss := updatePositionsAndVelocities(updateData{
			p:                     p,
			bestSwarm:             &bestSwarm,
			bestGlobal:            &bestGlobal,
			inertialWeight:        pti.InertialWeight,
			cognitiveWeight:       pti.CognitiveWeight,
			socialWeight:          pti.SocialWeight,
			globalWeight:          pti.GlobalWeight,
			weightRange:           pti.WeightRange,
			ridgeRegressionWeight: pti.RidgeRegressionWeight,
			ttSet: ttSet,
		})
		kfoldLossAvg += loss
	}
	kfoldLossAvg /= float64(len(ttSets))

	// for _, l := range p.nn.Layers {
	// 	log.Printf("%+v", l.WeightsAndBiases)
	// }
	// log.Printf("Iteration:%d <%d:%d> took %s. %f", iteration, p.swarmID, p.id, time.Since(start), kfoldLossAvg)

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
	} else {
		//The best don't die
		deathChance := p.r.Float64()
		if deathChance < pti.DeathRate {
			// log.Printf("<%d:%d:%d> died!", iteration, p.swarmID, p.id)
			p.nn.reset(p.r, p.layersTrainingInfo, pti.WeightRange)
			index := p.r.Intn(len(ttSets))

			loss := p.calculateMeanLoss(ttSets[index].train, pti.RidgeRegressionWeight)
			p.setBest(iteration, loss, pti.RidgeRegressionWeight)
		}

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
	rowCount := dataset.RowCount()
	if rowCount < k {
		k = rowCount
	}

	// log.Printf("%+v %+v", dataset.Inputs, dataset.Outputs)

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
	expected := DenseToRows(dataset.Outputs)
	actual := DenseToRows(p.nn.Activate(dataset.Inputs))
	// log.Printf("calculateMeanLoss \nExpected:%+v\nActual:%+v", expected, actual)
	loss := p.fn(expected, actual)
	l2Regularization := 0.0
	count := 0.0
	for _, layer := range p.nn.Layers {
		data := layer.WeightsAndBiases.Data().([]float64)
		for _, w := range data {
			l2Regularization += w * w
			count++
		}
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
