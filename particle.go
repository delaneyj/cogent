package cogent

import (
	"encoding/gob"
	fmt "fmt"
	"log"
	math "math"
	"math/rand"
	"os"
	"runtime"
	"strings"
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
	buckets                            DataBuckets
	testBucketIndex                    int
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

	loss := p.calculateMeanLoss(ud.testBucketIndex, ud.buckets, ud.ridgeRegressionWeight)
	return loss
}

func (p *particle) train(wg *sync.WaitGroup, iteration int, pti particleTrainingInfo, buckets DataBuckets) {
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

	kfoldLossAvg, bucketCount := 0.0, 0.0
	for testIndex := range buckets {
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
			testBucketIndex:       testIndex,
			buckets:               buckets,
		})
		kfoldLossAvg += loss
		bucketCount++
	}
	kfoldLossAvg /= bucketCount

	// for _, l := range p.nn.Layers {
	// 	log.Printf("%+v", l.WeightsAndBiases)
	// }
	// log.Printf("Iteration:%d <%d:%d> took %s. %f", iteration, p.swarmID, p.id, time.Since(start), kfoldLossAvg)

	wasSwarmBest, wasGlobalBest := p.setBest(iteration, kfoldLossAvg, pti.RidgeRegressionWeight, buckets, pti.StoreGlobalBest)
	if !wasGlobalBest && !wasSwarmBest {
		//The best don't die
		deathChance := p.r.Float64()
		if deathChance < pti.DeathRate {
			log.Printf("<%d> <Swarm%d:Particle%d> died!", iteration, p.swarmID, p.id)
			p.nn.reset(p.r, p.layersTrainingInfo, pti.WeightRange)
			randomIndex := p.r.Intn(len(buckets))
			loss := p.calculateMeanLoss(randomIndex, buckets, pti.RidgeRegressionWeight)
			p.setBest(iteration, loss, pti.RidgeRegressionWeight, buckets, pti.StoreGlobalBest)
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

//DataBuckets x
type DataBuckets []*DataBucket

//DataBucketToBuckets x
func DataBucketToBuckets(k int, dataset *DataBucket) DataBuckets {
	rowCount := dataset.RowCount()
	if rowCount < k {
		k = rowCount
	}

	// log.Printf("%+v %+v", dataset.Inputs, dataset.Outputs)

	inputs := dataset.Inputs.Data().([]float64)
	is := dataset.Inputs.Shape()
	iRowCount := is[0]
	iColCount := is[1]
	outputs := dataset.Outputs.Data().([]float64)
	rs := dataset.Outputs.Shape()
	oRowCount := rs[0]
	oColCount := rs[1]

	iTmp := make([]float64, iColCount)
	oTmp := make([]float64, oColCount)
	rand.Shuffle(rowCount, func(i, j int) {
		var x, y []float64

		iStartI := iColCount * i
		iEndI := iStartI + iColCount
		iStartJ := iColCount * j
		iEndJ := iStartJ + iColCount
		x = inputs[iStartI:iEndI]
		y = inputs[iStartJ:iEndJ]
		copy(iTmp, x)
		copy(x, y)
		copy(y, iTmp)

		oStartI := oColCount * i
		oEndI := oStartI + oColCount
		oStartJ := oColCount * j
		oEndJ := oStartJ + oColCount
		x = outputs[oStartI:oEndI]
		y = outputs[oStartJ:oEndJ]
		copy(oTmp, x)
		copy(x, y)
		copy(y, oTmp)
	})

	bucketRowCount := rowCount / k
	buckets := make(DataBuckets, k)
	iDelta := iColCount * iRowCount
	oDelta := oColCount * oRowCount
	for i := 0; i < k; i++ {
		iOffset := i * iDelta
		bucketInputs := inputs[iOffset : iOffset+iDelta]
		oOffset := i * oDelta
		bucketOutputs := outputs[oOffset : oOffset+iDelta]
		bucket := &DataBucket{
			Inputs: t.New(
				t.Of(Float),
				t.WithShape(bucketRowCount, iColCount),
				t.WithBacking(bucketInputs),
			),
			Outputs: t.New(
				t.Of(Float),
				t.WithShape(bucketRowCount, oColCount),
				t.WithBacking(bucketOutputs),
			),
		}
		buckets[i] = bucket
	}

	return buckets
}

func (p *particle) setBest(iteration int, loss float64, ridgeRegressionWeight float64, buckets DataBuckets, storeGlobalBest bool) (bool, bool) {
	p.nn.CurrentLoss = loss
	var wasSwarmBest, wasGlobalBest bool
	localBestLoss := p.nn.Best.Loss
	if loss < localBestLoss {
		blf := "max"
		if p.nn.Best.Loss != math.MaxFloat64 {
			blf = fmt.Sprintf("%0.16f", p.nn.Best.Loss)
		}
		log.Printf("<%d> Local best <Swarm%d:Particle%d> from %s->%f", iteration, p.swarmID, p.id, blf, loss)
		updatedBest := nnToPosition(loss, p.nn)
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
			log.Printf("<%d> Swarm best  <Swarm%d:Particle%d>> from %s->%f", iteration, p.swarmID, p.id, blf, loss)
			p.blackboard.Store(bestSwarmKey, updatedBest)
			wasSwarmBest = true

			res, ok = p.blackboard.Load(globalKey)
			checkOk(ok)
			bestGlobal := res.(Position)
			if loss < bestGlobal.Loss {
				blf := "max"
				if bestGlobal.Loss != math.MaxFloat64 {
					blf = fmt.Sprintf("%0.16f", bestGlobal.Loss)
				}
				log.Printf("<%d> Global best  <Swarm%d:Particle%d> from %s->%f", iteration, p.swarmID, p.id, blf, loss)
				p.blackboard.Store(globalKey, updatedBest)
				wasGlobalBest = true

				rmse := p.rmse(buckets)
				testAcc := p.nn.ClassificationAccuracy(buckets, -1)

				nodeCounts := make([]string, len(p.nn.Layers))
				for i, l := range p.nn.Layers {
					nodeCounts[i] = fmt.Sprint(l.NodeCount)
				}

				sb := strings.Builder{}
				sb.WriteString(strings.Join(nodeCounts, "x"))
				sb.WriteString(fmt.Sprintf("_KFX_%0.4f_RMSE_%0.4f_TACC%0.2f.nn", loss, rmse, 100*testAcc))

				filename := sb.String()
				// log.Printf("Iteration:%d <Swarm%d:Particle%d> New global best found", iteration, p.swarmID, p.id)

				if storeGlobalBest {
					f, err := os.Create(filename)
					checkErr(err)
					e := gob.NewEncoder(f)
					err = e.Encode(p.nn)
					checkErr(err)
					err = f.Close()
					checkErr(err)
				}
				log.Print(filename)
			}
		}
	}

	if wasGlobalBest {
		p.blackboard.Store(bestGlobalNNKey, *p.nn)
	}

	return wasSwarmBest, wasGlobalBest
}

func (p *particle) rmse(buckets DataBuckets) float64 {
	rmse, count := 0.0, 0.0
	for _, bucket := range buckets {
		expected := bucket.Outputs
		actual := p.nn.Activate(bucket.Inputs)

		// log.Printf("In rmse \nExpected:%+v\nActual:%+v", expected, actual)
		diff := must(actual.Sub(expected))
		backing := diff.Data().([]float64)

		for _, x := range backing {
			rmse += x * x
			count++
		}
	}
	rmse = math.Sqrt(rmse / count)
	return rmse
}

func (p *particle) calculateMeanLoss(testBucketIndex int, buckets DataBuckets, ridgeRegressionWeight float64) float64 {
	avgLoss, bucketCount := 0.0, 0.0
	for _, bucket := range buckets {
		expected := DenseToRows(bucket.Outputs)
		actual := DenseToRows(p.nn.Activate(bucket.Inputs))
		// log.Printf("calculateMeanLoss \nExpected:%+v\nActual:%+v", expected, actual)
		loss := p.fn(expected, actual)
		avgLoss += loss
		bucketCount++
	}
	avgLoss /= bucketCount
	l2Regularization := 0.0
	weightCount := 0.0
	for _, layer := range p.nn.Layers {
		data := layer.WeightsAndBiases.Data().([]float64)
		for _, w := range data {
			l2Regularization += w * w
			weightCount++
		}
	}
	l2Regularization /= weightCount
	l2Regularization *= ridgeRegressionWeight

	// log.Printf("<%02d:%02d>  LF:%f L2:%f", p.swarmID, p.id, loss, l2Regularization)
	return avgLoss + l2Regularization
}

func checkOk(ok bool) {
	if !ok {
		runtime.Breakpoint()
	}
}
