package cogent

import (
	fmt "fmt"
	"log"
	math "math"
	"math/rand"
	"sync"

	"github.com/dgraph-io/badger"
)

type particle struct {
	fn      lossFn
	data    *NeuralNetworkData
	db      *badger.DB
	swarmID []byte
}

//NewNeuralNetworkConfiguration x
func NewNeuralNetworkConfiguration(inputCount int, lc ...*LayerConfig) *NeuralNetworkConfiguration {
	nnc := NeuralNetworkConfiguration{
		Loss:         Cross,
		InputCount:   uint32(inputCount),
		LayerConfigs: lc,
	}
	return &nnc
}

func newParticle(swarmID []byte, db *badger.DB) *particle {
	var nnConfig NeuralNetworkConfiguration
	var trainingConfig TrainingConfiguration
	checkErr(db.View(func(txn *badger.Txn) error {
		i, err := txn.Get(neuralNetworkConfigPath)
		checkErr(err)
		b, err := i.Value()
		checkErr(err)
		err = nnConfig.Unmarshal(b)
		checkErr(err)

		i, err = txn.Get(trainingConfigPath)
		checkErr(err)
		b, err = i.Value()
		checkErr(err)
		err = trainingConfig.Unmarshal(b)
		checkErr(err)

		return nil
	}))

	fn := lossFns[nnConfig.Loss]
	if fn == nil {
		log.Fatalf("Invalid loss type '%s'", nnConfig.Loss)
	}
	data := NeuralNetworkData{
		Layers:      make([]*LayerData, len(nnConfig.LayerConfigs)),
		CurrentLoss: math.MaxFloat64,
		Best:        &Position{Loss: math.MaxFloat64},
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
		l.reset(trainingConfig.WeightRange)
		data.Layers[i] = &l
		previousLayerCount = layerConfig.NodeCount
	}

	return &particle{
		fn:      fn,
		data:    &data,
		db:      db,
		swarmID: swarmID,
	}
}

func (p *particle) train(wg *sync.WaitGroup) {
	var trainingData TrainingData
	var bestSwarm, bestGlobal Position
	var config TrainingConfiguration
	checkErr(p.db.View(func(txn *badger.Txn) error {
		var x *badger.Item

		x, err := txn.Get(trainingConfigPath)
		checkErr(err)
		b, _ := x.Value()
		err = config.Unmarshal(b)
		checkErr(err)

		x, err = txn.Get(globalBestPath)
		checkErr(err)
		b, _ = x.Value()
		err = bestGlobal.Unmarshal(b)
		checkErr(err)

		swarmBestPath := fmt.Sprintf(swarmBestFormat, p.swarmID)
		x, err = txn.Get([]byte(swarmBestPath))
		checkErr(err)
		b, _ = x.Value()
		err = bestSwarm.Unmarshal(b)
		checkErr(err)

		x, err = txn.Get(trainingDataPath)
		checkErr(err)
		b, _ = x.Value()
		err = trainingData.Unmarshal(b)
		checkErr(err)

		return nil
	}))

	flatArrayIndex := 0
	// Compute new velocity.  Depends on old velocity, best position of parrticle, and best position of any particle
	for _, l := range p.data.Layers {
		for i, currentLocalWeight := range l.WeightsAndBiases {
			bestGlobalPosition := bestGlobal.WeightsAndBiases[flatArrayIndex]
			bestSwarmPosition := bestGlobal.WeightsAndBiases[flatArrayIndex]
			bestLocalPosition := p.data.Best.WeightsAndBiases[flatArrayIndex]

			currentLocalVelocity := l.Velocities[i]

			oldVelocityFactor := config.InertiaWeight * currentLocalVelocity

			localRandomness := rand.Float64()
			bestLocationDelta := bestLocalPosition - currentLocalWeight
			localPositionFactor := config.CognitiveWeight * localRandomness * bestLocationDelta

			swarmRandomness := rand.Float64()
			bestSwarmlDelta := bestSwarmPosition - currentLocalWeight
			swarmPositionFactor := config.SocialWeight * swarmRandomness * bestSwarmlDelta

			globalRandomness := rand.Float64()
			bestGlobalDelta := bestGlobalPosition - currentLocalWeight
			globalPositionFactor := config.GlobalWeight * globalRandomness * bestGlobalDelta

			revisedVelocity := oldVelocityFactor + localPositionFactor + swarmPositionFactor + globalPositionFactor
			l.Velocities[i] = revisedVelocity

			flatArrayIndex++
		}
	}

	flatArrayIndex = 0
	for _, l := range p.data.Layers {
		for i, w := range l.WeightsAndBiases {
			v := l.Velocities[i]
			revisedPosition := w + v
			wr := config.WeightRange
			clamped := math.Max(-wr, math.Min(wr, revisedPosition)) // restriction
			decayed := clamped * (1 + config.WeightDecayRate)       // decay (large weights tend to overfit)

			l.WeightsAndBiases[i] = decayed
			flatArrayIndex++
		}
	}

	p.checkAndSetLoss(p.db, trainingData.Examples)

	deathChance := rand.Float64()
	if deathChance < config.ProbablityOfDeath {
		p.data.reset(config.WeightRange)
		p.checkAndSetLoss(p.db, trainingData.Examples)
	}

	wg.Done()
}

func (p *particle) checkAndSetLoss(db *badger.DB, data []*Data) float64 {
	loss := p.calculateMeanLoss(data)

	p.data.CurrentLoss = loss
	if loss < p.data.Best.Loss {
		bestPositions := p.data.weights()

		p.data.Best.Loss = loss
		p.data.Best.WeightsAndBiases = bestPositions

		checkErr(db.Update(func(txn *badger.Txn) error {
			var swarmBest, globalBest Position
			swarmBestPath := []byte(fmt.Sprintf(swarmBestFormat, p.swarmID))
			i, err := txn.Get(swarmBestPath)
			checkErr(err)
			x, err := i.Value()
			checkErr(swarmBest.Unmarshal(x))

			if loss < swarmBest.Loss {
				b, err := p.data.Best.Marshal()
				checkErr(err)

				err = txn.Set(swarmBestPath, b)
				checkErr(err)

				i, err := txn.Get(globalBestPath)
				checkErr(err)
				x, err := i.Value()
				checkErr(globalBest.Unmarshal(x))

				if loss < globalBest.Loss {
					err := txn.Set(globalBestPath, b)
					checkErr(err)
				}
			}
			return nil
		}))
	}
	return loss
}

func (p *particle) calculateMeanLoss(data []*Data) float64 {
	sum := 0.0
	for _, d := range data {
		actualOuputs := p.data.activate(d.Inputs...)
		err := p.fn(d.Outputs, actualOuputs)
		sum += err
	}
	loss := sum / float64(len(data))
	return loss
}
