package cogent

import (
	fmt "fmt"
	"log"
	math "math"
	"math/rand"
	"sync"

	"github.com/dgraph-io/badger"
	"github.com/google/uuid"
	"github.com/pkg/errors"
)

type particle struct {
	id      string
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
	data.Best = &Position{
		Loss:             math.MaxFloat64,
		WeightsAndBiases: data.weights(),
	}

	return &particle{
		id:      uuid.New().String(),
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

	maxIterations := int(math.MaxInt64)
	for iteration := 0; iteration < maxIterations; iteration++ {
		checkErr(p.db.View(func(txn *badger.Txn) error {
			var x *badger.Item

			x, err := txn.Get(trainingConfigPath)
			if err != nil {
				return errors.Wrap(err, "can't get training config")
			}
			b, _ := x.Value()
			err = config.Unmarshal(b)
			if err != nil {
				return errors.Wrap(err, "can't unmarshal training config")
			}

			x, err = txn.Get(globalBestPath)
			if err != nil {
				return errors.Wrap(err, "can't get global best")
			}
			b, _ = x.Value()
			err = bestGlobal.Unmarshal(b)
			if err != nil {
				return errors.Wrap(err, "can't unmarshal global best")
			}

			x, err = txn.Get(swarmBestPath(p.swarmID))
			if err != nil {
				return errors.Wrap(err, "can't get swarm best")
			}
			b, _ = x.Value()
			err = bestSwarm.Unmarshal(b)
			if err != nil {
				return errors.Wrap(err, "can't unmarshal swarm best")
			}

			x, err = txn.Get(trainingDataPath)
			if err != nil {
				return errors.Wrap(err, "can't get training data")
			}
			b, _ = x.Value()
			err = trainingData.Unmarshal(b)
			if err != nil {
				return errors.Wrap(err, "can't unmarshal training data")
			}

			return nil
		}))

		maxIterations = int(config.MaxIterations)
		if bestGlobal.Loss <= config.TargetAccuracy {
			log.Printf("early stop at %d iterations", iteration)
			break
		}

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

		err := badger.ErrConflict

		globalFound := false
		var from, to float64
		for err == badger.ErrConflict {
			err = db.Update(func(txn *badger.Txn) error {
				var swarmBest Position
				swarmBestPath := []byte(fmt.Sprintf(swarmBestFormat, p.swarmID))
				i, err := txn.Get(swarmBestPath)
				if err != nil {
					return errors.Wrap(err, "can't get swarm best")
				}
				x, err := i.Value()
				if err != nil {
					return errors.Wrap(err, "can't get value swarm best")
				}
				err = swarmBest.Unmarshal(x)
				if err != nil {
					return errors.Wrap(err, "can't unmarshal swarm best")
				}

				if loss < swarmBest.Loss {
					updatedBest := Position{
						Loss:             loss,
						WeightsAndBiases: bestPositions,
					}
					updatedBestBytes, err := updatedBest.Marshal()
					if err != nil {
						return errors.Wrap(err, "can't marshal new best")
					}

					err = txn.Set(swarmBestPath, updatedBestBytes)
					if err != nil {
						return errors.Wrap(err, "can't set swarm best")
					}

					var globalBest Position
					i, err := txn.Get(globalBestPath)
					if err != nil {
						return errors.Wrap(err, "can't get global best")
					}
					x, err := i.Value()
					if err != nil {
						return errors.Wrap(err, "can't get value global best")
					}
					err = globalBest.Unmarshal(x)
					if err != nil {
						return errors.Wrap(err, "can't unmarshal global best")
					}

					if loss < globalBest.Loss {
						err := txn.Set(globalBestPath, updatedBestBytes)
						if err != nil {
							return errors.Wrap(err, "can't set global best")
						}

						globalFound = true
						from = globalBest.Loss
						to = loss
					}
				}
				return nil
			})
		}

		if globalFound {
			log.Printf("New global best found %f->%f from %s", from, to, p.id)
		}
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
