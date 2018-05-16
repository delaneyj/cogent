package cogent

import (
	"fmt"
	"math"
	"os"
	"sync"

	"github.com/dgraph-io/badger"
	"github.com/google/uuid"
	"github.com/pkg/errors"
)

//MultiSwarm x
type MultiSwarm struct {
	folderName    string
	particleCount int
	db            *badger.DB
	swarms        []*swarm
}

type swarm struct {
	id        []byte
	particles []*particle
}

//NewMultiSwarm x
func NewMultiSwarm(config MultiSwarmConfiguration, trainingConfig TrainingConfiguration) *MultiSwarm {
	id := uuid.New().String()
	folderName := fmt.Sprintf("mso-%s", id)

	err := os.RemoveAll(folderName)
	checkErr(err)

	opts := badger.DefaultOptions
	opts.Dir = folderName
	opts.ValueDir = folderName
	db, err := badger.Open(opts)
	checkErr(err)

	if config.SwarmCount <= 0 {
		panic("No swarm count in config")
	}

	nnConfigBytes, err := config.NeuralNetworkConfiguration.Marshal()
	trainingConfigBytes, err := trainingConfig.Marshal()

	checkErr(db.Update(func(txn *badger.Txn) error {
		if err := txn.Set(neuralNetworkConfigPath, nnConfigBytes); err != nil {
			return errors.Wrap(err, "can't set nn config")
		}

		if err := txn.Set(trainingConfigPath, trainingConfigBytes); err != nil {
			return errors.Wrap(err, "can't set training config")
		}
		return nil
	}))

	ms := MultiSwarm{
		folderName:    folderName,
		db:            db,
		swarms:        make([]*swarm, config.SwarmCount),
		particleCount: int(config.SwarmCount * config.ParticleCount),
	}
	for i := range ms.swarms {
		uuid := uuid.New()
		swarmID, err := uuid.MarshalBinary()
		checkErr(err)

		s := &swarm{
			id:        swarmID,
			particles: make([]*particle, config.ParticleCount),
		}

		for i := 0; i < int(config.ParticleCount); i++ {
			p := newParticle(swarmID, db)
			s.particles[i] = p
		}
		ms.swarms[i] = s
	}

	p := newParticle([]byte{}, db)
	best := Position{
		Loss:             math.MaxFloat64,
		WeightsAndBiases: p.data.weights(),
	}
	b, err := best.Marshal()
	checkErr(err)

	checkErr(db.Update(func(txn *badger.Txn) error {
		if err := txn.Set(globalBestPath, b); err != nil {
			return errors.Wrap(err, "can't set global best")
		}

		for _, s := range ms.swarms {
			p.data.reset(trainingConfig.WeightRange)
			copy(best.WeightsAndBiases, p.data.weights())
			b, err = best.Marshal()

			sp := swarmBestPath(s.id)
			if err := txn.Set(sp, b); err != nil {
				return errors.Wrap(err, "can't set training config")
			}
		}

		return nil
	}))
	return &ms
}

func swarmBestPath(swarmID []byte) []byte {
	return []byte(fmt.Sprintf(swarmBestFormat, swarmID))
}

//Close x
func (ms *MultiSwarm) Close() {
	checkErr(ms.db.Close())
	checkErr(os.RemoveAll(ms.folderName))
}

//Train x
func (ms *MultiSwarm) Train(training *TrainingData) {
	{
		b, err := training.Marshal()
		checkErr(err)
		checkErr(ms.db.Update(func(txn *badger.Txn) error {
			err := txn.Set(trainingDataPath, b)
			if err != nil {
				return errors.Wrap(err, "can't set training data")
			}
			return nil
		}))
	}

	wg := &sync.WaitGroup{}
	wg.Add(ms.particleCount)
	for _, s := range ms.swarms {
		for _, p := range s.particles {
			go p.train(wg)
		}
	}
	wg.Wait()
}

//ClassificationAccuracy x
func (ms *MultiSwarm) ClassificationAccuracy(testData ...*Data) float64 {
	var globalBest *NeuralNetworkData

	for _, s := range ms.swarms {
		for _, p := range s.particles {
			if globalBest == nil || p.data.Best.Loss < globalBest.Best.Loss {
				globalBest = p.data
			}
		}
	}
	return globalBest.classificationAccuracy(testData)
}

//Predict x
func (ms *MultiSwarm) Predict(inputs ...float64) []float64 {
	var globalBest *NeuralNetworkData
	for _, s := range ms.swarms {
		for _, p := range s.particles {
			if globalBest == nil || p.data.Best.Loss < globalBest.Best.Loss {
				globalBest = p.data
			}
		}
	}
	return globalBest.activate(inputs...)
}
