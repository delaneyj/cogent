package cogent

import (
	fmt "fmt"
	"log"
	math "math"
	"math/rand"
	"sync"
)

type particle struct {
	id         int
	fn         lossFn
	data       *NeuralNetworkData
	blackboard *blackboard
	swarmID    int
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

func newParticle(swarmID, particleID int, blackboard *blackboard) *particle {
	var nnConfig NeuralNetworkConfiguration
	var trainingConfig TrainingConfiguration

	blackboard.mutex.RLock()
	nnConfig = blackboard.nnConfig
	trainingConfig = blackboard.trainingConfig
	blackboard.mutex.RUnlock()

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
		swarmID:    swarmID,
		id:         particleID,
		fn:         fn,
		data:       &data,
		blackboard: blackboard,
	}
}

func (p *particle) train(wg *sync.WaitGroup) {

	maxIterations := int(math.MaxInt64)
	for iteration := 0; iteration < maxIterations; iteration++ {
		p.blackboard.mutex.RLock()
		trainingConfig := p.blackboard.trainingConfig
		bestGlobal := p.blackboard.best[globalKey]
		bestSwarmKey := fmt.Sprintf(swarmKeyFormat, p.swarmID)
		bestSwarm := p.blackboard.best[bestSwarmKey]
		p.blackboard.mutex.RUnlock()

		maxIterations = int(trainingConfig.MaxIterations)
		if bestGlobal.Loss <= trainingConfig.TargetAccuracy {
			// log.Printf("early stop at %d iterations", iteration)
			break
		}

		flatArrayIndex := 0
		// Compute new velocity.  Depends on old velocity, best position of parrticle, and best position of any particle
		for _, l := range p.data.Layers {
			for i, currentLocalWeight := range l.WeightsAndBiases {
				bestGlobalPosition := bestGlobal.WeightsAndBiases[flatArrayIndex]
				bestSwarmPosition := bestSwarm.WeightsAndBiases[flatArrayIndex]
				bestLocalPosition := p.data.Best.WeightsAndBiases[flatArrayIndex]

				currentLocalVelocity := l.Velocities[i]

				oldVelocityFactor := trainingConfig.InertiaWeight * currentLocalVelocity

				localRandomness := rand.Float64()
				bestLocationDelta := bestLocalPosition - currentLocalWeight
				localPositionFactor := trainingConfig.CognitiveWeight * localRandomness * bestLocationDelta

				swarmRandomness := rand.Float64()
				bestSwarmlDelta := bestSwarmPosition - currentLocalWeight
				swarmPositionFactor := trainingConfig.SocialWeight * swarmRandomness * bestSwarmlDelta

				globalRandomness := rand.Float64()
				bestGlobalDelta := bestGlobalPosition - currentLocalWeight
				globalPositionFactor := trainingConfig.GlobalWeight * globalRandomness * bestGlobalDelta

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
				wr := trainingConfig.WeightRange
				clamped := math.Max(-wr, math.Min(wr, revisedPosition))   // restriction
				decayed := clamped * (1 + trainingConfig.WeightDecayRate) // decay (large weights tend to overfit)

				l.WeightsAndBiases[i] = decayed
				flatArrayIndex++
			}
		}

		p.checkAndSetLoss()

		deathChance := rand.Float64()
		if deathChance < trainingConfig.ProbablityOfDeath {
			p.data.reset(trainingConfig.WeightRange)
			p.checkAndSetLoss()
		}
	}

	wg.Done()
}

func (p *particle) checkAndSetLoss() float64 {
	loss := p.calculateMeanLoss()

	p.data.CurrentLoss = loss
	if loss < p.data.Best.Loss {
		updatedBest := Position{
			Loss:             loss,
			WeightsAndBiases: p.data.weights(),
		}

		p.data.Best = &updatedBest

		p.blackboard.mutex.Lock()
		defer p.blackboard.mutex.Unlock()

		swarmKey := fmt.Sprintf(swarmKeyFormat, p.swarmID)
		swarmBest := p.blackboard.best[swarmKey]

		if loss < swarmBest.Loss {
			p.blackboard.best[swarmKey] = updatedBest
			globalBest := p.blackboard.best[globalKey]

			globalLoss := globalBest.Loss
			if loss < globalLoss {
				p.blackboard.best[globalKey] = updatedBest

				log.Printf("<%d:%d> from %f->%f", p.swarmID, p.id, globalLoss, loss)
			}
		}
	}
	return loss
}

func (p *particle) calculateMeanLoss() float64 {
	p.blackboard.mutex.RLock()
	data := p.blackboard.trainingData.Examples
	p.blackboard.mutex.RUnlock()

	sum := 0.0
	for _, d := range data {
		actualOuputs := p.data.Activate(d.Inputs...)
		err := p.fn(d.Outputs, actualOuputs)
		sum += err
	}
	loss := sum / float64(len(data))
	return loss
}
