package cogent

import (
	"log"
	"math/rand"
)

//Particle x
type Particle struct {
	n                Network
	velocities, best []float64
}

//ParticleSwarmOptimization x
type ParticleSwarmOptimization struct {
	particles []Particle
	best      []float64
}

//NewParticleSwarmOptimization x
func NewParticleSwarmOptimization() *ParticleSwarmOptimization {
	particles := []Particle{}

	return &ParticleSwarmOptimization{
		particles: particles,
		best:      make([]float64, len(particles)),
	}
}

//Update x
func (pso *ParticleSwarmOptimization) Update(inputs, outputs []float64) {
	for i, p := range pso.particles {
		p.n.Activate(inputs)

		fitness, err := p.n.Error(outputs)
		if err != nil {
			log.Fatal(err)
		}

		if fitness > p.best[i] {
			p.best[i] = fitness

			if fitness > pso.best[i] {
				pso.best[i] = fitness
			}
		}
	}

	c1 := 2.0
	c2 := 2.0
	for _, p := range pso.particles {
		currentPosition := p.n.WeightsAndBias()
		updatedWeights := make([]float64, len(outputs))

		for i, v := range p.velocities {
			globalBest := pso.best[i]
			localBest := p.best[i]
			present := currentPosition[i]
			c1Result := c1 * rand.Float64() * (localBest - present)
			c2Result := c2 * rand.Float64() * (globalBest - present)
			updatedVelocity := v + c1Result + c2Result
			updatedPosition := present + updatedVelocity

			p.velocities[i] = updatedVelocity
			updatedWeights[i] = updatedPosition
		}

		p.n.UpdateWeightsAndBias(updatedWeights)
	}
}
