package cogent

import "math"

type activationFunction func([]float64) []float64

var activations = map[string]activationFunction{
	"hyperbolicTangent": func(values []float64) []float64 {
		result := make([]float64, len(values))
		for i, x := range values {
			switch {
			case x < -20:
				result[i] = -1
			case x > 20:
				result[i] = 1
			default:
				result[i] = math.Tanh(x)
			}

		}
		return result
	},
	"softmax": func(values []float64) []float64 {
		// does all output nodes at once so scale doesn't have to be re-computed each time
		// determine max output sum
		var max float64
		for _, x := range values {
			if x > max {
				max = x
			}
		}

		// determine scaling factor -- sum of exp(each val - max)
		scale := 0.0
		for _, x := range values {
			scale += math.Exp(x - max)
		}

		result := make([]float64, len(values))
		for i, x := range values {
			result[i] = math.Exp(x-max) / scale
		}

		return result // now scaled so that xi sum to 1.0
	},
}
