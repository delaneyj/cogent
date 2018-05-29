package cogent

import "math"

type activationFunction func(values []float64) []float64

var activations = map[ActivationMode]activationFunction{
	Identity: func(values []float64) []float64 {
		return values
	},
	BinaryStep: func(values []float64) []float64 {
		result := make([]float64, len(values))
		for i, x := range values {
			if x > 0 {
				result[i] = 1
			}
		}
		return result
	},
	Sigmoid: func(values []float64) []float64 {
		result := make([]float64, len(values))
		for i, x := range values {
			result[i] = 1 / (1 + math.Exp(-x))
		}
		return result
	},
	HyperbolicTangent: func(values []float64) []float64 {
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
	ArcTan: func(values []float64) []float64 {
		result := make([]float64, len(values))
		for i, x := range values {
			result[i] = math.Atan(x)
		}
		return result
	},
	Softsign: func(values []float64) []float64 {
		result := make([]float64, len(values))
		for i, x := range values {
			result[i] = x / (1 + math.Abs(x))
		}
		return result
	},
	ISRU: func(values []float64) []float64 {
		result := make([]float64, len(values))
		for i, x := range values {
			result[i] = x / math.Sqrt(1+x*x)
		}
		return result
	},
	ReLU: func(values []float64) []float64 {
		result := make([]float64, len(values))
		for i, x := range values {
			if x > 0 {
				result[i] = x
			}
		}
		return result
	},
	LeakyReLU: func(values []float64) []float64 {
		result := make([]float64, len(values))
		for i, x := range values {
			if x < 0 {
				result[i] = x * 0.01
			} else {
				result[i] = x
			}
		}
		return result
	},
	ELU: func(values []float64) []float64 {
		result := make([]float64, len(values))
		for i, x := range values {
			if x < 0 {
				result[i] = math.Exp(x) - 1
			} else {
				result[i] = x
			}
		}
		return result
	},
	SELU: func(values []float64) []float64 {
		const lambda, alpha = 1.0507, 1.67326
		result := make([]float64, len(values))
		for i, x := range values {
			y := lambda * x
			if x < 0 {
				y = lambda * alpha * (math.Exp(x) - 1)
			}

			if math.IsInf(y, 1) {
				y = math.MaxFloat64
			}
			result[i] = y
		}
		return result
	},
	SoftPlus: func(values []float64) []float64 {
		result := make([]float64, len(values))
		for i, x := range values {
			y := math.Log(1 + math.Exp(x))
			if math.IsInf(y, 1) {
				y = math.MaxFloat64
			}
			result[i] = y
		}
		return result
	},
	BentIdentity: func(values []float64) []float64 {
		result := make([]float64, len(values))
		for i, x := range values {
			y := (math.Sqrt(x*x+1)-1)/2 + x
			if math.IsInf(y, 1) {
				y = math.MaxFloat64
			}
			result[i] = y
		}
		return result
	},
	Sinusoid: func(values []float64) []float64 {
		result := make([]float64, len(values))
		for i, x := range values {
			y := math.Sin(x)
			if math.IsInf(y, 1) {
				y = math.MaxFloat64
			} else if math.IsInf(y, -1) {
				y = -math.MaxFloat64
			}
			result[i] = y
		}
		return result
	},
	Sinc: func(values []float64) []float64 {
		result := make([]float64, len(values))
		for i, x := range values {
			var y float64

			if x == 0 {
				y = 1
			} else {
				y = math.Sin(x) / x
			}

			if math.IsInf(y, 1) {
				y = math.MaxFloat64
			}
			result[i] = y
		}
		return result
	},
	Gaussian: func(values []float64) []float64 {
		result := make([]float64, len(values))
		for i, x := range values {
			result[i] = math.Exp(-(x * x))
		}
		return result
	},

	Softmax: func(values []float64) []float64 {
		// does all output nodes at once so scale doesn't have to be re-computed each time
		// determine max output sum
		max := -math.MaxFloat64
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

	Maxout: func(values []float64) []float64 {
		max := -math.MaxFloat64
		for _, x := range values {
			max = math.Max(x, max)
		}

		result := make([]float64, len(values))
		for i, x := range values {
			if x == max {
				result[i] = max
			}
		}
		return result
	},
}
