package cogent

import (
	"log"
	math "math"

	t "gorgonia.org/tensor"
)

type activationFunction func(values *t.Dense) *t.Dense

var activations = map[ActivationMode]activationFunction{
	Identity: func(values *t.Dense) *t.Dense {
		return values
	},
	BinaryStep: func(values *t.Dense) *t.Dense {
		log.Fatal("oh noes")
		return nil
		// result := make([]float64, len(values))
		// for i, x := range values {
		// 	if x > 0 {
		// 		result[i] = 1
		// 	}
		// }
		// return result
	},
	Sigmoid: func(values *t.Dense) *t.Dense {
		log.Fatal("oh noes")
		return nil
		// result := make([]float64, len(values))
		// for i, x := range values {
		// 	result[i] = 1 / (1 + math.Exp(-x))
		// }
		// return result
	},
	HyperbolicTangent: func(values *t.Dense) *t.Dense {
		log.Fatal("oh noes")
		return nil
		// result := make([]float64, len(values))
		// for i, x := range values {
		// 	switch {
		// 	case x < -20:
		// 		result[i] = -1
		// 	case x > 20:
		// 		result[i] = 1
		// 	default:
		// 		result[i] = math.Tanh(x)
		// 	}
		// }
		// return result
	},
	ArcTan: func(values *t.Dense) *t.Dense {
		log.Fatal("oh noes")
		return nil
		// result := make([]float64, len(values))
		// for i, x := range values {
		// 	result[i] = math.Atan(x)
		// }
		// return result
	},
	Softsign: func(values *t.Dense) *t.Dense {
		log.Fatal("oh noes")
		return nil
		// result := make([]float64, len(values))
		// for i, x := range values {
		// 	result[i] = x / (1 + math.Abs(x))
		// }
		// return result
	},
	ISRU: func(values *t.Dense) *t.Dense {
		log.Fatal("oh noes")
		return nil
		// result := make([]float64, len(values))
		// for i, x := range values {
		// 	result[i] = x / math.Sqrt(1+x*x)
		// }
		// return result
	},
	ReLU: func(tt *t.Dense) *t.Dense {
		activated := tt.Clone().(*t.Dense)
		data := activated.Data().([]float64)
		for i, x := range data {
			if x < 0 {
				data[i] = 0
			}
		}
		return activated
	},
	LeakyReLU: func(values *t.Dense) *t.Dense {
		log.Fatal("oh noes")
		return nil
		// result := make([]float64, len(values))
		// for i, x := range values {
		// 	if x < 0 {
		// 		result[i] = x * 0.01
		// 	} else {
		// 		result[i] = x
		// 	}
		// }
		// return result
	},
	ELU: func(values *t.Dense) *t.Dense {
		log.Fatal("oh noes")
		return nil
		// result := make([]float64, len(values))
		// for i, x := range values {
		// 	if x < 0 {
		// 		result[i] = math.Exp(x) - 1
		// 	} else {
		// 		result[i] = x
		// 	}
		// }
		// return result
	},
	SELU: func(values *t.Dense) *t.Dense {
		log.Fatal("oh noes")
		return nil
		// const lambda, alpha = 1.0507, 1.67326
		// result := make([]float64, len(values))
		// for i, x := range values {
		// 	y := lambda * x
		// 	if x < 0 {
		// 		y = lambda * alpha * (math.Exp(x) - 1)
		// 	}

		// 	if math.IsInf(y, 1) {
		// 		y = math.MaxFloat64
		// 	}
		// 	result[i] = y
		// }
		// return result
	},
	SoftPlus: func(values *t.Dense) *t.Dense {
		log.Fatal("oh noes")
		return nil
		// result := make([]float64, len(values))
		// for i, x := range values {
		// 	y := math.Log(1 + math.Exp(x))
		// 	if math.IsInf(y, 1) {
		// 		y = math.MaxFloat64
		// 	}
		// 	result[i] = y
		// }
		// return result
	},
	BentIdentity: func(values *t.Dense) *t.Dense {
		log.Fatal("oh noes")
		return nil
		// result := make([]float64, len(values))
		// for i, x := range values {
		// 	y := (math.Sqrt(x*x+1)-1)/2 + x
		// 	if math.IsInf(y, 1) {
		// 		y = math.MaxFloat64
		// 	}
		// 	result[i] = y
		// }
		// return result
	},
	Sinusoid: func(values *t.Dense) *t.Dense {
		log.Fatal("oh noes")
		return nil
		// result := make([]float64, len(values))
		// for i, x := range values {
		// 	y := math.Sin(x)
		// 	if math.IsInf(y, 1) {
		// 		y = math.MaxFloat64
		// 	} else if math.IsInf(y, -1) {
		// 		y = -math.MaxFloat64
		// 	}
		// 	result[i] = y
		// }
		// return result
	},
	Sinc: func(values *t.Dense) *t.Dense {
		log.Fatal("oh noes")
		return nil
		// result := make([]float64, len(values))
		// for i, x := range values {
		// 	var y float64

		// 	if x == 0 {
		// 		y = 1
		// 	} else {
		// 		y = math.Sin(x) / x
		// 	}

		// 	if math.IsInf(y, 1) {
		// 		y = math.MaxFloat64
		// 	}
		// 	result[i] = y
		// }
		// return result
	},
	Gaussian: func(values *t.Dense) *t.Dense {
		log.Fatal("oh noes")
		return nil
		// result := make([]float64, len(values))
		// for i, x := range values {
		// 	result[i] = math.Exp(-(x * x))
		// }
		// return result
	},

	Softmax: softmax,

	Maxout: func(values *t.Dense) *t.Dense {
		log.Fatal("oh noes")
		return nil
		// max := -math.MaxFloat64
		// for _, x := range values {
		// 	max = math.Max(x, max)
		// }

		// result := make([]float64, len(values))
		// for i, x := range values {
		// 	if x == max {
		// 		result[i] = max
		// 	}
		// }
		// return result
	},

	SplitSoftmax: func(values *t.Dense) *t.Dense {
		log.Fatal("oh noes")
		return nil
		// offset := len(values) / 2
		// result := make([]float64, len(values))
		// copy(result[:offset], softmax(values[:offset]))
		// copy(result[offset:], softmax(values[offset:]))
		// return result // now scaled so that xi sum to 1.0
	},
}

func softmax(tt *t.Dense) *t.Dense {
	result := tt.Clone().(*t.Dense)

	// does all output nodes at once so scale doesn't have to be re-computed each time
	// determine max output sum

	for _, row := range denseToRows(result) {
		max := -math.MaxFloat64
		for _, x := range row {
			if x > max {
				max = x
			}
		}

		// // determine scaling factor -- sum of exp(each val - max)
		scale := 0.0
		for _, x := range row {
			scale += math.Exp(x - max)
		}

		for i, x := range row {
			row[i] = math.Exp(x-max) / scale
		}
	}

	return result // now scaled so that xi sum to 1.0
}

func denseToRows(tt *t.Dense) [][]float64 {
	s := tt.Shape()
	rowCount := s[0]
	colCount := s[1]
	data := tt.Data().([]float64)

	results := make([][]float64, rowCount)
	for i := 0; i < rowCount; i++ {
		offset := i * colCount
		results[i] = data[offset : offset+colCount]
	}
	return results
}
