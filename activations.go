package cogent

import (
	math "math"

	t "gorgonia.org/tensor"
)

type activationFunction func(values *t.Dense) *t.Dense

var activations = map[ActivationMode]activationFunction{
	Identity: func(values *t.Dense) *t.Dense {
		return values
	},
	BinaryStep: func(tt *t.Dense) *t.Dense {
		activated := tt.Clone().(*t.Dense)
		data := activated.Data().([]float64)
		for i, x := range data {
			if x > 0 {
				data[i] = 1
			}
		}
		return activated
	},
	Sigmoid: func(tt *t.Dense) *t.Dense {
		activated := tt.Clone().(*t.Dense)
		data := activated.Data().([]float64)
		for i, x := range data {
			data[i] = 1 / (1 + math.Exp(-x))
		}
		return activated
	},
	HyperbolicTangent: func(tt *t.Dense) *t.Dense {
		activated := tt.Clone().(*t.Dense)
		data := activated.Data().([]float64)
		for i, x := range data {
			switch {
			case x < -20:
				data[i] = -1
			case x > 20:
				data[i] = 1
			default:
				data[i] = math.Tanh(x)
			}
		}
		return activated
	},
	ArcTan: func(tt *t.Dense) *t.Dense {
		activated := tt.Clone().(*t.Dense)
		data := activated.Data().([]float64)
		for i, x := range data {
			data[i] = math.Atan(x)
		}
		return activated
	},
	Softsign: func(tt *t.Dense) *t.Dense {
		activated := tt.Clone().(*t.Dense)
		data := activated.Data().([]float64)
		for i, x := range data {
			data[i] = x / (1 + math.Abs(x))
		}
		return activated
	},
	ISRU: func(tt *t.Dense) *t.Dense {
		activated := tt.Clone().(*t.Dense)
		data := activated.Data().([]float64)
		for i, x := range data {
			data[i] = x / math.Sqrt(1+x*x)
		}
		return activated
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
	LeakyReLU: func(tt *t.Dense) *t.Dense {
		activated := tt.Clone().(*t.Dense)
		data := activated.Data().([]float64)
		for i, x := range data {
			if x < 0 {
				data[i] = x * 0.01
			} else {
				data[i] = x
			}
		}
		return activated
	},
	ELU: func(tt *t.Dense) *t.Dense {
		activated := tt.Clone().(*t.Dense)
		data := activated.Data().([]float64)
		for i, x := range data {
			if x < 0 {
				data[i] = math.Exp(x) - 1
			} else {
				data[i] = x
			}
		}
		return activated
	},
	SELU: func(tt *t.Dense) *t.Dense {
		const lambda, alpha = 1.0507, 1.67326
		activated := tt.Clone().(*t.Dense)
		data := activated.Data().([]float64)
		for i, x := range data {
			y := lambda * x
			if x < 0 {
				y = lambda * alpha * (math.Exp(x) - 1)
			}

			if math.IsInf(y, 1) {
				y = math.MaxFloat64
			}
			data[i] = y
		}
		return activated
	},
	SoftPlus: func(tt *t.Dense) *t.Dense {
		activated := tt.Clone().(*t.Dense)
		data := activated.Data().([]float64)
		for i, x := range data {
			y := math.Log(1 + math.Exp(x))
			if math.IsInf(y, 1) {
				y = math.MaxFloat64
			}
			data[i] = y
		}
		return activated
	},
	BentIdentity: func(tt *t.Dense) *t.Dense {
		activated := tt.Clone().(*t.Dense)
		data := activated.Data().([]float64)
		for i, x := range data {
			y := (math.Sqrt(x*x+1)-1)/2 + x
			if math.IsInf(y, 1) {
				y = math.MaxFloat64
			}
			data[i] = y
		}
		return activated
	},
	Sinusoid: func(tt *t.Dense) *t.Dense {
		activated := tt.Clone().(*t.Dense)
		data := activated.Data().([]float64)
		for i, x := range data {
			y := math.Sin(x)
			if math.IsInf(y, 1) {
				y = math.MaxFloat64
			} else if math.IsInf(y, -1) {
				y = -math.MaxFloat64
			}
			data[i] = y
		}
		return activated
	},
	Sinc: func(tt *t.Dense) *t.Dense {
		activated := tt.Clone().(*t.Dense)
		data := activated.Data().([]float64)
		for i, x := range data {
			y := 1.0
			if x != 0 {
				y = math.Sin(x) / x
			}

			if math.IsInf(y, 1) {
				y = math.MaxFloat64
			}
			data[i] = y
		}
		return activated
	},
	Gaussian: func(tt *t.Dense) *t.Dense {
		activated := tt.Clone().(*t.Dense)
		data := activated.Data().([]float64)
		for i, x := range data {
			data[i] = math.Exp(-(x * x))
		}
		return activated
	},

	Softmax: func(tt *t.Dense) *t.Dense {
		result := tt.Clone().(*t.Dense)
		for _, row := range DenseToRows(result) {
			softmaxModifyRow(row)
		}
		return result
	},

	Maxout: func(tt *t.Dense) *t.Dense {
		activated := tt.Clone().(*t.Dense)
		data := activated.Data().([]float64)

		max := -math.MaxFloat64
		for _, x := range data {
			max = math.Max(x, max)
		}

		for i, x := range data {
			if x == max {
				data[i] = max
			}
		}
		return activated
	},

	SplitSoftmax: func(tt *t.Dense) *t.Dense {
		result := tt.Clone().(*t.Dense)
		for _, row := range DenseToRows(result) {
			offset := len(row) / 2
			softmaxModifyRow(row[:offset])
			softmaxModifyRow(row[offset:])

			softmaxModifyRow(row)
		}
		return result // now scaled so that xi sum to 1.0
	},
}

func softmaxModifyRow(row []float64) {
	exps := make([]float64, len(row))
	sum, max := 0.0, -math.MaxFloat64

	for _, x := range row {
		if x > max {
			max = x
		}
	}

	for i, x := range row {
		e := math.Exp(x - max)
		exps[i] = e
		sum += e
	}

	for i, e := range exps {
		row[i] = e / sum
	}
}

//DenseToRows x
func DenseToRows(tt *t.Dense) [][]float64 {
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
