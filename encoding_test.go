package cogent

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

//TestSwarm x
func Test_BinaryEncoding(t *testing.T) {
	assert.Equal(t, binaryEncoding(true), []float64{1})
	assert.Equal(t, binaryEncoding(false), []float64{-1})
	assert.Equal(t, binaryStringEncoding("male", "male"), []float64{1})
	assert.Equal(t, binaryStringEncoding("female", "male"), []float64{-1})
}

func Test_ClassificationEncoding(t *testing.T) {
	assert.Equal(t, classificationEncoding([]string{"foo", "bar", "baz", "foo"}), [][]float64{
		[]float64{0, 0, 1},
		[]float64{0, 1, 0},
		[]float64{1, 0, 0},
		[]float64{0, 0, 1},
	})
}

func Test_Normalization(t *testing.T) {
	assert.Equal(t,
		normalizeEncoding([]float64{60000, 24000, 30000, 30000, 18000, 56000}),
		[]float64{1.4892192139292129, -0.7760719847236746, -0.39852345161486, -0.39852345161486, -1.1536205178324892, 1.2375201918566698})
}

func Test_Combine(t *testing.T) {
	encodings := []Encoding{
		BinaryBoolean,
		BinaryString,
		Normalize,
		ClassifyString,
		Normalize,
		ClassifyString,
	}

	table := [][]string{
		[]string{"true", "male", "60000.00", "suburban", "54", "republican"},
		[]string{"false", "female", "24000.00", "city", "28", "democrat"},
		[]string{"true", "male", "30000.00", "rural", "31", "libertarian"},
		[]string{"false", "female", "30000.00", "suburban", "48", "republican"},
		[]string{"false", "female", "18000.00", "city", "22", "democrat"},
		[]string{"true", "male", "56000.00", "rural", "39", "other"},
	}

	assert.Equal(t, tableEncoding(encodings, table...), [][]float64{
		[]float64{
			1, 1,
			1.4892192139292129,
			0, 0, 1,
			1.5144803708370715,
			0, 0, 0, 1,
		},
		[]float64{
			-1, -1,
			-0.7760719847236746,
			0, 1, 0,
			-0.8017837257372732,
			0, 0, 1, 0,
		},
		[]float64{1, 1,
			-0.39852345161486,
			1, 0, 0,
			-0.5345224838248488,
			0, 1, 0, 0,
		},
		[]float64{-1, -1,
			-0.39852345161486,
			0, 0, 1,
			0.9799578870122228,
			0, 0, 0, 1,
		},
		[]float64{-1, -1,
			-1.1536205178324892,
			0, 1, 0,
			-1.3363062095621219,
			0, 0, 1, 0,
		},
		[]float64{1, 1,
			1.2375201918566698,
			1, 0, 0,
			0.1781741612749496,
			1, 0, 0, 0,
		},
	})
}
