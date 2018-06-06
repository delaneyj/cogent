package cogent

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
)

//TestSwarm x
func Test_BooleanEncoding(t *testing.T) {
	tx := []float64{1}
	fx := []float64{0}
	tests := []struct {
		with     string
		expected []float64
	}{
		{fmt.Sprint(true), tx},
		{fmt.Sprint(false), fx},
		{"f", fx},
		{"t", tx},
		{"arst", tx},
		{"0", fx},
		{"000", fx},
		{"-", fx},
		{"", fx},
	}

	be := booleanEncoding{}
	for _, tt := range tests {
		actual, err := be.Encode(tt.with)
		assert.Nil(t, err)
		assert.Equal(t, tt.expected, actual, tt.with)
	}
}

func Test_Ordinal(t *testing.T) {
	test := []string{"", "foo", "bar", "baz", "foo"}
	ohe := ordinalEncoding{}
	ohe.Learn(test...)

	for i, tt := range [][]float64{
		[]float64{0},
		[]float64{0.3333333333333333},
		[]float64{0.6666666666666666},
		[]float64{1},
		[]float64{0.3333333333333333},
	} {
		x := test[i]
		e, err := ohe.Encode(x)
		assert.Nil(t, err)
		assert.Equal(t, tt, e, x)
	}
}

func Test_Binary(t *testing.T) {
	test := []string{"", "foo", "bar", "baz", "foo"}
	ohe := ordinalEncoding{}
	ohe.Learn(test...)

	for i, tt := range [][]float64{
		[]float64{0},
		[]float64{0.3333333333333333},
		[]float64{0.6666666666666666},
		[]float64{1},
		[]float64{0.3333333333333333},
	} {
		x := test[i]
		e, err := ohe.Encode(x)
		assert.Nil(t, err)
		assert.Equal(t, tt, e, x)
	}
}

func Test_OneHotEncoding(t *testing.T) {
	test := []string{"foo", "bar", "baz", "foo"}
	ohe := oneHotEncoding{}
	ohe.Learn(test...)

	for i, tt := range [][]float64{
		[]float64{1, 0, 0},
		[]float64{0, 1, 0},
		[]float64{0, 0, 1},
		[]float64{1, 0, 0},
	} {
		x := test[i]
		e, err := ohe.Encode(x)
		assert.Nil(t, err)
		assert.Equal(t, tt, e, x)
	}
}

func Test_StringArrayEncoding(t *testing.T) {
	test := []struct {
		input    string
		expected []float64
	}{
		{
			"foo,baz",
			[]float64{1, 1, 0},
		},
		{
			"bar,foo",
			[]float64{1, 0, 1},
		},
		{
			"baz,foo",
			[]float64{1, 1, 0},
		},
		{
			"foo,foo",
			[]float64{2, 0, 0},
		},
	}

	sae := stringArrayEncoding{}
	for _, t := range test {
		sae.Learn(t.input)
	}

	for _, tt := range test {
		actual, err := sae.Encode(tt.input)
		assert.Nil(t, err)
		assert.Equal(t, tt.expected, actual, tt.input)
	}
}

func Test_Normalization(t *testing.T) {
	test := []string{"60000", "24000", "30000", "30000", "18000", "56000"}

	ne := normalizedEncoding{}
	for _, t := range test {
		ne.Learn(t)
	}
	assert.Equal(t, 36333.333333333336, ne.mean)
	assert.Equal(t, 15891.99658808029, ne.standardDeviation)

	ne = normalizedEncoding{}
	ne.Learn(test...)

	assert.Equal(t, 36333.333333333336, ne.mean)
	assert.Equal(t, 15891.99658808029, ne.standardDeviation)

	expected := [][]float64{
		{1.4892192139292129},
		{-0.7760719847236746},
		{-0.39852345161486},
		{-0.39852345161486},
		{-1.1536205178324892},
		{1.2375201918566698},
	}

	for i, e := range expected {
		x := test[i]
		a, err := ne.Encode(x)
		assert.Nil(t, err)
		assert.Equal(t, e, a, x)
	}

	ne = normalizedEncoding{}
	ne.Learn("-1", "0", "", "1")
	zero, err := ne.Encode("")
	assert.Nil(t, err)
	assert.Equal(t, []float64{0}, zero)
}

func Test_Combine(t *testing.T) {
	encodings := []EncodingMode{
		BooleanEncodingMode,
		OrdinalEncodingMode,
		NormalizedEncodingMode,
		OneHotEncodingMode,
		NormalizedEncodingMode,
		BinaryEncodingMode,
	}

	testTable := [][]string{
		[]string{"true", "male", "60000.00", "suburban", "54", "republican"},
		[]string{"false", "female", "24000.00", "city", "28", "democrat"},
		[]string{"true", "male", "30000.00", "rural", "31", "libertarian"},
		[]string{"false", "female", "30000.00", "suburban", "48", "republican"},
		[]string{"false", "female", "18000.00", "city", "22", "democrat"},
		[]string{"true", "male", "56000.00", "rural", "39", "other"},
	}

	expected := [][]float64{
		{
			1, 0,
			1.4892192139292129,
			1, 0, 0,
			1.5144803708370715,
			0, 0,
		},
		{
			0, 1,
			-0.7760719847236746,
			0, 1, 0,
			-0.8017837257372732,
			0, 1,
		},
		{
			1, 0,
			-0.39852345161486,
			0, 0, 1,
			-0.5345224838248488,
			1, 0,
		},
		{
			0, 1,
			-0.39852345161486,
			1, 0, 0,
			0.9799578870122228,
			0, 0,
		},
		{
			0, 1,
			-1.1536205178324892,
			0, 1, 0,
			-1.3363062095621219,
			0, 1,
		},
		{
			1, 0,
			1.2375201918566698,
			0, 0, 1,
			0.1781741612749496,
			1, 1,
		},
	}

	encoded, err := TableEncoding(encodings, testTable)
	assert.Nil(t, err)
	assert.Equal(t, expected, encoded)
}
