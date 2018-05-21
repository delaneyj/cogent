package cogent

import (
	"math"
	"strconv"

	"github.com/pkg/errors"
)

type normalizedEncoding struct {
	values            []float64
	mean              float64
	standardDeviation float64
}

func (n *normalizedEncoding) Learn(valueStrings ...string) error {
	for _, v := range valueStrings {
		x, err := strconv.ParseFloat(v, 64)
		if err != nil {
			return errors.Wrap(err, "can't convert to float")
		}
		n.values = append(n.values, x)
	}

	floatCount := float64(len(n.values))
	n.mean = 0
	for _, v := range n.values {
		n.mean += v
	}
	n.mean /= floatCount
	n.standardDeviation = 0
	for _, x := range n.values {
		diff := x - n.mean
		n.standardDeviation += diff * diff
	}
	n.standardDeviation /= floatCount
	n.standardDeviation = math.Sqrt(n.standardDeviation)

	return nil
}

func (n *normalizedEncoding) Encode(valueString string) ([]float64, error) {
	value, err := strconv.ParseFloat(valueString, 64)
	if err != nil {
		return nil, errors.Wrapf(err, "can't convert to float '%s'", valueString)
	}
	x := (value - n.mean) / n.standardDeviation
	return []float64{x}, nil
}
