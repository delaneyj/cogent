package cogent

import (
	"fmt"
	"math"
	"runtime"
	"strconv"
	strings "strings"

	"github.com/pkg/errors"
)

type normalizedEncoding struct {
	values            []float64
	mean              float64
	standardDeviation float64
}

func (n *normalizedEncoding) Learn(valueStrings ...string) error {
	for _, v := range valueStrings {
		var err error
		x := 0.0

		if trimmed := strings.TrimSpace(v); len(trimmed) > 0 {
			x, err = strconv.ParseFloat(trimmed, 64)
			if err != nil {
				return errors.Wrap(err, "can't convert to float")
			}
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
	var value float64
	var err error

	if trimmed := strings.TrimSpace(valueString); len(trimmed) > 0 {
		value, err = strconv.ParseFloat(trimmed, 64)
		if err != nil {
			return nil, errors.Wrapf(err, "can't convert to float '%s'", valueString)
		}
	}
	x := (value - n.mean)
	if n.standardDeviation != 0 {
		x /= n.standardDeviation
	}
	return []float64{x}, nil
}

type intRangeEncoding struct {
	min, max float64
	ohe      *oneHotEncoding
}

func (ire *intRangeEncoding) Learn(valueStrings ...string) error {
	if ire.min == 0 && ire.max == 0 {
		ire.min = math.MaxFloat64
		ire.max = -math.MaxFloat64
	}

	for _, v := range valueStrings {
		v = strings.TrimSpace(v)
		if v == "" {
			v = "0"
		}
		vf, err := strconv.ParseFloat(v, 64)
		if err != nil {
			return errors.Wrap(err, "can't parse string to float")
		}
		ire.min = math.Min(ire.min, vf)
		ire.max = math.Max(ire.max, vf)
	}
	return nil
}

func (ire *intRangeEncoding) Encode(valueString string) ([]float64, error) {
	if ire.ohe == nil {
		ire.ohe = &oneHotEncoding{}
		for i := ire.min; i <= ire.max; i++ {
			err := ire.ohe.Learn(fmt.Sprint(i))
			if err != nil {
				return nil, errors.Wrapf(err, "can't parse %d", i)
			}
		}
	}

	valueString = strings.TrimSpace(valueString)
	if valueString == "" {
		valueString = "0"
	}

	out, err := ire.ohe.Encode(valueString)
	for _, o := range out {
		if o == -1 {
			runtime.Breakpoint()
		}
	}
	return out, err
}
