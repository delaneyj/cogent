package cogent

import (
	"fmt"
	"runtime"
	"strconv"
	strings "strings"

	math "github.com/chewxy/math32"

	"github.com/pkg/errors"
)

type normalizedEncoding struct {
	values            []float32
	mean              float32
	standardDeviation float32
}

func (n *normalizedEncoding) Learn(valueStrings ...string) error {
	var x float32
	for _, v := range valueStrings {

		if trimmed := strings.TrimSpace(v); len(trimmed) > 0 {
			f, err := strconv.ParseFloat(trimmed, 32)
			if err != nil {
				return errors.Wrap(err, "can't convert to float")
			}
			x = float32(f)
		}

		n.values = append(n.values, x)
	}

	floatCount := float32(len(n.values))
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

func (n *normalizedEncoding) Encode(valueString string) ([]float32, error) {
	var value float32

	if trimmed := strings.TrimSpace(valueString); len(trimmed) > 0 {
		f, err := strconv.ParseFloat(trimmed, 32)
		if err != nil {
			return nil, errors.Wrapf(err, "can't convert to float '%s'", valueString)
		}
		value = float32(f)
	}
	x := (value - n.mean)
	if n.standardDeviation != 0 {
		x /= n.standardDeviation
	}
	return []float32{x}, nil
}

type intRangeEncoding struct {
	min, max float32
	ohe      *oneHotEncoding
}

func (ire *intRangeEncoding) Learn(valueStrings ...string) error {
	if ire.min == 0 && ire.max == 0 {
		ire.min = math.MaxFloat32
		ire.max = -math.MaxFloat32
	}

	for _, v := range valueStrings {
		v = strings.TrimSpace(v)
		if v == "" {
			v = "0"
		}
		vf, err := strconv.ParseFloat(v, 32)
		if err != nil {
			return errors.Wrap(err, "can't parse string to float")
		}
		ire.min = math.Min(ire.min, float32(vf))
		ire.max = math.Max(ire.max, float32(vf))
	}
	return nil
}

func (ire *intRangeEncoding) Encode(valueString string) ([]float32, error) {
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
