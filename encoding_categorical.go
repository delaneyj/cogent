package cogent

//Info from https://www.kdnuggets.com/2015/12/beyond-one-hot-exploration-categorical-variables.html
//shows most helpful to clearest algorithms to be Ordinal, OneHot and Binary.
//Heatmap is from my own research on using time based categories in non-recurrent neural networks.
import (
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/pkg/errors"
)

type booleanEncoding struct {
}

func (b *booleanEncoding) Learn(categories ...string) error {
	return nil
}

func (b *booleanEncoding) Encode(category string) ([]float64, error) {
	falseArray := []float64{-1}
	lower := strings.ToLower(category)
	if lower == "false" || lower == "f" {
		return falseArray, nil
	}

	fixed := strings.TrimSpace(lower)
	if len(fixed) == 0 {
		return falseArray, nil
	}

	if strings.ContainsAny(fixed[0:1], "0_- ") {
		return falseArray, nil
	}

	return []float64{1}, nil
}

type ordinalEncoding struct {
	mapping map[string]float64
	nextID  float64
}

func (o *ordinalEncoding) Learn(categories ...string) error {
	if o.mapping == nil {
		o.mapping = map[string]float64{}
	}
	for _, c := range categories {
		if _, ok := o.mapping[c]; !ok {
			o.mapping[c] = o.nextID
			o.nextID++
		}
	}
	return nil
}

func (o *ordinalEncoding) Encode(category string) ([]float64, error) {
	value, ok := o.mapping[category]
	if !ok {
		value = -1
	} else {
		value /= o.nextID - 1
	}
	return []float64{value}, nil
}

type oneHotEncoding struct {
	mapping map[string]int
	nextID  int
}

func (o *oneHotEncoding) Learn(categories ...string) error {
	if o.mapping == nil {
		o.mapping = map[string]int{}
	}
	for _, c := range categories {
		if _, ok := o.mapping[c]; !ok {
			o.mapping[c] = o.nextID
			o.nextID++
		}
	}

	return nil
}

func (o *oneHotEncoding) Encode(category string) ([]float64, error) {
	oneHot := make([]float64, len(o.mapping))

	index, ok := o.mapping[category]
	if !ok {
		for i := range oneHot {
			oneHot[i] = -1
		}
	} else {
		oneHot[index] = 1
	}
	return oneHot, nil
}

type binaryEncoding struct {
	mapping map[string]int64
	nextID  int64
}

func (b *binaryEncoding) Learn(categories ...string) error {
	if b.mapping == nil {
		b.mapping = map[string]int64{}
	}
	for _, c := range categories {
		if _, ok := b.mapping[c]; !ok {
			b.mapping[c] = b.nextID
			b.nextID++
		}
	}
	return nil
}

func (b *binaryEncoding) Encode(category string) ([]float64, error) {
	lf := float64(len(b.mapping))
	li := int(math.Ceil(math.Log2(lf)))
	binary := make([]float64, li)

	categoryValue, ok := b.mapping[category]
	if !ok {
		for i := range binary {
			binary[i] = -1
		}
		return binary, nil
	}

	format := fmt.Sprintf("%%0%ds", li)
	categoryBinaryString := strconv.FormatInt(categoryValue, 2)
	parts := fmt.Sprintf(format, categoryBinaryString)

	for i, p := range parts {
		if p == '1' {
			binary[i] = 1
		}
	}
	return binary, nil
}

type stringArrayEncoding struct {
	ohe oneHotEncoding
}

func (bsa *stringArrayEncoding) Learn(arr ...string) error {
	for _, c := range arr {
		for _, s := range strings.Split(c, ",") {
			bsa.ohe.Learn(s)
		}
	}
	return nil
}

func (bsa *stringArrayEncoding) Encode(arr string) ([]float64, error) {
	var response []float64
	for _, s := range strings.Split(arr, ",") {
		e, err := bsa.ohe.Encode(s)
		if err != nil {
			return nil, errors.Wrap(err, "can't encode string array")
		}

		if response == nil {
			response = make([]float64, len(e))
		}

		for i, x := range e {
			response[i] += x
		}
	}
	return response, nil
}

type heatMapEncoding struct {
	oneHot    oneHotEncoding
	nextValue float64
}

func (hm *heatMapEncoding) Learn(categories ...string) error {
	if hm.nextValue <= 0 {
		hm.nextValue = 0.5
		hm.oneHot = oneHotEncoding{}
	}
	return hm.oneHot.Learn(categories...)
}

func (hm *heatMapEncoding) Encode(category string) ([]float64, error) {
	arr, err := hm.oneHot.Encode(category)
	if err != nil {
		return nil, errors.Wrap(err, "can't encode heat map")
	}

	for i, x := range arr {
		if x == -1 {
			return arr, nil
		}
		arr[i] *= hm.nextValue
	}
	hm.nextValue /= 2
	return arr, nil
}

func (hm *heatMapEncoding) EncodeAll(categories []string, ascendingPriority bool) ([]float64, error) {
	ordered := categories
	if ascendingPriority {
		ordered = reverseStrings(ordered)
	}

	heatmap := make([]float64, len(hm.oneHot.mapping))
	for _, c := range ordered {
		encoded, err := hm.Encode(c)
		if err != nil {
			return nil, errors.Wrapf(err, "can't encode '%s'", c)
		}
		for i, h := range heatmap {
			e := encoded[i]
			if e < 0 || h < 0 {
				heatmap[i] = -1
			} else {
				heatmap[i] += encoded[i]
			}
		}
	}
	return heatmap, nil
}

func reverseStrings(input []string) []string {
	if len(input) == 0 {
		return input
	}
	return append(reverseStrings(input[1:]), input[0])
}
