package cogent

import (
	"log"
	"math"
	"strconv"
)

//Encoding to convert to float64
type Encoding int

const (
	//BinaryBoolean converts to -1,1
	BinaryBoolean Encoding = iota

	//BinaryString converts to -1,1
	BinaryString

	//Normalize confines to Mean 0 +- Standard Deviation
	Normalize

	//ClassifyString converts to one hot encoding
	ClassifyString
)

func binaryEncoding(value bool) []float64 {
	if value {
		return []float64{1}
	}
	return []float64{-1}
}

func binaryStringEncoding(value, trueValue string) []float64 {
	return binaryEncoding(value == trueValue)
}

func classificationEncoding(values []string) [][]float64 {
	set := map[string]int{}

	i := 0
	for _, s := range values {
		_, ok := set[s]
		if !ok {
			set[s] = i
			i++
		}
	}

	setLength := len(set)
	results := make([][]float64, 0, len(values))
	for _, value := range values {
		result := make([]float64, setLength)
		setIndex := set[value]
		result[setLength-setIndex-1] = 1
		results = append(results, result)
	}

	return results
}

func normalizeEncoding(raw []float64) []float64 {
	count := len(raw)
	floatCount := float64(count)

	mean := 0.0
	for _, x := range raw {
		mean += x
	}
	mean /= floatCount

	standardDeviation := 0.0
	for _, x := range raw {
		diff := x - mean
		standardDeviation += diff * diff
	}
	standardDeviation /= floatCount
	standardDeviation = math.Sqrt(standardDeviation)

	results := make([]float64, count)
	for i, x := range raw {
		results[i] = (x - mean) / standardDeviation
	}
	return results
}

func tableEncoding(encodings []Encoding, rows ...[]string) [][]float64 {
	results := make([][]float64, len(rows))

	columns := make([][][]float64, len(encodings))
	for i, encoding := range encodings {
		column := make([][]float64, len(rows))

		rawColumn := make([]string, len(rows))
		for j := range rows {
			// rawColumn[j] = rows[i]
			rawColumn[j] = rows[j][i]
		}

		switch encoding {
		case BinaryBoolean:
			for k, s := range rawColumn {
				b, _ := strconv.ParseBool(s)
				column[k] = binaryEncoding(b)
			}
		case BinaryString:
			for k, s := range rawColumn {
				column[k] = binaryStringEncoding(s, rawColumn[0])
			}
		case Normalize:
			floats := make([]float64, len(rawColumn))
			for i, s := range rawColumn {
				f, _ := strconv.ParseFloat(s, 64)
				floats[i] = f
			}
			normalized := normalizeEncoding(floats)
			for i, x := range normalized {
				column[i] = []float64{x}
			}
		case ClassifyString:
			column = classificationEncoding(rawColumn)
		default:
			log.Fatal(encoding)
		}

		columns[i] = column
	}

	columnsWidth := 0
	for _, c := range columns {
		columnsWidth += len(c[0])
	}

	for i := range rows {
		result := make([]float64, 0, columnsWidth)

		for _, c := range columns {
			result = append(result, c[i]...)
		}
		results[i] = result
	}

	return results
}
