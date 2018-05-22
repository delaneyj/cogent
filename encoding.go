package cogent

import (
	"fmt"
	"log"
	"math"

	"github.com/pkg/errors"
)

//EncodingMode determines what style of encoding to use
type EncodingMode int

const (
	//BooleanEncoding returns either -1 to 1
	BooleanEncoding EncodingMode = iota

	//OrdinalEncoding outputs array from first to last seen from 0-1
	OrdinalEncoding

	//OneHotEncoding one column per category, with a 1 or 0 in each cell for if the row contained that column’s category
	OneHotEncoding

	//BinaryEncoding first the categories are encoded as ordinal.
	// Those integers are converted into binary code.
	//The digits from that binary string are split into separate columns.
	//This encodes the data in fewer dimensions that one-hot, but with some distortion of the distances.
	BinaryEncoding

	//HeatMapEncoding will take categories from lowest to highest priority and make a weight heatmap
	HeatMapEncoding

	//StringArrayEncoding converts a single csv into OneHot array
	StringArrayEncoding

	//NormalizedEncoding normalized to mean and std deviation
	NormalizedEncoding
)

type valueEncoding interface {
	Learn(categories ...string) error
	Encode(category string) ([]float64, error)
}

//TableEncoding converts strings from usually an excel file to data ready for neural network
func TableEncoding(encodings []EncodingMode, table [][]string) ([][]float64, error) {
	rowCount := len(table)
	if rowCount == 0 {
		return nil, errors.New("no rows in table")
	}

	columnCount := len(table[0])
	if len(table[0]) == 0 {
		return nil, errors.New("no columns in table")
	}

	log.Printf("Create encoding instance for all %d columns", columnCount)
	columnEncodings := make([]valueEncoding, columnCount)
	for i, encoding := range encodings {
		switch encoding {
		case BooleanEncoding:
			columnEncodings[i] = &booleanEncoding{}
		case OrdinalEncoding:
			columnEncodings[i] = &ordinalEncoding{}
		case OneHotEncoding:
			columnEncodings[i] = &oneHotEncoding{}
		case BinaryEncoding:
			columnEncodings[i] = &binaryEncoding{}
		case HeatMapEncoding:
			columnEncodings[i] = &heatMapEncoding{}
		case StringArrayEncoding:
			columnEncodings[i] = &stringArrayEncoding{}
		case NormalizedEncoding:
			columnEncodings[i] = &normalizedEncoding{}
		default:
			msgFormat := "can't find valid encoding '%s' for column %d"
			msg := fmt.Sprintf(msgFormat, encoding, i)
			return nil, errors.New(msg)
		}
	}

	log.Printf("Learn encodings from %d rows of examples.", rowCount)
	for _, row := range table {
		for c, col := range row {
			ce := columnEncodings[c]
			err := ce.Learn(col)
			if err != nil {
				return nil, errors.Wrap(err, "can't learn from column")
			}
		}
	}

	log.Print("Start encoding.")
	encodedRows := make([][]float64, rowCount)
	for r, row := range table {
		firstRow := r == 0
		encodedRow := []float64{}
		for c, col := range row {
			ce := columnEncodings[c]

			encoded, err := ce.Encode(col)
			if err != nil {
				return nil, errors.Wrap(err, "can't encode column")
			}
			encodedLength := len(encoded)

			if encodedLength <= 0 {
				return nil, errors.New("empty encoding")
			}

			if firstRow {
				log.Printf("Column %d will use %d floats.", c, encodedLength)
			}

			for _, x := range encoded {
				if math.IsInf(x, 0) {
					msg := "%s is encoding as infinity, bad news <%s:%s>"
					return nil, errors.Wrapf(err, msg, col, c, r)
				}

				if math.IsNaN(x) {
					msg := "%s is encoding as NaN, bad news <%s:%s>"
					return nil, errors.Wrapf(err, msg, col, c, r)
				}
			}

			encodedRow = append(encodedRow, encoded...)
		}

		encodedRows[r] = encodedRow
		if r == 0 {
			log.Printf("Each row will have %d floats.", len(encodedRow))
		}
	}
	return encodedRows, nil
}
