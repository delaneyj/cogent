package cogent

import (
	"runtime"

	math "github.com/chewxy/math32"
)

//LossFns x
var LossFns = map[LossMode]lossFn{
	SquaredLoss:                              squaredLoss,
	HingeLoss:                                hinge,
	CrossLoss:                                crossLoss,
	ExponentialLoss:                          exponentialLoss,
	HellingerDistanceLoss:                    hellingerDistanceLoss,
	KullbackLeiblerDivergenceLoss:            kullbackLeiblerDivergenceLoss,
	GeneralizedKullbackLeiblerDivergenceLoss: generalizedKullbackLeiblerDivergenceLoss,
	ItakuraSaitoDistanceLoss:                 itakuraSaitoDistanceLoss,
}

type lossFn func(expected, actual [][]float32) float32

func squaredLoss(expected, actual [][]float32) float32 {
	sum, count := float32(0), float32(len(actual))
	for i, actualRow := range actual {
		expectedRow := expected[i]

		for i, e := range expectedRow {
			a := actualRow[i]
			x := a - e
			sum += x * x
		}
		count++
	}
	return sum / count
}

func crossLoss(expected, actual [][]float32) float32 {
	sum, count := float32(0), float32(len(actual))
	epsilon := float32(0.000001)
	for i, actualRow := range actual {
		expectedRow := expected[i]

		for i, e := range expectedRow {
			a := actualRow[i]

			p := math.Max(epsilon, math.Min(a, 1-epsilon))
			var x float32
			if e == 1 {
				x = -math.Log(p)
			} else {
				x = -math.Log(1 - p)
			}
			if math.IsInf(x, 0) || x < 0 {
				runtime.Breakpoint()
			}
			sum += x
		}
		count++
	}

	return sum / count
}

func hinge(expected, actual [][]float32) float32 {
	sum, count := float32(0), float32(len(actual))
	for i, actualRow := range actual {
		expectedRow := expected[i]
		for i, e := range expectedRow {
			a := actualRow[i]
			sum += math.Max(0, 1-a*e)
		}
	}
	return sum / count
}

func exponentialLoss(expected, actual [][]float32) float32 {
	return math.Exp(squaredLoss(expected, actual))
}

func hellingerDistanceLoss(expected, actual [][]float32) float32 {
	sum, count := float32(0), float32(len(actual))
	for i, actualRow := range actual {
		expectedRow := expected[i]
		for i, e := range expectedRow {
			a := actualRow[i]
			b := math.Sqrt(math.Max(0, a)) - math.Sqrt(e)
			sum += b * b
		}
	}
	return ((1 / math.Sqrt2) * math.Sqrt(sum)) / count
}

func kullbackLeiblerDivergenceLoss(expected, actual [][]float32) float32 {
	sum, count := float32(0), float32(len(actual))
	for i, actualRow := range actual {
		expectedRow := expected[i]
		for i, e := range expectedRow {
			a := actualRow[i]
			l := math.Log(e / a)
			if !math.IsNaN(l) && !math.IsInf(l, 0) {
				sum += e * l
			}
		}
	}
	return sum / count
}

func generalizedKullbackLeiblerDivergenceLoss(expected, actual [][]float32) float32 {
	var xSum, ySum, zSum float32
	count := float32(len(actual))
	for i, actualRow := range actual {
		expectedRow := expected[i]
		for i, e := range expectedRow {
			a := actualRow[i]
			l := e * math.Log(e/a)
			if !math.IsNaN(l) && !math.IsInf(l, 0) {
				xSum += l
				ySum += e
				zSum += a
			}
		}
	}
	return (xSum - ySum + zSum) / count
}

func itakuraSaitoDistanceLoss(expected, actual [][]float32) float32 {
	count := float32(len(actual))
	nonSymmetric := func(eX, aY [][]float32) float32 {
		sum := float32(0)
		for i, actualRow := range actual {
			expectedRow := expected[i]
			for i, e := range expectedRow {
				a := actualRow[i]
				x := (e * e) / (a * a)
				if y := math.Log(x); !math.IsNaN(y) && !math.IsInf(y, 0) {
					sum += x - y - 1
				}
			}
		}

		return (1 / 2 * math.Pi) + sum
	}
	a := nonSymmetric(expected, actual)
	b := nonSymmetric(actual, expected)
	return ((a + b) / 2) / count
}
