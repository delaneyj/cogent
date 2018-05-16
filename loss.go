package cogent

import "math"

var lossFns = map[Loss]lossFn{
	Squared:                              squaredLoss,
	Cross:                                crossLoss,
	Exponential:                          exponentialLoss,
	HellingerDistance:                    hellingerDistanceLoss,
	KullbackLeiblerDivergence:            kullbackLeiblerDivergenceLoss,
	GeneralizedKullbackLeiblerDivergence: generalizedKullbackLeiblerDivergenceLoss,
	ItakuraSaitoDistance:                 itakuraSaitoDistanceLoss,
}

type lossFn func(want, actual []float64) float64

func squaredLoss(want, actual []float64) float64 {
	sum := 0.0
	for i, w := range want {
		a := actual[i]
		e := a - w
		sum += e * e
	}
	return sum
}

func crossLoss(want, actual []float64) float64 {
	sum := 0.0
	for i, w := range want {
		a := actual[i]
		if l := math.Log(a); !math.IsNaN(l) && !math.IsInf(l, 0) {
			sum += l * w
		}
	}
	return -sum
}

func exponentialLoss(want, actual []float64) float64 {
	return math.Exp(squaredLoss(want, actual))
}

func hellingerDistanceLoss(want, actual []float64) float64 {
	wl, al := len(want), len(actual)
	if wl != al || wl == 0 {
		panic("expected and actual need to be same length")
	}

	sum := 0.0
	for i, w := range want {
		a := actual[i]
		b := math.Sqrt(a) - math.Sqrt(w)
		sum += b * b
	}
	return (1 / math.Sqrt2) * math.Sqrt(sum)
}

func kullbackLeiblerDivergenceLoss(want, actual []float64) float64 {
	wl, al := len(want), len(actual)
	if wl != al || wl == 0 {
		panic("expected and actual need to be same length")
	}

	sum := 0.0
	for i, w := range want {
		a := actual[i]
		l := math.Log(w / a)
		if !math.IsNaN(l) && !math.IsInf(l, 0) {
			sum += w * l
		}
	}
	return sum
}

func generalizedKullbackLeiblerDivergenceLoss(want, actual []float64) float64 {
	wl, al := len(want), len(actual)
	if wl != al || wl == 0 {
		panic("expected and actual need to be same length")
	}

	xSum, ySum, zSum := 0.0, 0.0, 0.0
	for i, w := range want {
		a := actual[i]

		l := w * math.Log(w/a)
		if !math.IsNaN(l) && !math.IsInf(l, 0) {
			xSum += l
			ySum += w
			zSum += a
		}
	}

	return xSum - ySum + zSum
}

func itakuraSaitoDistanceLoss(want, actual []float64) float64 {
	wl, al := len(want), len(actual)
	if wl != al || wl == 0 {
		panic("expected and actual need to be same length")
	}

	sum := 0.0
	for i, w := range want {
		a := actual[i]
		x := (w * w) / (a * a)
		if y := math.Log(x); !math.IsNaN(y) && !math.IsInf(y, 0) {
			sum += x - y - 1
		}
	}
	return (1 / 2 * math.Pi) + sum
}
