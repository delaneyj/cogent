package cogent

import "math"

//EntropyType x
type EntropyType string

//Entropy Constants
const (
	SquaredEntropy                              = "squaredEntropy"
	CrossEntropy                                = "crossEntropy"
	ExponentialEntropy                          = "exponentialEntropy"
	HellingerDistanceEntropy                    = "hellingerDistanceEntropy"
	KullbackLeiblerDivergenceEntropy            = "kullbackLeiblerDivergenceEntropy"
	GeneralizedKullbackLeiblerDivergenceEntropy = "generalizedKullbackLeiblerDivergenceEntropy"
	ItakuraSaitoDistanceEntropy                 = "itakuraSaitoDistanceEntropy"
)

var entropyFns = map[EntropyType]entropyFn{
	SquaredEntropy:                              squaredEntropy,
	CrossEntropy:                                crossEntropy,
	ExponentialEntropy:                          exponentialEntropy,
	HellingerDistanceEntropy:                    hellingerDistanceEntropy,
	KullbackLeiblerDivergenceEntropy:            kullbackLeiblerDivergenceEntropy,
	GeneralizedKullbackLeiblerDivergenceEntropy: generalizedKullbackLeiblerDivergenceEntropy,
	ItakuraSaitoDistanceEntropy:                 itakuraSaitoDistanceEntropy,
}

type entropyFn func(want, actual []float64) float64

func squaredEntropy(want, actual []float64) float64 {
	wl, al := len(want), len(actual)
	if wl != al || wl == 0 {
		panic("expected and actual need to be same length")
	}

	sum := 0.0
	for i, w := range want {
		a := actual[i]
		e := a - w
		sum += e * e
	}
	return sum
}

func crossEntropy(want, actual []float64) float64 {
	wl, al := len(want), len(want)
	if wl != al || wl == 0 {
		panic("expected and actual need to be same length")
	}

	sum := 0.0
	for i, w := range want {
		a := actual[i]
		if l := math.Log(a); !math.IsNaN(l) && !math.IsInf(l, 0) {
			sum += l * w
		}
	}
	return -sum
}

func exponentialEntropy(want, actual []float64) float64 {
	return math.Exp(squaredEntropy(want, actual))
}

func hellingerDistanceEntropy(want, actual []float64) float64 {
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

func kullbackLeiblerDivergenceEntropy(want, actual []float64) float64 {
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

func generalizedKullbackLeiblerDivergenceEntropy(want, actual []float64) float64 {
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

func itakuraSaitoDistanceEntropy(want, actual []float64) float64 {
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
