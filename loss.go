package cogent

import (
	"log"
	math "math"
	"runtime"

	t "gorgonia.org/tensor"
)

//LossFns x
var LossFns = map[LossMode]lossFn{
	Squared:                              squaredLoss,
	Hinge:                                hinge,
	Cross:                                crossLoss,
	Exponential:                          exponentialLoss,
	HellingerDistance:                    hellingerDistanceLoss,
	KullbackLeiblerDivergence:            kullbackLeiblerDivergenceLoss,
	GeneralizedKullbackLeiblerDivergence: generalizedKullbackLeiblerDivergenceLoss,
	ItakuraSaitoDistance:                 itakuraSaitoDistanceLoss,
}

type lossFn func(want, actual *t.Dense) float64

func squaredLoss(want, actual *t.Dense) float64 {
	log.Fatal("oh noes")
	return 0
	// sum := 0.0
	// for i, w := range want {
	// 	a := actual[i]
	// 	e := a - w
	// 	sum += e * e
	// }
	// return sum
}

func crossLoss(expected, actual *t.Dense) float64 {
	sum := 0.0
	epsilon := float64(7.)/3 - float64(4.)/3 - float64(1.)
	ar := denseToRows(actual)
	er := denseToRows(expected)
	for i, actualRow := range ar {
		expectedRow := er[i]

		for i, e := range expectedRow {
			a := actualRow[i]

			p := math.Max(epsilon, math.Min(a, 1-epsilon))
			var x float64
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
	}

	return sum / float64(len(ar))
}

func hinge(want, actual *t.Dense) float64 {
	log.Fatal("oh noes")
	return 0
	// sum := 0.0
	// for i, w := range want {
	// 	a := actual[i]
	// 	sum += math.Max(0, 1-a*w)
	// }
	// return sum
}

func exponentialLoss(want, actual *t.Dense) float64 {
	log.Fatal("oh noes")
	return 0
	// return math.Exp(squaredLoss(want, actual))
}

func hellingerDistanceLoss(want, actual *t.Dense) float64 {
	log.Fatal("oh noes")
	return 0
	// wl, al := len(want), len(actual)
	// if wl != al || wl == 0 {
	// 	panic("expected and actual need to be same length")
	// }

	// sum := 0.0
	// for i, w := range want {
	// 	a := actual[i]
	// 	b := math.Sqrt(a) - math.Sqrt(w)
	// 	sum += b * b
	// }
	// return (1 / math.Sqrt2) * math.Sqrt(sum)
}

func kullbackLeiblerDivergenceLoss(want, actual *t.Dense) float64 {
	log.Fatal("oh noes")
	return 0
	// wl, al := len(want), len(actual)
	// if wl != al || wl == 0 {
	// 	panic("expected and actual need to be same length")
	// }

	// sum := 0.0
	// for i, w := range want {
	// 	a := actual[i]
	// 	l := math.Log(w / a)
	// 	if !math.IsNaN(l) && !math.IsInf(l, 0) {
	// 		sum += w * l
	// 	}
	// }
	// return sum
}

func generalizedKullbackLeiblerDivergenceLoss(want, actual *t.Dense) float64 {
	log.Fatal("oh noes")
	return 0
	// wl, al := len(want), len(actual)
	// if wl != al || wl == 0 {
	// 	panic("expected and actual need to be same length")
	// }

	// xSum, ySum, zSum := 0.0, 0.0, 0.0
	// for i, w := range want {
	// 	a := actual[i]

	// 	l := w * math.Log(w/a)
	// 	if !math.IsNaN(l) && !math.IsInf(l, 0) {
	// 		xSum += l
	// 		ySum += w
	// 		zSum += a
	// 	}
	// }

	// return xSum - ySum + zSum
}

func itakuraSaitoDistanceLoss(want, actual *t.Dense) float64 {
	log.Fatal("oh noes")
	return 0
	// wl, al := len(want), len(actual)
	// if wl != al || wl == 0 {
	// 	panic("expected and actual need to be same length")
	// }

	// nonSymmetric := func(wX, aY *t.Dense) float64 {
	// 	sum := 0.0
	// 	for i, w := range wX {
	// 		a := aY[i]
	// 		x := (w * w) / (a * a)
	// 		if y := math.Log(x); !math.IsNaN(y) && !math.IsInf(y, 0) {
	// 			sum += x - y - 1
	// 		}
	// 	}
	// 	return (1 / 2 * math.Pi) + sum
	// }

	// a := nonSymmetric(want, actual)
	// b := nonSymmetric(actual, want)
	// return (a + b) / 2
}
