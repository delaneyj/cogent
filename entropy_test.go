package cogent

import (
	"fmt"
	"reflect"
	"runtime"
	"testing"

	"github.com/stretchr/testify/assert"
)

func Test_entropy(t *testing.T) {
	xE, yE, zE := []float64{0, 0, 1}, []float64{0, 1, 0}, []float64{1, 0, 0}
	xA, yA, zA := []float64{0.1, 0.3, 0.6}, []float64{0.2, 0.6, 0.2}, []float64{0.3, 0.4, 0.3}

	jE, kE, lE := []float64{0.125, 0.125, 0.5}, []float64{0.125, 0.5, 0.125}, []float64{0.5, 0.125, 0.125}
	jA, kA, lA := []float64{0.1, 0.3, 0.6}, []float64{0.2, 0.6, 0.2}, []float64{0.3, 0.4, 0.3}

	type args struct {
		want   []float64
		actual []float64
	}
	tests := []struct {
		et   EntropyType
		args args
		want float64
	}{
		{SquaredEntropy, args{xE, xA}, 0.26},
		{SquaredEntropy, args{yE, yA}, 0.24000000000000005},
		{SquaredEntropy, args{zE, zA}, 0.7399999999999999},
		{CrossEntropy, args{xE, xA}, 0.5108256237659907},
		{CrossEntropy, args{yE, yA}, 0.5108256237659907},
		{CrossEntropy, args{zE, zA}, 1.2039728043259361},
		{ExponentialEntropy, args{xE, xA}, 1.2969300866657718},
		{ExponentialEntropy, args{yE, yA}, 1.2712491503214047},
		{ExponentialEntropy, args{zE, zA}, 2.0959355144943643},
		{HellingerDistanceEntropy, args{xE, xA}, 0.47476660661688985},
		{HellingerDistanceEntropy, args{yE, yA}, 0.47476660661688985},
		{HellingerDistanceEntropy, args{zE, zA}, 0.6725157563171542},
		{KullbackLeiblerDivergenceEntropy, args{xE, xA}, 0.5108256237659907},
		{KullbackLeiblerDivergenceEntropy, args{yE, yA}, 0.5108256237659907},
		{KullbackLeiblerDivergenceEntropy, args{zE, zA}, 1.2039728043259361},
		{GeneralizedKullbackLeiblerDivergenceEntropy, args{xE, xA}, 0.1108256237659907},
		{GeneralizedKullbackLeiblerDivergenceEntropy, args{yE, yA}, 0.1108256237659907},
		{GeneralizedKullbackLeiblerDivergenceEntropy, args{zE, zA}, 0.5039728043259362},
		{ItakuraSaitoDistanceEntropy, args{xE, xA}, 0.7561265302457962},
		{ItakuraSaitoDistanceEntropy, args{yE, yA}, 0.7561265302457962},
		{ItakuraSaitoDistanceEntropy, args{zE, zA}, 7.703165502459239},

		{SquaredEntropy, args{jE, jA}, 0.041249999999999995},
		{SquaredEntropy, args{kE, kA}, 0.021249999999999998},
		{SquaredEntropy, args{lE, lA}, 0.14625000000000002},
		{CrossEntropy, args{jE, jA}, 0.6937325490479931},
		{CrossEntropy, args{kE, kA}, 0.6577722899915205},
		{CrossEntropy, args{lE, lA}, 0.8670193441879794},
		{ExponentialEntropy, args{jE, jA}, 1.0421126011324575},
		{ExponentialEntropy, args{kE, kA}, 1.0214773890662867},
		{ExponentialEntropy, args{lE, lA}, 1.1574855232632035},
		{HellingerDistanceEntropy, args{jE, jA}, 0.14773244839734279},
		{HellingerDistanceEntropy, args{kE, kA}, 0.1051174413596333},
		{HellingerDistanceEntropy, args{lE, lA}, 0.26541608903551517},
		{KullbackLeiblerDivergenceEntropy, args{jE, jA}, -0.17270142665193855},
		{KullbackLeiblerDivergenceEntropy, args{kE, kA}, -0.2086616857084112},
		{KullbackLeiblerDivergenceEntropy, args{lE, lA}, 0.0005853684880477716},
		{GeneralizedKullbackLeiblerDivergenceEntropy, args{jE, jA}, 0.07729857334806145},
		{GeneralizedKullbackLeiblerDivergenceEntropy, args{kE, kA}, 0.04133831429158885},
		{GeneralizedKullbackLeiblerDivergenceEntropy, args{lE, lA}, 0.2505853684880478},
		{ItakuraSaitoDistanceEntropy, args{jE, jA}, 1.0998490412228452},
		{ItakuraSaitoDistanceEntropy, args{kE, kA}, 0.7203520750152961},
		{ItakuraSaitoDistanceEntropy, args{lE, lA}, 3.1046329856760693},
	}
	for _, tt := range tests {
		fn := entropyFns[tt.et]
		p := reflect.ValueOf(fn).Pointer()
		rf := runtime.FuncForPC(p)
		msg := fmt.Sprintf("%s%v%f", rf.Name(), tt.args.want, tt.want)
		assert.Equal(t, tt.want, fn(tt.args.want, tt.args.actual), msg)
	}
}
