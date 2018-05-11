package cogent

import (
	"fmt"
	"reflect"
	"runtime"
	"testing"

	"github.com/stretchr/testify/assert"
)

func Test_Loss(t *testing.T) {
	xE, yE, zE := []float64{0, 0, 1}, []float64{0, 1, 0}, []float64{1, 0, 0}
	xA, yA, zA := []float64{0.1, 0.3, 0.6}, []float64{0.2, 0.6, 0.2}, []float64{0.3, 0.4, 0.3}

	jE, kE, lE := []float64{0.125, 0.125, 0.5}, []float64{0.125, 0.5, 0.125}, []float64{0.5, 0.125, 0.125}
	jA, kA, lA := []float64{0.1, 0.3, 0.6}, []float64{0.2, 0.6, 0.2}, []float64{0.3, 0.4, 0.3}

	type args struct {
		want   []float64
		actual []float64
	}
	tests := []struct {
		et   LossType
		args args
		want float64
	}{
		{SquaredLoss, args{xE, xA}, 0.26},
		{SquaredLoss, args{yE, yA}, 0.24000000000000005},
		{SquaredLoss, args{zE, zA}, 0.7399999999999999},
		{CrossLoss, args{xE, xA}, 0.5108256237659907},
		{CrossLoss, args{yE, yA}, 0.5108256237659907},
		{CrossLoss, args{zE, zA}, 1.2039728043259361},
		{ExponentialLoss, args{xE, xA}, 1.2969300866657718},
		{ExponentialLoss, args{yE, yA}, 1.2712491503214047},
		{ExponentialLoss, args{zE, zA}, 2.0959355144943643},
		{HellingerDistanceLoss, args{xE, xA}, 0.47476660661688985},
		{HellingerDistanceLoss, args{yE, yA}, 0.47476660661688985},
		{HellingerDistanceLoss, args{zE, zA}, 0.6725157563171542},
		{KullbackLeiblerDivergenceLoss, args{xE, xA}, 0.5108256237659907},
		{KullbackLeiblerDivergenceLoss, args{yE, yA}, 0.5108256237659907},
		{KullbackLeiblerDivergenceLoss, args{zE, zA}, 1.2039728043259361},
		{GeneralizedKullbackLeiblerDivergenceLoss, args{xE, xA}, 0.1108256237659907},
		{GeneralizedKullbackLeiblerDivergenceLoss, args{yE, yA}, 0.1108256237659907},
		{GeneralizedKullbackLeiblerDivergenceLoss, args{zE, zA}, 0.5039728043259362},
		{ItakuraSaitoDistanceLoss, args{xE, xA}, 0.7561265302457962},
		{ItakuraSaitoDistanceLoss, args{yE, yA}, 0.7561265302457962},
		{ItakuraSaitoDistanceLoss, args{zE, zA}, 7.703165502459239},

		{SquaredLoss, args{jE, jA}, 0.041249999999999995},
		{SquaredLoss, args{kE, kA}, 0.021249999999999998},
		{SquaredLoss, args{lE, lA}, 0.14625000000000002},
		{CrossLoss, args{jE, jA}, 0.6937325490479931},
		{CrossLoss, args{kE, kA}, 0.6577722899915205},
		{CrossLoss, args{lE, lA}, 0.8670193441879794},
		{ExponentialLoss, args{jE, jA}, 1.0421126011324575},
		{ExponentialLoss, args{kE, kA}, 1.0214773890662867},
		{ExponentialLoss, args{lE, lA}, 1.1574855232632035},
		{HellingerDistanceLoss, args{jE, jA}, 0.14773244839734279},
		{HellingerDistanceLoss, args{kE, kA}, 0.1051174413596333},
		{HellingerDistanceLoss, args{lE, lA}, 0.26541608903551517},
		{KullbackLeiblerDivergenceLoss, args{jE, jA}, -0.17270142665193855},
		{KullbackLeiblerDivergenceLoss, args{kE, kA}, -0.2086616857084112},
		{KullbackLeiblerDivergenceLoss, args{lE, lA}, 0.0005853684880477716},
		{GeneralizedKullbackLeiblerDivergenceLoss, args{jE, jA}, 0.07729857334806145},
		{GeneralizedKullbackLeiblerDivergenceLoss, args{kE, kA}, 0.04133831429158885},
		{GeneralizedKullbackLeiblerDivergenceLoss, args{lE, lA}, 0.2505853684880478},
		{ItakuraSaitoDistanceLoss, args{jE, jA}, 1.0998490412228452},
		{ItakuraSaitoDistanceLoss, args{kE, kA}, 0.7203520750152961},
		{ItakuraSaitoDistanceLoss, args{lE, lA}, 3.1046329856760693},
	}
	for _, tt := range tests {
		fn := lossFns[tt.et]
		p := reflect.ValueOf(fn).Pointer()
		rf := runtime.FuncForPC(p)
		msg := fmt.Sprintf("%s%v%f", rf.Name(), tt.args.want, tt.want)
		assert.Equal(t, tt.want, fn(tt.args.want, tt.args.actual), msg)
	}
}
