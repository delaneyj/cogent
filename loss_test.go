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
		et   Loss
		args args
		want float64
	}{
		{Squared, args{xE, xA}, 0.26},
		{Squared, args{yE, yA}, 0.24000000000000005},
		{Squared, args{zE, zA}, 0.7399999999999999},
		{Cross, args{xE, xA}, 29.24046029502877},
		{Cross, args{yE, yA}, 29.52814203414752},
		{Cross, args{zE, zA}, 31.31990092004245},
		{Exponential, args{xE, xA}, 1.2969300866657718},
		{Exponential, args{yE, yA}, 1.2712491503214047},
		{Exponential, args{zE, zA}, 2.0959355144943643},
		{HellingerDistance, args{xE, xA}, 0.47476660661688985},
		{HellingerDistance, args{yE, yA}, 0.47476660661688985},
		{HellingerDistance, args{zE, zA}, 0.6725157563171542},
		{KullbackLeiblerDivergence, args{xE, xA}, 0.5108256237659907},
		{KullbackLeiblerDivergence, args{yE, yA}, 0.5108256237659907},
		{KullbackLeiblerDivergence, args{zE, zA}, 1.2039728043259361},
		{GeneralizedKullbackLeiblerDivergence, args{xE, xA}, 0.1108256237659907},
		{GeneralizedKullbackLeiblerDivergence, args{yE, yA}, 0.1108256237659907},
		{GeneralizedKullbackLeiblerDivergence, args{zE, zA}, 0.5039728043259362},
		{ItakuraSaitoDistance, args{xE, xA}, 0.5688888888888888},
		{ItakuraSaitoDistance, args{yE, yA}, 0.5688888888888888},
		{ItakuraSaitoDistance, args{zE, zA}, 4.600555555555555},

		{Squared, args{jE, jA}, 0.041249999999999995},
		{Squared, args{kE, kA}, 0.021249999999999998},
		{Squared, args{lE, lA}, 0.14625000000000002},
		{Cross, args{jE, jA}, 1.2809331454625148},
		{Cross, args{kE, kA}, 1.1223281819524886},
		{Cross, args{lE, lA}, 2.5494440209261597},
		{Exponential, args{jE, jA}, 1.0421126011324575},
		{Exponential, args{kE, kA}, 1.0214773890662867},
		{Exponential, args{lE, lA}, 1.1574855232632035},
		{HellingerDistance, args{jE, jA}, 0.14773244839734279},
		{HellingerDistance, args{kE, kA}, 0.1051174413596333},
		{HellingerDistance, args{lE, lA}, 0.26541608903551517},
		{KullbackLeiblerDivergence, args{jE, jA}, -0.17270142665193855},
		{KullbackLeiblerDivergence, args{kE, kA}, -0.2086616857084112},
		{KullbackLeiblerDivergence, args{lE, lA}, 0.0005853684880477716},
		{GeneralizedKullbackLeiblerDivergence, args{jE, jA}, 0.07729857334806145},
		{GeneralizedKullbackLeiblerDivergence, args{kE, kA}, 0.04133831429158885},
		{GeneralizedKullbackLeiblerDivergence, args{lE, lA}, 0.2505853684880478},
		{ItakuraSaitoDistance, args{jE, jA}, 2.1352777777777776},
		{ItakuraSaitoDistance, args{kE, kA}, 1.0178472222222226},
		{ItakuraSaitoDistance, args{lE, lA}, 6.704522569444445},
	}
	for _, tt := range tests {
		fn := lossFns[tt.et]
		p := reflect.ValueOf(fn).Pointer()
		rf := runtime.FuncForPC(p)
		msg := fmt.Sprintf("%s%v%f", rf.Name(), tt.args.want, tt.want)
		loss := fn(tt.args.want, tt.args.actual)
		assert.Equal(t, tt.want, loss, msg)
	}
}
