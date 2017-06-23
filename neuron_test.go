package cogent

import (
	"math/rand"
	"testing"

	"encoding/json"

	"fmt"

	. "github.com/smartystreets/goconvey/convey"
)

func TestNeurons(t *testing.T) {
	Convey("Given a seed 0", t, func() {
		rand.Seed(0)
		Convey("When neuron created with 3 inputs and no activation", func() {
			n := newNeuron(3, nil)

			Convey("The weight is be set", func() {
				So(n.Weights, ShouldResemble, []float64{-0.28158587086436215, 0.570933095808067, -1.6920196326157044})
			})

			Convey("The bias is set", func() {
				So(n.Bias, ShouldEqual, 0.1996229111693099)
			})

			Convey("The output is zero", func() {
				So(n.Output, ShouldEqual, 0)
			})

			Convey("The activation defaults to ReLU", func() {
				So(n.Activation, ShouldEqual, "relu")
			})

			Convey("encodes to JSON", func() {
				b, err := json.MarshalIndent(n, "", "  ")
				So(err, ShouldBeNil)
				So(string(b), ShouldEqual, `{
  "weights": [
    -0.28158587086436215,
    0.570933095808067,
    -1.6920196326157044
  ],
  "bias": 0.1996229111693099,
  "output": 0,
  "activation": "relu"
}`)
			})
		})

		Convey("With sigmoid activation", func() {
			a := "sigmoid"
			n := newNeuron(3, &a)

			Convey("with good inputs", func() {
				inputs := []float64{1, 2, 3}
				Convey(fmt.Sprintf("activation %v works", inputs), func() {
					err := n.activate(inputs)
					So(err, ShouldBeNil)
					So(n.Output, ShouldEqual, 0.01770306823606807)
				})
			})

			Convey("with bad inputs", func() {
				inputs := []float64{1, 3}
				Convey(fmt.Sprintf("activation %v works", inputs), func() {
					err := n.activate(inputs)
					So(err, ShouldBeError)
					So(n.Output, ShouldEqual, 0)
				})
			})
		})
	})
}
