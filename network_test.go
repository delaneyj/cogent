package cogent

import (
	"encoding/json"
	"math/rand"
	"testing"

	"fmt"

	. "github.com/smartystreets/goconvey/convey"
)

func TestNetworks(t *testing.T) {
	Convey("Given a new network with seed 0", t, func() {
		layerCounts := []int{2, 3, 1}
		Convey(fmt.Sprintf("When starting values are %v", layerCounts), func() {
			rand.Seed(0)
			n := newNetwork(`sigmoid`, layerCounts...)
			Convey("JSON should be known", func() {
				bytes, _ := json.MarshalIndent(n, ``, `  `)
				So(string(bytes), ShouldEqual, `{
  "Layers": [
    [
      {
        "weights": [
          -0.28158587086436215, 
          0.570933095808067
        ],
        "bias": -1.6920196326157044,
        "output": 0,
        "activation": "sigmoid"
      },
      {
        "weights": [
          0.1996229111693099,
          1.9195199291234621
        ],
        "bias": 0.8954838794918353,
        "output": 0,
        "activation": "sigmoid"
      },
      {
        "weights": [
          0.41457072128813166,
          -0.48700161491544713
        ],
        "bias": -0.1684059662402393,
        "output": 0,
        "activation": "sigmoid"
      }
    ],
    [
      {
        "weights": [
          0.37056410998929545,
          1.0156889027029008,
          -0.5174422210625114
        ],
        "bias": -0.5565834214413804,
        "output": 0,
        "activation": "sigmoid"
      }
    ]
  ],
  "ErrorName": "meanSquared"
}`)
			})

			Convey("When activating with good inputs", func() {
				inputs := []float64{1, 0}
				outputs, err := n.Activate(inputs)
				expected := []float64{0.48983125507322217}
				Convey("Outputs should be known", func() {
					So(err, ShouldBeNil)
					So(outputs, ShouldResemble, expected)
				})

				Convey("Error() is known", func() {
					val, err := n.Error(expected)
					So(val, ShouldEqual, 0.23993465844660805)
					So(err, ShouldBeNil)
				})
			})

			Convey("When activating with bad inputs", func() {
				inputs := []float64{1}
				outputs, err := n.Activate(inputs)
				Convey("Outputs should be known", func() {
					So(err, ShouldBeError)
					So(outputs, ShouldBeNil)
				})
			})
		})
	})
}
