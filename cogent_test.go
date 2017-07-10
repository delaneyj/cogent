package cogent

import (
	"log"
	"math/rand"

	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

//TestSwarm x
func TestSwarm(t *testing.T) {
	log.SetFlags(log.Lshortfile | log.LstdFlags)

	Convey("Basic Math", t, func() {
		basicMathConfig := &SwarmConfiguration{
			NeuralNetworkConfiguration: NeuralNetworkConfiguration{
				InputCount: 2,
				LayerConfigs: []LayerConfig{
					LayerConfig{
						NodeCount:      3,
						ActivationName: "hyperbolicTangent",
					},
					LayerConfig{
						NodeCount:      2,
						ActivationName: "softmax",
					},
				},
			},
			ParticleCount: 8,
		}

		Convey("XOR", func() {
			rand.Seed(1)
			trainData := []Data{
				Data{
					Inputs:  []float64{0, 0},
					Outputs: []float64{0, 1},
				},
				Data{
					Inputs:  []float64{0, 1},
					Outputs: []float64{1, 0},
				},
				Data{
					Inputs:  []float64{1, 0},
					Outputs: []float64{1, 0},
				},
				Data{
					Inputs:  []float64{1, 1},
					Outputs: []float64{0, 1},
				},
			}

			s := NewSwarm(basicMathConfig)
			tries := s.Train(trainData)

			accuracy := s.ClassificationAccuracy(trainData)
			So(accuracy, ShouldEqual, 1)
			So(tries, ShouldEqual, 3096)
		})

		Convey("AND", func() {
			rand.Seed(0)
			trainData := []Data{
				Data{
					Inputs:  []float64{0, 0},
					Outputs: []float64{0, 1},
				},
				Data{
					Inputs:  []float64{0, 1},
					Outputs: []float64{0, 1},
				},
				Data{
					Inputs:  []float64{1, 0},
					Outputs: []float64{0, 1},
				},
				Data{
					Inputs:  []float64{1, 1},
					Outputs: []float64{1, 0},
				},
			}

			s := NewSwarm(basicMathConfig)
			tries := s.Train(trainData)

			accuracy := s.ClassificationAccuracy(trainData)
			So(accuracy, ShouldEqual, 1)
			So(tries, ShouldEqual, 160)
		})

		Convey("NOT", func() {
			rand.Seed(1)
			trainData := []Data{
				Data{
					Inputs:  []float64{0, 0},
					Outputs: []float64{1, 0},
				},
				Data{
					Inputs:  []float64{0, 1},
					Outputs: []float64{0, 1},
				},
				Data{
					Inputs:  []float64{1, 0},
					Outputs: []float64{0, 1},
				},
				Data{
					Inputs:  []float64{1, 1},
					Outputs: []float64{0, 1},
				},
			}

			s := NewSwarm(basicMathConfig)
			tries := s.Train(trainData)

			accuracy := s.ClassificationAccuracy(trainData)
			So(accuracy, ShouldEqual, 1)
			So(tries, ShouldEqual, 56)
		})

		Convey("OR", func() {
			rand.Seed(1)
			trainData := []Data{
				Data{
					Inputs:  []float64{0, 0},
					Outputs: []float64{0, 1},
				},
				Data{
					Inputs:  []float64{0, 1},
					Outputs: []float64{1, 0},
				},
				Data{
					Inputs:  []float64{1, 0},
					Outputs: []float64{1, 0},
				},
				Data{
					Inputs:  []float64{1, 1},
					Outputs: []float64{1, 0},
				},
			}

			s := NewSwarm(basicMathConfig)
			tries := s.Train(trainData)

			accuracy := s.ClassificationAccuracy(trainData)
			So(accuracy, ShouldEqual, 1)
			So(tries, ShouldEqual, 16)
		})

		Convey("NAND", func() {
			rand.Seed(1)
			trainData := []Data{
				Data{
					Inputs:  []float64{0, 0},
					Outputs: []float64{1, 0},
				},
				Data{
					Inputs:  []float64{0, 1},
					Outputs: []float64{1, 0},
				},
				Data{
					Inputs:  []float64{1, 0},
					Outputs: []float64{1, 0},
				},
				Data{
					Inputs:  []float64{1, 1},
					Outputs: []float64{0, 1},
				},
			}

			s := NewSwarm(basicMathConfig)
			tries := s.Train(trainData)

			accuracy := s.ClassificationAccuracy(trainData)
			So(accuracy, ShouldEqual, 1)
			So(tries, ShouldEqual, 64)
		})

		Convey("NOR", func() {
			rand.Seed(1)
			trainData := []Data{
				Data{
					Inputs:  []float64{0, 0},
					Outputs: []float64{1, 0},
				},
				Data{
					Inputs:  []float64{0, 1},
					Outputs: []float64{0, 1},
				},
				Data{
					Inputs:  []float64{1, 0},
					Outputs: []float64{0, 1},
				},
				Data{
					Inputs:  []float64{1, 1},
					Outputs: []float64{0, 1},
				},
			}

			s := NewSwarm(basicMathConfig)
			tries := s.Train(trainData)

			accuracy := s.ClassificationAccuracy(trainData)
			So(accuracy, ShouldEqual, 1)
			So(tries, ShouldEqual, 56)
		})

		Convey("XNOR", func() {
			rand.Seed(1)
			trainData := []Data{
				Data{
					Inputs:  []float64{0, 0},
					Outputs: []float64{1, 0},
				},
				Data{
					Inputs:  []float64{0, 1},
					Outputs: []float64{0, 1},
				},
				Data{
					Inputs:  []float64{1, 0},
					Outputs: []float64{0, 1},
				},
				Data{
					Inputs:  []float64{1, 1},
					Outputs: []float64{1, 0},
				},
			}

			s := NewSwarm(basicMathConfig)
			tries := s.Train(trainData)

			accuracy := s.ClassificationAccuracy(trainData)
			So(accuracy, ShouldEqual, 1)
			So(tries, ShouldEqual, 232)
		})
	})

	Convey("Flowers", t, func() {
		rand.Seed(1)
		trainData := []Data{
			Data{
				Inputs:  []float64{6.3, 2.9, 5.6, 1.8},
				Outputs: []float64{1, 0, 0},
			},
			Data{
				Inputs:  []float64{6.9, 3.1, 4.9, 1.5},
				Outputs: []float64{0, 1, 0},
			},
			Data{
				Inputs:  []float64{4.6, 3.4, 1.4, 0.3},
				Outputs: []float64{0, 0, 1},
			},
			Data{
				Inputs:  []float64{7.2, 3.6, 6.1, 2.5},
				Outputs: []float64{1, 0, 0},
			},
			Data{
				Inputs:  []float64{4.7, 3.2, 1.3, 0.2},
				Outputs: []float64{0, 0, 1},
			},
			Data{
				Inputs:  []float64{4.9, 3, 1.4, 0.2},
				Outputs: []float64{0, 0, 1},
			},
			Data{
				Inputs:  []float64{7.6, 3, 6.6, 2.1},
				Outputs: []float64{1, 0, 0},
			},
			Data{
				Inputs:  []float64{4.9, 2.4, 3.3, 1},
				Outputs: []float64{0, 1, 0},
			},
			Data{
				Inputs:  []float64{5.4, 3.9, 1.7, 0.4},
				Outputs: []float64{0, 0, 1},
			},
			Data{
				Inputs:  []float64{4.9, 3.1, 1.5, 0.1},
				Outputs: []float64{0, 0, 1},
			},
			Data{
				Inputs:  []float64{5, 3.6, 1.4, 0.2},
				Outputs: []float64{0, 0, 1},
			},
			Data{
				Inputs:  []float64{6.4, 3.2, 4.5, 1.5},
				Outputs: []float64{0, 1, 0},
			},
			Data{
				Inputs:  []float64{4.4, 2.9, 1.4, 0.2},
				Outputs: []float64{0, 0, 1},
			},
			Data{
				Inputs:  []float64{5.8, 2.7, 5.1, 1.9},
				Outputs: []float64{1, 0, 0},
			},
			Data{
				Inputs:  []float64{6.3, 3.3, 6, 2.5},
				Outputs: []float64{1, 0, 0},
			},
			Data{
				Inputs:  []float64{5.2, 2.7, 3.9, 1.4},
				Outputs: []float64{0, 1, 0},
			},
			Data{
				Inputs:  []float64{7, 3.2, 4.7, 1.4},
				Outputs: []float64{0, 1, 0},
			},
			Data{
				Inputs:  []float64{6.5, 2.8, 4.6, 1.5},
				Outputs: []float64{0, 1, 0},
			},
			Data{
				Inputs:  []float64{4.9, 2.5, 4.5, 1.7},
				Outputs: []float64{1, 0, 0},
			},
			Data{
				Inputs:  []float64{5.7, 2.8, 4.5, 1.3},
				Outputs: []float64{0, 1, 0},
			},
			Data{
				Inputs:  []float64{5, 3.4, 1.5, 0.2},
				Outputs: []float64{0, 0, 1},
			},
			Data{
				Inputs:  []float64{6.5, 3, 5.8, 2.2},
				Outputs: []float64{1, 0, 0},
			},
			Data{
				Inputs:  []float64{5.5, 2.3, 4, 1.3},
				Outputs: []float64{0, 1, 0},
			},
			Data{
				Inputs:  []float64{6.7, 2.5, 5.8, 1.8},
				Outputs: []float64{1, 0, 0},
			},
		}

		testData := []Data{
			Data{
				Inputs:  []float64{4.6, 3.1, 1.5, 0.2},
				Outputs: []float64{0, 0, 1},
			},
			Data{
				Inputs:  []float64{7.1, 3, 5.9, 2.1},
				Outputs: []float64{1, 0, 0},
			},
			Data{
				Inputs:  []float64{5.1, 3.5, 1.4, 0.2},
				Outputs: []float64{0, 0, 1},
			},
			Data{
				Inputs:  []float64{6.3, 3.3, 4.7, 1.6},
				Outputs: []float64{0, 1, 0},
			},
			Data{
				Inputs:  []float64{6.6, 2.9, 4.6, 1.3},
				Outputs: []float64{0, 1, 0},
			},
			Data{
				Inputs:  []float64{7.3, 2.9, 6.3, 1.8},
				Outputs: []float64{1, 0, 0},
			},
		}

		config := SwarmConfiguration{
			NeuralNetworkConfiguration: NeuralNetworkConfiguration{
				InputCount: 4,
				LayerConfigs: []LayerConfig{
					LayerConfig{
						NodeCount:      10,
						ActivationName: "hyperbolicTangent",
					},
					LayerConfig{
						NodeCount:      3,
						ActivationName: "softmax",
					},
				},
			},
			ParticleCount: 8,
		}
		s := NewSwarm(&config)
		tries := s.Train(trainData)

		accuracy := s.ClassificationAccuracy(testData)
		So(accuracy, ShouldEqual, 1)
		So(tries, ShouldEqual, 560)
	})
}
