package pso

import (
	"log"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestNeuralNetwork(t *testing.T) {
	log.SetFlags(log.Lshortfile | log.LstdFlags)

	Convey("Training Demo", t, func() {
		// log.Println("")
		// log.Println("Begin neural network training with particle swarm optimization demo")
		// log.Println("")
		// log.Println("Data is a 30-item subset of the famous Iris flower set")
		// log.Println("Data is sepal length, width, petal length, width -> iris species")
		// log.Println("Iris setosa = 0 0 1, Iris versicolor = 0 1 0, Iris virginica = 1 0 0 ")
		// log.Println("Predicting species from sepal length & width, petal length & width")
		// log.Println("")

		// // this is a 30-item subset of the 150-item set
		// // data has been randomly assign to train (80%) and test (20%) sets
		// // y-value (species) encoding: (0,0,1) = setosa; (0,1,0) = versicolor; (1,0,0) = virginica
		// // for simplicity, data has not been normalized as you would in a real scenario

		trainData := [][]float64{
			[]float64{6.3, 2.9, 5.6, 1.8, 1, 0, 0},
			[]float64{6.9, 3.1, 4.9, 1.5, 0, 1, 0},
			[]float64{4.6, 3.4, 1.4, 0.3, 0, 0, 1},
			[]float64{7.2, 3.6, 6.1, 2.5, 1, 0, 0},
			[]float64{4.7, 3.2, 1.3, 0.2, 0, 0, 1},
			[]float64{4.9, 3, 1.4, 0.2, 0, 0, 1},
			[]float64{7.6, 3, 6.6, 2.1, 1, 0, 0},
			[]float64{4.9, 2.4, 3.3, 1, 0, 1, 0},
			[]float64{5.4, 3.9, 1.7, 0.4, 0, 0, 1},
			[]float64{4.9, 3.1, 1.5, 0.1, 0, 0, 1},
			[]float64{5, 3.6, 1.4, 0.2, 0, 0, 1},
			[]float64{6.4, 3.2, 4.5, 1.5, 0, 1, 0},
			[]float64{4.4, 2.9, 1.4, 0.2, 0, 0, 1},
			[]float64{5.8, 2.7, 5.1, 1.9, 1, 0, 0},
			[]float64{6.3, 3.3, 6, 2.5, 1, 0, 0},
			[]float64{5.2, 2.7, 3.9, 1.4, 0, 1, 0},
			[]float64{7, 3.2, 4.7, 1.4, 0, 1, 0},
			[]float64{6.5, 2.8, 4.6, 1.5, 0, 1, 0},
			[]float64{4.9, 2.5, 4.5, 1.7, 1, 0, 0},
			[]float64{5.7, 2.8, 4.5, 1.3, 0, 1, 0},
			[]float64{5, 3.4, 1.5, 0.2, 0, 0, 1},
			[]float64{6.5, 3, 5.8, 2.2, 1, 0, 0},
			[]float64{5.5, 2.3, 4, 1.3, 0, 1, 0},
			[]float64{6.7, 2.5, 5.8, 1.8, 1, 0, 0},
		}

		testData := [][]float64{
			[]float64{4.6, 3.1, 1.5, 0.2, 0, 0, 1},
			[]float64{7.1, 3, 5.9, 2.1, 1, 0, 0},
			[]float64{5.1, 3.5, 1.4, 0.2, 0, 0, 1},
			[]float64{6.3, 3.3, 4.7, 1.6, 0, 1, 0},
			[]float64{6.6, 2.9, 4.6, 1.3, 0, 1, 0},
			[]float64{7.3, 2.9, 6.3, 1.8, 1, 0, 0},
		}

		// log.Println("The training data is:")
		// labels := []string{"sepal length", "sepal width", "petal length", "petal width", "setosa", "versicolor", "virginica"}
		// ShowMatrix(labels, trainData)
		// log.Println("The test data is:")
		// ShowMatrix(labels, testData)

		numInput := 4
		numHidden := 6
		numOutput := 3
		nn := NewNeuralNetwork(numInput, numHidden, numOutput)
		// log.Print("")
		// log.Printf("Creating a %d-input, %d-hidden, %d-output neural network", numInput, numHidden, numOutput)
		// log.Println("Using tanh and softmax activations")
		So(nn.getWeights(), ShouldResemble, []float64{
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		})

		numParticles := 8
		maxEpochs := 700
		exitError := 0.0060
		probDeath := 0.005

		// log.Printf("Setting numParticles = %d", numParticles)
		// log.Printf("Setting maxEpochs = %d", maxEpochs)
		// log.Printf("Setting early exit MSE error = %3f", exitError)
		// log.Printf("Setting probDeath = %3f", probDeath)
		// other optional PSO parameters (weight decay, death, etc) here

		// log.Println("")
		// log.Println("Beginning training using a particle swarm")
		// log.Println("")
		// start := time.Now()
		bestWeights, epochs := nn.Train(trainData, numParticles, maxEpochs, exitError, probDeath)
		// log.Printf("Training complete, took %d tries in %s.", epochs*numParticles, time.Since(start))
		// log.Println("Final neural network weights and bias values:")

		// ShowVector(bestWeights, 10, 3)
		So(epochs, ShouldEqual, 4968)
		So(bestWeights, ShouldResemble, []float64{
			-1.758750764280674, 6.167729335101607, 7.041242956331752, 10, -0.7854738538109363, -8.377882474785384, -10, 9.36896893116969, 6.910045760583305, 10, 4.543859315413998, 6.39613991567246, 10, -9.51183222367704, 10, 9.859044321878889, -10, 3.8658810013116414, 9.475952164445735, -10, 1.7650412773922217, 10, 5.046827704947329, 9.393419808802244, -5.889280706334276, 1.5959063931500639, -9.967117037684213, 10, -10, -9.831034176697825, 3.8129904791291453, 0.46607253853096964, -10, -6.250331525823661, 7.789206696657659, 6.699456164775033, -1.7758755029137947, 1.1872583633568337, -8.986396292276936, 10, -8.51839872932893, 10, -7.811349545130188, -9.963585887274384, -10, 10, -7.884823072757726, 10, 0.8955392035496308, -10, 1.4577166406473108,
		})

		nn.SetWeights(bestWeights)
		trainAcc := nn.Accuracy(trainData)
		// log.Printf("\nAccuracy on training data = %4f", trainAcc)
		So(trainAcc, ShouldEqual, 1)

		testAcc := nn.Accuracy(testData)
		// log.Printf("\nAccuracy on test data = %4f", testAcc)
		So(testAcc, ShouldEqual, 1)

		// log.Println("\nEnd neural network training with particle swarm optimization demo")
	})
}
