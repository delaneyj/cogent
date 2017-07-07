package pso

import (
	"log"
	"testing"
	"time"

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
		// So(nn.getWeights(), ShouldResemble, []float64{
		// 	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		// })

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
		start := time.Now()
		bestWeights, epochs := nn.Train(trainData, numParticles, maxEpochs, exitError, probDeath)
		log.Printf("Training complete, took %d tries in %s.", epochs*numParticles, time.Since(start))
		// log.Println("Final neural network weights and bias values:")

		// ShowVector(bestWeights, 10, 3)
		// So(epochs, ShouldEqual, 1384)
		// So(bestWeights, ShouldResemble, []float64{
		// 	0.0508924655570497, -7.2394852994539844, 10, 9.884812599709054, -6.383703231219537, -10, -10, 5.201582531371728, -1.474064166471636, 5.6691271289299205, 0.35949972311568323, -7.39614778232059, -1.1437723032863123, 10, 0.5032735029603613, 4.175115530354129, 10, 3.6480759492273043, -5.418904231357591, -1.3060532105280098, 10, 9.991695114930653, -4.254050721820939, 4.167872861493218, -4.781489224130403, -5.540349128316362, -1.650496675887277, -0.0256199120189822, -3.6797101113189457, 5.509064178293798, -10, 3.873695872730174, -6.855414507479212, -4.114366096697146, 3.392092264316361, -5.737853122538639, -3.7761203551449514, -2.8156256598563676, -0.435854491509225, 4.991424882469882, 6.393787619071993, -10, 10, -2.3211954314739134, -7.125366902878934, -3.6921911074848364, -7.412342664255363, -4.508832868674209, -10, -0.10080334953483647, 6.04611945609111,
		// })

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
