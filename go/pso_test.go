package pso

import (
	"log"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestNeuralNetwork(t *testing.T) {
	log.SetFlags(log.Lshortfile | log.LstdFlags)

	Convey("Training Demo", t, func() {
		log.Println("")
		log.Println("Begin neural network training with particle swarm optimization demo")
		log.Println("")
		log.Println("Data is a 30-item subset of the famous Iris flower set")
		log.Println("Data is sepal length, width, petal length, width -> iris species")
		log.Println("Iris setosa = 0 0 1, Iris versicolor = 0 1 0, Iris virginica = 1 0 0 ")
		log.Println("Predicting species from sepal length & width, petal length & width")
		log.Println("")

		// this is a 30-item subset of the 150-item set
		// data has been randomly assign to train (80%) and test (20%) sets
		// y-value (species) encoding: (0,0,1) = setosa; (0,1,0) = versicolor; (1,0,0) = virginica
		// for simplicity, data has not been normalized as you would in a real scenario

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

		labels := []string{"sepal length", "sepal width", "petal length", "petal width", "setosa", "versicolor", "virginica"}

		log.Println("The training data is:")
		ShowMatrix(labels, trainData)

		log.Println("The test data is:")
		ShowMatrix(labels, testData)

		log.Println("\nCreating a 4-input, 6-hidden, 3-output neural network")
		log.Println("Using tanh and softmax activations")
		numInput := 4
		numHidden := 6
		numOutput := 3
		nn := NewNeuralNetwork(numInput, numHidden, numOutput)

		numParticles := 8
		maxEpochs := 700
		exitError := 0.0060
		probDeath := 0.005

		log.Printf("Setting numParticles = %d", numParticles)
		log.Printf("Setting maxEpochs = %d", maxEpochs)
		log.Printf("Setting early exit MSE error = %3f", exitError)
		log.Printf("Setting probDeath = %3f", probDeath)
		// other optional PSO parameters (weight decay, death, etc) here

		log.Println("")
		log.Println("Beginning training using a particle swarm")
		log.Println("")
		bestWeights, epochs := nn.Train(trainData, numParticles, maxEpochs, exitError, probDeath)
		log.Printf("Training complete, took %d tries.", epochs*numParticles)
		log.Println("Final neural network weights and bias values:")
		ShowVector(bestWeights, 10, 3)

		nn.SetWeights(bestWeights)
		trainAcc := nn.Accuracy(trainData)
		log.Printf("\nAccuracy on training data = %4f", trainAcc)

		testAcc := nn.Accuracy(testData)
		log.Printf("\nAccuracy on test data = %4f", testAcc)

		log.Println("\nEnd neural network training with particle swarm optimization demo")
	})
}
