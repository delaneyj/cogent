package main

import (
	"log"
)

func main() {
	log.SetFlags(log.Lshortfile | log.LstdFlags)
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

	s := NewSwarm(8, 4, 6, 3)
	// log.Println(s.JSON())

	s.Train(700, 0.006, trainData)

	log.Println(s.ClassificationAccuracy(testData))
}
