package cogent

// func Test_Activations(t *testing.T) {
// 	log.SetFlags(log.Lshortfile | log.LstdFlags)

// 	expected := []struct {
// 		at   ActivationMode
// 		want [][]float64
// 	}{
// 		{
// 			Identity,
// 			[][]float64{
// 				{0, 0},
// 				{0, 1},
// 				{1, 0},
// 				{1, 1},
// 				{1, 2, 3, 4},
// 				{100, 200, 300, 400},
// 				{-1.7976931348623157e+308, 1.7976931348623157e+308},
// 			}},
// 		{
// 			BinaryStep,
// 			[][]float64{
// 				{0, 0},
// 				{0, 1},
// 				{1, 0},
// 				{1, 1},
// 				{1, 1, 1, 1},
// 				{1, 1, 1, 1},
// 				{0, 1},
// 			}},
// 		{
// 			Sigmoid,
// 			[][]float64{
// 				{0.5, 0.5},
// 				{0.5, 0.7310585786300049},
// 				{0.7310585786300049, 0.5},
// 				{0.7310585786300049, 0.7310585786300049},
// 				{0.7310585786300049, 0.8807970779778823, 0.9525741268224334, 0.9820137900379085},
// 				{1, 1, 1, 1},
// 				{0, 1},
// 			}},
// 		{
// 			HyperbolicTangent,
// 			[][]float64{
// 				{0, 0},
// 				{0, 0.7615941559557649},
// 				{0.7615941559557649, 0},
// 				{0.7615941559557649, 0.7615941559557649},
// 				{0.7615941559557649, 0.9640275800758169, 0.9950547536867305, 0.999329299739067},
// 				{1, 1, 1, 1},
// 				{-1, 1},
// 			}},
// 		{
// 			ArcTan,
// 			[][]float64{
// 				{0, 0},
// 				{0, 0.7853981633974483},
// 				{0.7853981633974483, 0},
// 				{0.7853981633974483, 0.7853981633974483},
// 				{0.7853981633974483, 1.1071487177940904, 1.2490457723982544, 1.3258176636680323},
// 				{1.5607966601082313, 1.5657963684609382, 1.56746300580716, 1.5682963320032104},
// 				{-1.5707963267948966, 1.5707963267948966},
// 			}},
// 		{
// 			Softsign,
// 			[][]float64{
// 				{0, 0},
// 				{0, 0.5},
// 				{0.5, 0},
// 				{0.5, 0.5},
// 				{0.5, 0.6666666666666666, 0.75, 0.8},
// 				{0.9900990099009901, 0.9950248756218906, 0.9966777408637874, 0.9975062344139651},
// 				{-1, 1},
// 			}},
// 		{
// 			ISRU,
// 			[][]float64{
// 				{0, 0},
// 				{0, 0.7071067811865475},
// 				{0.7071067811865475, 0},
// 				{0.7071067811865475, 0.7071067811865475},
// 				{0.7071067811865475, 0.8944271909999159, 0.9486832980505138, 0.9701425001453319},
// 				{0.9999500037496876, 0.9999875002343701, 0.9999944444907404, 0.9999968750146484},
// 				{0, 0},
// 			}},
// 		{
// 			ReLU,
// 			[][]float64{
// 				{0, 0},
// 				{0, 1},
// 				{1, 0},
// 				{1, 1},
// 				{1, 2, 3, 4},
// 				{100, 200, 300, 400},
// 				{0, 1.7976931348623157e+308},
// 			}},
// 		{
// 			LeakyReLU,
// 			[][]float64{
// 				{0, 0},
// 				{0, 1},
// 				{1, 0},
// 				{1, 1},
// 				{1, 2, 3, 4},
// 				{100, 200, 300, 400},
// 				{-1.7976931348623156e+306, 1.7976931348623157e+308},
// 			}},
// 		{
// 			ELU,
// 			[][]float64{
// 				{0, 0},
// 				{0, 1},
// 				{1, 0},
// 				{1, 1},
// 				{1, 2, 3, 4},
// 				{100, 200, 300, 400},
// 				{-1, 1.7976931348623157e+308},
// 			}},
// 		{
// 			SELU,
// 			[][]float64{
// 				{0, 0},
// 				{0, 1.0507},
// 				{1.0507, 0},
// 				{1.0507, 1.0507},
// 				{1.0507, 2.1014, 3.1521, 4.2028},
// 				{105.07, 210.14, 315.21, 420.28},
// 				{-1.758094282, 1.7976931348623157e+308},
// 			}},
// 		{
// 			SoftPlus,
// 			[][]float64{
// 				{0.6931471805599453, 0.6931471805599453},
// 				{0.6931471805599453, 1.3132616875182228},
// 				{1.3132616875182228, 0.6931471805599453},
// 				{1.3132616875182228, 1.3132616875182228},
// 				{1.3132616875182228, 2.1269280110429727, 3.048587351573742, 4.0181499279178094},
// 				{100, 200, 300, 400},
// 				{0, 1.7976931348623157e+308},
// 			}},
// 		{
// 			BentIdentity,
// 			[][]float64{
// 				{0, 0},
// 				{0, 1.2071067811865475},
// 				{1.2071067811865475, 0},
// 				{1.2071067811865475, 1.2071067811865475},
// 				{1.2071067811865475, 2.618033988749895, 4.08113883008419, 5.561552812808831},
// 				{149.50249993750313, 299.5012499921876, 449.5008333310185, 599.5006249990234},
// 				{1.7976931348623157e+308, 1.7976931348623157e+308},
// 			}},
// 		{
// 			Sinusoid,
// 			[][]float64{
// 				{0, 0},
// 				{0, 0.8414709848078965},
// 				{0.8414709848078965, 0},
// 				{0.8414709848078965, 0.8414709848078965},
// 				{0.8414709848078965, 0.9092974268256816, 0.1411200080598672, -0.7568024953079282},
// 				{-0.5063656411097588, -0.8732972972139945, -0.9997558399011496, -0.8509193596391765},
// 				{-1.7976931348623157e+308, 1.7976931348623157e+308},
// 			}},
// 		{
// 			Sinc,
// 			[][]float64{
// 				{1, 1},
// 				{1, 0.8414709848078965},
// 				{0.8414709848078965, 1},
// 				{0.8414709848078965, 0.8414709848078965},
// 				{0.8414709848078965, 0.4546487134128408, 0.0470400026866224, -0.18920062382698205},
// 				{-0.005063656411097588, -0.004366486486069973, -0.0033325194663371654, -0.0021272983990979414},
// 				{1.7976931348623157e+308, 1.7976931348623157e+308},
// 			}},
// 		{
// 			Gaussian,
// 			[][]float64{
// 				{1, 1},
// 				{1, 0.36787944117144233},
// 				{0.36787944117144233, 1},
// 				{0.36787944117144233, 0.36787944117144233},
// 				{0.36787944117144233, 0.01831563888873418, 0.00012340980408667956, 1.1253517471925912e-07},
// 				{0, 0, 0, 0},
// 				{0, 0},
// 			}},
// 		{
// 			Softmax,
// 			[][]float64{
// 				{0.5, 0.5},
// 				{0.2689414213699951, 0.7310585786300049},
// 				{0.7310585786300049, 0.2689414213699951},
// 				{0.5, 0.5},
// 				{0.03205860328008499, 0.08714431874203257, 0.23688281808991013, 0.6439142598879724},
// 				{5.148200222412013e-131, 1.3838965267367376e-87, 3.720075976020836e-44, 1},
// 				{0, 1},
// 			}},
// 		{
// 			Maxout,
// 			[][]float64{
// 				{0, 0},
// 				{0, 1},
// 				{1, 0},
// 				{1, 1},
// 				{0, 0, 0, 4},
// 				{0, 0, 0, 400},
// 				{0, 1.7976931348623157e+308},
// 			},
// 		},
// 	}

// 	examples := [][]float64{
// 		{0, 0},
// 		{0, 1},
// 		{1, 0},
// 		{1, 1},
// 		{1, 2, 3, 4},
// 		{100, 200, 300, 400},
// 		{-math.MaxFloat64, math.MaxFloat64},
// 	}

// 	for _, e := range expected {
// 		for i, example := range examples {
// 			fn := activations[e.at]
// 			assert.NotNil(t, fn, e.at)
// 			actual := fn(example)
// 			expected := e.want[i]
// 			assert.Equal(t, expected, actual, fmt.Sprintf("%d:%02d", e.at, i))
// 		}
// 	}
// }

// func Test_ActivationsSpeed(t *testing.T) {
// 	log.SetFlags(log.Lshortfile | log.LstdFlags)

// 	expected := []struct {
// 		at   ActivationMode
// 		want float64
// 	}{
// 		{BinaryStep, 29},
// 		{Sigmoid, 56},
// 		{HyperbolicTangent, 77},
// 		{ArcTan, 60},
// 		{Softsign, 40},
// 		{ISRU, 19},
// 		{ReLU, 20},
// 		{LeakyReLU, 18},
// 		{ELU, 27},
// 		{SELU, 13.8},
// 		{SoftPlus, 40},
// 		{BentIdentity, 35},
// 		{Sinusoid, 100},
// 		{Sinc, 80},
// 		{Gaussian, 55},
// 		{Softmax, 107},
// 		{Maxout, 45},
// 	}

// 	examples := make([][]float64, 1000000)
// 	for i := range examples {
// 		examples[i] = make([]float64, rand.Intn(25))
// 		for j := range examples[i] {
// 			examples[i][j] = rand.Float64()
// 		}
// 	}

// 	timeToActivate := func(at ActivationMode) time.Duration {
// 		start := time.Now()
// 		activate := activations[at]
// 		assert.NotNil(t, activate)
// 		for _, e := range examples {
// 			activate(e)
// 		}
// 		d := time.Since(start)
// 		return d
// 	}

// 	identity := timeToActivate(Identity)

// 	labels := []string{
// 		"BinaryStep",
// 		"Sigmoid",
// 		"HyperbolicTangent",
// 		"ArcTan",
// 		"Softsign",
// 		"ISRU",
// 		"ReLU",
// 		"LeakyReLU",
// 		"ELU",
// 		"SELU",
// 		"SoftPlus",
// 		"BentIdentity",
// 		"Sinusoid",
// 		"Sinc",
// 		"Gaussian",
// 		"Softmax",
// 		"Maxout",
// 	}

// 	for i, e := range expected {
// 		duration := timeToActivate(e.at)
// 		ratio := float64(duration) / float64(identity)
// 		maxDelta := e.want * 0.5
// 		assert.InDelta(t, e.want, ratio, maxDelta, labels[i])
// 	}
// }
