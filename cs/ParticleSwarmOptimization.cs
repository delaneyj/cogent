using System;

namespace cs
{
    static class Rand{
        private static UInt32[] state = new UInt32[]{7919, 104729, 1299709, 15485863};

public static UInt32 Next() {
	var t = state[3];
	t ^= t << 11;
	t ^= t >> 8;
	state[3] = state[2];
	state[2] = state[1];
	state[1] = state[0];

	t ^= state[0];
	t ^= state[0] >> 19;
	state[0] = t;
	return t;
}

public static double NextDouble(){
	return (double)Next() / (double)(UInt32.MaxValue);
}

public static int NextMax(int max) {
	return (int)(Next() % (UInt32)max); 
}

public static int NextRange(int min, int max){
    var diff = max - min;
    return NextMax(diff) + min;
}
    }
    
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("\nBegin neural network training with particle swarm optimization demo\n");
            Console.WriteLine("Data is a 30-item subset of the famous Iris flower set");
            Console.WriteLine("Data is sepal length, width, petal length, width -> iris species");
            Console.WriteLine("Iris setosa = 0 0 1, Iris versicolor = 0 1 0, Iris virginica = 1 0 0 ");
            Console.WriteLine("Predicting species from sepal length & width, petal length & width\n");

            // this is a 30-item subset of the 150-item set
            // data has been randomly assign to train (80%) and test (20%) sets
            // y-value (species) encoding: (0,0,1) = setosa; (0,1,0) = versicolor; (1,0,0) = virginica
            // for simplicity, data has not been normalized as you would in a real scenario

            var trainData = new double[24][];
            trainData[0] = new double[] { 6.3, 2.9, 5.6, 1.8, 1, 0, 0 };
            trainData[1] = new double[] { 6.9, 3.1, 4.9, 1.5, 0, 1, 0 };
            trainData[2] = new double[] { 4.6, 3.4, 1.4, 0.3, 0, 0, 1 };
            trainData[3] = new double[] { 7.2, 3.6, 6.1, 2.5, 1, 0, 0 };
            trainData[4] = new double[] { 4.7, 3.2, 1.3, 0.2, 0, 0, 1 };
            trainData[5] = new double[] { 4.9, 3, 1.4, 0.2, 0, 0, 1 };
            trainData[6] = new double[] { 7.6, 3, 6.6, 2.1, 1, 0, 0 };
            trainData[7] = new double[] { 4.9, 2.4, 3.3, 1, 0, 1, 0 };
            trainData[8] = new double[] { 5.4, 3.9, 1.7, 0.4, 0, 0, 1 };
            trainData[9] = new double[] { 4.9, 3.1, 1.5, 0.1, 0, 0, 1 };
            trainData[10] = new double[] { 5, 3.6, 1.4, 0.2, 0, 0, 1 };
            trainData[11] = new double[] { 6.4, 3.2, 4.5, 1.5, 0, 1, 0 };
            trainData[12] = new double[] { 4.4, 2.9, 1.4, 0.2, 0, 0, 1 };
            trainData[13] = new double[] { 5.8, 2.7, 5.1, 1.9, 1, 0, 0 };
            trainData[14] = new double[] { 6.3, 3.3, 6, 2.5, 1, 0, 0 };
            trainData[15] = new double[] { 5.2, 2.7, 3.9, 1.4, 0, 1, 0 };
            trainData[16] = new double[] { 7, 3.2, 4.7, 1.4, 0, 1, 0 };
            trainData[17] = new double[] { 6.5, 2.8, 4.6, 1.5, 0, 1, 0 };
            trainData[18] = new double[] { 4.9, 2.5, 4.5, 1.7, 1, 0, 0 };
            trainData[19] = new double[] { 5.7, 2.8, 4.5, 1.3, 0, 1, 0 };
            trainData[20] = new double[] { 5, 3.4, 1.5, 0.2, 0, 0, 1 };
            trainData[21] = new double[] { 6.5, 3, 5.8, 2.2, 1, 0, 0 };
            trainData[22] = new double[] { 5.5, 2.3, 4, 1.3, 0, 1, 0 };
            trainData[23] = new double[] { 6.7, 2.5, 5.8, 1.8, 1, 0, 0 };

            var testData = new double[6][]; 
            testData[0] = new double[] { 4.6, 3.1, 1.5, 0.2, 0, 0, 1 };
            testData[1] = new double[] { 7.1, 3, 5.9, 2.1, 1, 0, 0 };
            testData[2] = new double[] { 5.1, 3.5, 1.4, 0.2, 0, 0, 1 };
            testData[3] = new double[] { 6.3, 3.3, 4.7, 1.6, 0, 1, 0 };
            testData[4] = new double[] { 6.6, 2.9, 4.6, 1.3, 0, 1, 0 };
            testData[5] = new double[] { 7.3, 2.9, 6.3, 1.8, 1, 0, 0 };

            Console.WriteLine("The training data is:");
            ShowMatrix(trainData, trainData.Length, 1, true);

            Console.WriteLine("The test data is:");
            ShowMatrix(testData, testData.Length, 1, true);

            Console.WriteLine("\nCreating a 4-input, 6-hidden, 3-output neural network");
            Console.WriteLine("Using tanh and softmax activations");
            const int numInput = 4;
            const int numHidden = 6;
            const int numOutput = 3;
            NeuralNetwork nn = new NeuralNetwork(numInput, numHidden, numOutput);

            var numParticles = 12;
            var maxEpochs = 700;
            var exitError = 0.060;
            var probDeath = 0.005;

            Console.WriteLine("Setting numParticles = " + numParticles);
            Console.WriteLine("Setting maxEpochs = " + maxEpochs);
            Console.WriteLine("Setting early exit MSE error = " + exitError.ToString("F3"));
            Console.WriteLine("Setting probDeath = " + probDeath.ToString("F3"));
            // other optional PSO parameters (weight decay, death, etc) here

            Console.WriteLine("\nBeginning training using a particle swarm\n");
            var bestWeights  = nn.Train(trainData, numParticles, maxEpochs, exitError, probDeath);
            Console.WriteLine("Training complete");
            Console.WriteLine("Final neural network weights and bias values:");
            ShowVector(bestWeights, 10, 3, true);

            nn.SetWeights(bestWeights);
            var trainAcc = nn.Accuracy(trainData);
            Console.WriteLine("\nAccuracy on training data = " + trainAcc.ToString("F4"));

            var testAcc = nn.Accuracy(testData);
            Console.WriteLine("\nAccuracy on test data = " + testAcc.ToString("F4"));

            Console.WriteLine("\nEnd neural network training with particle swarm optimization demo\n");
            // Console.ReadLine();

        } // Main

        static void ShowVector(double[] vector, int valsPerRow, int decimals, bool newLine)
        {
            for (var i = 0; i < vector.Length; ++i)
            {
                if (i % valsPerRow == 0) Console.WriteLine("");
                Console.Write(vector[i].ToString("F" + decimals).PadLeft(decimals + 4) + " ");
            }
            if (newLine == true) Console.WriteLine("");
        }

        static void ShowMatrix(double[][] matrix, int numRows, int decimals, bool newLine)
        {
            for (var i = 0; i < numRows; ++i)
            {
                Console.Write(i.ToString().PadLeft(3) + ": ");
                for (var j = 0; j < matrix[i].Length; ++j)
                {
                    if (matrix[i][j] >= 0.0) Console.Write(" "); else Console.Write("-"); ;
                    Console.Write(Math.Abs(matrix[i][j]).ToString("F" + decimals) + " ");
                }
                Console.WriteLine("");
            }
            if (newLine == true) Console.WriteLine("");
        }
    }

    public class NeuralNetwork
    {
        //private static Random rnd; // for BP to initialize wts, in PSO 
        private int numInput;
        private int numHidden;
        private int numOutput;
        private double[] inputs;
        private double[][] ihWeights; // input-hidden
        private double[] hBiases;
        private double[] hOutputs;
        private double[][] hoWeights; // hidden-output
        private double[] oBiases;
        private double[] outputs;

        public NeuralNetwork(int numInput, int numHidden, int numOutput)
        {
            //rnd = new Random(16); // for particle initialization. 16 just gives nice demo
            this.numInput = numInput;
            this.numHidden = numHidden;
            this.numOutput = numOutput;
            this.inputs = new double[numInput];
            this.ihWeights = MakeMatrix(numInput, numHidden);
            this.hBiases = new double[numHidden];
            this.hOutputs = new double[numHidden];
            this.hoWeights = MakeMatrix(numHidden, numOutput);
            this.oBiases = new double[numOutput];
            this.outputs = new double[numOutput];
        } // ctor

        private static double[][] MakeMatrix(int rows, int cols) // helper for ctor
        {
           var result = new double[rows][];
            for (var r = 0; r < result.Length; ++r)
                result[r] = new double[cols];
            return result;
        }

        public void SetWeights(double[] weights)
        {
            // copy weights and biases in weights[] array to i-h weights, i-h biases, h-o weights, h-o biases
            var numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
            if (weights.Length != numWeights)
                throw new Exception("Bad weights array length: ");

            var k = 0; // points into weights param

            for (var i = 0; i < numInput; ++i)
                for (var j = 0; j < numHidden; ++j)
                    ihWeights[i][j] = weights[k++];
            for (var i = 0; i < numHidden; ++i)
                hBiases[i] = weights[k++];
            for (var i = 0; i < numHidden; ++i)
                for (var j = 0; j < numOutput; ++j)
                    hoWeights[i][j] = weights[k++];
            for (var i = 0; i < numOutput; ++i)
                oBiases[i] = weights[k++];
        }

        public double[] GetWeights()
        {
            // returns the current set of wweights, presumably after training
            var numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
           var result = new double[numWeights];
            var k = 0;
            for (var i = 0; i < ihWeights.Length; ++i)
                for (var j = 0; j < ihWeights[0].Length; ++j)
                    result[k++] = ihWeights[i][j];
            for (var i = 0; i < hBiases.Length; ++i)
                result[k++] = hBiases[i];
            for (var i = 0; i < hoWeights.Length; ++i)
                for (var j = 0; j < hoWeights[0].Length; ++j)
                    result[k++] = hoWeights[i][j];
            for (var i = 0; i < oBiases.Length; ++i)
                result[k++] = oBiases[i];
            return result;
        }

        // ----------------------------------------------------------------------------------------

        public double[] ComputeOutputs(double[] xValues)
        {
            if (xValues.Length != numInput)
                throw new Exception("Bad xValues array length");

           var hSums = new double[numHidden]; // hidden nodes sums scratch array
            var oSums = new double[numOutput]; // output nodes sums

            for (var i = 0; i < xValues.Length; ++i) // copy x-values to inputs
                this.inputs[i] = xValues[i];

            for (var j = 0; j < numHidden; ++j)  // compute i-h sum of weights * inputs
                for (var i = 0; i < numInput; ++i)
                    hSums[j] += this.inputs[i] * this.ihWeights[i][j]; // note +=

            for (var i = 0; i < numHidden; ++i)  // add biases to input-to-hidden sums
                hSums[i] += this.hBiases[i];

            for (var i = 0; i < numHidden; ++i)   // apply activation
                this.hOutputs[i] = HyperTanFunction(hSums[i]); // hard-coded

            for (var j = 0; j < numOutput; ++j)   // compute h-o sum of weights * hOutputs
                for (var i = 0; i < numHidden; ++i)
                    oSums[j] += hOutputs[i] * hoWeights[i][j];

            for (var i = 0; i < numOutput; ++i)  // add biases to input-to-hidden sums
                oSums[i] += oBiases[i];

           var softOut = Softmax(oSums); // softmax activation does all outputs at once for efficiency
            Array.Copy(softOut, outputs, softOut.Length);

            var retResult = new double[numOutput]; // could define a GetOutputs method instead
            Array.Copy(this.outputs, retResult, retResult.Length);
            return retResult;
        } // ComputeOutputs

        private static double HyperTanFunction(double x)
        {
            if (x < -20.0) return -1.0; // approximation is correct to 30 decimals
            else if (x > 20.0) return 1.0;
            else return Math.Tanh(x);
        }

        private static double[] Softmax(double[] oSums)
        {
            // does all output nodes at once so scale doesn't have to be re-computed each time
            // determine max output sum
            var max = oSums[0];
            for (var i = 0; i < oSums.Length; ++i)
                if (oSums[i] > max) max = oSums[i];

            // determine scaling factor -- sum of exp(each val - max)
            var scale = 0.0;
            for (var i = 0; i < oSums.Length; ++i)
                scale += Math.Exp(oSums[i] - max);

           var result = new double[oSums.Length];
            for (var i = 0; i < oSums.Length; ++i)
                result[i] = Math.Exp(oSums[i] - max) / scale;

            return result; // now scaled so that xi sum to 1.0
        }

        // ----------------------------------------------------------------------------------------

        public double[] Train(double[][] trainData, int numParticles, int maxEpochs, double exitError, double probDeath)
        {
            // PSO version training. best weights stored into NN and returned
            // particle position == NN weights

            // Random rnd = new Random(16); // 16 just gives nice demo

            var numWeights = (this.numInput * this.numHidden) + (this.numHidden * this.numOutput) + this.numHidden + this.numOutput;

            // use PSO to seek best weights
            var epoch = 0;
            var minX = -10.0; // for each weight. assumes data has been normalized about 0
            var maxX = 10.0;
            var w = 0.729; // inertia weight
            var c1 = 1.49445; // cognitive/local weight
            var c2 = 1.49445; // social/global weight
            // var r1, r2; // cognitive and social randomizations

            var swarm = new Particle[numParticles];
            // best solution found by any particle in the swarm. implicit initialization to all 0.0
           var bestGlobalPosition = new double[numWeights];
            var bestGlobalError = double.MaxValue; // smaller values better

            //double minV = -0.01 * maxX;  // velocities
            //double maxV = 0.01 * maxX;

            // swarm initialization
            // initialize each Particle in the swarm with random positions and velocities
            for (var i = 0; i < swarm.Length; ++i)
            {
               var randomPosition = new double[numWeights];
                for (var j = 0; j < randomPosition.Length; ++j)
                {
                    //double lo = minX;
                    //double hi = maxX;
                    //randomPosition[j] = (hi - lo) * rnd.NextDouble() + lo;
                    randomPosition[j] = (maxX - minX) * Rand.NextDouble() + minX;
                }

                // randomPosition is a set of weights; sent to NN
                //double error = MeanCrossEntropy(trainData, randomPosition);
                var error = MeanSquaredError(trainData, randomPosition);
                var randomVelocity = new double[numWeights];

                for (var j = 0; j < randomVelocity.Length; ++j)
                {
                    //double lo = -1.0 * Math.Abs(maxX - minX);
                    //double hi = Math.Abs(maxX - minX);
                    //randomVelocity[j] = (hi - lo) * rnd.NextDouble() + lo;
                    var lo = 0.1 * minX;
                    var hi = 0.1 * maxX;
                    randomVelocity[j] = (hi - lo) * Rand.NextDouble() + lo;

                }
                swarm[i] = new Particle(randomPosition, error, randomVelocity, randomPosition, error); // last two are best-position and best-error

                // does current Particle have global best position/solution?
                if (swarm[i].error < bestGlobalError)
                {
                    bestGlobalError = swarm[i].error;
                    swarm[i].position.CopyTo(bestGlobalPosition, 0);
                }
            }
            // initialization



            //Console.WriteLine("Entering main PSO weight estimation processing loop");

            // main PSO algorithm

            var sequence = new int[numParticles]; // process particles in random order
            for (var i = 0; i < sequence.Length; ++i)
                sequence[i] = i;

            while (epoch < maxEpochs)
            {
                if (bestGlobalError < exitError) break; // early exit (MSE error)

                var newVelocity = new double[numWeights]; // step 1
                var newPosition = new double[numWeights]; // step 2
                // var newError; // step 3

                Shuffle(sequence); // move particles in random sequence

                for (var pi = 0; pi < swarm.Length; ++pi) // each Particle (index)
                {
                    var i = sequence[pi];
                    var currP = swarm[i]; // for coding convenience

                    // 1. compute new velocity
                    for (var j = 0; j < currP.velocity.Length; ++j) // each x value of the velocity
                    {
                       var r1 = Rand.NextDouble();
                        var r2 = Rand.NextDouble();

                        // velocity depends on old velocity, best position of parrticle, and 
                        // best position of any particle
                        newVelocity[j] = (w * currP.velocity[j]) +
                          (c1 * r1 * (currP.bestPosition[j] - currP.position[j])) +
                          (c2 * r2 * (bestGlobalPosition[j] - currP.position[j]));
                    }

                    newVelocity.CopyTo(currP.velocity, 0);

                    // 2. use new velocity to compute new position
                    for (var j = 0; j < currP.position.Length; ++j)
                    {
                        newPosition[j] = currP.position[j] + newVelocity[j];  // compute new position
                        if (newPosition[j] < minX) // keep in range
                            newPosition[j] = minX;
                        else if (newPosition[j] > maxX)
                            newPosition[j] = maxX;
                    }

                    newPosition.CopyTo(currP.position, 0);

                    // 2b. optional: apply weight decay (large weights tend to overfit) 

                    // 3. use new position to compute new error
                    //newError = MeanCrossEntropy(trainData, newPosition); // makes next check a bit cleaner
                   var newError = MeanSquaredError(trainData, newPosition);
                    currP.error = newError;

                    if (newError < currP.bestError) // new particle best?
                    {
                        newPosition.CopyTo(currP.bestPosition, 0);
                        currP.bestError = newError;
                    }

                    if (newError < bestGlobalError) // new global best?
                    {
                        newPosition.CopyTo(bestGlobalPosition, 0);
                        bestGlobalError = newError;
                    }

                    // 4. optional: does curr particle die?
                    var die = Rand.NextDouble();
                    if (die < probDeath)
                    {
                        // new position, leave velocity, update error
                        for (var j = 0; j < currP.position.Length; ++j)
                            currP.position[j] = (maxX - minX) * Rand.NextDouble() + minX;
                        currP.error = MeanSquaredError(trainData, currP.position);
                        currP.position.CopyTo(currP.bestPosition, 0);
                        currP.bestError = currP.error;

                        if (currP.error < bestGlobalError) // global best by chance?
                        {
                            bestGlobalError = currP.error;
                            currP.position.CopyTo(bestGlobalPosition, 0);
                        }
                    }

                } // each Particle

                ++epoch;

            } // while

            this.SetWeights(bestGlobalPosition);  // best position is a set of weights
            var retResult = new double[numWeights];
            Array.Copy(bestGlobalPosition, retResult, retResult.Length);
            return retResult;return retResult;

        } // Train

        private static void Shuffle(int[] sequence)
        {
            for (var i = 0; i < sequence.Length; ++i)
            {
                var r = Rand.NextRange(i, sequence.Length);
                var tmp = sequence[r];
                sequence[r] = sequence[i];
                sequence[i] = tmp;
            }
        }

        private double MeanSquaredError(double[][] trainData, double[] weights)
        {
            // assumes that centroids and widths have been set!
            this.SetWeights(weights); // copy the weights to evaluate in

            var xValues = new double[numInput]; // inputs
            var tValues = new double[numOutput]; // targets
            var sumSquaredError = 0.0;
            for (var i = 0; i < trainData.Length; ++i) // walk through each training data item
            {
                // following assumes data has all x-values first, followed by y-values!
                Array.Copy(trainData[i], xValues, numInput); // extract inputs
                Array.Copy(trainData[i], numInput, tValues, 0, numOutput); // extract targets
               var yValues = this.ComputeOutputs(xValues); // compute the outputs using centroids, widths, weights, bias values
                for (var j = 0; j < yValues.Length; ++j)
                    sumSquaredError += ((yValues[j] - tValues[j]) * (yValues[j] - tValues[j]));
            }
            return sumSquaredError / trainData.Length;
        }

        public double Accuracy(double[][] testData)
        {
            // percentage correct using winner-takes all
            var numCorrect = 0;
            var numWrong = 0;
            var xValues = new double[numInput]; // inputs
            var tValues = new double[numOutput]; // targets
            // var yValues; // computed Y

            for (var i = 0; i < testData.Length; ++i)
            {
                Array.Copy(testData[i], xValues, numInput); // parse test data into x-values and t-values
                Array.Copy(testData[i], numInput, tValues, 0, numOutput);
               var yValues = this.ComputeOutputs(xValues);
                var maxIndex = MaxIndex(yValues); // which cell in yValues has largest value?

                if (tValues[maxIndex] == 1.0) // ugly. consider AreEqual(double x, double y)
                    ++numCorrect;
                else
                    ++numWrong;
            }
            return (numCorrect * 1.0) / (numCorrect + numWrong); // ugly 2 - check for divide by zero
        }

        private static int MaxIndex(double[] vector) // helper for Accuracy()
        {
            // index of largest value
            var bigIndex = 0;
            var biggestVal = vector[0];
            for (var i = 0; i < vector.Length; ++i)
            {
                if (vector[i] > biggestVal)
                {
                    biggestVal = vector[i]; bigIndex = i;
                }
            }
            return bigIndex;
        }

    } // NeuralNetwork

    // ==============================================================================================

    public class Particle
    {
        public double[] position; // equivalent to NN weights
        public double error; // measure of fitness
        public double[] velocity;

        public double[] bestPosition; // best position found so far by this Particle
        public double bestError;

        //public double age; // optional used to determine death-birth

        public Particle(double[] position, double error, double[] velocity,
          double[] bestPosition, double bestError)
        {
            this.position = new double[position.Length];
            position.CopyTo(this.position, 0);
            this.error = error;
            this.velocity = new double[velocity.Length];
            velocity.CopyTo(this.velocity, 0);
            this.bestPosition = new double[bestPosition.Length];
            bestPosition.CopyTo(this.bestPosition, 0);
            this.bestError = bestError;

            //this.age = 0;
        }

    } 
} 