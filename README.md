# Forrest Gump for PSI:ML

This repository represents Forrest Gump simulation for training RL Agents to successfully steer within corridors (pathways).

Prerequisites:
* [Visual Studio 2022](https://visualstudio.microsoft.com/) w/ Unity Development Toolkit
* [Unity 2020.1.2f1](https://unity3d.com/get-unity/download/archive)
* [CUDA 11.7](https://developer.nvidia.com/cuda-toolkit-archive)

Once installed, simply open the project with Unity. Once Unity opens, navigate to the Scenes folder and open (double click) 'Main' scene to open it. Simply press Play button on the top to run the simulation (or Stop when it is running to stop it). Navigate to Scripts folder and double click on any script to open C# scripts in Visual Studio. Scripts that are of your interest: `Logic/AI/Agent.cs` & `Logic/AI/Simulation.cs`.

Simulation consists of multiple agents and easy/medium/hard terrain level (difficulty). Inside Unity Editor, you can change the number of agents & terrain level (0 - Easy, 1 - Medium, 2 - Hard) when you select GameController:

![image](https://user-images.githubusercontent.com/13545172/179606002-ca1be1e4-c3eb-457f-8e64-4621d3016a95.png)

Your task is to implement Agent & Simulation classes, which drive decisions when the runners should steer left/right.

![image](https://user-images.githubusercontent.com/13545172/179607080-f72f2def-e7a5-4ab4-af84-7368722af941.png)

Hint: While the simulation is running, press 'X' to lock the camera on another (alive) runner, 'C' to change Camera mode (3rd person or bird view), and 'Q' to kill all alive runners.

## Agent Class

In the example above, a random RL Agent is implemented. When an episode starts, `Agent` will receive `Runner` object via `AssignRunner()` method. This is the runner which the agent will be driving **within this episode**. `Agent` objects are created at the beginning of the simulation and are kept throughout the entire simulation (i.e. same objects are used across all episodes). At the start of **every episode**, new `Runner` objects are created, and are distributed to `Agent` objects this way, thus `AssignRunner()` method is called once per episode.

Another method which `Agent` class has is `Think()`. While the Runner which this agent is driving is alive (within the episode), `Think()` method is being called repeatedly (i.e. 60 times per second), for every agent (you can think of it being called every frame essentially). Calling `runner.Steer(-1)` here will effectively make the runner steer a bit to the left in that frame, and `runner.Steer(+1)` will effectively make the runner steer a bit to the right in that frame. If `runner.Steer()` doesn't get called, runner will continue forward in that frame but will have a chance to 'rethink' again upon the next call, which will happen essentially in next frame (very soon). In the example above, we are giving a runner random decision to either steer left/right or move forward **every frame**, so since this is happening repeatedly, there is a chance that several consecutive calls will end up having i.e. -1 returned, meaning the runner's steering to the left will be more noticable.

## Simulation Class

This class exposes few simulation events that you can use for any purpose. `Begin()` method is called once per simulation, at the very beginning (essentially when you click Play button). As an argument, it receives an array of agents, and the length of this array basically is the number of runners which you set from the GameController inside UnityEditor. This array, including the objects in it are used throughout the entire simulation. Similarly, `End()` is called when the simulation ends (i.e. when you click Stop button).

Method `EpisodeStart()` is called at the beginning of every episode. Method `RunnerTerminated()` is called whenever a runner dies (by hitting a wall), and as an argument, it receives the number of runners left (not counting the one which just died).

## Grand Intelligence API

In order to implement evolution based neural training, you can use Grand Intelligence API. To start using it, at the beginning of the file, add:

`using GrandIntelligence`

Proposed way to implement RL agents is to use basic NeuroEvolution technique. This includes usages of `BasicBrain` which consists of Neural Networks with Fully Connected layers and Darwin Basic Genetic Evolution Algorithm `DarwinBgea`, which implies that every neural network is of the same structure within the population.

In order to create a new `BasicBrain`, it is necessary to create a `NeuralBuilder` object where it is possible to specify the architecture of the Neural Network:

```
var prototype = new NeuralBuilder(Shape.As2D(1, 7));
prototype.FCLayer(24, ActivationFunction.ELU);
prototype.FCLayer(8, ActivationFunction.ELU);
prototype.FCLayer(3, ActivationFunction.Sigmoid);

var brain1 = new BasicBrain(prototype);
var brain2 = new BasicBrain(prototype);
...
var brain100 = new BasicBrain(prototype);

prototype.Dispose();
```

In the example above, we are defining that our Neural Network defined by this Neural Builder has 7 inputs (2D tensor of 1x7 size => basically 7 float values - if the first layer is Fully Connected layer, input shape should always be 2D as 1xN). Then, we are specifying that the first layer is FullyConnected (`.FCLayer()`), with 24 neurons and ELU activation. Similar can be said for the second layer, just that it has 8 neurons. The third layer has 3 neurons with Sigmoid activation, and also represents the output layer - the last layer specified is also the output layer. After that, the `prototype` object can be reused for multiple creations of `BasicBrain` objects.

Creating new `BasicBrain` objects this way will not initialize parameters. In order to randomly initialize parameters of the given Neural Network (brain), use:

```
brain1.Randomize();
```

`BasicBrain` represents a Darwinean Genetic aspect of a Neural Network, which means it can be viewed as an individual inside a population, or in other words as a DNA (inside a population). From class perspectives, `BasicBrain` class is directly inheriting from `DNA`. Essentially, `NeuralBuilder` can be used to 'compile' a `NeuralNetwork` object, i.e. `NeuralNetwork nn = prototype.Compile();`, but Neural Networks cannot be used directly inside Evolution Algorithms (i.e you cannot add a NeuralNetwork to the population), hence, `BasicBrain` is used for this purpose. In order to create a new `Population` it is possible to do:

```
var firstGeneration = new Population();
firstGeneration.Add(brain1);
firstGeneration.Add(brain2);
...
firstGeneration.Add(brain100);
```

`Population` is essentially an array, so you can retrieve the individuals back by indexing it, but they are retrieved as `DNA` objects.

```
DNA brain1dna = firstGeneration[0];
BasicBrain brain1basic = (BasicBrain)firstGeneration[0];

brain1dna == brain1basic == brain1 // All of these are the same objects
```

In order to use Darwin Basic Genetic Evolution Algorithm, we have to define Selection & Crossover algorithms:

```
Selection selection = Selection.RandFit(4);
```

`RandFit` is a selection algorithm which randomly chooses individuals based on their fitness value for every offspring. As an argument, pass in the number of parents needed for creating 1 offspring (i.e. in the example above, 4 parents are used to crossover and create an offspring).

Then, it is necessary to define a Reproduction (crossover) algorithm:

```
// These two numbers are used only for optimization purposes, they are required to pass in but don't affect the logic of the algorithms
var populationSize = // Set the size of the population, i.e. 'var populationSize = firstGeneration.Size';
var individualSize = // Set the size of the individual inside the population (number of parameters), i.e. 'var individualSize = brain1.NeuralNetwork.Params'

Reproduction reproduction = BasicBrain.SequentialEvenCrossover(populationSize, individualSize);
--or--
Reproduction reproduction = BasicBrain.RandomFullCrossover(populationSize, individualSize);
--or--
Reproduction reproduction = BasicBrain.RandomUniformCrossover(populationSize, individualSize);
```

```
// Arg1: First generation
// Arg2: Selection algorithm
// Arg3: Reproduction algorithm
// Arg4: Total generations
// Arg5: Mutation rate (chance, 0-100%)
var evolution = new DarwinBgea(firstGeneration, selection, reproduction, 1000, 20f);
```

To access the current generation of the evolution algorithm, use:

```
Population currentGeneration = evolution.Generation;
// In the beginning, the above object will be essentially the same as 'firstGeneration' object (holding the same objects)
firstGeneration[0] == currentGeneration[0] // This is true!
```

To give certain individuals fitness value, use `EvolutionValue` property from `BasicBrain` objects. This directly impacts selection algorithm, as this is used when selecting parents to create offsprings; must be a non-negative number, and higher value means the individual is more advanced/more evolved/more adapted/is better in general:

```
brain1.EvolutionValue = 30.5f;
brain2.EvolutionValue = 18.3f;
--or--
var currentBrain1 = (BasicBrain)currentGeneration[0];
currentBrain1.EvolutionValue = 30.5f;
```

Once *all* individuals have their EvolutionValue (fitness) set, to run one evolution cycle and create a new generation, use:

```
evolution.Cycle();
```

Now, using `evolution.Current` will yield with new population of new individuals which were created based on the previous generation, using the specified algorithms.

To access the Neural Network of a `BasicBrain`, use:

```
NeuralNetwork nn1 = brain1.NeutalNetwork;

// To transfer inputs to the Neural Network
float[] inputs = new float[7];
inputs[0] = // ...
...
inputs[6] = // ...
nn1.Input.Transfer(inputs);

// Evaluate neural network
nn1.Eval();

// Retrieve the actual outputs (results) of the Neural Network
float[] outputs = new float[3];
nn1.Output.Retrieve(outputs);
```

GrandIntelligence API contains more functionallity, though this is the basic to get started with. Contact me for further questions.
