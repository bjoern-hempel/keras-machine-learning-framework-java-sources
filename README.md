# Keras Machine Learning Framework (Java Sources)

This repository contains the source code for the Keras Java import of the [Keras Machine Learning Framework Repository](https://github.com/bjoern-hempel/keras-machine-learning-framework). See there for more information.

# Installation

```bash
$ git clone https://github.com/bjoern-hempel/keras-machine-learning-framework-java-sources.git
$ cd keras-machine-learning-framework-java-sources
$ mvn clean install
$ mvn package
$ mvn exec:java -Dexec.mainClass="de.ixno.kmls.start.Hello" -Dexec.args="John"
Hello world! Your name is John.
```

# Nine Points Example

You can find more information here: [Nine Points Example @ keras-machine-learning-framework](https://github.com/bjoern-hempel/keras-machine-learning-framework/blob/master/markdown/demos/nine_points.md)

```bash
$ mvn exec:java -Dexec.mainClass="de.ixno.kmls.nine.points.Exec"

Prediction results: x_1 ∈ {0, 0.5, 1} ∧ x_2 ∈ {0, 0.5, 1}
---------------------------------------------------------
x1:  0,00;   x2:  0,00;   prediction:  1,00
x1:  0,00;   x2:  0,50;   prediction:  1,00
x1:  0,00;   x2:  1,00;   prediction:  1,00
x1:  0,50;   x2:  0,00;   prediction:  1,00
x1:  0,50;   x2:  0,50;   prediction:  0,00
x1:  0,50;   x2:  1,00;   prediction:  1,00
x1:  1,00;   x2:  0,00;   prediction:  1,00
x1:  1,00;   x2:  0,50;   prediction:  1,00
x1:  1,00;   x2:  1,00;   prediction:  1,00
---------------------------------------------------------

```

## A. Further Tutorials

* [An introduction to artificial intelligence](https://github.com/friends-of-ai/an-introduction-to-artificial-intelligence)

## B. Sources

Currently there are no sources available.

## C. Authors

* Björn Hempel <bjoern@hempel.li> - _Initial work_ - [https://github.com/bjoern-hempel](https://github.com/bjoern-hempel)

## D. License

This tutorial is licensed under the MIT License - see the [LICENSE.md](/LICENSE.md) file for details

## E. Closing words

Have fun! :)
