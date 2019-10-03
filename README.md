# Keras Machine Learning Suite (Java Import)

This repository contains the source code for the Keras Java import of the [Keras Machine Learning Suite Repository](https://github.com/bjoern-hempel/keras-machine-learning-suite). See there for more information.

# Installation

```bash
$ git clone https://github.com/bjoern-hempel/keras-machine-learning-suite-java-import.git
$ cd keras-machine-learning-suite-java-import
$ mvn clean install
$ mvn package
$ mvn exec:java -Dexec.mainClass="de.ixno.kmls.start.Hello" -Dexec.args="John"
Hello world! Your name is John.
```

# Nine Points Example

You can find more information here: [Nine Points Example @ keras-machine-learning-suite](https://github.com/bjoern-hempel/keras-machine-learning-suite/blob/master/markdown/demos/nine_points.md)

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
