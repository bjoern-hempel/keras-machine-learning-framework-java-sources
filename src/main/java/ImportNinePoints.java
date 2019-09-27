/** 
 * The ImportNinePoints class imports a keras model for a nine points example.
 *
 * @author Björn Hempel <bjoern@hempel.li>
 * @version 1.0
 * @web: https://github.com/bjoern-hempel/machine-learning-keras-suite
 *
 * LICENSE
 *
 * MIT License
 *
 * Copyright (c) 2019 Björn Hempel <bjoern@hempel.li>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import java.io.IOException;

public class ImportNinePoints {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(ImportNinePoints.class);

    private static String modelPathFull="C:\\Users\\bjoern\\Development\\keras-machine-learning-suite\\model.h5";

    public static void main(String [] args) throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        log.info("\n\nImport Nine Points\n\n");

        MultiLayerNetwork model = KerasModelImport.importKerasSequentialModelAndWeights(modelPathFull);

        double prediction = 0;
        double x1 = 0;
        double x2 = 0;
        String output = "";

        int inputs = 2;
        INDArray features = Nd4j.create(1, inputs);

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                x1 = i * 0.5;
                x2 = j * 0.5;
                features.putScalar(0, 0, x1);
                features.putScalar(0, 1, x2);
                prediction = model.output(features).getDouble(0);
                output = String.format("x1 = %.2f; x2 = %.2f; prediction = %.2f", x1, x2, prediction);
                log.info(output);
            }
        }
    }
}
