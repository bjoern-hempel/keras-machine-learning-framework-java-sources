/**
 * The de.ixno.kmls.evaluate.SinglePicture class imports a given keras transfer
 * learning model for evaluation a given image.
 *
 * @author Björn Hempel <bjoern@hempel.li>
 * @author David Urbansky <david@davidurbansky.com>
 * @version 1.1
 * @web: https://github.com/bjoern-hempel/machine-learning-keras-suite
 *
 *       LICENSE
 *
 *       MIT License
 *
 *       Copyright (c) 2019 Björn Hempel <bjoern@hempel.li>
 *
 *       Permission is hereby granted, free of charge, to any person obtaining a copy
 *       of this software and associated documentation files (the "Software"), to deal
 *       in the Software without restriction, including without limitation the rights
 *       to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *       copies of the Software, and to permit persons to whom the Software is
 *       furnished to do so, subject to the following conditions:
 *
 *       The above copyright notice and this permission notice shall be included in all
 *       copies or substantial portions of the Software.
 *
 *       THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *       IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *       FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *       AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *       LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *       OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *       SOFTWARE.
 */
package de.ixno.kmls.evaluate;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.apache.commons.lang3.tuple.Pair;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.nd4j.linalg.api.ndarray.INDArray;

import ws.palladian.helper.io.FileHelper;
import ws.palladian.helper.math.MathHelper;
import ws.palladian.retrieval.parser.json.JsonArray;
import ws.palladian.retrieval.parser.json.JsonException;
import ws.palladian.retrieval.parser.json.JsonObject;

public class Classifier {
    /**
     * Classify an image with a given model description.
     * 
     * @param jsonFilePath JSON file describing the model an its classes.
     * @param imageFilePath The path to the image that should be classified.
     * @throws IOException
     * @throws JsonException
     */
    public void classify(String jsonFilePath, String imageFilePath) throws IOException, JsonException, UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {
        JsonObject json = new JsonObject(FileHelper.readFileToString(jsonFilePath));

        String modelName = json.tryQueryString("data/model_file_best/model_file");

        // json: environment.classes
        Collection classes = json.tryQueryJsonArray("environment/classes");

        // load model
        ComputationGraph model = KerasModelImport.importKerasModelAndWeights(modelName, false);

        // load image to predict
        // json: transfer_learning.input_dimension
        Integer inputSize = json.tryQueryInt("transfer_learning/input_dimension");
        NativeImageLoader loader = new NativeImageLoader(inputSize, inputSize, 3);
        File file = new File(imageFilePath);
        INDArray image = loader.asMatrix(file);
        image.divi(255);

        // predict given image
        INDArray[] prediction = model.output(image);

        // print out the predicted classes
        List<Pair<String, Double>> pairs = new ArrayList<>();
        int i;
        for (i = 0; i < classes.size(); i++) {
            double value = MathHelper.round(prediction[0].getScalar(0, i).getDouble(0) * 100, 2);
            String className = (String)((JsonArray)classes).get(i);
            pairs.add(Pair.of(className, value));
        }

        // sort pairs
        pairs.sort((o1, o2) -> Double.compare(o2.getValue(), o1.getValue()));

        for (Pair<String, Double> pair : pairs) {
            System.out.println(pair.getKey() + ": " + pair.getValue() + "%");
        }
    }

    public static void main(String[] args) throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException, JsonException {
        new Classifier().classify("food_2.inceptionv3.json", "src/main/resources/cake.jpg");
        new Classifier().classify("food_2.inceptionv3.json", "src/main/resources/burger.jpg");
    }
}
