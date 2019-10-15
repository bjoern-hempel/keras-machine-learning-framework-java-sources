/**
 * The de.ixno.kmls.nine.points.ImportNinePoints class imports a keras model for a nine points example
 * and does some predictions.
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
package de.ixno.kmls.evaluate;

import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.nd4j.linalg.io.ClassPathResource;

import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.awt.image.BufferedImage;
import java.io.IOException;
import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.WindowConstants;

public class SinglePicture {
    private static final String jsonPath="food_2.inceptionv3.json";
    private static final String modelPath="food_2.inceptionv3.best.12-0.80.h5";
    private static final String predictionPathFull = "predict.image.jpg";

    public static void main(String [] args) throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        String modelPathRessource = new ClassPathResource(modelPath).getFile().getPath();

        /* json: environment.classes */
        String[] classes = {
            "baked_beans",
            "baked_salmon",
            "beef_stew",
            "beef_stroganoff",
            "brownies",
            "bundt_cake",
            "burger",
            "burrito",
            "buttermilk_biscuits",
            "caesar_salad",
            "calzone",
            "cheesecake",
            "chicken_piccata",
            "chicken_wings",
            "cinnamon_roll",
            "cobb_salad",
            "coleslaw",
            "corn_dog",
            "creamed_spinach",
            "donut",
            "empanada",
            "french_fries",
            "frittata",
            "granola_bar",
            "grilled_cheese_sandwich",
            "guacamole",
            "ice_cream",
            "kebabs",
            "key_lime_pie",
            "lasagne",
            "macaroni_and_cheese",
            "margarita",
            "martini",
            "mashed_potatoes",
            "meatballs",
            "meatloaf",
            "muffin",
            "nachos",
            "omelet",
            "pancakes",
            "pizza",
            "popcorn",
            "quesadilla",
            "salad",
            "sloppy_joe",
            "smoothie",
            "soup",
            "spaghetti",
            "stuffed_pepper",
            "waffles"
        };

        /* load model */
        ComputationGraph model = KerasModelImport.importKerasModelAndWeights(modelPathRessource, false);

        /* load image to predict */
        /* json: transfer_learning.input_dimension */
        Integer inputSize = 299;
        NativeImageLoader loader = new NativeImageLoader(inputSize, inputSize, 3);
        File file = new File(predictionPathFull);
        INDArray image = loader.asMatrix(file);
        image.divi(255);

        /* predict given image */
        INDArray[] prediction = model.output(image);

        /* print out the predicted classes */
        int i;
        for (i = 0; i < classes.length; i++) {
            double value = prediction[0].getScalar(0, i).getDouble(0) * 100;
            System.out.println(classes[i] + ": " + value + "%");
        }

        /* optional: show given image (remove the following lines) */
        BufferedImage img = ImageIO.read(new File(predictionPathFull));
        JFrame f = new JFrame();
        JLabel picLabel = new JLabel(new ImageIcon(img));
        f.add(picLabel);
        f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        f.pack();
        f.setVisible(true);
    }
}
