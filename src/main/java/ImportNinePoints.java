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
