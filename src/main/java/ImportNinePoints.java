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

        int inputs = 2;
        INDArray features = Nd4j.create(1, inputs);

        features.putScalar(0, 0, 0.5);
        features.putScalar(0, 1, 0.5);

        prediction = model.output(features).getDouble(0);
        log.info(String.valueOf(prediction));
    }
}
