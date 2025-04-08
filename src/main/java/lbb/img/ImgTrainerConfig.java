/**
 * 
 */
package lbb.img;

import java.io.File;
import java.util.Random;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * 
 */
public class ImgTrainerConfig {
	
    public static final int IMAGE_HEIGHT = 150;
    public static final int IMAGE_WIDTH = 150;
    public static final int IMAGE_CHANNELS = 3;
    public static final int BATCH_SIZE = 32;
    public static final int EPOCHS = 10;
    public static final double UNKNOWN_THRESHOLD = 0.6;
    
    public static final String PYTHON_FILE =  "C:/999.python/PyChatGpt/machine_learning/img_download.py";

    public static final File ROOT_DIR = new File("data");
    public static final File TRAINING_DATA_DIR = new File(ROOT_DIR, "training_data/dataset");
    public static final File MODEL_FILE = new File(ROOT_DIR, "img_classification_model.zip");
    public static final File CLASS_NAMES_FILE = new File(ROOT_DIR, "class_names.txt");

    /**get image trainer config
     * @return
     */
    public static MultiLayerConfiguration configuration(int outputClasses) {
    	
        // 모델 구성
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(42)
            .updater(new Adam(0.001))  // ✅ Adam 사용
            .weightInit(WeightInit.XAVIER)
            .list()
            .layer(0, new ConvolutionLayer.Builder(3, 3)
                .nIn(IMAGE_CHANNELS)
                .nOut(32)
                .stride(1, 1)
                .activation(Activation.RELU)
                .build())
            .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build())
            .layer(2, new ConvolutionLayer.Builder(3, 3)
                .nOut(64)
                .stride(1, 1)
                .activation(Activation.RELU)
                .build())
            .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build())
            .layer(4, new DenseLayer.Builder()
                .nOut(128)
                .activation(Activation.RELU)
                .build())
            .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(outputClasses)
                .activation(Activation.SOFTMAX)
                .build())
            .setInputType(InputType.convolutional(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
            .build();
    	
    	return conf;
    }
    
//    private static MultiLayerNetwork createModel(int outputClasses) {
//        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
//            .seed(123)
//            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//            .updater(new Adam(0.001))
//            .list()
//            .layer(0, new ConvolutionLayer.Builder(3, 3)
//                    .nIn(IMAGE_CHANNELS)
//                    .nOut(32)
//                    .activation(Activation.RELU)
//                    .weightInit(WeightInit.XAVIER)
//                    .build())
//            .layer(1, new DenseLayer.Builder().nOut(100).activation(Activation.RELU).build())
//            .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
//                    .nOut(outputClasses)
//                    .activation(Activation.SOFTMAX)
//                    .build())
//            .setInputType(InputType.convolutional(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
//            .build();
//
//        MultiLayerNetwork model = new MultiLayerNetwork(conf);
//        model.init();
//        return model;
//    }
    
    /**ImageRecordReader
     * @return
     * @throws Exception
     */
    public static ImageRecordReader recordReader() throws Exception {
        FileSplit fileSplit = new FileSplit(TRAINING_DATA_DIR, NativeImageLoader.ALLOWED_FORMATS, new Random(42));
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        
        ImageRecordReader recordReader = new ImageRecordReader(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, labelMaker);
        recordReader.initialize(fileSplit);
        return recordReader; 
    }
    
    /**DataSetIterator
     * @param recordReader
     * @return
     */
    public static DataSetIterator trainingDataSet(ImageRecordReader recordReader) {
        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, BATCH_SIZE, 1, recordReader.numLabels());
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);
    	return dataIter;
    }
	
}
