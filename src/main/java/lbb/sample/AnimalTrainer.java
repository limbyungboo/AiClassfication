package lbb.sample;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;

import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Random;

import javax.imageio.ImageIO;

import org.bytedeco.javacv.*;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;
import org.bytedeco.javacv.Frame;
import static org.bytedeco.opencv.global.opencv_objdetect.*;

public class AnimalTrainer {
    public static final int HEIGHT = 100;
    public static final int WIDTH = 100;
    public static final int CHANNELS = 3;
    public static final int BATCH_SIZE = 16;
    public static final int EPOCHS = 10;
    public static final int NUM_CLASSES = 4; // cats, dogs, lion, tiger

    private static final CascadeClassifier faceDetector = new CascadeClassifier("haarcascade_frontalcatface.xml");

    public static void main(String[] args) throws Exception {
        File trainDir = new File("training_data/dataset/train");

        // Load label map
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        FileSplit fileSplit = new FileSplit(trainDir, NativeImageLoader.ALLOWED_FORMATS, new Random(123));
        BalancedPathFilter pathFilter = new BalancedPathFilter(new Random(123), labelMaker, NUM_CLASSES, BATCH_SIZE, 1);
        InputSplit[] inputSplit = fileSplit.sample(pathFilter, 1.0);
        InputSplit trainData = inputSplit[0];

        ImageRecordReader recordReader = new ImageRecordReader(HEIGHT, WIDTH, CHANNELS, labelMaker) {
            //@Override
            public INDArray asMatrix(File imageFile) throws IOException {
                BufferedImage img = ImageIO.read(imageFile);
                BufferedImage face = detectAndCropFace(img);
                if (face == null) return null;

                NativeImageLoader loader = new NativeImageLoader(AnimalTrainer.HEIGHT, AnimalTrainer.WIDTH, AnimalTrainer.CHANNELS);
                return loader.asMatrix(face);
            }
        };

        recordReader.initialize(trainData);

        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, BATCH_SIZE, 1, NUM_CLASSES);
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);

        MultiLayerNetwork model = buildModel();
        model.fit(dataIter, EPOCHS);
        ModelSerializer.writeModel(model, "animal_model.zip", true);
    }

    public static BufferedImage detectAndCropFace(BufferedImage image) {
        Mat mat = bufferedImageToMat(image);
        RectVector faces = new RectVector();
        faceDetector.detectMultiScale(mat, faces);

        if (faces.size() > 0) {
            Rect rect = faces.get(0);
            BufferedImage face = image.getSubimage(rect.x(), rect.y(), rect.width(), rect.height());
            BufferedImage copied = new BufferedImage(rect.width(), rect.height(), BufferedImage.TYPE_3BYTE_BGR);
            Graphics2D g = copied.createGraphics();
            g.drawImage(face, 0, 0, null);
            g.dispose();
            return copied;
        }
        return null;
    }

    public static MultiLayerNetwork buildModel() {
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
            .seed(123)
            .updater(new Adam(0.001))
            .list()
            .layer(new ConvolutionLayer.Builder(5, 5).nIn(CHANNELS).nOut(20).activation(Activation.RELU).build())
            .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).build())
            .layer(new ConvolutionLayer.Builder(5, 5).nOut(50).activation(Activation.RELU).build())
            .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).build())
            .layer(new DenseLayer.Builder().nOut(500).activation(Activation.RELU).build())
            .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                   .nOut(NUM_CLASSES).activation(Activation.SOFTMAX).build())
            .setInputType(InputType.convolutional(HEIGHT, WIDTH, CHANNELS))
            .build();

        return new MultiLayerNetwork(config);
    }

    public static Mat bufferedImageToMat(BufferedImage bi) {
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
        Java2DFrameConverter java2dConverter = new Java2DFrameConverter();
        Frame frame = java2dConverter.convert(bi);
        return converter.convert(frame);
    }
}