/**
 * 
 */
package com.animal;

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
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class AnimalTrainer {
    private static final int IMAGE_HEIGHT = 64;
    private static final int IMAGE_WIDTH = 64;
    private static final int IMAGE_CHANNELS = 3;
    private static final int OUTPUT_CLASSES = 4;
    private static final int BATCH_SIZE = 32;
    private static final int EPOCHS = 10;
    
    
    public static void main(String[] args) throws Exception {
        // 데이터셋 로드
        File dataDir = new File("training_data/dataset/train");
        
        
        FileSplit fileSplit = new FileSplit(dataDir, NativeImageLoader.ALLOWED_FORMATS, new Random(42));
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        
        ImageRecordReader recordReader = new ImageRecordReader(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, labelMaker);
        recordReader.initialize(fileSplit);
        System.out.println("데이터 개수: " + recordReader.numLabels());
        
        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, BATCH_SIZE, 1, OUTPUT_CLASSES);
        dataIter.setPreProcessor(new ImagePreProcessingScaler(0, 1));

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
                .nOut(OUTPUT_CLASSES)
                .activation(Activation.SOFTMAX)
                .build())
            .setInputType(InputType.convolutional(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
            .build();
        
        File modelFile = new File("animal_model.zip");
        MultiLayerNetwork model = null;
        if(modelFile.exists() == true) {
        	model = ModelSerializer.restoreMultiLayerNetwork(modelFile);
            System.out.println("✅ 기존 모델을 불러왔습니다.");
        }
        else {
        	model = new MultiLayerNetwork(conf);
        	 System.out.println("🚀 새 모델을 생성했습니다.");
        }
        
        //MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));

        // 모델 학습
        System.out.println("🚀 학습 시작!");
        for (int i = 0; i < EPOCHS; i++) {
            model.fit(dataIter);
            System.out.println("✅ Epoch " + (i + 1) + " 완료!");
        }
        
        if(modelFile.exists() == true) {
        	ModelSerializer.writeModel(model, modelFile, true);
        	System.out.println("✅ 업데이트된 모델을 저장했습니다.");
        }
        else {
            // 모델 저장
            model.save(new File("animal_model.zip"));
            System.out.println("🎉 모델 저장 완료!");
        }

    }
}

