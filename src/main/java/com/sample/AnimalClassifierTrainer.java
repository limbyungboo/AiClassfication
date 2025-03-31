package com.sample;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;

import com.sample.block.AnimalBlock;
import com.sample.dataset.ImageDataset;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;

public class AnimalClassifierTrainer {

    private static final int BATCH_SIZE = 32;
    private static final int EPOCHS = 100;
    private static final float LEARNING_RATE = 0.001f;
    
    public static final String[] CLASS_NAMES = {"고양이", "강아지", "사자", "호랑이"};
    
    /**
     * 훈련모델 저장 디렉토리
     */
    public static final File MODEL_PATH = new File("trained_model");
    
    public static final String MODEL_NAME = "animal_classifier";
    
    /**
     * 훈련데이터
     */
    private RandomAccessDataset trainDataset;
    
    /**
     * valid 데이터
     */
    private RandomAccessDataset validDataset;
    
    /**
     * dataset 디렉토리
     */
    private Path datasetRoot;
    
    /**
     * 디바이스 : cpu
     */
    private Device device = Device.cpu();
    
    /**constructor
     * @param datasetPath
     */
    public AnimalClassifierTrainer(String datasetPathRoot) {
    	datasetRoot = Paths.get(datasetPathRoot);
    	initDataset();
    }
    
    /**
     * 
     */
    private void initDataset() {
    	Path trainPath = datasetRoot.resolve("train");
    	Path validPath = datasetRoot.resolve("valid");
    	
    	//train dataset
    	trainDataset = ImageDataset.builder()
				        .setSampling(BATCH_SIZE, true)
				        .addDirectory(trainPath.resolve("cats"), 0)
				        .addDirectory(trainPath.resolve("dogs"), 1)
				        .addDirectory(trainPath.resolve("lion"), 2)
				        .addDirectory(trainPath.resolve("tiger"), 3)
				        .build();
    	
    	//valid dataset
    	validDataset = ImageDataset.builder()
		        .setSampling(BATCH_SIZE, true)
		        .addDirectory(validPath.resolve("cats"), 0)
		        .addDirectory(validPath.resolve("dogs"), 1)
		        .addDirectory(validPath.resolve("lion"), 2)
		        .addDirectory(validPath.resolve("tiger"), 3)
		        .build();
    }
    
    /**
     * @throws Exception
     */
    public void trainImage() throws Exception {
    	Model model = Model.newInstance(MODEL_NAME);
    	model.setBlock(new AnimalBlock(CLASS_NAMES.length));
    	
    	//ai.djl.mxnet.engine.MxModel
    	System.out.println("##### " + model.getClass().getName());
    	
        DefaultTrainingConfig config = new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                .addEvaluator(new Accuracy())
                .optDevices(new Device[]{device})
                .optOptimizer(Optimizer.adam().optLearningRateTracker(Tracker.fixed(LEARNING_RATE)).build())
                .addTrainingListeners(TrainingListener.Defaults.logging());
        
        try (Trainer trainer = model.newTrainer(config)) {
        	Shape inputShape = new Shape(BATCH_SIZE, 3, ImageDataset.IMAGE_WIDTH, ImageDataset.IMAGE_HEIGHT);
            trainer.initialize(inputShape);
            
            //trainning
            EasyTrain.fit(trainer, EPOCHS, trainDataset, validDataset);
            
            System.out.println(">>> 학습을 완료하였습니다.");
            
            // 모델 저장 디렉토리 생성
            if (!MODEL_PATH.exists()) {
            	MODEL_PATH.mkdirs();
            }
            
            // 모델 저장
            model.setProperty("Epoch", String.valueOf(EPOCHS));
            model.save(MODEL_PATH.toPath(), MODEL_NAME);
            System.out.println(">>> 학습한 모델을 저장 하였습니다.");
        }
    }
    
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		String dataset = "C:/001.dev/999.workspace/Ai_Sample1/training_data/dataset";
    	try {
    		AnimalClassifierTrainer trainer = new AnimalClassifierTrainer(dataset);
    		trainer.trainImage();
    	}
    	catch(Exception e) {
    		System.out.println("에러 발생: " + e.getMessage());
    		e.printStackTrace();
    	}
    }
    
} 

