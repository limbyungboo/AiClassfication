package com.example;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.BatchNorm;
import ai.djl.nn.pooling.Pool;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.Activation;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.mxnet.engine.MxSymbolBlock;
import ai.djl.mxnet.engine.MxEngine;
import ai.djl.mxnet.engine.MxNDManager;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDArray;
import ai.djl.translate.NoopTranslator;
import ai.djl.inference.Predictor;
import ai.djl.util.PairList;
import ai.djl.training.ParameterStore;

import java.nio.file.Path;
import java.nio.file.Paths;

import javax.imageio.ImageIO;

import java.awt.image.BufferedImage;
import java.io.File;

public class AnimalClassifierTrainer {
    private static final int BATCH_SIZE = 32;
    private static final int EPOCHS = 10;
    private static final float LEARNING_RATE = 0.001f;
    
    public static void main(String[] args) {
        try {
            // 데이터셋 경로 설정
            if (args.length != 1) {
                System.out.println("사용법: java AnimalClassifierTrainer <데이터셋_경로>");
                System.exit(1);
            }
            Path datasetPath = Paths.get(args[0]);
            
            // CPU 모드로 강제 설정
            Device device = Device.cpu();
            System.out.println("CPU 모드로 실행됩니다.");
            
            // 모델 생성 (MXNet 엔진 명시)
            Model model = Model.newInstance("animal_classifier");
            model.setBlock(getBlock());
            
            // 데이터셋 로드
            RandomAccessDataset trainDataset = getDataset(datasetPath.resolve("train"));
            RandomAccessDataset validDataset = getDataset(datasetPath.resolve("valid"));
            
            // 학습 설정
            DefaultTrainingConfig config = new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                .addEvaluator(new Accuracy())
                .optDevices(new Device[]{device})
                .optOptimizer(Optimizer.adam().optLearningRateTracker(Tracker.fixed(LEARNING_RATE)).build())
                .addTrainingListeners(TrainingListener.Defaults.logging());
            
            // 학습 시작
            try (Trainer trainer = model.newTrainer(config)) {
                // 입력 형태 설정 [배치크기, 채널, 높이, 너비]
                Shape inputShape = new Shape(BATCH_SIZE, 3, AnimalDataset.IMAGE_WIDTH, AnimalDataset.IMAGE_HEIGHT);
                trainer.initialize(inputShape);

//                test(model, device, "C:/999.python/PyChatGpt/cursor_tensoflow/test_images/12.jpg");
//                System.out.println("");
//                test(model, device, "C:/999.python/PyChatGpt/cursor_tensoflow/test_images/22.jpg");

                System.out.println("학습을 시작합니다...");
                System.out.printf("사용 중인 디바이스: %s%n", device);
                
                EasyTrain.fit(trainer, EPOCHS, trainDataset, validDataset);
                
//                test(model, device, "C:/999.python/PyChatGpt/cursor_tensoflow/test_images/11.jpg");
//                System.out.println("");
//                test(model, device, "C:/999.python/PyChatGpt/cursor_tensoflow/test_images/21.jpg");
                
                // 모델 저장 디렉토리 생성
                File modelDir = new File("trained_model");
                if (!modelDir.exists()) {
                    modelDir.mkdirs();
                }
                
                // 모델 저장 (절대 경로 사용)
                Path modelPath = modelDir.toPath().toAbsolutePath();
                
                // MXNet 엔진 확인
                if (model.getNDManager().getEngine() instanceof MxEngine) {
                    System.out.println("MXNet 엔진을 사용하여 모델을 저장합니다...");
                    
                    try {
                        // 학습된 모델을 MxSymbolBlock으로 변환
                        Block trainedBlock = model.getBlock();
                        MxNDManager manager = (MxNDManager) model.getNDManager();
                        
                        // 입력 설정
                        String inputName = "data";
                        Shape sampleInputShape = new Shape(1, 3, AnimalDataset.IMAGE_WIDTH, AnimalDataset.IMAGE_HEIGHT);
                        NDArray sampleInput = manager.ones(sampleInputShape);
                        NDList sampleInputs = new NDList(sampleInput);
                        
                        // Forward pass를 통해 Symbol 생성
                        ParameterStore parameterStore = new ParameterStore(manager, false);
                        NDList result = trainedBlock.forward(parameterStore, sampleInputs, false);
                        
                        // 모델 저장
                        model.setProperty("Epoch", String.valueOf(EPOCHS));
                        model.save(modelPath, "animal_classifier");
                        System.out.println("모델이 성공적으로 저장되었습니다.");
                        
                        // 메모리 정리
                        sampleInput.close();
                        result.close();
                        
                        // 저장된 파일 확인
                        File symbolFile = modelPath.resolve("animal_classifier-symbol.json").toFile();
                        File paramsFile = modelPath.resolve("animal_classifier-0000.params").toFile();
                        
                        System.out.println("\n저장된 모델 파일 확인:");
                        System.out.println("- symbol.json: " + (symbolFile.exists() ? "존재함" : "없음"));
                        System.out.println("- params: " + (paramsFile.exists() ? "존재함" : "없음"));
                        
                    } catch (Exception e) {
                        System.out.println("모델 저장 중 오류 발생: " + e.getMessage());
                        e.printStackTrace();
                    }
                } else {
                    System.out.println("경고: MXNet 엔진을 사용하고 있지 않습니다.");
                }
                
                // 학습 결과 출력
                System.out.println("\n학습이 완료되었습니다.");
                
                
                //test(model, device);
            }
            
            model.close();
        } catch (Exception e) {
            System.out.println("에러 발생: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static void test(Model model, Device device, String imgPath) throws Exception {
        System.out.println("############################################");
        System.out.println("############################################");
        
        
        //String imgPath = "C:/999.python/PyChatGpt/cursor_tensoflow/test_images/11.jpg";
        Path path = Paths.get(imgPath);
        Predictor<Path, float[]> predictor = model.newPredictor(new AnimalTranslator(), device);
        float[] probabilities = predictor.predict(path);
        
        if(probabilities == null) {
        	System.out.println(">> probabilities is null");
        }
        else {
        	for(float p : probabilities) {
        		System.out.println("predict : " + p);
        	}
        }
    	
    }
    
    private static Block getBlock() {
        return new SequentialBlock()
            // 첫 번째 컨볼루션 블록
            .add(Conv2d.builder()
                .setKernelShape(new Shape(3, 3))
                .setFilters(32)
                .optPadding(new Shape(1, 1))
                .build())
            //.add(BatchNorm.builder().build())
            .add(Activation.reluBlock())
            .add(Pool.maxPool2dBlock(new Shape(2, 2), new Shape(2, 2)))
            
            // 두 번째 컨볼루션 블록
            .add(Conv2d.builder()
                .setKernelShape(new Shape(3, 3))
                .setFilters(64)
                .optPadding(new Shape(1, 1))
                .build())
            //.add(BatchNorm.builder().build())
            .add(Activation.reluBlock())
            .add(Pool.maxPool2dBlock(new Shape(2, 2), new Shape(2, 2)))
            
            // 세 번째 컨볼루션 블록
            .add(Conv2d.builder()
                .setKernelShape(new Shape(3, 3))
                .setFilters(128)
                .optPadding(new Shape(1, 1))
                .build())
            //.add(BatchNorm.builder().build())
            .add(Activation.reluBlock())
            .add(Pool.maxPool2dBlock(new Shape(2, 2), new Shape(2, 2)))
            
            // Flatten 및 완전연결층
            .add(Blocks.batchFlattenBlock())
            .add(Linear.builder().setUnits(512).build())
            .add(Activation.reluBlock())
            .add(Linear.builder().setUnits(2).build()); // 2 classes: cat and dog
    }
    
    private static RandomAccessDataset getDataset(Path datasetPath) throws Exception {
        return AnimalDataset.builder()
            .setSampling(BATCH_SIZE, true)
            .addDirectory(datasetPath.resolve("cats"), 0)
            .addDirectory(datasetPath.resolve("dogs"), 1)
            .build();
    }
} 

