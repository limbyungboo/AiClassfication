package com.example;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.Blocks;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.BatchNorm;
import ai.djl.nn.pooling.Pool;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.Activation;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.BlockList;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.file.Paths;
import java.util.Arrays;

public class AnimalClassifier {
    private static final int TARGET_WIDTH = 150;
    private static final int TARGET_HEIGHT = 150;
    private static final String[] CLASS_NAMES = {"고양이", "강아지"};

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

    public static void main(String[] args) {
        if (args.length != 1) {
            System.out.println("사용법: java AnimalClassifier <이미지_경로>");
            System.exit(1);
        }

        String imagePath = args[0];
        String modelPath = "trained_model";

        try {
            // 이미지 로드
            BufferedImage img = ImageIO.read(new File(imagePath));
        	
            Model model1 = Model.newInstance("animal_classifier");
            model1.setBlock(new SequentialBlock());
            model1.load(Paths.get(modelPath));
        	
            // 모델 로드 및 예측
            Criteria<BufferedImage, float[]> criteria = Criteria.builder()
                    .setTypes(BufferedImage.class, float[].class)
                    .optModelPath(Paths.get(modelPath))
                    .optModelName("animal_classifier")
                    .optBlock(getBlock())
                    //.optBlock(new SequentialBlock())
                    .optTranslator(new AnimalTranslator())
                    .optDevice(Device.cpu())
                    .optProgress(new ProgressBar())
                    .build();

            try (ZooModel<BufferedImage, float[]> model = criteria.loadModel()) {
            	
                try (Predictor<BufferedImage, float[]> predictor = model.newPredictor()) {
                    float[] probabilities = predictor.predict(img);
                    
                    // 디버그 정보 출력
                    System.out.println("\n=== 디버그 정보 ===");
                    System.out.println("probabilities 배열 크기: " + probabilities.length);
                    //System.out.println("probabilities 값: " + Arrays.toString(probabilities));
                    
                    int predictedClass = argmax(probabilities);
                    System.out.println("predictedClass: " + predictedClass);
                    System.out.println("==================\n");
                    
                    // 배열 크기 검증
                    if (predictedClass >= CLASS_NAMES.length) {
                        System.out.println("오류: 예측된 클래스 인덱스가 CLASS_NAMES 배열 크기를 초과합니다.");
                        System.out.println("예측된 클래스: " + predictedClass);
                        System.out.println("CLASS_NAMES 크기: " + CLASS_NAMES.length);
                        return;
                    }
                    
                    String predictedAnimal = CLASS_NAMES[predictedClass];
                    
                    System.out.println("\n예측 결과:");
                    System.out.println("동물: " + predictedAnimal);
                    System.out.printf("확률: %.2f%%%n", probabilities[predictedClass] * 100);
                    
                    // 각 클래스별 확률 출력
                    System.out.println("\n각 클래스별 확률:");
                    for (int i = 0; i < CLASS_NAMES.length; i++) {
                        System.out.printf("%s: %.2f%%%n", CLASS_NAMES[i], probabilities[i] * 100);
                    }
                }
            }
        } catch (Exception e) {
            System.out.println("예측 중 오류 발생: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static int argmax(float[] arr) {
        int maxIdx = 0;
        float maxVal = arr[0];
        
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > maxVal) {
                maxVal = arr[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }
    
    private static class AnimalTranslator implements Translator<BufferedImage, float[]> {
        @Override
        public NDList processInput(TranslatorContext ctx, BufferedImage input) {
            //NDManager manager = ctx.getNDManager();
            NDManager manager = ctx.getPredictorManager();
            
            // 이미지 크기 조정
            //BufferedImage resized = ImageUtil.resizeImage(input, TARGET_WIDTH, TARGET_HEIGHT);
            BufferedImage resized = ImageUtil.resizeImage(input, TARGET_WIDTH, TARGET_HEIGHT);
            
            // NCHW 형식으로 직접 배열 생성 [batch=1][channels=3][height][width]
//            float[] imageData = new float[1 * 3 * TARGET_HEIGHT * TARGET_WIDTH];
//            
//            // RGB 값 추출 및 정규화
//            for (int h = 0; h < TARGET_HEIGHT; h++) {
//                for (int w = 0; w < TARGET_WIDTH; w++) {
//                    int rgb = resized.getRGB(w, h);
//                    float r = ((rgb >> 16) & 0xFF) / 255.0f;
//                    float g = ((rgb >> 8) & 0xFF) / 255.0f;
//                    float b = (rgb & 0xFF) / 255.0f;
//                    
//                    // NCHW 형식으로 인덱스 계산
//                    int rIdx = h * TARGET_WIDTH + w;
//                    int gIdx = (TARGET_HEIGHT * TARGET_WIDTH) + (h * TARGET_WIDTH + w);
//                    int bIdx = (2 * TARGET_HEIGHT * TARGET_WIDTH) + (h * TARGET_WIDTH + w);
//                    
//                    imageData[rIdx] = r;  // R 채널
//                    imageData[gIdx] = g;  // G 채널
//                    imageData[bIdx] = b;  // B 채널
//                }
//            }
            float[] imageData = ImageUtil.imageToPixels(resized);
            
            
            // NDArray 생성 및 reshape
            NDArray array = manager.create(imageData);
            array = array.reshape(3, TARGET_HEIGHT, TARGET_WIDTH);
            System.out.println("최종 array shape: " + array.getShape());
            System.out.println("데이터 타입: " + array.getDataType());
            
            float[] result = array.toFloatArray();
            for(float a : result) {
            	System.out.println("###### processInput #######    " + a);
            }
            
            NDList rtn = new NDList(array);
            return rtn;
//        	
        	
//        	input = ImageUtil.resizeImage(input, 224, 224);
//        	NDManager p_manager = ctx.getPredictorManager();
//        	float[] p_imageData = ImageUtil.imageToPixels(input);
//        	
////        	Shape inputShape = new Shape(1, 3, 224, 224);
////        	NDArray n1 = p_manager.create(inputShape);
////        	n1.set(p_imageData);
////        	
////        	long[] n1s = n1.getShape().getShape();
//        	
//        	
//        	NDArray p_array = p_manager.create(p_imageData);
//        	Shape s = p_array.getShape();
//        	long[] aa =s.getShape();
////        	
//        	
//        	p_array = p_array.reshape(3, 224, 224);
//        	long[] cc =p_array.getShape().getShape();
//        	
//            return new NDList(p_array);
        }
        
        @Override
        public float[] processOutput(TranslatorContext ctx, NDList list) {
            NDArray probabilities = list.get(0);
            
            float[] result = null;
            
            // 배치 차원 제거
            probabilities = probabilities.squeeze();
//            result = probabilities.toFloatArray();
//            for(float a : result) {
//            	System.out.println("###### squeeze #######    " + a);
//            }
            
            // 소프트맥스 적용하여 확률값으로 변환
            probabilities = probabilities.softmax(0);
//            result = probabilities.toFloatArray();
//            for(float a : result) {
//            	System.out.println("###### softmax #######    " + a);
//            }
            
            // 결과를 float 배열로 변환
            result = probabilities.toFloatArray();
            for(float a : result) {
            	System.out.println("####### result ######    " + a);
            }
            
            
            
            // 디버그 정보
            //System.out.println("예측 확률: " + Arrays.toString(result));
            
            return result;
        }
        
        @Override
        public Batchifier getBatchifier() {
            return Batchifier.STACK;
        }
    }
} 