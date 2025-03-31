package com.example;

import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.file.Paths;
import java.util.Arrays;

import javax.imageio.ImageIO;

import ai.djl.Device;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.SequentialBlock;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

public class AnimalClassifier_BACK {
    private static final int TARGET_WIDTH = 150;
    private static final int TARGET_HEIGHT = 150;
    private static final String[] CLASS_NAMES = {"고양이", "강아지"};

    public static void main(String[] args) {
    	
    	
        if (args.length != 1) {
            System.out.println("사용법: java AnimalClassifier <이미지_경로>");
            System.exit(1);
        }

        String imagePath = args[0];
        //String modelPath = "animal_classifier_model";
        String modelPath = "trained_model";

        try {
            // 이미지 로드
            BufferedImage img = ImageIO.read(new File(imagePath));
            
            // 모델 로드 및 예측
            Criteria<BufferedImage, float[]> criteria = Criteria.builder()
                    .setTypes(BufferedImage.class, float[].class)
                    .optModelPath(Paths.get(modelPath))
                    .optTranslator(new AnimalTranslator())
                    .optDevice(Device.cpu())
                    .optProgress(new ProgressBar())
                    .optModelName("animal_classifier")
                    .optBlock(new SequentialBlock())
                    .build();

            try (ZooModel<BufferedImage, float[]> model = criteria.loadModel()) {
                try (Predictor<BufferedImage, float[]> predictor = model.newPredictor()) {
                    float[] probabilities = predictor.predict(img);
                    
                    
                    // 디버그 정보 출력
                    System.out.println("\n=== 디버그 정보 ===");
                    System.out.println("원본 예측값: " + Arrays.toString(probabilities));
                    System.out.println("예측값 shape: [1, 2]");
                    int argmax = argmax(probabilities);
                    System.out.println("argmax 인덱스: " + argmax);
                    System.out.println("==================\n");
                    
                    // 예측값 순서를 뒤집음 (TensorFlow와 PyTorch의 출력 순서가 다름)
                    float temp = probabilities[0];
                    probabilities[0] = probabilities[1];
                    probabilities[1] = temp;
                    argmax = argmax(probabilities);
                    
                    // 결과 출력
                    System.out.println("\n예측 결과:");
                    System.out.printf("이 이미지는 %s입니다. (확률: %.2f%%)\n",
                            CLASS_NAMES[argmax],
                            probabilities[argmax] * 100);
                    
                    System.out.println("\n각 클래스별 확률:");
                    for (int i = 0; i < CLASS_NAMES.length; i++) {
                        System.out.printf("%s: %.2f%%\n",
                                CLASS_NAMES[i],
                                probabilities[i] * 100);
                    }
                }
            }
        } catch (Exception e) {
            System.out.println("에러 발생: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static int argmax(float[] array) {
        int maxIndex = 0;
        float maxValue = array[0];
        
        for (int i = 1; i < array.length; i++) {
            if (array[i] > maxValue) {
                maxValue = array[i];
                maxIndex = i;
            }
        }
        
        return maxIndex;
    }
    
    private static class AnimalTranslator implements Translator<BufferedImage, float[]> {
        @Override
        public NDList processInput(TranslatorContext ctx, BufferedImage input) {
            NDManager manager = ctx.getNDManager();
            
            // 이미지 크기 조정 및 정규화
            BufferedImage resized = ImageUtil.resizeImage(input, TARGET_WIDTH, TARGET_HEIGHT);
            float[] pixels = ImageUtil.imageToPixels(resized);
            
            // 텐서 생성 및 reshape
            NDArray array = manager.create(pixels);
            array = array.reshape(1, TARGET_HEIGHT, TARGET_WIDTH, 3);
            
            return new NDList(array);
        }
        
        @Override
        public float[] processOutput(TranslatorContext ctx, NDList list) {
            NDArray probabilities = list.get(0);
            float[] result = probabilities.toFloatArray();
            return result;
        }
        
        @Override
        public Batchifier getBatchifier() {
            return Batchifier.STACK;
        }
    }
} 









