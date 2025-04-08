package com.animal;

import java.io.File;
import java.io.IOException;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;

public class AnimalClassifier {
    private static final int IMAGE_HEIGHT = 64;
    private static final int IMAGE_WIDTH = 64;
    private static final int IMAGE_CHANNELS = 3; // RGB
    private static final String[] LABELS = {"Cat", "Dog", "Lion", "Tiger"}; // 학습한 동물 라벨

    public static void main(String[] args) throws IOException {
//        if (args.length == 0) {
//            System.out.println("사용법: java AnimalClassifier <이미지 파일 경로>");
//            return;
//        }
    	
//    	String imgFile = "training_data/dataset/test/tiger01.jpg";
//        File imageFile = new File(imgFile);
//        if (!imageFile.exists()) {
//            System.out.println("파일이 존재하지 않습니다: " + args[0]);
//            return;
//        }

        // 저장된 모델 로드
        File modelFile = new File("animal_model.zip");
        if (!modelFile.exists()) {
            System.out.println("저장된 모델이 없습니다. 먼저 학습을 수행하세요.");
            return;
        }

        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(modelFile);
        
        // 이미지 로드 및 전처리
        NativeImageLoader loader = new NativeImageLoader(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS);
        
    	File testDir = new File("training_data/dataset/test");
    	File[] imgFiles = testDir.listFiles();
    	for(File f : imgFiles) {
            INDArray image = loader.asMatrix(f);
            ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
            scaler.transform(image);

            // 모델 예측 수행
            INDArray output = model.output(image);
            int predictedClass = Nd4j.argMax(output, 1).getInt(0);

            // 결과 출력
            String result = String.format("실제파일 [%s]  :::: 예측동물 [%s]", f.getName(), LABELS[predictedClass]);
            System.out.println(result);
    	}
        
        
    }
}

