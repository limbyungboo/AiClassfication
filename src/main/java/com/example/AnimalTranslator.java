package com.example;

import java.nio.file.Path;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

public class AnimalTranslator implements Translator<Path, float[]> {

//    private static final int TARGET_WIDTH = 224;
//    private static final int TARGET_HEIGHT = 224;
//    private static final String[] CLASS_NAMES = {"고양이", "강아지"};

    @Override
    public NDList processInput(TranslatorContext ctx, Path input) {
        NDManager manager = ctx.getNDManager();
        //NDManager manager = ctx.getPredictorManager();
        
        try {
        	NDArray image = ImageUtil.loadImageToNDArray(manager, input, AnimalDataset.IMAGE_WIDTH, AnimalDataset.IMAGE_HEIGHT);
        	//NDList rtn = new NDList(NDImageUtils.toTensor(image));
        	NDList rtn = new NDList(image);
            return rtn;
        }
        catch(Exception e) {
        	throw new IllegalStateException("Failed to load image: " + input, e);
        }
        
        
        // 이미지 크기 조정
        //BufferedImage resized = ImageUtil.resizeImage(input, TARGET_WIDTH, TARGET_HEIGHT);
        
        // NCHW 형식으로 직접 배열 생성 [batch=1][channels=3][height][width]
//        float[] imageData = new float[1 * 3 * TARGET_HEIGHT * TARGET_WIDTH];
//        
//        // RGB 값 추출 및 정규화
//        for (int h = 0; h < TARGET_HEIGHT; h++) {
//            for (int w = 0; w < TARGET_WIDTH; w++) {
//                int rgb = resized.getRGB(w, h);
//                float r = ((rgb >> 16) & 0xFF) / 255.0f;
//                float g = ((rgb >> 8) & 0xFF) / 255.0f;
//                float b = (rgb & 0xFF) / 255.0f;
//                
//                // NCHW 형식으로 인덱스 계산
//                int rIdx = h * TARGET_WIDTH + w;
//                int gIdx = (TARGET_HEIGHT * TARGET_WIDTH) + (h * TARGET_WIDTH + w);
//                int bIdx = (2 * TARGET_HEIGHT * TARGET_WIDTH) + (h * TARGET_WIDTH + w);
//                
//                imageData[rIdx] = r;  // R 채널
//                imageData[gIdx] = g;  // G 채널
//                imageData[bIdx] = b;  // B 채널
//            }
//        }
//        float[] imageData = ImageUtil.imageToPixels(resized);
//        
//        
//        // NDArray 생성 및 reshape
//        NDArray array = manager.create(imageData);
//        array = array.reshape(3, TARGET_HEIGHT, TARGET_WIDTH);
//        System.out.println("최종 array shape: " + array.getShape());
//        System.out.println("데이터 타입: " + array.getDataType());
//        
//        float[] result = array.toFloatArray();
////        for(float a : result) {
////        	System.out.println("###### processInput #######    " + a);
////        }
        
       
//    	
    	
//    	input = ImageUtil.resizeImage(input, 224, 224);
//    	NDManager p_manager = ctx.getPredictorManager();
//    	float[] p_imageData = ImageUtil.imageToPixels(input);
//    	
////    	Shape inputShape = new Shape(1, 3, 224, 224);
////    	NDArray n1 = p_manager.create(inputShape);
////    	n1.set(p_imageData);
////    	
////    	long[] n1s = n1.getShape().getShape();
//    	
//    	
//    	NDArray p_array = p_manager.create(p_imageData);
//    	Shape s = p_array.getShape();
//    	long[] aa =s.getShape();
////    	
//    	
//    	p_array = p_array.reshape(3, 224, 224);
//    	long[] cc =p_array.getShape().getShape();
//    	
//        return new NDList(p_array);
    }
    
    @Override
    public float[] processOutput(TranslatorContext ctx, NDList list) {
    	
    	System.out.println("###### NDList size #######    " + list.size());
        NDArray probabilities = list.get(0);
        
        float[] result = null;
//        result = probabilities.toFloatArray();
//        for(float a : result) {
//        	System.out.println("###### probabilities #######    " + a);
//        }

        
        // 배치 차원 제거
        probabilities = probabilities.squeeze();
//        result = probabilities.toFloatArray();
//        for(float a : result) {
//        	System.out.println("###### squeeze #######    " + a);
//        }
        
        // 소프트맥스 적용하여 확률값으로 변환
        probabilities = probabilities.softmax(0);
//        result = probabilities.toFloatArray();
//        for(float a : result) {
//        	System.out.println("###### softmax #######    " + a);
//        }
        
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