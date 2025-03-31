package com.sample;

import java.io.File;
import java.nio.file.Path;

import com.sample.block.AnimalBlock;
import com.sample.translator.AnimalTranslator;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.repository.zoo.Criteria;
import ai.djl.training.util.ProgressBar;

/**
 * 
 */
public class AnimalClassifier {
	
	private Model model;
	
	/**constructor
	 * @throws Exception
	 */
	private AnimalClassifier() throws Exception{
		model = loadModel();
		//ai.djl.repository.zoo.ZooModel
		System.out.println("##### " + model.getClass().getName());
	}

	/**
	 * @param imgPath
	 * @throws Exception
	 */
	public void predict(String imgPath) throws Exception {
		
		File imgFile = new File(imgPath);
		if(imgFile.exists() == false) {
			System.out.println("테스트 이미지 파일이 존재하지 않습니다.");
			return;
		}
		
		int realIdx = -1;
		if(imgFile.getName().indexOf("cat") >= 0) {
			realIdx = 0;
		}
		else if(imgFile.getName().indexOf("dog") >= 0) {
			realIdx = 1;
		}
		else if(imgFile.getName().indexOf("lion") >= 0) {
			realIdx = 2;
		}
		else if(imgFile.getName().indexOf("tiger") >= 0) {
			realIdx = 3;
		}
		
		
        Predictor<Path, float[]> predictor = model.newPredictor(new AnimalTranslator());
        float[] probabilities = predictor.predict(imgFile.toPath());
        
        int idx = argmax(probabilities);
        String className = AnimalClassifierTrainer.CLASS_NAMES[idx];
        String realclassName = AnimalClassifierTrainer.CLASS_NAMES[realIdx];
        
        System.out.println(String.format("실제동몰 : %s,  예측동물 : %s", realclassName, className));
        System.out.printf("확률: %.2f%%%n", probabilities[idx] * 100);
	}
	
	/**load model
	 * @return
	 * @throws Exception
	 */
	private Model loadModel() throws Exception {
		Criteria<Path, float[]> criteria = Criteria.builder()
                .setTypes(Path.class, float[].class)
                .optModelPath(AnimalClassifierTrainer.MODEL_PATH.toPath())
                .optModelName(AnimalClassifierTrainer.MODEL_NAME)
                .optBlock(new AnimalBlock(AnimalClassifierTrainer.CLASS_NAMES.length))
                .optTranslator(new AnimalTranslator())
                .optDevice(Device.cpu())
                .optProgress(new ProgressBar())
                .build();
		
		
		
		return criteria.loadModel();
	}
	
	
    /**
     * @param arr
     * @return
     */
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
    
    /**
     * @param args
     */
    public static void main(String[] args) {
    	try {
    		AnimalClassifier test = new AnimalClassifier();
    		
        	String testImgDir = "C:/001.dev/999.workspace/Ai_Sample1/training_data/dataset/test";
        	
        	File f = new File(testImgDir);
        	File[] images = f.listFiles();
        	
        	for(int i=0; i<images.length; i++) {
    			System.out.println("image = " + images[i].getName());
    			test.predict(images[i].getAbsolutePath());
    			System.out.println("-------------------------------------");
    			System.out.println("");
        	}
    	}
    	catch(Exception e) {
    		e.printStackTrace();
    	}
    	
    }
	
} 