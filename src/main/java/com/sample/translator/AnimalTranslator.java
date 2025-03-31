package com.sample.translator;

import java.nio.file.Path;

import com.sample.dataset.ImageDataset;
import com.sample.utils.ImageUtil;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

public class AnimalTranslator implements Translator<Path, float[]> {

    @Override
    public NDList processInput(TranslatorContext ctx, Path input) {
        //NDManager manager = ctx.getNDManager();
        NDManager manager = ctx.getPredictorManager();
        try {
        	NDArray image = ImageUtil.loadImageToNDArray(manager, input, ImageDataset.IMAGE_WIDTH, ImageDataset.IMAGE_HEIGHT);
        	NDList rtn = new NDList(image);
            return rtn;
        }
        catch(Exception e) {
        	throw new IllegalStateException("Failed to load image: " + input, e);
        }
    }
    
    @Override
    public float[] processOutput(TranslatorContext ctx, NDList list) {
        NDArray probabilities = list.get(0);
        
        // 배치 차원 제거
        probabilities = probabilities.squeeze();
        
        // 소프트맥스 적용하여 확률값으로 변환
        probabilities = probabilities.softmax(0);
        
        // 결과를 float 배열로 변환
        return probabilities.toFloatArray();
    }
    
    @Override
    public Batchifier getBatchifier() {
        return Batchifier.STACK;
    }
} 