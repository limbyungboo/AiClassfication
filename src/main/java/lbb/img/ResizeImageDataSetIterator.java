/**
 * 
 */
package lbb.img;

import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;

/**
 * 
 */
public class ResizeImageDataSetIterator {

	/**
	 * @param datasetDir
	 * @param width
	 * @param height
	 * @param channels
	 * @param batchSize
	 * @return
	 * @throws Exception
	 */
	public static DataSetIterator createIterator(File datasetDir, int width, int height, int channels, int batchSize, Map<String, Integer> labelIndexMap) throws Exception {
        List<DataSet> dataSetList = new ArrayList<>();
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
        NativeImageLoader loader = new NativeImageLoader(height, width, channels);

        File[] labelDirs = datasetDir.listFiles(File::isDirectory);
        if (labelDirs == null) throw new IllegalArgumentException("Invalid dataset directory");

        if(labelIndexMap == null) {
        	labelIndexMap = new HashMap<>();
        }
        
        int labelIdx = labelIndexMap.size();

        for (File labelDir : labelDirs) {
            String labelName = labelDir.getName();
            
            if(labelIndexMap.containsKey(labelName) == false) {
            	labelIndexMap.put(labelName, labelIdx++);
            }
            
            File[] imageFiles = labelDir.listFiles((dir, name) -> name.endsWith(".jpg") || name.endsWith(".png"));
            if (imageFiles == null) continue;

            for (File imgFile : imageFiles) {
                //BufferedImage img = ImageIO.read(imgFile);
            	BufferedImage img = ImageUtils.resizeAndPadImage(imgFile, width, height);
                if (img == null) continue;

                // Resize and convert to INDArray
                INDArray feature = loader.asMatrix(img);
                scaler.transform(feature);

                // One-hot label
                INDArray label = Nd4j.zeros(1, labelDirs.length);
                label.putScalar(0, labelIndexMap.get(labelName), 1.0);

                dataSetList.add(new DataSet(feature, label));
            }
        }

//        Map<String, Integer> labelIndexMap = new HashMap<>();
//        int labelIdx = 0;
//
//        for (File labelDir : labelDirs) {
//            String labelName = labelDir.getName();
//            labelIndexMap.put(labelName, labelIdx++);
//
//            File[] imageFiles = labelDir.listFiles((dir, name) -> name.endsWith(".jpg") || name.endsWith(".png"));
//            if (imageFiles == null) continue;
//
//            for (File imgFile : imageFiles) {
//                //BufferedImage img = ImageIO.read(imgFile);
//            	BufferedImage img = ImageUtils.resizeAndPadImage(imgFile, width, height);
//                if (img == null) continue;
//
//                // Resize and convert to INDArray
//                INDArray feature = loader.asMatrix(img);
//                scaler.transform(feature);
//
//                // One-hot label
//                INDArray label = Nd4j.zeros(labelDirs.length);
//                label.putScalar(labelIndexMap.get(labelName), 1.0);
//
//                dataSetList.add(new DataSet(feature, label));
//            }
//        }

        // Create and return DataSetIterator
        Collections.shuffle(dataSetList, new Random(123));
        return new ListDataSetIterator<>(dataSetList, batchSize);
    }
}
