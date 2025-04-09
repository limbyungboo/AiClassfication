/**
 * 
 */
package lbb.sample;

import ai.djl.Application;
import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.modality.cv.output.DetectedObjects.DetectedObject;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Collectors;

import javax.imageio.ImageIO;

/**
 * 
 */
public class AnimalDetector {

	/**
	 * @param args
	 */
	public static void main(String[] args) throws IOException, TranslateException, Exception {
        String imagePath = "input/animals.jpg";         // 입력 이미지 경로
        String outputDir = "output/";                   // 크롭 이미지 저장 폴더
        float detectionThreshold = 0.3f;                // 신뢰도 임계값

        // 출력 폴더 없으면 생성
        new File(outputDir).mkdirs();

        Image image = ImageFactory.getInstance().fromFile(Paths.get(imagePath));

        Criteria<Image, DetectedObjects> criteria = Criteria.builder()
                .optApplication(Application.CV.OBJECT_DETECTION)
                .setTypes(Image.class, DetectedObjects.class)
                .optFilter("backbone", "mobilenet0.25")    // 기본 YOLO 모델
                .optEngine("PyTorch")
                .optProgress(new ai.djl.training.util.ProgressBar())
                .build();

        try (ZooModel<Image, DetectedObjects> model = ModelZoo.loadModel(criteria);
             Predictor<Image, DetectedObjects> predictor = model.newPredictor()) {

            DetectedObjects detections = predictor.predict(image);
            List<DetectedObjects.DetectedObject> animalObjects = detections.items();

//            List<DetectedObject> animalObjects = detections.items().stream()
//                    .filter(obj -> obj.getProbability() > detectionThreshold)
//                    .filter(obj -> obj.getClassName().matches("cat|dog|bird|cow|sheep|horse"))
//                    .collect(Collectors.toList());

            System.out.println("Detected animals: " + animalObjects.size());

            BufferedImage original = (BufferedImage) image.getWrappedImage();
            
            // 박스 크롭 후 저장
            for (int i = 0; i < animalObjects.size(); i++) {
                DetectedObject obj = animalObjects.get(i);
                BoundingBox box = obj.getBoundingBox();
                
                String s = obj.getClassName();
                
                Rectangle rect = box.getBounds();

                int x = (int) (rect.getX() * original.getWidth());
                int y = (int) (rect.getY() * original.getHeight());
                int w = (int) (rect.getWidth() * original.getWidth());
                int h = (int) (rect.getHeight() * original.getHeight());
                
                Image subImg = image.getSubImage(x, y, w, h);

                String className = obj.getClassName();
                String fileName = className + "_" + i + ".png";
                File outputFile = new File(outputDir, fileName);
                
                //subImg.save(outputFile, "png");
                subImg.save(new FileOutputStream(outputFile), "png");
                System.out.println("Saved cropped: " + outputFile.getAbsolutePath());
            }
        }
    }

}
