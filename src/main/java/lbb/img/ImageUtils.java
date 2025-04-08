/**
 * 
 */
package lbb.img;

import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import javax.imageio.ImageIO;

import ai.djl.Application;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;

/**
 * 
 */
public class ImageUtils {

	
	/**image 에 있는 모든 물체(object) 별로 이미지를 자름
	 * @param imgFile
	 * @return
	 * @throws Exception
	 */
	public static List<BufferedImage> cutBoxObject(File imgFile) throws Exception {
		List<BufferedImage> boxList = new ArrayList<>();
		
		
		Image img = ImageFactory.getInstance().fromFile(Paths.get(imgFile.getAbsolutePath()));
		
        Criteria<Image, DetectedObjects> criteria = Criteria.builder()
                .optApplication(Application.CV.OBJECT_DETECTION)
                .setTypes(Image.class, DetectedObjects.class)
                .optFilter("backbone", "mobilenet1.0") // 또는 yolov5, resnet50 등
                .optProgress(new ProgressBar())
                .build();

        try (ZooModel<Image, DetectedObjects> model = criteria.loadModel();
             Predictor<Image, DetectedObjects> predictor = model.newPredictor()) {
        	
            DetectedObjects detections = predictor.predict(img);
            List<DetectedObjects.DetectedObject> items = detections.items();

            // BufferedImage로 변환
            BufferedImage original = (BufferedImage) img.getWrappedImage();

            for (DetectedObjects.DetectedObject obj : items) {
                BoundingBox box = obj.getBoundingBox();
                Rectangle rect = box.getBounds();

                int x = (int) (rect.getX() * original.getWidth());
                int y = (int) (rect.getY() * original.getHeight());
                int w = (int) (rect.getWidth() * original.getWidth());
                int h = (int) (rect.getHeight() * original.getHeight());

                // 박스 잘라내기
                BufferedImage cropped = original.getSubimage(x, y, w, h);
                boxList.add(cropped);
            }
    		return boxList;
        }
	}
	
	/**BufferedImage 를 파일로 저장 (무손실 : png)
	 * @param img
	 * @param saveFile
	 * @throws Exception
	 */
	public static void saveBufferedImg(BufferedImage img, File saveFile) throws Exception {
		ImageIO.write(img, "png", saveFile);
	}
	
}
