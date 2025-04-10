/**
 * 
 */
package lbb.img;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
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

        	//------------------------------------------------
        	//yolo 모델에서 인식가능한 객체 전체 리스트 목록 출력
        	//classes.txt
//        	Path p = model.getModelPath();
//        	System.out.println("model path = " + p.toString());
//        	Path c = p.resolve("classes.txt");
//        	if(Files.exists(c)) {
//        		List<String> classNames = Files.readAllLines(c);
//                System.out.println("총 클래스 수: " + classNames.size());
//                for (int i = 0; i < classNames.size(); i++) {
//                    System.out.println(i + ": " + classNames.get(i));
//                }
//        	}
        	//------------------------------------------------
        	
            DetectedObjects detections = predictor.predict(img);
            List<DetectedObjects.DetectedObject> items = detections.items();
            //필터링을 해서 객체를 인식할경우 
            //items = items.stream().filter(obj -> obj.getClassName().matches("dog|bird")).collect(Collectors.toList());
            
            
            // BufferedImage로 변환
            BufferedImage original = (BufferedImage) img.getWrappedImage();

            for (DetectedObjects.DetectedObject obj : items) {
                BoundingBox box = obj.getBoundingBox();
                Rectangle rect = box.getBounds();
                String className = obj.getClassName();
                System.out.println("className = " + className);

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
	
	/**
	 * @param inputFile
	 * @param targetWidth
	 * @param targetHeight
	 * @return
	 * @throws IOException
	 */
	public static BufferedImage resizeAndPadImage(File inputFile, int targetWidth, int targetHeight) throws IOException {
        // 원본 이미지 읽기
        BufferedImage originalImage = ImageIO.read(inputFile);
        int originalWidth = originalImage.getWidth();
        int originalHeight = originalImage.getHeight();

        // 원본 비율 기준으로 리사이즈 크기 계산
        double widthRatio = (double) targetWidth / originalWidth;
        double heightRatio = (double) targetHeight / originalHeight;
        double scale = Math.min(widthRatio, heightRatio);

        int newWidth = (int) (originalWidth * scale);
        int newHeight = (int) (originalHeight * scale);

        // 이미지 리사이즈
        java.awt.Image scaledImage = originalImage.getScaledInstance(newWidth, newHeight, java.awt.Image.SCALE_SMOOTH);

        // 새 BufferedImage 생성 (배경 흰색으로)
        BufferedImage outputImage = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = outputImage.createGraphics();
        g2d.setColor(Color.WHITE); // 배경 색상
        g2d.fillRect(0, 0, targetWidth, targetHeight);

        // 가운데 정렬하여 이미지 배치
        int x = (targetWidth - newWidth) / 2;
        int y = (targetHeight - newHeight) / 2;
        g2d.drawImage(scaledImage, x, y, null);
        g2d.dispose();
        return outputImage;
    }
}
