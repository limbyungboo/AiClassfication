package lbb;

import java.io.File;

import org.opencv.core.Core;

import lbb.img.ImgTrainer;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class Test {

//	static {
//        System.loadLibrary("opencv_java4110");
//    }
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		try {
//			System.out.println("OpenCV loaded: opencv_java4110 -- " + Core.NATIVE_LIBRARY_NAME);
			
			
			ImgTrainer trainer = ImgTrainer.instance();
			
//			boolean rtn = ((ImgTrainer)trainer).runImageDownload();
//			System.out.println("다운로드 결과 = " + rtn);
			
//			System.out.println("학습 시작");
//			trainer.fit();
//			System.out.println("모델 저장");
//			trainer.modelSave();
			
//			File f = new File("C:\\001.dev\\999.mywork\\AiClassfication\\training_data\\dataset\\test\\cat001.jpg");
//			String result = trainer.predict(f);
//			System.out.println(String.format("원본 : [%s] , 예측결과 : [%s]",  f.getName(), result));
			
			File testDir = new File("C:\\001.dev\\999.mywork\\AiClassfication\\training_data\\dataset\\test1");
			File[] files = testDir.listFiles();
			
			for(File f : files) {
				String result = trainer.predict(f);
				System.out.println(String.format("원본 : [%s] , predict 예측결과 : [%s]",  f.getName(), result));
			}
			
			
//			File dir = new File("C:\\001.dev\\999.mywork\\AiClassfication\\training_data\\dataset\\test1");
//			File[] files = dir.listFiles();
//			
//			for(File f : files) {
//				int idx = f.getName().lastIndexOf("."); 
//				String fileNm = f.getName().substring(0, idx);
//				String exp = f.getName().substring(idx + 1);
//				File outFile = new File(dir, String.format("%s_resize.%s", fileNm, exp));
//				
//				resizeAndPadImage(f, outFile, 100, 100);
//			}
			
		}
		catch(Exception e) {
			e.printStackTrace();
		}
	}

	public static void resizeAndPadImage(File inputFile, File outFile, int targetWidth, int targetHeight) throws IOException {
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
        Image scaledImage = originalImage.getScaledInstance(newWidth, newHeight, Image.SCALE_SMOOTH);

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
        
        // 저장
        
        String format = outFile.getName().substring(outFile.getName().lastIndexOf(".") + 1);
        ImageIO.write(outputImage, format, inputFile);
        System.out.println("이미지를 비율 유지 + 패딩하여 저장했습니다: " + outFile.getName());
    }
}
