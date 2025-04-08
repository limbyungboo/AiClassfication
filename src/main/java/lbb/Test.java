package lbb;

import java.io.File;

import lbb.img.ImgTrainer;

public class Test {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		try {
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
		}
		catch(Exception e) {
			e.printStackTrace();
		}
	}

}
