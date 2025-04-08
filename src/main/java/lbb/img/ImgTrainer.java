/**
 * 
 */
package lbb.img;

import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.lang3.StringUtils;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;

import lbb.Trainer;

/**
 */
public class ImgTrainer extends Trainer{
	
	/**
	 * image trainer instance
	 */
	private static ImgTrainer _instance = null;
	
	//모델 생성에 실패했을경우 instance 제거
	static {
		_instance = new ImgTrainer();
		try {
			_instance.loadModel();
		}
		catch(Exception e) {
			_instance = null;
		}
	}

	/**
	 * 학습 모델
	 */
	private MultiLayerNetwork model = null;
	
	/**
	 * classes 
	 */
	private List<String> labels;
	
	/**
	 * 모델파일
	 */
	//private File modelFile = new File("img_classification_model.zip");
	
	/**singleton instance
	 * @return
	 */
	public static ImgTrainer instance() {
		return _instance;
	}
	
	/**
	 * constructor : singleton
	 */
	private ImgTrainer() {}
	
	/**@Override 
	 * @see lbb.Trainer#loadModel()
	 */
	private void loadModel() throws Exception {
		if(model != null || ImgTrainerConfig.MODEL_FILE.exists() == false) {
			return;
		}
		ImageRecordReader recordReader = ImgTrainerConfig.recordReader();
		labels = recordReader.getLabels();
		model = MultiLayerNetwork.load(ImgTrainerConfig.MODEL_FILE, true);
//		//저장된 학습 모델이 있는 경우
//		if(ImgTrainerConfig.MODEL_FILE.exists() == true) {
//			model = ModelSerializer.restoreMultiLayerNetwork(modelFile);
//			
//		} 
//		//저장 학습 모델이 없는경우
//		else {
//			model = new MultiLayerNetwork(ImgTrainerConfig.configuration());
//			model.init();
//	        model.setListeners(new ScoreIterationListener(10));
//		}
	}

	/**@Override 
	 * @see lbb.Trainer#fit(org.nd4j.linalg.dataset.DataSet)
	 */
	@Override
	public void fit() throws Exception {
		//이미지 파일 다운로드
//		if(runImageDownload() == false) {
//			throw new Exception("이미지 다운로드 (데이터셋 생성) 실패");
//		}
		
		//학습
		ImageRecordReader recordReader = ImgTrainerConfig.recordReader();
		labels = recordReader.getLabels();
		
		DataSetIterator dataIter = ImgTrainerConfig.trainingDataSet(recordReader);
		
		if(model == null) {
			model = new MultiLayerNetwork(ImgTrainerConfig.configuration(recordReader.numLabels()));
			model.init();
			model.setListeners(new ScoreIterationListener(10));
		}
		model.fit(dataIter, ImgTrainerConfig.EPOCHS);
	}
	
	/**@Override 
	 * @see lbb.Trainer#modelSave()
	 */
	@Override
	public void modelSave() throws Exception {
		//저장된 학습 모델이 있는 경우
		if(ImgTrainerConfig.MODEL_FILE.exists() == true) {
			ModelSerializer.writeModel(model, ImgTrainerConfig.MODEL_FILE, true);
		}
		//저장 학습 모델이 없는경우
		else {
			model.save(ImgTrainerConfig.MODEL_FILE);
		}
	}
	
	/**@Override 
	 * @see lbb.Trainer#predict(java.io.File)
	 */
//	@Override
//	public String predict(File imgFile) throws Exception {
//		if(model == null || labels == null || labels.size() == 0) {
//			throw new Exception("아직 학습을 진행하지 않았습니다. 먼저 학습을 진행해 주세요");
//		}
//		NativeImageLoader loader = new NativeImageLoader(ImgTrainerConfig.IMAGE_HEIGHT, ImgTrainerConfig.IMAGE_WIDTH, ImgTrainerConfig.IMAGE_CHANNELS);
//		INDArray image = loader.asMatrix(imgFile);
//		int[] preInt = model.predict(image);
//		ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
//		scaler.transform(image);
//		
////		DataSet a = new DataSet();
////		a.setFeatures(image)
//		preInt = model.predict(image);
//
//		INDArray output = model.output(image, false);
//		int predictedClassIdx = Nd4j.argMax(output, 1).getInt(0);
//		return labels.get(predictedClassIdx);
//	}
	@Override
	public String predict(File imgFile) throws Exception {
		if(model == null || labels == null || labels.size() == 0) {
			throw new Exception("아직 학습을 진행하지 않았습니다. 먼저 학습을 진행해 주세요");
		}
		List<BufferedImage> imgList = ImageUtils.cutBoxObject(imgFile);
		
		String result = "";
		System.out.println(">> object length = " + imgList.size());
		
		for(BufferedImage img : imgList) {
			NativeImageLoader loader = new NativeImageLoader(ImgTrainerConfig.IMAGE_HEIGHT, ImgTrainerConfig.IMAGE_WIDTH, ImgTrainerConfig.IMAGE_CHANNELS);
			INDArray image = loader.asMatrix(img);
			ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
			scaler.transform(image);
			
			INDArray output = model.output(image, false);
			int predictedClassIdx = Nd4j.argMax(output, 1).getInt(0);
			double confidence = output.getDouble(0, predictedClassIdx);
			System.out.println(String.format("예측확률 : [%s], 예측Class : [%s]", Double.toString(confidence), labels.get(predictedClassIdx)));
			
			if (confidence < ImgTrainerConfig.UNKNOWN_THRESHOLD) {
				result = result + "UNKNOWN >> ";
			}
			else {
				result = result + labels.get(predictedClassIdx) + " >> ";
			}
		}
		return result;
	}
	
	/**
	 * 파이선 연동 : 이미지 데이터를 웹에서 검색하여 다운로드함
	 */
	public boolean runImageDownload() throws Exception {
		List<String> classNameList = loadClassNames();
		String trainingDir = ImgTrainerConfig.TRAINING_DATA_DIR.getAbsolutePath();
		boolean rtn = false;
		for(String className : classNameList) {
			String cmd = String.format("python %s %s %s", ImgTrainerConfig.PYTHON_FILE, className, trainingDir);
			System.out.println("python cmd = " + cmd);
			//@SuppressWarnings("deprecation")
			//Process process = Runtime.getRuntime().exec(cmd);
			int exitCode = runPython(className);
			System.out.println("python exit code = " + exitCode);
			if(exitCode == 0) {
				rtn = true;
			}
		}
		return rtn;
	}
	
	/**
	 * @param className
	 * @return
	 * @throws Exception
	 */
	private int runPython(String className) throws Exception {
		String trainingDir = ImgTrainerConfig.TRAINING_DATA_DIR.getAbsolutePath();
		String cmd = String.format("python %s %s %s", ImgTrainerConfig.PYTHON_FILE, className, trainingDir);
		String home = System.getProperty("user.home");
		ProcessBuilder pb = new ProcessBuilder();
		pb.directory(new File(home));
		pb.command("cmd.exe", "/c", cmd);
		Process process = pb.start();
//		// 프로세스의 출력 스트림을 읽음
//	    BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
//	    String line;
//	    while ((line = reader.readLine()) != null) {
//	        System.out.println(line);
//	    }
//
//	    // 프로세스의 오류 스트림을 읽음
//	    BufferedReader errorReader = new BufferedReader(new InputStreamReader(process.getErrorStream()));
//	    while ((line = errorReader.readLine()) != null) {
//	        System.err.println(line);
//	    }
		return process.waitFor();
	}
	
	/**read class_names.txt
	 * @return
	 * @throws Exception
	 */
	private List<String> loadClassNames() throws Exception {
		//ImgTrainerConfig.CLASS_NAMES_FILE
		List<String> classNameList = new ArrayList<>();
		try(BufferedReader reader = new BufferedReader(new FileReader(ImgTrainerConfig.CLASS_NAMES_FILE))) {
			String line = null;
			while(StringUtils.isNotBlank((line = reader.readLine())) == true) {
				classNameList.add(line);
			} 
		}
		return classNameList;
	}

}
