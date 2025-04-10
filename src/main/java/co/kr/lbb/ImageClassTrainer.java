package co.kr.lbb;

import java.io.File;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

/**
 * 
 */
public class ImageClassTrainer {

    public static final int IMAGE_HEIGHT = 100;
    public static final int IMAGE_WIDTH = 100;
    public static final int IMAGE_CHANNELS = 3;
    public static final int BATCH_SIZE = 32;
    public static final int EPOCHS = 20;
    public static final double UNKNOWN_THRESHOLD = 0.6;

    public static final File ROOT_DIR = new File("data");
    public static final File DATASET_DIR = new File(ROOT_DIR, "training_data/dataset");
    public static final File MODEL_FILE = new File(ROOT_DIR, "img_classification_model.zip");
    public static final File LABEL_FILE = new File(ROOT_DIR, "img_classification_label.json");
    
	/**
	 * image trainer instance
	 */
	private static ImageClassTrainer _instance = null;
	
	//모델 생성에 실패했을경우 instance 제거
	static {
		_instance = new ImageClassTrainer();
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
	 * label 정보
	 */
	private LabelInfo labeInfo;
	
	/**singleton instance
	 * @return
	 */
	public static ImageClassTrainer instance() {
		return _instance;
	}
	
	/**
	 * constructor : singleton
	 */
	private ImageClassTrainer() {}
	
	/**@Override 
	 * @see lbb.Trainer#loadModel()
	 */
	private void loadModel() throws Exception {
		
		//label info load
		labeInfo = new LabelInfo(DATASET_DIR, LABEL_FILE);

		//추가학습 (모델파일이 이미 존재)
		if(MODEL_FILE.exists() == true) {
			// model load
			model = MultiLayerNetwork.load(MODEL_FILE, true);
		}
		//최초 학습시 (모델파일 없음.)
		else {
			//model create
			MultiLayerConfiguration configuration = LayerConfiguration.configuration(labeInfo.getLabelCount(), IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS);
			model = new MultiLayerNetwork(configuration);
			model.init();
			model.setListeners(new ScoreIterationListener(10));
		}
	}

	/**
	 * @throws Exception
	 */
	public void fit() throws Exception {
		System.out.println("---------------------------------- dataset create.");
		DataSetIterator dataIter = ResizeImageDataSetIterator.createIterator(DATASET_DIR, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS, BATCH_SIZE, labeInfo);
		
		System.out.println("---------------------------------- model update check :: label count = " + labeInfo.getLabelCount());
		model = LayerConfiguration.updateOutput(model, labeInfo.getLabelCount());
		
		System.out.println("---------------------------------- start training.");
		model.fit(dataIter, EPOCHS);
		
		System.out.println("---------------------------------- model save.");
		modelSave();
		System.out.println("");
	}
	
	/**
	 * @throws Exception
	 */
	public void modelSave() throws Exception {
		//모델 저장
		model.save(MODEL_FILE);
		
		//label 정보 저장
		labeInfo.save();
	}
	
	/**@Override 
	 * @see lbb.Trainer#predict(java.io.File)
	 */
	/**
	 * @param imgFile
	 * @return
	 * @throws Exception
	 */
	public PredictResult predict(File imgFile) throws Exception {
		
		if(MODEL_FILE.exists() == false) {
			throw new Exception("아직 학습을 진행하지 않았습니다. 먼저 학습을 진행해 주세요");
		}
		
		NativeImageLoader loader = new NativeImageLoader(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS);
		INDArray features = loader.asMatrix(imgFile);
		
		// 스케일링 (0~1)
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
        scaler.transform(features);
		
        // 예측
        INDArray output = model.output(features, false);
        
        //예측 결과 return
        return new PredictResult(labeInfo, output);
	}
	
	
	/**test
	 * @param args
	 */
	public static void main(String[] args) {
		try {
			ImageClassTrainer trainer = ImageClassTrainer.instance();
			//trainer.fit();
			
			File testDir = new File(ROOT_DIR, "test_data");
			File[] files = testDir.listFiles();
			
			for(File f : files) {
				PredictResult result = trainer.predict(f);
				System.out.println("--------------------------------------------------------------------------");
				System.out.println(String.format(":: 원본파일 : [%s]",  f.getName()));
				System.out.println(String.format(":: 예측결과 : [%s]",  result.getLabelName()));
				System.out.println(String.format(":: 결과확률 : [%s]",  result.getConfidence()));
			}
		}
		catch(Exception e) {
			e.printStackTrace();
		}
	}
}
