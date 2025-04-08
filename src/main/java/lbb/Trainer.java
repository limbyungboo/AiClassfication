/**
 * 
 */
package lbb;

import java.io.File;

/**
 * trainer
 */
public abstract class Trainer {
	
	/**모델 학습
	 */
	public abstract void fit() throws Exception;
	
	/**
	 * 학습모델 저장
	 */
	public abstract void modelSave() throws Exception;
	
	/**test
	 * @param imgFile
	 * @return
	 * @throws Exception
	 */
	public abstract String predict(File imgFile) throws Exception;
}
