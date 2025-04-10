/**
 * 
 */
package co.kr.lbb;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.fasterxml.jackson.databind.ObjectMapper;

/**
 * 
 */
public class LabelInfo {

	private Map<String, Integer> labelMap;
	
	private List<String> labelsNameList;
	
	private File labelFile;
	
	private ObjectMapper mapper = new ObjectMapper();
	
	/**
	 * @param trainDataDir
	 * @param labelFile
	 * @throws Exception
	 */
	public LabelInfo(File datasetDir, File labelFile) throws Exception {
		this.labelFile = labelFile;
		
		//label 정보 파일이 존재할경우
		if(labelFile.exists() == true) {
			labelMap = mapper.readValue(labelFile, mapper.getTypeFactory().constructMapType(Map.class, String.class, Integer.class));
		}
		//label 정보 파일이 없고, 데이터셋 디렉토리가 존재하는 경우
		else if(datasetDir.exists() == true) {
			File[] labelDirs = datasetDir.listFiles(File::isDirectory);
			if (labelDirs == null || labelDirs.length == 0) { 
				throw new IllegalArgumentException("Invalid dataset directory");
			}
			labelMap = new HashMap<>();
			int labelIdx = labelMap.size();
			for (File labelDir : labelDirs) {
				labelMap.put(labelDir.getName(), labelIdx++);
			}
		}
		else {
			throw new IllegalArgumentException("Invalid dataset directory");
		}
		
		//labelMap 의 인덱스 값으로 정렬하여 label name 을 리스트에 저장
		//테스트시 받은 인덱스 값으로 label name 취득
		resetLabelsNameList();
	}
	
	/**
	 * labelNamesList 재정의
	 */
	private void resetLabelsNameList() {
		// 인덱스 기준으로 정렬된 라벨 리스트 만들기
        String[] labels = new String[labelMap.size()];

        for (Map.Entry<String, Integer> entry : labelMap.entrySet()) {
            labels[entry.getValue()] = entry.getKey();
        }
        labelsNameList = new ArrayList<>(Arrays.asList(labels));
	}
	
	/**
	 * @param labelName
	 */
	public void putLabel(String labelName) {
		//이미 존재하는 label 이면 아무처리도 하지 않음.
		if(labelMap.containsKey(labelName) == true) {
			return;
		}
		labelMap.put(labelName, labelMap.size());
		labelsNameList.add(labelName);
	}
	
	/**label index
	 * @param labelName
	 * @return
	 */
	public int getLabelIndex(String labelName) {
		return labelMap.get(labelName);
	}
	
	/**label name 취득
	 * @param idx
	 * @return
	 */
	public String getLabelName(int idx) {
		return labelsNameList.get(idx);
	}
	
	/**label 갯수
	 * @return
	 */
	public int getLabelCount() {
		return labelMap.size();
	}
	
	/**저장
	 * @throws Exception
	 */
	public void save() throws Exception {
		mapper.writerWithDefaultPrettyPrinter().writeValue(labelFile, labelMap);
	}
	
}
