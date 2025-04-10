/**
 * 
 */
package co.kr.lbb;

import java.util.List;
import java.util.stream.Collectors;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * layer configuration
 */
public class LayerConfiguration {

    /**image trainner layer configuration
     * @param outputClassesCount
     * @param imageWidth
     * @param imageHeight
     * @param imageChannels
     * @return
     */
    public static MultiLayerConfiguration configuration(int outputClassesCount, int imageWidth, int imageHeight, int imageChannels) {
    	
        // 모델 구성
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(42)
            .updater(new Adam(0.001))  // ✅ Adam 사용
            .weightInit(WeightInit.XAVIER)
            .list()
            .layer(0, new ConvolutionLayer.Builder(3, 3)
                .nIn(imageChannels)
                .nOut(32)
                .stride(1, 1)
                .activation(Activation.RELU)
                .build())
            .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build())
            .layer(2, new ConvolutionLayer.Builder(3, 3)
                .nOut(64)
                .stride(1, 1)
                .activation(Activation.RELU)
                .build())
            .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build())
            .layer(4, new DenseLayer.Builder()
                .nOut(128)
                .activation(Activation.RELU)
                .build())
            .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(outputClassesCount)
                .activation(Activation.SOFTMAX)
                .build())
            .setInputType(InputType.convolutional(imageHeight, imageWidth, imageChannels))
            .build();
    	
    	return conf;
    }
    
    
    /**output 클래스가 변경되었을경우 model update
     * @param model
     * @param nOut
     * @return
     */
    public static MultiLayerNetwork updateOutput(MultiLayerNetwork model, int nOut) {
		MultiLayerConfiguration configure = model.getLayerWiseConfigurations();
		List<org.deeplearning4j.nn.conf.layers.Layer> layers = configure.getConfs()
				.stream()
				.map(c -> c.getLayer())
				.collect(Collectors.toList());
		
		OutputLayer outputLayer = (OutputLayer)layers.get(layers.size() - 1);
		
		//기존 out 가 동일할경우
		if(nOut == outputLayer.getNOut()) {
			return model;
		}
		
		System.out.println("=================>>> model update");
		FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
			    .updater(new Nesterovs(0.01, 0.9))  // 옵티마이저
			    .seed(123)
			    .build();
		
		model = new TransferLearning.Builder(model)
				.fineTuneConfiguration(fineTuneConf)
			    .removeLayersFromOutput(1) // 마지막 OutputLayer 제거
			    .addLayer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
			        .nIn(128)   // 이전 레이어의 출력 수
			        .nOut(nOut)  // 새로운 클래스 개수
			        .activation(Activation.SOFTMAX)
			        .build())
			    .build();		
		return model;
    }
}
