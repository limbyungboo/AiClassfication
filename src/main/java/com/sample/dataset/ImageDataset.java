package com.sample.dataset;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.dataset.Record;
import ai.djl.util.Progress;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import com.sample.utils.ImageUtil;

/**
 * 
 */
public class ImageDataset extends RandomAccessDataset {
    private final List<Path> imagePaths;
    private final List<Integer> labels;
    
    public static final int IMAGE_WIDTH = 150;
    public static final int IMAGE_HEIGHT = 150;

    /**
     * @param builder
     */
    public ImageDataset(Builder builder) {
        super(builder);
        this.imagePaths = builder.imagePaths;
        this.labels = builder.labels;
    }

    /**
     *
     */
    @Override
    public void prepare(Progress progress) throws IOException {
        // 이미지 데이터셋은 이미 메모리에 로드되어 있으므로 추가 준비가 필요 없음
    }

    /**
     *
     */
    @Override
    public long availableSize() {
        return imagePaths.size();
    }

    /**
     *
     */
    @Override
    public Record get(NDManager manager, long index) {
        Path imagePath = imagePaths.get((int) index);
        int label = labels.get((int) index);
        try {
            // 이미지를 로드하고 전처리
            NDArray image = ImageUtil.loadImageToNDArray(manager, imagePath, IMAGE_WIDTH, IMAGE_HEIGHT);
            
            // 레이블을 생성
            NDArray labelArray = manager.create(new float[] {label});

            return new Record(new NDList(image), new NDList(labelArray));
        } 
        catch (Exception e) {
            throw new IllegalStateException("Failed to load image: " + imagePath, e);
        }
    }

    /**
     * @return
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * 
     */
    public static final class Builder extends RandomAccessDataset.BaseBuilder<Builder> {
        private List<Path> imagePaths;
        private List<Integer> labels;

        Builder() {
            imagePaths = new ArrayList<>();
            labels = new ArrayList<>();
        }

        public Builder addDirectory(Path directory, int label) {
            File dir = directory.toFile();
            if (dir.exists() && dir.isDirectory()) {
                File[] files = dir.listFiles((d, name) -> 
                    name.toLowerCase().endsWith(".jpg") || 
                    name.toLowerCase().endsWith(".jpeg") || 
                    name.toLowerCase().endsWith(".png"));
                
                if (files != null) {
                    for (File file : files) {
                        imagePaths.add(file.toPath());
                        labels.add(label);
                    }
                }
            }
            return this;
        }

        @Override
        protected Builder self() {
            return this;
        }

        public ImageDataset build() {
            if (imagePaths.isEmpty()) {
                throw new IllegalStateException("No images found in the specified directories");
            }
            setSampling(imagePaths.size(), true);
            return new ImageDataset(this);
        }
    }
} 