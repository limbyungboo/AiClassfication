package com.example;

import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.Image;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.io.File;
import java.io.IOException;
import java.nio.file.Path;

public class ImageUtil {
    public static BufferedImage resizeImage(BufferedImage originalImage, int targetWidth, int targetHeight) {
        BufferedImage resizedImage = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB);
        Graphics2D graphics = resizedImage.createGraphics();
        graphics.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        graphics.drawImage(originalImage, 0, 0, targetWidth, targetHeight, null);
        graphics.dispose();
        return resizedImage;
    }
    
    public static float[] imageToPixels(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();
        float[] pixels = new float[width * height * 3];
        
        int index = 0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = image.getRGB(x, y);
                pixels[index++] = ((rgb >> 16) & 0xFF) / 255.0f;  // R
                pixels[index++] = ((rgb >> 8) & 0xFF) / 255.0f;   // G
                pixels[index++] = (rgb & 0xFF) / 255.0f;          // B
            }
        }
        
        return pixels;
    }

    public static float[][][][] loadAndPreprocessImage(String imagePath, int targetWidth, int targetHeight) throws IOException {
        // 이미지 로드
        BufferedImage originalImage = ImageIO.read(new File(imagePath));
        
        // 이미지 크기 조정
        BufferedImage resizedImage = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB);
        Graphics2D graphics = resizedImage.createGraphics();
        graphics.drawImage(originalImage.getScaledInstance(targetWidth, targetHeight, Image.SCALE_SMOOTH), 0, 0, null);
        graphics.dispose();
        
        // 픽셀 데이터 추출 및 정규화
        float[][][][] imageData = new float[1][targetHeight][targetWidth][3];
        
        for (int y = 0; y < targetHeight; y++) {
            for (int x = 0; x < targetWidth; x++) {
                int rgb = resizedImage.getRGB(x, y);
                float red = ((rgb >> 16) & 0xFF) / 255.0f;
                float green = ((rgb >> 8) & 0xFF) / 255.0f;
                float blue = (rgb & 0xFF) / 255.0f;
                
                imageData[0][y][x][0] = red;
                imageData[0][y][x][1] = green;
                imageData[0][y][x][2] = blue;
            }
        }
        
        return imageData;
    }

    public static NDArray loadImageToNDArray(NDManager manager, Path imagePath, int width, int height) throws IOException {
        BufferedImage img = ImageIO.read(imagePath.toFile());
        BufferedImage resized = resizeImage(img, width, height);
        float[] pixels = imageToPixels(resized);
        
        // Create NDArray with shape [channels, height, width] (PyTorch format)
        NDArray array = manager.create(pixels, new Shape(3, height, width));
        return array.toType(DataType.FLOAT32, false);
    }
} 