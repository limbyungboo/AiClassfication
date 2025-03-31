package com.sample.block;

import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.Linear;
import ai.djl.nn.pooling.Pool;

public class AnimalBlock extends SequentialBlock {
	
	/**
	 * sample = 2 unit (cat, dog)
	 */
	private int unitCnt;

	/**
	 * constructor
	 */
	public AnimalBlock() {
		super();
		this.unitCnt = 2;
		init();
	}
	
	/**constructor
	 * @param unitCnt
	 */
	public AnimalBlock(int unitCnt) {
		super();
		this.unitCnt = unitCnt;
		init();
	}
	
	private void init() {
		super
			// 첫 번째 컨볼루션 블록
			.add(Conv2d.builder()
                .setKernelShape(new Shape(3, 3))
                .setFilters(32)
                .optPadding(new Shape(1, 1))
                .build())
            .add(Activation.reluBlock())
            .add(Pool.maxPool2dBlock(new Shape(2, 2), new Shape(2, 2)))
            
            // 두 번째 컨볼루션 블록
            .add(Conv2d.builder()
                .setKernelShape(new Shape(3, 3))
                .setFilters(64)
                .optPadding(new Shape(1, 1))
                .build())
            .add(Activation.reluBlock())
            .add(Pool.maxPool2dBlock(new Shape(2, 2), new Shape(2, 2)))
            
            // 세 번째 컨볼루션 블록
            .add(Conv2d.builder()
                .setKernelShape(new Shape(3, 3))
                .setFilters(128)
                .optPadding(new Shape(1, 1))
                .build())
            .add(Activation.reluBlock())
            .add(Pool.maxPool2dBlock(new Shape(2, 2), new Shape(2, 2)))
            
            // Flatten 및 완전연결층
            .add(Blocks.batchFlattenBlock())
            .add(Linear.builder().setUnits(512).build())
            .add(Activation.reluBlock())
            .add(Linear.builder().setUnits(unitCnt).build()); // 2 classes: cat and dog
	}
}
