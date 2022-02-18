# 耳垂表型分类模型

**使用流程：**

1. 将文件夹`InferEarlobe`复制到工作目录下。
   **注意**：必须下载v1.0.0 release的3个必需的模型文件压缩包，解压缩后放置于`InferEarlobe\models`文件夹内

2. 环境配置
   
   1）使用anaconda和目录下的环境配置文件`environment.yaml`配置名为`InferEarlobe`的工作环境：
   
   ```bash
   conda env create -f environment.yaml
   ```
   
   2）或者使用anaconda手动配置
   
   ```bash
   conda create -n InferEarlobe python=3.9
   conda activate InferEarlobe
   pip install torch opencv-python pandas requests pyyaml tqdm numpy pillow torchvision matplotlib seaborn
   ##Install the CPU version of tensorflow
   wget https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow_cpu-2.7.0-cp39-cp39-manylinux2010_x86_64.whl
   pip install tensorflow_cpu-2.7.0-cp39-cp39-manylinux2010_x86_64.whl
   ```

3. 将目标检测照片放置于`data/images`目录下

4. 运行`inference.py`，结果存储与生成的`runs`文件下

5. 每次运行结束后更新目标检测照片到`data/images`，并移除或修改`runs`文件夹，再进行下一次检测和表型预测。

**输入文件：**

多种格式的二维图像文件（.jpg, .bmp, et al.）都可以识别

**输出文件：**

`runs`输出结果文件夹内包括：

+ 所有耳垂区域标签标注后的原始照片

+ `CropEarLobe_info.csv`文件，存储了耳垂部位在原始图片中的标准化后的中心点的`x`和`y`值以及`width`和`height`

+ 和`Predicted_phenos.csv`的结果文件，存储了耳垂图像最终的分类结果

+ 子目录`runs/crops`存储了切割后的耳垂图像

+ 子目录`runs/labels`存储了`CropEarLobe_info.csv`汇总前各图像对应的耳垂部位位置信息文件
