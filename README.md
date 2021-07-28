# Brain Tumor Segmentation

In this project, 4 MR sequence images (T1, T1ce, T2, Flair) were used train 2D Unet. This model segment the image voxels into three types of tumor (Edema, Enhanced tumor, Necrotic or Non-enhaning tumor).
The dataset are from BRaTs. 

### Unet
Unet is a well known model for Bio Medical Image Segmentation. It has encoding parts and decoding part. Encoders are composed with convolutional layers and max pooling layers while Decoders are composed with transposed convolutions. In decoding parts, encoding layers are concatenated to preserved the location information. Conventional unet output has 1 channel, but I changed output shape to have 3 channel since we have 3 labels to classify. 

### Training Details
- Input shape (240, 240, 4) 
- output shape (240, 240, 3)
- Loss function  : Binary cross entropy
- Metrics : Dice coefficient , Accuracy
- Batch size : 10
- Learning rate : 0.0001 ( decay until 0.000001 )
- Epoch : 100 ( early stop if there is no lr update)

### Results
This result would not be optimal due to small data.
![result](https://github.com/kohheekyung/BrainTumorSegmentation/blob/main/resource/result.png){: width="100%" height="100%"}
![loss](https://github.com/kohheekyung/BrainTumorSegmentation/blob/main/resource/loss.png){: width="50" height="50"}

<img src="https://github.com/kohheekyung/BrainTumorSegmentation/blob/main/resource/result.png" width="500">
<img src="https://github.com/kohheekyung/BrainTumorSegmentation/blob/main/resource/loss.png" width="500">

### Prerequisite
- tensorflow-gpu 1.15.0
- keras 2.3.1
- nibabel

### Reference
[1] B. H. Menze, A. Jakab, S. Bauer, J. Kalpathy-Cramer, K. Farahani, J. Kirby, et al. "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)", IEEE Transactions on Medical Imaging 34(10), 1993-2024 (2015) DOI: 10.1109/TMI.2014.2377694

[2] Iqbal, Sajid, et al. "Brain tumor segmentation in multi‐spectral MRI using convolutional neural networks (CNN)." Microscopy research and technique 81.4 (2018): 419-427.
