# Wavelet Integrated CNNs for Noise Robust Image Classification

This project is part of the `Wavelet Course` of [Kévin Polisano](https://polisano.pages.math.cnrs.fr/) and focuses reproducing the results of the paper [Wavelet Integrated CNNs for Noise Robust Image Classification](https://arxiv.org/pdf/2005.03337). The papers talks about on enhancing the robustness of Convolutional Neural Networks (CNNs) for image classification tasks, specifically in noisy environments. By integrating Discrete Wavelet Transform (DWT) layers into popular CNN architectures, such as AlexNet and VGG, they aim to improve performance on corrupted images. See the paper in `doc/` for more details.

## Project Structure

```
├── doc
│   └── Wavelet Integrated CNNs for Noise Robust Image Classification.df  # The original paper for reference, which this project is based on
├── environment.yml  # Conda environment file for dependencies
├── Project Report.pdf  # Master project report
├── README.md  # This file
└── src
    ├── alexnet.py  # AlexNet model with integrated wavelet layer
    ├── downsample.py  # Utility for wavelet downsampling
    ├── DWT_IDWT_layer.py  # Discrete Wavelet Transform (DWT) and Inverse DWT layers
    ├── resnet.py  # ResNet model (future extension)
    ├── train_nets.py  # Main training script
    ├── utils.py  # Utility functions (checkpointing, accuracy metrics, etc.)
    └── vgg.py  # VGG model with integrated wavelet layer
```

## How to Run the Project

### 1. Setup the Environment
Make sure you have [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) installed, then create the environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate wavelets-cnn
```

### 2. Prepare the Dataset
This project uses the [Tiny-ImageNet-C](https://github.com/hendrycks/robustness) dataset, which consists of corrupted versions of Tiny-ImageNet. Ensure the dataset is placed in the following structure:

```
data/
└── Tiny-ImageNet-C/
    ├── brightness/
    ├── gaussian_noise/
    ├── motion_blur/
    └── ...
```

### 3. Training the Models
To train AlexNet or VGG with wavelet integration, use the following command:

```bash
cd src
python train_nets.py --model alexnet --num_epochs 100 --batch_size 128 --lr 0.01 --save_model
```

Here are some important arguments:

- `--model`: Choose between `alexnet` and `vgg`.
- `--num_epochs`: Set the number of training epochs.
- `--batch_size`: Batch size for training.
- `--lr`: Learning rate.
- `--save_model`: Add this flag to save the model during training.

### 4. Resuming Training from a Checkpoint
If training was interrupted, you can resume from the last saved checkpoint:

```bash
python train_nets.py --model alexnet --num_epochs 100 --save_model
```

The script will automatically load the latest checkpoint from the `checkpoints/` folder.

### 5. Evaluating the Model
After training, the model is tested on a hold-out test set to evaluate its robustness to noisy inputs. The final accuracy is logged, and the model is saved.

### 6. TensorBoard Visualization
You can visualize the training metrics using TensorBoard:

```bash
tensorboard --logdir=runs/
```

Navigate to `http://localhost:6006` to view the metrics.

## What Does This Project Do?

- **Noise Robust Image Classification**: The core of this project lies in improving the classification performance of CNN models under noisy conditions, using wavelet-based feature extraction techniques.
- **Wavelet Integration**: By adding Discrete Wavelet Transform (DWT) layers to CNNs like AlexNet and VGG, we analyze the effect of multi-scale feature extraction on robustness to common noise distortions.
- **Model Checkpointing**: During training, models are saved at each epoch, allowing for easy resumption.
- **Performance Logging**: Both training and validation metrics are logged using TensorBoard for easy tracking of model performance.

## Project Goals

The aim of this project is to:
- Explore the integration of wavelets into CNNs.
- Evaluate the robustness of CNNs on corrupted datasets.
- Analyze the impact of different wavelets (e.g., Haar, Daubechies) on model performance.

## Future Work

- **ResNet Integration**: Extend the wavelet integration approach to more advanced architectures like ResNet.
- **Additional Corruptions**: Evaluate on other datasets or create custom corruptions to extend the scope of the study.
- **Hyperparameter Tuning**: Investigate deeper tuning of wavelet parameters, network architectures, and regularization techniques for improved performance.

For more information, refer to the project report in `Project Report.pdf`.

## Credits

[Wavelet Integrated CNNs for Noise Robust Image Classification](https://arxiv.org/pdf/2005.03337)