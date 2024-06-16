## Being More Practical and Lightweight: Mini-sized Contrastive Learning Pre-trained Models for Fine-grained Traffic Tasks

This repository contains the PyTorch implementation of MiniTraffic, our proposed pre-trained model specifically designed for fine-grained traffic tasks.

### Key Designs:

- **Frequency Domain Stability Augmentation:** This technique uses the statistical correlation between road-level and lane-level data to simulate the scarce lane-level traffic data, forming the basis for effective cross-granularity transfer prediction.
- **Contrastive Clustering Backbone:** By leveraging the similarity between lane segments, this approach clusters spatio-temporally similar patches through contrastive learning. Graph attention convolution is then applied within these clusters, partitioning the entire spatio-temporal graph into multiple smaller graphs, thereby reducing computational complexity.

### Requirements:
- Python 3.8
- Matplotlib
- NumPy
- SciPy
- Pandas
- Argparse
- Scikit-learn
- PyTorch==1.11.0

### Datasets:
All datasets should be placed in the `dataset` directory. If you wish to use your own data, please ensure it is also placed in this directory.

### Pre-training:
After verifying the correct setup of your hyperparameters and datasets, you can start pre-training with the following command:
```
python minitraffic_pretrain.py
```

### Fine-tuning:
To fine-tune the model on your target lane-level or road-level training set, modify the training set accordingly and use the following command:
```
python minitraffic_finetune.py
```

This comprehensive guide ensures you can effectively implement and utilize the MiniTraffic model for your fine-grained traffic prediction tasks.
