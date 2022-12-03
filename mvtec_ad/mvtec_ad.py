"""MvTec-AD dataset."""
import json
import random

import tensorflow as tf
import tensorflow_datasets.public_api as tfds

_DESCRIPTION = """
MVTec AD is a dataset for benchmarking anomaly detection methods with a focus on industrial inspection. It contains over 5000 high-resolution images divided into fifteen different object and texture categories. Each category comprises a set of defect-free training images and a test set of images with various kinds of defects as well as images without defects.

Pixel-precise annotations of all anomalies are also provided. More information can be in our paper "MVTec AD – A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection" and its extended version "The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection".
"""

_CITATION = """
1. Paul Bergmann, Kilian Batzner, Michael Fauser, David Sattlegger, Carsten Steger: The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection; in: International Journal of Computer Vision 129(4):1038-1059, 2021, DOI: 10.1007/s11263-020-01400-4 \
2. Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger: MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection; in: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 9584-9592, 2019, DOI: 10.1109/CVPR.2019.00982.\
"""

class MvtecAd(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for MvTecAD dataset."""

  VERSION = tfds.core.Version('1.0.0')

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    t_shape = (224,224,3)
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Image(shape=t_shape),
            # 'label': tfds.features.ClassLabel(names=['bottle', 'cable', 'capsule', 'carpte', 'grid',
                #'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']),
            'label': tfds.features.ClassLabel(num_classes=15),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('image', 'label'),  # Set to `None` to disable
        homepage='https://www.mvtec.com/company/research/datasets/mvtec-ad',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    path = dl_manager.download_and_extract('https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz')
    return {
        'train': self._generate_examples(path, 'train'),
        'test': self._generate_examples(path, 'test'),
    }

  def _generate_examples(self, path, tag):
    # Load labels and image path.
    label_int = 0
    for label in tf.io.gfile.listdir(path):
        label_folder = tf.io.gfile.join(path, label, tag)
        count = 0
        for i in tf.io.gfile.glob(str(label_folder)+'/*/*.png'):
            count = count + 1
            key = str(label)+tag+str(count)
            yield key, {
                'image': image,
                'label': label_int,
            }
        label_int = label_int + 1