# coding=utf-8
# Copyright 2022 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BeeDataset dataset."""
import json
import random
import tarfile

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
    num_classes = 3
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Image(shape=t_shape),
            # 'label': tfds.features.ClassLabel(names=['bottle', 'cable', 'capsule', 'carpte', 'grid',
                #'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']),
            'label': tfds.features.ClassLabel(num_classes=num_classes),
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
    path = dl_manager.download('https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.gz')

    #if Extracted directory already exists, remove it to avoid permission error
    dirpath = './Extracted'
    #if os.path.exists(dirpath) and os.path.isdir(dirpath):
    #      shutil.rmtree(dirpath)

    #Extract contents of tar.gz file into the Extracted directory
    #file = tarfile.open(path)
    #file.extractall('./Extracted')
    #file.close()

    #remove the files which are not folders (data cleaning)
    os.remove('./Extracted/readme*')
    os.remove('./Extracted/licesnce.txt')
    
    ex_path = "./Extracted"
    return {
        'train': self._generate_examples(ex_path, 'train'),
        'test': self._generate_examples(ex_path, 'test'),
    }

  def _generate_examples(self, path, tag):
    # Load labels and image path.
    label_int = 0
    
    for label in tf.io.gfile.listdir(path):
        if label_int >= num_classes:
          break;
      
        label_folder = tf.io.gfile.join(path, label, tag)
        count = 0
        for image in tf.io.gfile.glob(str(label_folder)+'/*/*.png'):
            count = count + 1
            key = str(label)+tag+str(count)
            yield key, {
                'image': image,
                'label': label_int,
            }
        label_int = label_int + 1
