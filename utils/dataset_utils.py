# -*- coding: utf-8 -*-
__author__ = 'socurites@gmail.com'

import tensorflow as tf
import tensorflow.contrib.slim as slim
from datasets import tf_record_dataset


def load_batch(dataset, batch_size=32, height=28, width=28, num_classes=10, is_training=True):
  """Loads a single batch of data.

  Args:
    dataset: The dataset to load.
    batch_size: The number of images in the batch.
    height: The size of each image after preprocessing.
    width: The size of each image after preprocessing.
    is_training: Whether or not we're currently training or evaluating.

  Returns:
    images: A Tensor of size [batch_size, height, width, 3], image samples that have been preprocessed.
    images_raw: A Tensor of size [batch_size, height, width, 3], image samples that can be used for visualization.
    labels: A Tensor of size [batch_size], whose values range between 0 and dataset.num_classes.
  """
  # Creates a TF-Slim DataProvider which reads the dataset in the background during both training and testing.
  provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
  [image, label] = provider.get(['image', 'label'])

  # image: resize with crop
  image = tf.image.resize_image_with_crop_or_pad(image, height, width)
  image = tf.to_float(image)

  # label: one-hot encoding
  one_hot_labels = slim.one_hot_encoding(label, num_classes)

  # Batch it up.
  images, labels = tf.train.batch(
    [image, one_hot_labels],
    batch_size=batch_size,
    num_threads=1,
    capacity=2 * batch_size)

  return images, labels


def load_tfrecord_batch(tfrecord_dir, dataset_name, num_classes, split_name, batch_size, image_size):
  # TFReocrd 데이터셋 객체 생성
  tfrecord_dataset = tf_record_dataset.TFRecordDataset(tfrecord_dir=tfrecord_dir,
                                                       dataset_name=dataset_name, num_classes=num_classes)
  # 로드할 TFRecord 데이터셋 생성('train' or 'validation')
  dataset = tfrecord_dataset.get_split(split_name=split_name)

  # batch 객체 생성
  images, labels = load_batch(dataset=dataset, batch_size=batch_size,
                              height=image_size, width=image_size, num_classes=num_classes)

  return images, labels