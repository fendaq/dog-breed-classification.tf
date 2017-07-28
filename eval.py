# -*- coding: utf-8 -*-
__author__ = 'socurites@gmail.com'

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets
from utils.dataset_utils import load_batch


# Load the images and labels.
from datasets import tf_record_dataset
# create MNIST dataset
mnist_tfrecord_dataset = tf_record_dataset.TFRecordDataset(
  tfrecord_dir='/home/itrocks/Git/Tensorflow/dog-breed-classification.tf/raw_data/dog/tfrecord',
  dataset_name='dog',
  num_classes=120)
# Selects the 'train' dataset.
dataset = mnist_tfrecord_dataset.get_split(split_name='validation')
images, labels = load_batch(dataset=dataset, batch_size=32, height=224, width=224, num_classes=120)

# 모델 생성
vgg = tf.contrib.slim.nets.vgg
logits, end_points = vgg.vgg_16(inputs=images, num_classes=120, is_training=False)
logits = tf.argmax(logits, 1)
labels = tf.argmax(labels, 1)

# Define the metrics:
names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
  'eval/Accuracy': slim.metrics.streaming_accuracy(logits, labels),
  #'eval/Recall@5': slim.metrics.streaming_recall_at_k(logits, labels, 5),
})

tf.logging.set_verbosity(tf.logging.DEBUG)
print('Running evaluation Loop...')
log_dir = '/home/itrocks/Downloads/dog_model/'
eval_dir = '/home/itrocks/Downloads/dog_model-eval/'
checkpoint_path = tf.train.latest_checkpoint(log_dir)
metric_values = slim.evaluation.evaluate_once(
  master='',
  checkpoint_path=checkpoint_path,
  logdir=eval_dir,
  num_evals=20,
  eval_op=names_to_updates.values(),
  final_op=names_to_values.values())

names_to_values = dict(zip(names_to_values.keys(), metric_values))
for name in names_to_values:
  print('%s: %f' % (name, names_to_values[name]))