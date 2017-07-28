# -*- coding: utf-8 -*-
__author__ = 'socurites@gmail.com'

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets
import utils.dataset_utils as dutils

'''
Define Environment Constants
'''
# A directory containing TFRecord datasets.
tfrecord_dir = '/home/itrocks/Git/Tensorflow/dog-breed-classification.tf/raw_data/dog/tfrecord'
# The name of the dataset prefix.
dataset_name = 'dog'
# The dataset split(train or validation).
split_name = 'train'
# The number of classes of datasets.
num_classes = 120
# A batch size
batch_size = 32
# A image size(width and height are same).
image_size = 224
# A phase of learning(True for training False for validation)..
is_training = True
# The directory for logging events and checkpoint files.
logdir = '/home/itrocks/Downloads/dog_model/'
# The path where pre-trained mode resides in.
model_path = '/home/itrocks/Backup/Model/TF-Slim/vgg_16.ckpt'

'''
Define 학습 하이퍼파라미터
'''
# 학습률
learning_rate = 0.0002

'''
run()
'''
# logging 설정
tf.logging.set_verbosity(tf.logging.INFO)

# 데이터셋 배치 로드
images, labels = dutils.load_tfrecord_batch(tfrecord_dir=tfrecord_dir,
                                            dataset_name=dataset_name, num_classes=num_classes,
                                            split_name=split_name, batch_size=batch_size, image_size=image_size)

# 모델 생성
vgg = tf.contrib.slim.nets.vgg
logits, end_points = vgg.vgg_16(inputs=images, num_classes=num_classes, is_training=is_training)

# Restore할 변수 정의하기
exculde = ['fc6', 'fc7', 'fc8']
variables_to_restore = slim.get_variables_to_restore(exclude=exculde)

# 손실 함수 정의
loss = slim.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels)
total_loss = slim.losses.get_total_loss()

# 옵티마이저 정의
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

# 훈련 오퍼레이션 정의
train_op = slim.learning.create_train_op(total_loss, optimizer)


print(logits)
print(labels)

# 모니터링할 메트릭 정의
predictions = tf.argmax(logits, 1)
targets = tf.argmax(labels, 1)

print(predictions)
print(targets)

accuracy, _ = tf.contrib.metrics.streaming_accuracy(predictions, targets)

tf.summary.scalar('losses/Total', total_loss)
tf.summary.scalar('accuracy', accuracy)
summary_op = tf.summary.merge_all()


# Restore pre-trained model from checkpoint file
saver = tf.train.Saver(variables_to_restore)
def restore_fn(sess):
    return saver.restore(sess, model_path)


# 훈련 시작
if not tf.gfile.Exists(logdir):
  tf.gfile.MakeDirs(logdir)

final_loss = slim.learning.train(train_op=train_op,
                                 logdir=logdir,
                                 number_of_steps=500000,
                                 summary_op=summary_op,
                                 save_summaries_secs=300,
                                 save_interval_secs=600)

print('Finished training. Fianl batch lose %f' %final_loss)


# FLAGS = tf.app.flags.DEFINE_string(
#   'tfrecord_dir',
#   '/home/itrocks/Git/Tensorflow/dog-breed-classification.tf/raw_data/dog/tfrecord',
#   'A directory containing TFRecord datasets.')
#
# def main():
#   if not FLAGS.tfrecord_dir:
#     raise ValueError('You must supply the dataset name with --tfrecord_dir')
#
# if __name__ == '__main__':
#   tf.app.run()