"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import keras
from keras_retinanet.utils.colors import colors
from keras_retinanet.utils.visualization import draw_box, draw_detections, draw_annotations
import numpy as np
import cv2
import os

from ..utils.eval import evaluate


class Evaluate(keras.callbacks.Callback):
    """ Evaluation callback for arbitrary datasets.
    """

    def __init__(
        self,
        generator,
        iou_threshold=0.5,
        score_threshold=0.05,
        max_detections=100,
        save_path=None,
        tensorboard=None,
        weighted_average=False,
        verbose=1
    ):
        """ Evaluate a given dataset using a given model at the end of every epoch during training.

        # Arguments
            generator        : The generator that represents the dataset to evaluate.
            iou_threshold    : The threshold used to consider when a detection is positive or negative.
            score_threshold  : The score confidence threshold to use for detections.
            max_detections   : The maximum number of detections to use per image.
            save_path        : The path to save images with visualized detections to.
            tensorboard      : Instance of keras.callbacks.TensorBoard used to log the mAP value.
            weighted_average : Compute the mAP using the weighted average of precisions among classes.
            verbose          : Set the verbosity level, by default this is set to 1.
        """
        self.generator       = generator
        self.iou_threshold   = iou_threshold
        self.score_threshold = score_threshold
        self.max_detections  = max_detections
        self.save_path       = save_path
        self.tensorboard     = tensorboard
        self.weighted_average = weighted_average
        self.verbose         = verbose

        super(Evaluate, self).__init__()





    def image_resize(self, image, width = None, height = None, inter = cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]
    
        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image
    
        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)
    
        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))
    
        # resize the image
        resized = cv2.resize(image, dim, interpolation = inter)
    
        # return the resized image
        return resized
    






    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # run evaluation
        average_precisions, _ = evaluate(
            self.generator,
            self.model,
            iou_threshold=self.iou_threshold,
            score_threshold=self.score_threshold,
            max_detections=self.max_detections,
            save_path=self.save_path
        )

        # compute per class average precision
        total_instances = []
        precisions = []
        for label, (average_precision, num_annotations) in average_precisions.items():
            if self.verbose == 1:
                print('{:.0f} instances of class'.format(num_annotations),
                      self.generator.label_to_name(label), 'with average precision: {:.4f}'.format(average_precision))
            total_instances.append(num_annotations)
            precisions.append(average_precision)
        if self.weighted_average:
            self.mean_ap = sum([a * b for a, b in zip(total_instances, precisions)]) / sum(total_instances)
        else:
            self.mean_ap = sum(precisions) / sum(x > 0 for x in total_instances)

        if self.tensorboard:
            import tensorflow as tf
            if tf.version.VERSION < '2.0.0' and self.tensorboard.writer:
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = self.mean_ap
                summary_value.tag = "mAP"
                self.tensorboard.writer.add_summary(summary, epoch)
            else:
                images_to_generate = min(100, self.generator.size())
                images_index = 0
                writer = tf.summary.create_file_writer(self.tensorboard.log_dir)
                with writer.as_default():
                    while images_index < images_to_generate:
                        raw_image = self.generator.load_image(images_index)
                        image = self.generator.preprocess_image(raw_image.copy())
                        image, scale = self.generator.resize_image(image)
                        # prediction
                        boxes, scores, labels = self.model.predict_on_batch(np.expand_dims(image, axis=0))[:3]
                        # correct boxes for image scale
                        boxes /= scale
                        # select indices which have a score above the threshold
                        score_threshold = 0.4
                        indices = np.where(scores[0, :] > score_threshold)[0]
                        # select those scores
                        scores = scores[0][indices]
                        # find the order with which to sort the scores
                        max_detections = 20
                        scores_sort = np.argsort(-scores)[:max_detections]
                        # select detections
                        image_boxes = boxes[0, indices[scores_sort], :]
                        image_scores = scores[scores_sort]
                        image_labels = labels[0, indices[scores_sort]]
                        draw_detections(raw_image, image_boxes, image_scores, image_labels,
                            label_to_name=self.generator.label_to_name, score_threshold=score_threshold)
                        # create tensorboard image
                        image_name = self.generator.image_names[images_index]
                        tensorboard_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
                        tensorboard_image = self.image_resize(tensorboard_image, height=800)
                        tensorboard_image = tf.expand_dims(tensorboard_image, 0)
                        tf.summary.image(image_name, tensorboard_image, step=epoch,
                            description='ScoreThreshold: {} MaxDetections: {}'.format(score_threshold, max_detections))

                        images_index += 1
                    writer.flush()

        logs['mAP'] = self.mean_ap

        if self.verbose == 1:
            print('mAP: {:.4f}'.format(self.mean_ap))
