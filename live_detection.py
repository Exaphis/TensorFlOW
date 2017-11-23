import numpy as np
import tensorflow as tf
import cv2
import time
from mss import mss

# Initial configuration of screenshot and TensorFlow
SCREENSHOT_SIZE = 400
MIN_SCORE_THRESH = 0.97
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6

monitor = {'left': 960 - (SCREENSHOT_SIZE // 2), 'top': 540 - (SCREENSHOT_SIZE // 2), 'width': SCREENSHOT_SIZE,
           'height': SCREENSHOT_SIZE}
sct = mss()

# # Model preparation

# ## Variables
#
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by 
# changing `PATH_TO_CKPT` to point to a new .pb file.

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = "PATH_TO frozen_inference_graph.pb"

# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

total_time = 0
count = 0
with detection_graph.as_default():
    with tf.Session(graph=detection_graph, config=config) as sess:
        while True:
            last = time.time()
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # Grab screenshot and convert to RGB
            image_np = np.array(sct.grab(monitor))[:, :, :3]
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            # ## Visualization of the results of a detection.
            # Create map of boxes and their respective scores
            boxes, scores = np.squeeze(boxes), np.squeeze(scores)
            box_to_score_map = {}

            for i in range(boxes.shape[0]):
                if scores is None or scores[i] > MIN_SCORE_THRESH:
                    box = tuple([int(x * SCREENSHOT_SIZE) for x in boxes[i].tolist()])
                    box_to_score_map[box] = int(100 * scores[i])

            # Convert image back to BGR for OpenCV to display it correctly
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            # Draw boxes on to image
            for box, score in box_to_score_map.items():
                y_min, x_min, y_max, x_max = box
                # Draw rectangle around detected health bar
                cv2.rectangle(image_np,
                              (x_min, y_min),
                              (x_max, y_max),
                              (0, 255, 0),
                              thickness=2)

                # Write score above detected health bar
                font_face = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                score_text = "Score: {}".format(score)
                font_thickness = 2

                text_size = cv2.getTextSize(score_text, font_face, font_scale, font_thickness)[0]

                cv2.rectangle(image_np,
                              (x_min, y_min - 10),
                              (x_min + text_size[0], y_min - 10 + text_size[1]),
                              (0, 255, 0),
                              -1)

                cv2.putText(image_np,
                            score_text,
                            (x_min, y_min),
                            font_face,
                            font_scale,
                            (0, 0, 0),
                            thickness=1)
            total_time += time.time() - last
            count += 1
            # Show image
            cv2.imshow('window', image_np)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
print("fps: {}".format(count // total_time))
