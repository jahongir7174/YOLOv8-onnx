import os
import warnings
from argparse import ArgumentParser

import cv2
import numpy

warnings.filterwarnings("ignore")


class ONNXDetect:
    def __init__(self, args, onnx_path, session=None):
        self.session = session
        if self.session is None:
            assert onnx_path is not None
            assert os.path.exists(onnx_path)
            from onnxruntime import InferenceSession
            self.session = InferenceSession(onnx_path,
                                            providers=['CUDAExecutionProvider'])

        self.inputs = self.session.get_inputs()[0]
        self.confidence_threshold = 0.25
        self.iou_threshold = 0.7
        self.input_size = args.input_size
        shape = (1, 3, self.input_size, self.input_size)
        image = numpy.zeros(shape, dtype='float32')
        for _ in range(10):
            self.session.run(output_names=None,
                             input_feed={self.inputs.name: image})

    def __call__(self, image):
        image, scale = self.resize(image, self.input_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose((2, 0, 1))[::-1]
        image = image.astype('float32') / 255
        image = image[numpy.newaxis, ...]

        outputs = self.session.run(output_names=None,
                                   input_feed={self.inputs.name: image})
        outputs = numpy.transpose(numpy.squeeze(outputs[0]))

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_indices = []

        # Iterate over each row in the outputs array
        for i in range(outputs.shape[0]):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]

            # Find the maximum score among the class scores
            max_score = numpy.amax(classes_scores)

            # If the maximum score is above the confidence threshold
            if max_score >= self.confidence_threshold:
                # Get the class ID with the highest score
                class_id = numpy.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                image, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # Calculate the scaled coordinates of the bounding box
                left = int((image - w / 2) / scale)
                top = int((y - h / 2) / scale)
                width = int(w / scale)
                height = int(h / scale)

                # Add the class ID, score, and box coordinates to the respective lists
                class_indices.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_threshold, self.iou_threshold)

        # Iterate over the selected indices after non-maximum suppression
        nms_outputs = []
        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            score = scores[i]
            class_id = class_indices[i]
            nms_outputs.append([*box, score, class_id])
        return nms_outputs

    @staticmethod
    def resize(image, input_size):
        shape = image.shape

        ratio = float(shape[0]) / shape[1]
        if ratio > 1:
            h = input_size
            w = int(h / ratio)
        else:
            w = input_size
            h = int(w * ratio)
        scale = float(h) / shape[0]
        resized_image = cv2.resize(image, (w, h))
        det_image = numpy.zeros((input_size, input_size, 3), dtype=numpy.uint8)
        det_image[:h, :w, :] = resized_image
        return det_image, scale


def export(args):
    import onnx  # noqa
    import torch  # noqa
    import onnxsim  # noqa

    filename = './weights/v8_n.pt'

    model = torch.load(filename)['model'].float()
    image = torch.zeros((1, 3, args.input_size, args.input_size))

    torch.onnx.export(model,
                      image,
                      filename.replace('pt', 'onnx'),
                      verbose=False,
                      opset_version=12,
                      do_constant_folding=True,
                      input_names=['inputs'],
                      output_names=['outputs'],
                      dynamic_axes=None)

    # Checks
    model_onnx = onnx.load(filename.replace('pt', 'onnx'))  # load onnx model

    # Simplify
    try:
        model_onnx, check = onnxsim.simplify(model_onnx)
        assert check, 'Simplified ONNX model could not be validated'
    except Exception as e:
        print(e)

    onnx.save(model_onnx, filename.replace('pt', 'onnx'))


def test(args):
    # Load model
    model = ONNXDetect(args, onnx_path='./weights/v8_n.onnx')

    frame = cv2.imread('zidane.jpg')
    image = frame.copy()
    outputs = model(image)
    for output in outputs:
        x, y, w, h, score, index = output
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
    cv2.imwrite('output.jpg', frame)


def main():
    parser = ArgumentParser()
    parser.add_argument('--input-size', default=640, type=int)
    parser.add_argument('--export', action='store_true')
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()

    if args.export:
        export(args)
    if args.test:
        test(args)


if __name__ == "__main__":
    main()
