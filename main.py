import os
import warnings
from argparse import ArgumentParser

import cv2
import numpy
from onnxruntime import InferenceSession

warnings.filterwarnings("ignore")


class ONNXDetect:
    def __init__(self, args, onnx_path, session=None):
        self.session = session
        if self.session is None:
            assert onnx_path is not None
            assert os.path.exists(onnx_path)
            self.session = InferenceSession(onnx_path,
                                            providers=['CUDAExecutionProvider'])

        self.inputs = self.session.get_inputs()[0].name
        self.confidence_threshold = 0.25
        self.iou_threshold = 0.7
        self.input_size = args.input_size
        shape = (1, 3, self.input_size, self.input_size)
        image = numpy.zeros(shape, dtype='float32')
        for _ in range(10):
            self.session.run(output_names=None,
                             input_feed={self.inputs: image})

    def __call__(self, image):
        x, pad, gain = self.resize(image, image.shape)
        x = x.transpose((2, 0, 1))[::-1]
        x = x.astype('float32') / 255
        x = x[numpy.newaxis, ...]

        outputs = self.session.run(output_names=None,
                                   input_feed={self.inputs: x})[0]
        outputs = outputs[0].transpose(1, 0)

        outputs[:, 0] -= pad[1]
        outputs[:, 1] -= pad[0]

        # Extract class scores (all rows, columns 4 onwards)
        class_scores = outputs[:, 4:]  # Shape: (8400, num_classes)

        # Find maximum score and corresponding class ID for each detection
        max_scores = numpy.amax(class_scores, axis=1)  # Shape: (8400,)
        class_indices = numpy.argmax(class_scores, axis=1)  # Shape: (8400,)

        # Filter detections based on confidence threshold
        mask = max_scores >= self.confidence_threshold
        if not numpy.any(mask):
            return []

        # Apply mask to filter valid detections
        outputs = outputs[mask]  # Shape: (N, 4 + num_classes)
        scores = max_scores[mask]  # Shape: (N,)
        class_indices = class_indices[mask]  # Shape: (N,)

        # Extract bounding box coordinates (cx, cy, w, h)
        cx, cy, w, h = outputs[:, 0], outputs[:, 1], outputs[:, 2], outputs[:, 3]

        # Calculate scaled bounding box coordinates
        left = ((cx - w / 2) / gain).astype(int)
        top = ((cy - h / 2) / gain).astype(int)
        width = (w / gain).astype(int)
        height = (h / gain).astype(int)

        # Stack boxes into list of [left, top, width, height]
        boxes = numpy.stack(arrays=[left, top, width, height], axis=1).tolist()
        scores = scores.tolist()
        class_indices = class_indices.tolist()

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

    def resize(self, image, shape):
        r = min(self.input_size / shape[0], self.input_size / shape[1])

        # Compute padding
        pad = int(round(shape[1] * r)), int(round(shape[0] * r))
        w = (self.input_size - pad[0]) / 2  # w padding
        h = (self.input_size - pad[1]) / 2  # h padding

        if shape[::-1] != pad:
            image = cv2.resize(image, pad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(h - 0.1)), int(round(h + 0.1))
        left, right = int(round(w - 0.1)), int(round(w + 0.1))
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return image, (top, left), min(self.input_size / shape[0], self.input_size / shape[1])


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
