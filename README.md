YOLOv8 inference using ONNX Runtime

### Installation

```
conda create -n ONNX python=3.8.10
conda activate ONNX
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
pip install opencv-python
pip install onnx
pip install onnxsim
pip install onnxruntime-gpu
```

### Train

* Follow [YOLOv8](https://github.com/jahongir7174/YOLOv8-pt) for training

### Export to ONNX

* Configure your model path in `main.py` for exporting
* Run `python main.py --export`

### Test

* Configure your exported onnx model path and image path in `main.py` for testing
* Run `python main.py --test` for testing

#### Reference

* https://github.com/jahongir7174/YOLOv8-pt
