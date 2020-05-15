# ONNX-Conversion
This repository shows an example of exporting an ONNX model from PyTorch.

## Virtual Environment
First, you should create a virtual environment if you do not have one for ONNX conversion. You can change the versions of some of the packages in requirements.txt, however you may also have to make some changes on the export parameters. For example, opset_version=11 is not supported in torch 1.1.0

~~~
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
~~~

## Export to ONNX
Then, make the necessary changes on the below lines in both pytorch_to_onnx.py and difference_check.py

~~~
from model import Model # model.py is where Model class is defined

onnxfile = "filename.onnx"
weight_file = "weight_file.pth"
~~~

After doing the changes, you can execute the program with

~~~
python pytorch_to_onnx.py 1
~~~

(Optional) If you also want to optimize the ONNX graph, you can execute the program with

~~~
python pytorch_to_onnx.py 2
~~~

## Difference Check with ONNX-Runtime
If you want to check whether the exported ONNX model gives the same output as the original model, you can simply execute the difference_check.py with

~~~
python difference_check.py
~~~
