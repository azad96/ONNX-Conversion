import onnx
import onnxruntime
import numpy as np
import torch
from model import Model


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


onnxfile = "filename.onnx"
weight_file = "weight_file.pth"

model = Model()
map_location = (lambda storage, loc:storage)
if torch.cuda.is_available():
    map_location = None
ckpt = torch.load(weight_file, map_location=map_location)
model.load_state_dict(ckpt)

x = torch.randn(1, 3, 224, 224)

if torch.cuda.is_available():
    x = x.cuda()
    model = model.cuda()

model.eval()

with torch.no_grad():
    res = model(x)

ort_session = onnxruntime.InferenceSession(onnxfile)
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(res), ort_outs[0], rtol=1e-03, atol=1e-05)

