from sys import argv

onnxfile = "filename.onnx"
weight_file = "weight_file.pth"

def export():
    import torch
    from model import Model # model.py is where Model class is defined
    
    model = Model()
    map_location = (lambda storage, loc:storage)
	if torch.cuda.is_available():
	    map_location = None
	ckpt = torch.load(weight_file, map_location=map_location)
	model.load_state_dict(ckpt)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    input_names = [ "input" ]
    output_names = [ "output" ]

    torch.onnx.export(model, dummy_input, onnxfile,
                    input_names=input_names, 
                    output_names=output_names,
                    verbose=True,
                    opset_version=11,
                    # keep_initializers_as_inputs=True,
                    export_params=True)

    print('Exported.')

def optimize():
    import onnx
    from onnx import optimizer
    model = onnx.load(onnxfile)
    onnx.checker.check_model(model)
    print('Checked.')
    passes = ["extract_constant_to_initializer", "eliminate_unused_initializer"]
    optimized_model = optimizer.optimize(model, passes)
    print('Optimized.')
    onnx.save(optimized_model, onnxfile)


if __name__ == "__main__":
    if len(argv) < 2 or argv[1] not in ('1', '2'):
        print('python pytorch_to_onnx.py 1 : export')
        print('python pytorch_to_onnx.py 2 : optimize')
    elif argv[1] == '1':
        export()
    elif argv[1] == '2':
        optimize()
        