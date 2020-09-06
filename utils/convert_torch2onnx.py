import torch
import torch.onnx as onnx
from opts import parse_opts
from model import generate_model

if __name__ == '__main__':
    opt = parse_opts()
    if opt.resume_path is None:
        raise NotImplemented()

    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)

    # Load pretrained model
    model, _ = generate_model(opt)
    # model = model.cuda()
    checkpoint = torch.load(opt.resume_path)
    model.load_state_dict(checkpoint['state_dict'])

    # dummy input
    dummy_input = torch.randn(
        1, 3, opt.sample_duration, opt.sample_size, opt.sample_size)
    onnx.export(model, dummy_input, opt.model +
                "-" + str(opt.model_depth) + ".onnx")
