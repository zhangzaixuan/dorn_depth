from DORNnet import ResNet
import torch.onnx

batch_size = 2  # 可以取其他值
torch_model = ResNet()

# torch_model.load_state_dict(torch.load('model_data/hrnetv2_w32_weights_voc.pth'), strict=False)
torch_model.eval()
# # (b, 2048, 33, 45)
# 3, 257, 353
# x = torch.randn(batch_size, 3, 480, 480, requires_grad=True)  # 模拟输入数据尺寸
x = torch.randn(batch_size, 3, 257, 353, requires_grad=True)  # 模拟输入数据尺寸

torch_out = torch_model(x)

torch.onnx.export(torch_model,  # model being run
                  x,  # model input (or a tuple for multiple inputs)
                  "./basic_test.onnx",  # where to save the model (can be a file or file-like object)
                  export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=11,
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input'],  # the model's input names
                  output_names=['output'],  # the model's output names
                  dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                'output': {0: 'batch_size'}})
print("onnx is saved")
