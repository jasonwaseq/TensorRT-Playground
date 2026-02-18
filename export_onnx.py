import torch
import torchvision.models as models
import os

def export_onnx(output_path="models/resnet50.onnx"):
    print(f"Loading ResNet50 model...")
    model = models.resnet50(pretrained=True)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    input_names = ["input"]
    output_names = ["output"]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Exporting model to {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        opset_version=11,
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print("Export successful!")

if __name__ == "__main__":
    export_onnx()
