import torch

from src.models.factory import get_model


def main():
    config = {
        'name': 'unet',
        'params': {
            "model_backbone": "resnet18",
            'in_channels': 3,
            'num_classes': 1,
            # 'init_features': 32,
            # 'return_logits': True,
        }
    }

    model = get_model(config)

    input_tensor = torch.rand(32, 3, 512, 512)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)


if __name__ == '__main__':
    main()