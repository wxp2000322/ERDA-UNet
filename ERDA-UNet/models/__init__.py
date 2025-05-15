from models.network import *

def get_segmentation_model(name):
    if name == 'ERDA':
        net = get_ERDA()
    else:
        raise NotImplementedError

    return net


if __name__ == '__main__':
    net = get_segmentation_model('transformer')
    from torchsummary import summary

    summary(net, (3, 256, 256), device='cpu')
