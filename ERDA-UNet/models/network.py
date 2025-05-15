from models.ERDA import ERDA

def get_ERDA(num_class=1):
    return ERDA(num_class)


if __name__ == '__main__':
    model = get_ERDA(1)
    from torchsummary import summary

    summary(model, (3, 256, 256), device='cpu')
