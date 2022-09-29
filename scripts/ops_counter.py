from torchinfo import summary

from models.wide_resnet import wide_resnet28_10
from utils.constants import INPUT_SIZE, NUM_CLASSES


def to_readable(num: int) -> str:
    """
    Converts a number to thousands, millions, billions, or trillions.
    This is a slight modification of the original function in torchinfo.ModelStatistics
    """
    if num >= 1e12:
        return f"{num / 1e12:.3f}T"
    if num >= 1e9:
        return f"{num / 1e9:.3f}G"
    if num >= 1e6:
        return f"{num / 1e6:.3f}M"
    return f"{num / 1e3:.3f}K"

if __name__ == '__main__':
    ds_name = 'CIFAR100'
    c, h, w = INPUT_SIZE[ds_name]
    num_classes = NUM_CLASSES[ds_name]
    cfg_name = '100_baseline_single_late'
    full_net = wide_resnet28_10(num_classes=num_classes)
    summary(full_net, (1, c, h, w))
