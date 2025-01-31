import torch
import torch.nn as nn
import torchvision


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class MessageDecoder(nn.Module):
    def __init__(self, message_len=48, pretrained_weights= None):
        super().__init__()

        if pretrained_weights != None:
            self.decoder = torchvision.models.resnet50(pretrained=False, progress=False)
            checkpoint = torch.load(pretrained_weights)
            self.decoder.load_state_dict(checkpoint)
            self.decoder.fc = nn.Linear(self.decoder.fc.in_features, message_len)
        else:
            self.decoder = torchvision.models.resnet50(pretrained=True, progress=False)
            self.decoder.fc = nn.Linear(self.decoder.fc.in_features, message_len) 

    def forward(self, image):
        x = self.decoder(image)
        return x
