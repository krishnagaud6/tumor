"""Best-effort architecture reconstruction for the provided tumor_cnn checkpoint.

IMPORTANT: This architecture was inferred from state_dict key names and tensor
shapes, not recovered from the original training source. It loads all tensors in
the supplied checkpoint, but the original creator should confirm non-parameter
layers, image preprocessing, and label order before using predictions.
"""

from torch import nn


class TumorCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Inferred from the checkpoint keys:
        # features.0.weight -> Conv2d(3, 16, 3, ...)
        # features.3.weight -> Conv2d(16, 32, 3, ...)
        # features.6 has no weights; AdaptiveAvgPool2d is a plausible fit.
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.AdaptiveAvgPool2d((8, 8)),
        )

        # The stored tensor classifier.1.weight has shape [64, 2048],
        # and classifier.3.weight has shape [2, 64].
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def create_model():
    """Factory used by app.py when loading the state_dict checkpoint."""
    return TumorCNN()
