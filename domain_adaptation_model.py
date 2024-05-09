import torch.nn as nn
from torch.autograd import Function

# Custom Function for gradient reversal
class GradientReversalFunction(Function):
    @staticmethod
    # Forward pass is a no-op; saves the alpha value for use in the backward pass
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # In the backward pass, gradients are multiplied by -alpha (gradient reversal)
        output = grad_output.neg() * ctx.alpha  # Reverse and scale the gradient
        return output, None  # Only gradients w.r.t. x are returned; None for alpha

def grad_reverse(x, alpha): # Alpha can and should be tuned
    return GradientReversalFunction.apply(x, alpha)

class DomainAdaptationModel(nn.Module):
    def __init__(self, unet_model, domain_classifier, alpha=1):
        super(DomainAdaptationModel, self).__init__()
        self.unet_model = unet_model
        self.domain_classifier = domain_classifier
        self.alpha = alpha

    def forward(self, x):
        # Obtain segmentation output and bottleneck features from the UNet model
        seg_output, features = self.unet_model(x)
        
        # Apply gradient reversal to the bottleneck features for domain adaptation
        reversed_features = grad_reverse(features, self.alpha)

        # Flatten the reversed features (domain classifier expects a flat vector)
        reversed_features_flat = reversed_features.view(reversed_features.size(0), -1)

        # Pass the reversed features through the domain classifier
        domain_output = self.domain_classifier(reversed_features_flat)

        return seg_output, domain_output # Return both segmentation and domain classification outputs
