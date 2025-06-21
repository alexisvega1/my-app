# --- 6. Loss Function ---
class DiceBCELoss(nn.Module):
    """
    Combined Dice and Binary Cross-Entropy loss for segmentation.
    This is highly effective for imbalanced datasets.
    """
    def __init__(self, weight=0.5, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.weight = weight

    def forward(self, inputs, targets, smooth=1):
        # Flatten label and prediction tensors
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)

        # Binary Cross-Entropy
        BCE = F.binary_cross_entropy(inputs_flat, targets_flat, reduction='mean')
        
        # Dice Coefficient
        intersection = (inputs_flat * targets_flat).sum()
        dice_score = (2. * intersection + smooth) / (inputs_flat.sum() + targets_flat.sum() + smooth)
        Dice_loss = 1 - dice_score
        
        # Combined loss
        Dice_BCE = self.weight * BCE + (1 - self.weight) * Dice_loss
        return Dice_BCE

# We'll keep the old name for compatibility with the config
class MathematicalLossFunction(DiceBCELoss):
    def __init__(self, cfg):
        super().__init__() 