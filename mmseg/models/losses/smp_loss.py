import segmentation_models_pytorch.losses as smp

from ..builder import LOSSES


@LOSSES.register_module()
class SMPDiceLoss(smp.DiceLoss):
    def __init__(self,
                 weight=1.0,
                 **kwargs):
        super(SMPDiceLoss, self).__init__(**kwargs)
        self.weight = weight

    def forward(self, *args):
        loss = super(SMPDiceLoss, self).forward(*args)
        return loss * self.weight


@LOSSES.register_module()
class SMPFocalLoss(smp.FocalLoss):
    def __init__(self,
                 weight=1.0,
                 **kwargs):
        self.weight = weight
        super(SMPFocalLoss, self).__init__(**kwargs)

    def forward(self, *args):
        loss = super(SMPFocalLoss, self).forward(*args)
        return loss * self.weight


@LOSSES.register_module()
class SMPSoftBCEWithLogitsLoss(smp.SoftBCEWithLogitsLoss):
    def __init__(self,
                 weight=1.0,
                 **kwargs):
        super(SMPSoftBCEWithLogitsLoss, self).__init__(**kwargs)
        self.loss_weight = weight

    def forward(self, *args):
        loss = super(SMPSoftBCEWithLogitsLoss, self).forward(*args)
        return loss * self.loss_weight