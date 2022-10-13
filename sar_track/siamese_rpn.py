from mmtrack.models import MODELS, SiamRPN


@MODELS.register_module()
class SiameseRPN(SiamRPN):
    """Implement Siamese RPN which can be inited with pretrained init_cfg.
    NOTE The init_weights official implementation only init submodules.
    """

    def init_weights(self):
        if self.init_cfg is None:
            super().init_weights()
        else:
            super(SiamRPN, self).init_weights()
