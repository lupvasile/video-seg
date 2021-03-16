import numpy as np
import torch

from model.vcn import VCN


class VCN_Wrapped(VCN):
    """L is previous frame, R is current frame, flow is L refference frame"""

    # check partial GRU implementation commit
    def __init__(self, size, md, fac, meanL, meanR):
        """
        md - maximum disparity, flow for a pixel can be -md to md pixels
        fac - width of search windows is unchanged, height is divided by fac
        """
        assert size[0] == 1, 'VCN_Wrapped can only support one image at a time'
        super().__init__(size, md, fac)
        self.register_buffer('meanL', self.__get_mean_buffer(meanL))
        self.register_buffer('meanR', self.__get_mean_buffer(meanR))

    def forward(self, im_prev: torch.Tensor, im_new: torch.Tensor, disc_aux=None):
        """im_prev, im_new should be dim(3,height,width) and RGB format, normalized in [0..1]"""
        assert im_prev.dim() == 3 and im_new.dim() == 3, 'VCN_Wrapped can only support one image at a time'

        im_prev = im_prev.flip(0)
        im_new = im_new.flip(0)

        im_prev = im_prev - self.meanL.view(3, 1, -1)
        im_new = im_new - self.meanR.view(3, 1, -1)

        im = torch.stack((im_prev, im_new))
        assert im.dtype is torch.float

        return super().forward(im, disc_aux)

    def __get_mean_buffer(self, meanArray: np.ndarray):
        if meanArray is None:
            return torch.tensor([0.33, 0.33, 0.33]).float()

        meanArray = np.asarray(meanArray)
        if meanArray.ndim == 1:
            meanArray = meanArray[np.newaxis, :]

        return torch.from_numpy(meanArray.mean(0)).float()
