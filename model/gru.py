import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from utils.image_viewer import IM


class WarpModuleGru:
    """adapted from PWCNet"""

    def __init__(self, size=0):
        super(WarpModuleGru, self).__init__()
        self.grid = torch.empty(size)

    def _make_grid(self, size):
        B, C, H, W = size

        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        self.grid = torch.cat((xx, yy), 1).float()

    def __call__(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        assert x.dim() == 4 and flo.dim() == 4

        if self.grid.shape[0] != x.shape[0] or self.grid.shape[2:] != x.shape[2:]:
            self._make_grid(x.size())
        if self.grid.device != x.device:
            self.grid = self.grid.to(x.device)

        B, C, H, W = x.size()
        vgrid = self.grid + flo

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid, align_corners=True, padding_mode='zeros', mode='bilinear')
        return output


def get_gaussian_kernel(shape=(3, 3), sigma=0.7):
    x = np.arange(-(shape[0] // 2), shape[0] // 2 + 1, 1)
    y = np.arange(-(shape[1] // 2), shape[1] // 2 + 1, 1)
    xx, yy = np.meshgrid(x, y)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * (sigma ** 2)))
    kernel = kernel / (2 * np.pi * (sigma ** 2))
    return torch.tensor(kernel)


class GRU(nn.Module):
    def __init__(self, conv_size, nr_channels, show_timeout=None):
        """nr_channels - number of segmentation tensor classes"""
        super(GRU, self).__init__()

        assert conv_size[0] % 2 == 1 and conv_size[1] % 2 == 1  # current padding only works for odd sizes

        self.conv_size = conv_size
        padding_size = (conv_size[0] // 2, conv_size[1] // 2)  # in tf used 'SAME' padding, this is different from tf
        self.padding_size = padding_size

        identity = torch.zeros((nr_channels, nr_channels, conv_size[0], conv_size[1]), dtype=torch.float32)
        for k in range(nr_channels):
            identity[k, k, conv_size[0] // 2, conv_size[1] // 2] = 1.

        self.conv_ir = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=conv_size, stride=1, padding=padding_size, bias=True)  # includes bias_r
        with torch.no_grad():
            self.conv_ir.weight.copy_(get_gaussian_kernel(conv_size) + 4 * identity[0, 0] + torch.zeros_like(self.conv_ir.weight).normal_(std=0.001))
        nn.init.zeros_(self.conv_ir.bias)  # initializations must be done with no_grad, no_grad is caled in nn.init

        self.conv_xh = nn.Conv2d(in_channels=nr_channels, out_channels=nr_channels, kernel_size=conv_size, stride=1, padding=padding_size, bias=False)
        with torch.no_grad():
            self.conv_xh.weight.copy_(6. * identity + torch.zeros_like(identity).normal_(std=0.01))

        self.conv_hh = nn.Conv2d(in_channels=nr_channels, out_channels=nr_channels, kernel_size=conv_size, stride=1, padding=padding_size, bias=False)
        with torch.no_grad():
            self.conv_hh.weight.copy_(6. * identity + torch.zeros_like(identity).normal_(std=0.01))

        self.conv_xz = nn.Conv2d(in_channels=nr_channels, out_channels=1, kernel_size=conv_size, stride=1, padding=padding_size, bias=False)
        nn.init.normal_(self.conv_xz.weight, std=0.01)

        self.conv_hz = nn.Conv2d(in_channels=nr_channels, out_channels=1, kernel_size=conv_size, stride=1, padding=padding_size, bias=False)
        nn.init.normal_(self.conv_hz.weight, std=0.01)

        self.lamda = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        nn.init.constant_(self.lamda, 4)

        self.bias_z = nn.Parameter(torch.zeros((nr_channels, 1, 1), dtype=torch.float32))
        nn.init.zeros_(self.bias_z)

        self.warp_with_flow = WarpModuleGru()

        self.show_timeout = show_timeout
        self.image_viewer = None
        if show_timeout is not None:
            self.image_viewer = IM

    def forward(self, hidden_prev, curr_segmentation, flow, prev_image, curr_image):
        """hidden_prev is the previous segmentation
        images should be in dimension N*C*H*W, (currently N should be only 1)
        """
        assert prev_image.ndimension() == 4 and hidden_prev.ndimension() == 4 and flow.ndimension() == 4
        assert prev_image.size()[0] == 1

        hidden_prev = F.softmax(hidden_prev, dim=1)
        curr_segmentation = F.softmax(curr_segmentation, dim=1)

        img_diff = curr_image - self.warp_with_flow(prev_image, flow)

        hidden_prev_warped = self.warp_with_flow(hidden_prev, flow)

        r = 1.0 - torch.tanh(torch.abs(self.conv_ir(img_diff)))  # conv_ir includes bias_r

        hidden_prev_reset = hidden_prev_warped * r

        hidden_tilde = self.conv_xh(curr_segmentation) + self.conv_hh(hidden_prev_reset)

        z = torch.sigmoid(self.conv_xz(curr_segmentation) + self.conv_hz(hidden_prev_reset) + self.bias_z)

        h = self.lamda * (1 - z) * hidden_prev_reset + z * hidden_tilde

        if self.image_viewer is not None:
            self.image_viewer.showImage(6, prev_image, imgtitle='prev_image')
            self.image_viewer.showImage(3, curr_image, imgtitle='curr_image')
            self.image_viewer.showImage(0, flow, 'flow', imgtitle='flow')
            self.image_viewer.showImage(4, self.warp_with_flow(prev_image, flow), imgtitle='warped_image')
            self.image_viewer.showImage(1, img_diff.abs(), imgtitle='img_diff abs')
            self.image_viewer.showImage(5, r, imgtitle='r')

            self.image_viewer.showImage(7, hidden_prev_warped, 'seg', imgtitle='hidden prev warped')
            self.image_viewer.showImage(8, hidden_prev_reset, 'seg', imgtitle='hidden prev reset')
            self.image_viewer.showImage(9, hidden_prev, 'seg', imgtitle='prev_state')
            self.image_viewer.showImage(10, curr_segmentation, 'seg', imgtitle='curr_segmentation')
            self.image_viewer.showImage(11, h, 'seg', imgtitle='curr_refined_segmentation')
            self.image_viewer.show(self.show_timeout)

        return h


def run_gru_sequence(gru: GRU, seg_frames, flow_frames, orig_frames, prev_cell_state, prev_image, all_outputs: bool):
    nr_frames = orig_frames.size()[0]
    offset = int(prev_cell_state is None)  # 1 when prev_cell_state is not provided, 0 otherwise

    assert len(flow_frames) + offset == nr_frames, 'not enough flow frames'

    outputs = []
    if prev_cell_state is None:
        hidden_state = seg_frames[0:1]
    else:
        hidden_state = gru(prev_cell_state, seg_frames[0:1], flow_frames[0], prev_image, orig_frames[0:1])

    if all_outputs:
        outputs.append(hidden_state)

    for i in range(1, nr_frames):
        hidden_state = gru(hidden_state, seg_frames[i:i + 1], flow_frames[i - offset], orig_frames[i - 1:i], orig_frames[i:i + 1])
        if all_outputs:
            outputs.append(hidden_state)

    if not all_outputs:
        return hidden_state
    else:
        return torch.cat(outputs)


def _tests():
    GRU((7, 7), 20)
    w = WarpModuleGru()
    img_in = torch.arange(1, 17).float().view(1, 1, 4, -1)
    flow = torch.tensor([
        [[-1, -1, -1, -1],
         [-1, -1, -1, -1],
         [-1, -0.5, -1, -1],
         [-1, -1, -1, -1]],

        [[-1, -1, -1, -1],
         [-1, -1, -1, -1],
         [-1, -0.5, -1, -1],
         [-1, -1, -1, -1]]
    ]).float().unsqueeze(0)
    print(img_in, img_in.shape)
    print(flow, flow.shape)
    print(w(img_in, flow), w(img_in, flow).shape)
