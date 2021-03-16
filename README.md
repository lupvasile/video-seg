# Video Semantic Segmentation leveraging Dense Optical Flow
> V. Lup and S. Nedevschi, "Video Semantic Segmentation leveraging Dense Optical Flow," 2020 IEEE 16th International Conference on Intelligent Computer Communication and Processing (ICCP), Cluj-Napoca, Romania, 2020, pp. 369-376, doi: 10.1109/ICCP51029.2020.9266150.

Publication [link](https://ieeexplore.ieee.org/abstract/document/9266150).

## Abstract
Semantic segmentation is a key step in scene understanding for any autonomous system, as it provides the class of each pixel of an image. The best results on this task are achieved by deep neural networks, however they require in training a significant number of annotated frames which are difficult to create. This work is focused on improving the semantic segmentation network on sparsely labeled video sequences by including optical flow information between consecutive frames, and to provide a meaningful evaluation using densely labeled sequences. The system is composed of a static semantic segmentation, an optical flow and a linking network, which are chosen from existing architectures based on their high accuracy and low computational needs, but were never used together before. After conducting various experiments on the novel Virtual KITTI 2 dataset, we find that the optical flow compensates for the sparsity of annotations in video training sets, and by using just 25% of the labels we achieve the same video semantic segmentation quality as the static network trained on the fully labeled dataset. Since fewer annotations are needed, the cost of ground truth generation is reduced significantly. Moreover, taking advantage of the optical flow ground truth availability, we find the limits of the flow approach despite future advances, when more accurate optical flow networks will be developed. The system's improved capabilities are also validated on the Cityscapes dataset.

## Acknowledgement
This code uses the following repositories:
* Erfnet [implementation](https://github.com/Eromera/erfnet_pytorch)
* VCN [implementation](https://github.com/gengshan-y/VCN/)
* Cityscapes Scripts [implementation](https://github.com/mcordts/cityscapesScripts)

Please see [NOTICE](NOTICE) for details about their license.