import torch

from models.common import DarkFPN, DarkNet, Head, OBBHead, Conv, fuse_conv


class YOLO(torch.nn.Module):
    def __init__(self, width, depth, csp, num_classes, exclude_post_process=False, activation='silu', imgsz=640, task='detect'):
        super().__init__()
        self.task = task
        self.net = DarkNet(width, depth, csp, act=activation)
        self.fpn = DarkFPN(width, depth, csp, act=activation)

        img_dummy = torch.zeros(1, width[0], imgsz, imgsz)
        if task == 'obb':
            self.head = OBBHead(num_classes, (width[3], width[4], width[5]), ne=1,
                               exclude_post_process=exclude_post_process, act=activation, imgsz=imgsz)
        else:
            self.head = Head(num_classes, (width[3], width[4], width[5]),
                            exclude_post_process=exclude_post_process, act=activation, imgsz=imgsz)
        with torch.no_grad():
            fpn_out = self.fpn(self.net(img_dummy))
        self.head.stride = torch.tensor([imgsz / x.shape[-2] for x in fpn_out])
        self.stride = self.head.stride
        if not exclude_post_process:
            self.head.initialize_biases()

    def forward(self, x):
        x = self.net(x)
        x = self.fpn(x)
        return self.head(list(x))

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self


def yolo_v11_n(num_classes: int = 80, exclude_post_process: bool = False, activation: str = 'silu', imgsz: int = 640, task: str = 'detect'):
    csp = [False, True]
    depth = [1, 1, 1, 1, 1, 1]
    width = [3, 16, 32, 64, 128, 256]
    return YOLO(width, depth, csp, num_classes, exclude_post_process, activation, imgsz, task)


def yolo_v11_n_obb(num_classes: int = 80, exclude_post_process: bool = False, activation: str = 'silu', imgsz: int = 640):
    return yolo_v11_n(num_classes, exclude_post_process, activation, imgsz, task='obb')


def yolo_v11_t(num_classes: int = 80, exclude_post_process: bool = False, activation: str = 'silu', imgsz: int = 640, task: str = 'detect'):
    csp = [False, True]
    depth = [1, 1, 1, 1, 1, 1]
    width = [3, 24, 48, 96, 192, 384]
    return YOLO(width, depth, csp, num_classes, exclude_post_process, activation, imgsz, task)


def yolo_v11_t_obb(num_classes: int = 80, exclude_post_process: bool = False, activation: str = 'silu', imgsz: int = 640):
    return yolo_v11_t(num_classes, exclude_post_process, activation, imgsz, task='obb')


def yolo_v11_s(num_classes: int = 80, exclude_post_process: bool = False, activation: str = 'silu', imgsz: int = 640, task: str = 'detect'):
    csp = [False, True]
    depth = [1, 1, 1, 1, 1, 1]
    width = [3, 32, 64, 128, 256, 512]
    return YOLO(width, depth, csp, num_classes, exclude_post_process, activation, imgsz, task)


def yolo_v11_s_obb(num_classes: int = 80, exclude_post_process: bool = False, activation: str = 'silu', imgsz: int = 640):
    return yolo_v11_s(num_classes, exclude_post_process, activation, imgsz, task='obb')


def yolo_v11_m(num_classes: int = 80, exclude_post_process: bool = False, activation: str = 'silu', imgsz: int = 640, task: str = 'detect'):
    csp = [True, True]
    depth = [1, 1, 1, 1, 1, 1]
    width = [3, 64, 128, 256, 512, 512]
    return YOLO(width, depth, csp, num_classes, exclude_post_process, activation, imgsz, task)


def yolo_v11_m_obb(num_classes: int = 80, exclude_post_process: bool = False, activation: str = 'silu', imgsz: int = 640):
    return yolo_v11_m(num_classes, exclude_post_process, activation, imgsz, task='obb')


def yolo_v11_l(num_classes: int = 80, exclude_post_process: bool = False, activation: str = 'silu', imgsz: int = 640, task: str = 'detect'):
    csp = [True, True]
    depth = [2, 2, 2, 2, 2, 2]
    width = [3, 64, 128, 256, 512, 512]
    return YOLO(width, depth, csp, num_classes, exclude_post_process, activation, imgsz, task)


def yolo_v11_l_obb(num_classes: int = 80, exclude_post_process: bool = False, activation: str = 'silu', imgsz: int = 640):
    return yolo_v11_l(num_classes, exclude_post_process, activation, imgsz, task='obb')


def yolo_v11_x(num_classes: int = 80, exclude_post_process: bool = False, activation: str = 'silu', imgsz: int = 640, task: str = 'detect'):
    csp = [True, True]
    depth = [2, 2, 2, 2, 2, 2]
    width = [3, 96, 192, 384, 768, 768]
    return YOLO(width, depth, csp, num_classes, exclude_post_process, activation, imgsz, task)


def yolo_v11_x_obb(num_classes: int = 80, exclude_post_process: bool = False, activation: str = 'silu', imgsz: int = 640):
    return yolo_v11_x(num_classes, exclude_post_process, activation, imgsz, task='obb')
