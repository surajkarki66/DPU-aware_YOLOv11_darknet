import math
import torch

from utils.util import make_anchors


def get_activation(act='silu'):
    """Get activation function instance
    Args:
        act: 'silu', 'relu', or 'hardswish' or 'hardsigmoid'
    """
    act = act.lower()
    if act == 'silu':
        return torch.nn.SiLU()
    elif act == 'relu':
        return torch.nn.ReLU()
    elif act == 'hardswish':
        return torch.nn.Hardswish()
    elif act == 'hardsigmoid':
        return torch.nn.Hardsigmoid()
    else:
        raise ValueError(f"Unsupported activation: {act}. Use 'silu', 'relu', 'hardsigmoid', or 'hardswish'")


def fuse_conv(conv, norm):
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 kernel_size=conv.kernel_size,
                                 stride=conv.stride,
                                 padding=conv.padding,
                                 groups=conv.groups,
                                 bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm) # type: ignore

    return fused_conv


class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, activation, k=1, s=1, p=0, g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False)
        self.norm = torch.nn.BatchNorm2d(out_ch, eps=0.001, momentum=0.03)
        self.relu = activation

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

    def fuse_forward(self, x):
        return self.relu(self.conv(x))


class Residual(torch.nn.Module):
    def __init__(self, ch, e=0.5, act='silu'):
        super().__init__()
        self.conv1 = Conv(ch, int(ch * e), get_activation(act), k=3, p=1)
        self.conv2 = Conv(int(ch * e), ch, get_activation(act), k=3, p=1)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class CSPModule(torch.nn.Module):
    def __init__(self, in_ch, out_ch, act='silu'):
        super().__init__()
        self.conv1 = Conv(in_ch, out_ch // 2, get_activation(act))
        self.conv2 = Conv(in_ch, out_ch // 2, get_activation(act))
        self.conv3 = Conv(2 * (out_ch // 2), out_ch, get_activation(act))
        self.res_m = torch.nn.Sequential(Residual(out_ch // 2, e=1.0, act=act),
                                         Residual(out_ch // 2, e=1.0, act=act))

    def forward(self, x):
        y = self.res_m(self.conv1(x))
        return self.conv3(torch.cat((y, self.conv2(x)), dim=1))


class CSP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, n, csp, r, act='silu'):
        super().__init__()
        # Replace single conv + chunk with two separate convolutions (DPU Compatibility)
        self.conv1_a = Conv(in_ch, out_ch // r, get_activation(act))
        self.conv1_b = Conv(in_ch, out_ch // r, get_activation(act))
        self.conv2 = Conv((2 + n) * (out_ch // r), out_ch, get_activation(act))

        if not csp:
            self.res_m = torch.nn.ModuleList(Residual(out_ch // r, act=act) for _ in range(n))
        else:
            self.res_m = torch.nn.ModuleList(CSPModule(out_ch // r, out_ch // r, act=act) for _ in range(n))

    def forward(self, x):
        # use separate convolutions instead of chunk
        y = [self.conv1_a(x), self.conv1_b(x)]
        y.extend(m(y[-1]) for m in self.res_m)
        return self.conv2(torch.cat(y, dim=1))


class SPP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=5, act='silu'):
        super().__init__()
        self.conv1 = Conv(in_ch, in_ch // 2, get_activation(act))
        self.conv2 = Conv(in_ch * 2, out_ch, get_activation(act))
        self.res_m = torch.nn.MaxPool2d(k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.res_m(x)
        y2 = self.res_m(y1)
        return self.conv2(torch.cat(tensors=[x, y1, y2, self.res_m(y2)], dim=1))


class Attention(torch.nn.Module):
    """DPU-compatible Attention Block"""
    def __init__(self, ch, num_head):
        super().__init__()
        self.num_head = num_head
        self.dim_head = ch // num_head
        self.dim_key = self.dim_head // 2
        self.scale = self.dim_key ** -0.5

        self.qkv = Conv(ch, ch + self.dim_key * num_head * 2, torch.nn.Identity())

        self.conv1 = Conv(ch, ch, torch.nn.Identity(), k=3, p=1, g=ch)
        self.conv2 = Conv(ch, ch, torch.nn.Identity())
        
        total_channels = ch + self.dim_key * num_head * 2
        self.q_channels = self.dim_key * num_head
        self.k_channels = self.dim_key * num_head
        self.v_channels = ch
        
        # Create conv layers to extract q, k, v
        self.extract_q = torch.nn.Conv2d(total_channels, self.q_channels, 1, bias=False)
        self.extract_k = torch.nn.Conv2d(total_channels, self.k_channels, 1, bias=False)
        self.extract_v = torch.nn.Conv2d(total_channels, self.v_channels, 1, bias=False)
        
        # Projection layer to match attention channels with value channels
        self.attn_proj = torch.nn.Conv2d(self.q_channels, self.v_channels, 1, bias=False)
        
        # Initialize weights
        with torch.no_grad():
            # Q extraction
            self.extract_q.weight.zero_()
            for i in range(self.q_channels):
                self.extract_q.weight[i, i, 0, 0] = 1.0
            
            # K extraction
            self.extract_k.weight.zero_()
            for i in range(self.k_channels):
                self.extract_k.weight[i, self.q_channels + i, 0, 0] = 1.0
            
            # V extraction
            self.extract_v.weight.zero_()
            for i in range(self.v_channels):
                self.extract_v.weight[i, self.q_channels + self.k_channels + i, 0, 0] = 1.0
            
            # Initialize projection to average pooling
            self.attn_proj.weight.fill_(1.0 / self.q_channels)
        
        # Hardsigmoid
        self.attn_act = torch.nn.Hardsigmoid()

    def forward(self, x):
        qkv = self.qkv(x)

        # Extract q, k, v using convolutions
        q = self.extract_q(qkv)  # [b, q_channels, h, w]
        k = self.extract_k(qkv)  # [b, k_channels, h, w]
        v = self.extract_v(qkv)  # [b, v_channels=ch, h, w]
        
        # Compute attention: q*k produces [b, q_channels, h, w]
        # Project to match v's channel dimension using 1x1 conv
        qk = q * k * self.scale  # [b, q_channels, h, w]
        attn = self.attn_act(self.attn_proj(qk))  # [b, v_channels=ch, h, w]
        
        # Apply attention with element-wise multiplication
        x = v * attn + self.conv1(v)
        return self.conv2(x)


class PSABlock(torch.nn.Module):

    def __init__(self, ch, num_head, act='silu'):
        super().__init__()
        self.conv1 = Attention(ch, num_head)
        self.conv2 = torch.nn.Sequential(Conv(ch, ch * 2, get_activation(act)),
                                         Conv(ch * 2, ch, torch.nn.Identity()))

    def forward(self, x):
        x = x + self.conv1(x)
        return x + self.conv2(x)


class PSA(torch.nn.Module):
    def __init__(self, ch, n, act='silu'):
        super().__init__()
        
        self.conv1a = Conv(ch, ch // 2, get_activation(act))
        self.conv1b = Conv(ch, ch // 2, get_activation(act))
        
        self.conv2 = Conv(2 * (ch // 2), ch, get_activation(act))
        self.res_m = torch.nn.Sequential(*(PSABlock(ch // 2, ch // 128, act=act) for _ in range(n)))

    def forward(self, x):
        x_a = self.conv1a(x)
        x_b = self.conv1b(x)
        return self.conv2(torch.cat(tensors=(x_a, self.res_m(x_b)), dim=1))
        


class DarkNet(torch.nn.Module):
    def __init__(self, width, depth, csp, act='silu'):
        super().__init__()
        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.p4 = []
        self.p5 = []

        # p1/2
        self.p1.append(Conv(width[0], width[1], get_activation(act), k=3, s=2, p=1))
        # p2/4
        self.p2.append(Conv(width[1], width[2], get_activation(act), k=3, s=2, p=1))
        self.p2.append(CSP(width[2], width[3], depth[0], csp[0], r=4, act=act))
        # p3/8
        self.p3.append(Conv(width[3], width[3], get_activation(act), k=3, s=2, p=1))
        self.p3.append(CSP(width[3], width[4], depth[1], csp[0], r=4, act=act))
        # p4/16
        self.p4.append(Conv(width[4], width[4], get_activation(act), k=3, s=2, p=1))
        self.p4.append(CSP(width[4], width[4], depth[2], csp[1], r=2, act=act))
        # p5/32
        self.p5.append(Conv(width[4], width[5], get_activation(act), k=3, s=2, p=1))
        self.p5.append(CSP(width[5], width[5], depth[3], csp[1], r=2, act=act))
        self.p5.append(SPP(width[5], width[5], act=act))
        self.p5.append(PSA(width[5], depth[4], act=act))

        self.p1 = torch.nn.Sequential(*self.p1)
        self.p2 = torch.nn.Sequential(*self.p2)
        self.p3 = torch.nn.Sequential(*self.p3)
        self.p4 = torch.nn.Sequential(*self.p4)
        self.p5 = torch.nn.Sequential(*self.p5)

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return p3, p4, p5


class DarkFPN(torch.nn.Module):
    def __init__(self, width, depth, csp, act='silu'):
        super().__init__()
        self.up = torch.nn.Upsample(scale_factor=2)
        self.h1 = CSP(width[4] + width[5], width[4], depth[5], csp[0], r=2, act=act)
        self.h2 = CSP(width[4] + width[4], width[3], depth[5], csp[0], r=2, act=act)
        self.h3 = Conv(width[3], width[3], get_activation(act), k=3, s=2, p=1)
        self.h4 = CSP(width[3] + width[4], width[4], depth[5], csp[0], r=2, act=act)
        self.h5 = Conv(width[4], width[4], get_activation(act), k=3, s=2, p=1)
        self.h6 = CSP(width[4] + width[5], width[5], depth[5], csp[1], r=2, act=act)

    def forward(self, x):
        p3, p4, p5 = x
        p4 = self.h1(torch.cat(tensors=[self.up(p5), p4], dim=1))
        p3 = self.h2(torch.cat(tensors=[self.up(p4), p3], dim=1))
        p4 = self.h4(torch.cat(tensors=[self.h3(p3), p4], dim=1))
        p5 = self.h6(torch.cat(tensors=[self.h5(p4), p5], dim=1))
        return p3, p4, p5


class DFL(torch.nn.Module):
    # Generalized Focal Loss
    # https://ieeexplore.ieee.org/document/9792391
    def __init__(self, ch=16):
        super().__init__()
        self.ch = ch
        self.conv = torch.nn.Conv2d(ch, out_channels=1, kernel_size=1, bias=False).requires_grad_(False)
        x = torch.arange(ch, dtype=torch.float).view(1, ch, 1, 1)
        self.conv.weight.data[:] = torch.nn.Parameter(x)

    def forward(self, x):
        b, _, a = x.shape
        x = x.view(b, 4, self.ch, a).transpose(2, 1)
        return self.conv(x.softmax(1)).view(b, 4, a)


class Head(torch.nn.Module):
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, filters=(), exclude_post_process=False, act='silu'):
        super().__init__()
        self.ch = 16  # DFL channels
        self.nc = nc  # number of classes
        self.nl = len(filters)  # number of detection layers
        self.no = nc + self.ch * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        self.exclude_post_process = exclude_post_process

        box = max(64, filters[0] // 4)
        cls = max(80, filters[0], self.nc)

        # DFL is only needed for post-processing
        if not exclude_post_process:
            self.dfl = DFL(self.ch)
        else:
            self.dfl = None
        
        self.box = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, box, get_activation(act), k=3, p=1),
                                                           Conv(box, box, get_activation(act), k=3, p=1),
                                                           torch.nn.Conv2d(box, out_channels=4 * self.ch,
                                                                           kernel_size=1)) for x in filters)
        self.cls = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, x, get_activation(act), k=3, p=1, g=x),
                                                           Conv(x, cls, get_activation(act)),
                                                           Conv(cls, cls, get_activation(act), k=3, p=1, g=cls),
                                                           Conv(cls, cls, get_activation(act)),
                                                           torch.nn.Conv2d(cls, out_channels=self.nc,
                                                                           kernel_size=1)) for x in filters)

    def forward(self, x):
        # Apply box and cls convolutions
        for i, (box, cls) in enumerate(zip(self.box, self.cls)):
            x[i] = torch.cat(tensors=(box(x[i]), cls(x[i])), dim=1)
        
        # For DPU deployment
        if self.exclude_post_process:
            return x
        
        # Training
        if self.training:
            return x

        self.anchors, self.strides = (i.transpose(0, 1) for i in make_anchors(x, self.stride))
        x = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], dim=2)
        box, cls = x.split(split_size=(4 * self.ch, self.nc), dim=1)

        a, b = self.dfl(box).chunk(2, 1) # type: ignore
        a = self.anchors.unsqueeze(0) - a
        b = self.anchors.unsqueeze(0) + b
        box = torch.cat(tensors=((a + b) / 2, b - a), dim=1)

        return torch.cat(tensors=(box * self.strides, cls.sigmoid()), dim=1)

    def initialize_biases(self):
        # Initialize biases
        # WARNING: requires stride availability
        for box, cls, s in zip(self.box, self.cls, self.stride):
            # box
            box[-1].bias.data[:] = 1.0
            # cls (.01 objects, 80 classes, 640 image)
            cls[-1].bias.data[:self.nc] = math.log(5 / self.nc / (640 / s) ** 2)


class YOLO(torch.nn.Module):
    def __init__(self, width, depth, csp, num_classes, exclude_post_process=False, activation='silu'):
        super().__init__()
        self.net = DarkNet(width, depth, csp, act=activation)
        self.fpn = DarkFPN(width, depth, csp, act=activation)

        img_dummy = torch.zeros(1, width[0], 256, 256)
        self.head = Head(num_classes, (width[3], width[4], width[5]), exclude_post_process=exclude_post_process, act=activation)
        self.head.stride = torch.tensor([256 / x.shape[-2] for x in self.forward(img_dummy)])
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


def yolo_v11_n(num_classes: int = 80, exclude_post_process: bool = False, activation: str = 'silu'):
    csp = [False, True]
    depth = [1, 1, 1, 1, 1, 1]
    width = [3, 16, 32, 64, 128, 256]
    return YOLO(width, depth, csp, num_classes, exclude_post_process, activation)


def yolo_v11_t(num_classes: int = 80, exclude_post_process: bool = False, activation: str = 'silu'):
    csp = [False, True]
    depth = [1, 1, 1, 1, 1, 1]
    width = [3, 24, 48, 96, 192, 384]
    return YOLO(width, depth, csp, num_classes, exclude_post_process, activation)


def yolo_v11_s(num_classes: int = 80, exclude_post_process: bool = False, activation: str = 'silu'):
    csp = [False, True]
    depth = [1, 1, 1, 1, 1, 1]
    width = [3, 32, 64, 128, 256, 512]
    return YOLO(width, depth, csp, num_classes, exclude_post_process, activation)


def yolo_v11_m(num_classes: int = 80, exclude_post_process: bool = False, activation: str = 'silu'):
    csp = [True, True]
    depth = [1, 1, 1, 1, 1, 1]
    width = [3, 64, 128, 256, 512, 512]
    return YOLO(width, depth, csp, num_classes, exclude_post_process, activation)


def yolo_v11_l(num_classes: int = 80, exclude_post_process: bool = False, activation: str = 'silu'):
    csp = [True, True]
    depth = [2, 2, 2, 2, 2, 2]
    width = [3, 64, 128, 256, 512, 512]
    return YOLO(width, depth, csp, num_classes, exclude_post_process, activation)


def yolo_v11_x(num_classes: int = 80, exclude_post_process: bool = False, activation: str = 'silu'):
    csp = [True, True]
    depth = [2, 2, 2, 2, 2, 2]
    width = [3, 96, 192, 384, 768, 768]
    return YOLO(width, depth, csp, num_classes, exclude_post_process, activation)
