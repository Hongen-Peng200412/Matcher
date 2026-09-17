"""用带回收状态、注意力门与三轴旋转位置编码的 RAUNet 编码候选体素。

主要入口 RaUnetBackbone.forward 接收通道优先的 48 立方体候选输入，经过四级
下采样、最深网格全连接自注意力、三级注意力门控跳跃连接和多尺度卷积分支，
返回同分辨率体素特征。recycle_iterations 大于一时，前置轮显式放在 torch.no_grad
中，只把最高分辨率 decoded0 作为下一轮状态；最后一轮不受该内部 no_grad 包裹，
是否建立计算图和参与反向传播仍服从调用位置的外层梯度模式。
本模块只处理无物理坐标附加信息的规则体素网格，不读取世界原点或体素大小。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


# 九项整数元组，依次为 enc0、enc1、enc2、enc3、bottleneck、dec3、dec2、dec1、dec0 的默认通道数。
DEFAULT_PLANES = (64, 256, 256, 256, 256, 256, 128, 64, 64)


def _largest_divisor_not_above(value: int, maximum: int) -> int:
    """选择不超过上限且能整除通道数的最大正整数。

    输入参数：
        - value: 正整数；待分头的通道数，正式调用来自已验证的 RAUNet 通道。
        - maximum: 正整数；允许返回的最大头数，当前调用固定为 8。

    返回值：
        - divisor: 正整数；范围 [1,min(value,maximum)]，并满足 value % divisor == 0。

    本函数不显式检查两个输入为正；直接传入非正值会遍历空区间并触发末尾 AssertionError。
    """

    for candidate in range(min(value, maximum), 0, -1):
        if value % candidate == 0:
            return candidate
    raise AssertionError("正整数至少能被 1 整除。")


def _grid_positions(
    depth: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """建立三维旋转位置编码使用的规则网格位置。

    输入参数：
        - depth: 正整数 Z；网格深度位置数。
        - height: 正整数 Y；网格高度位置数。
        - width: 正整数 X；网格宽度位置数。
        - device: torch.device；返回张量所在设备。
        - dtype: torch.dtype；三个位置分量的数据类型，通常与当前体素特征相同。

    返回值：
        - positions: 浮点张量，(L,3)，L=Z×Y×X；最后一轴依次为 Z/Y/X，展平顺序使 X 最先变化，原点为 (0,0,0)，每轴相邻位置差为 1.5。

    这里的 1.5 是模型内部无物理单位位置间隔，不是 Å，也不读取 CandidateBatch 的真实体素大小。本函数依赖调用者提供正空间尺寸，不单独检查。
    """

    # 长度分别为 Z、Y、X 的三个一维坐标张量；元素从 0 递增到对应大小减一。
    axes = [torch.arange(size, device=device, dtype=dtype) for size in (depth, height, width)]
    # meshgrid 先产生 (Z,Y,X,3)，reshape 按 NumPy C-order 等价顺序展平空间轴，再乘模型内部间隔 1.5。
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3) * 1.5


def _apply_three_axis_rope(values: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """把每个注意力头的通道均分给 Z/Y/X，并对相邻通道对执行旋转。

    输入参数：
        - values: 浮点张量，(B,K,L,C_h)；查询或键，C_h 必须能被 6 整除，使三轴各有偶数通道。
        - positions: 浮点张量，(L,3)；与 values 的位置轴逐项对齐，最后一轴依次为 Z/Y/X。

    返回值：
        - rotated_values: 浮点张量，(B,K,L,C_h)；三段轴通道各自按对应位置相位旋转后重新连接。

    每轴通道数 C_axis=C_h/3，相邻偶数/奇数通道组成二维向量。第 j 个通道对的逆频率为 exp(-log(100)×j/max(C_axis/2,1))。本函数只显式检查计算出的 C_axis 为偶数；正式 RotarySelfAttention3d 构造器保证 C_h 可被 6 整除并提供匹配的 L。
    """

    # int，每个空间轴分到的头内通道数；正式调用满足 C_h=3×axis_channels。
    axis_channels = values.shape[-1] // 3
    if axis_channels % 2 != 0:
        raise ValueError("每个空间轴的旋转位置编码通道必须为偶数。")
    # list[Tensor]，依次收集 Z、Y、X 三段旋转后的 (B,K,L,C_axis) 通道。
    rotated_parts = []
    # float32，(C_axis/2,)，每个相邻通道对的零基频率编号；先用 FP32 计算指数以稳定频率。
    frequency_index = torch.arange(
        axis_channels // 2,
        device=values.device,
        dtype=torch.float32,
    )
    # 与 values 同 dtype，(C_axis/2,)，从 1 到约 1/100 的指数间隔逆频率。
    inverse_frequency = torch.exp(
        -math.log(100.0) * frequency_index / max(axis_channels // 2, 1)
    ).to(dtype=values.dtype)
    for axis, part in enumerate(values.split(axis_channels, dim=-1)):
        # 浮点，(L,C_axis/2)，当前位置轴分量与各通道对逆频率的外积相位。
        angles = positions[:, axis, None] * inverse_frequency[None, :]
        # 浮点，(1,1,L,C_axis/2)，向批次和头轴广播的余弦系数。
        cosine = angles.cos()[None, None]
        # 浮点，(1,1,L,C_axis/2)，向批次和头轴广播的正弦系数。
        sine = angles.sin()[None, None]
        # 浮点，(B,K,L,C_axis/2)，当前轴通道段的偶数位置分量。
        even = part[..., 0::2]
        # 浮点，(B,K,L,C_axis/2)，当前轴通道段的奇数位置分量。
        odd = part[..., 1::2]
        # 浮点，[B,K,L,C_axis/2] -> [B,K,L,C_axis/2,2] -> [B,K,L,C_axis]，按二维旋转公式更新每对通道。
        rotated = torch.stack(
            (even * cosine - odd * sine, even * sine + odd * cosine),
            dim=-1,
        ).flatten(-2)
        rotated_parts.append(rotated)
    return torch.cat(rotated_parts, dim=-1)


def _upsample_add(coarse: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
    """把低分辨率特征插值到跳跃特征空间大小后逐元素相加。

    输入参数：
        - coarse: 浮点张量，(B,C,Z_c,Y_c,X_c)；待上采样的粗尺度特征。
        - skip: 浮点张量，(B,C,Z_s,Y_s,X_s)；提供目标空间大小和逐元素跳跃值，批次与通道必须可和 coarse 插值结果相加。

    返回值：
        - fused: 浮点张量，(B,C,Z_s,Y_s,X_s)；align_corners=True 三线性插值结果与 skip 的逐元素和。

    本函数不投影通道，也不显式检查批次或通道相等；不相容形状由张量加法失败。
    """

    return skip + F.interpolate(
        coarse,
        size=skip.shape[-3:],
        mode="trilinear",
        align_corners=True,
    )


class ResidualBottleneck3d(nn.Module):
    """用一乘一、三乘三、一乘一卷积主路和捷径执行残差下采样。

    构造参数：
        - input_channels: 整数 C_in；输入通道数。
        - output_channels: 整数 C_out；输出通道数，必须能被固定 expansion=4 整除。
        - stride: 整数 s；主路三乘三卷积和投影捷径的共同步幅。

    前向输入：
        - inputs: 浮点张量，(B,C_in,Z,Y,X)。

    前向输出：
        - outputs: 浮点张量，(B,C_out,ceil(Z/s),ceil(Y/s),ceil(X/s))；主路和捷径相加后经过 ReLU。
    """

    expansion = 4

    def __init__(self, input_channels: int, output_channels: int, stride: int) -> None:
        """建立瓶颈主路和按需投影的残差捷径。

        输入参数：
            - input_channels: 整数 C_in；主路首层和捷径读取宽度。
            - output_channels: 整数 C_out；必须能被 4 整除，中间宽度为 C_out/4。
            - stride: 整数 s；传给主路三乘三卷积，且在通道或步幅变化时传给捷径一乘一卷积。

        实例状态：
            - self.main: Sequential；C_in 到 C_out/4 到 C_out 的三层卷积及 InstanceNorm/ReLU。
            - self.shortcut: Identity 或 Conv3d；仅 s=1 且 C_in=C_out 时为恒等映射，否则投影到 C_out 并使用步幅 s。
            - self.activation: ReLU；作用于主路和捷径之和。
        """

        super().__init__()
        if output_channels % self.expansion != 0:
            raise ValueError("ResidualBottleneck3d 的输出通道必须能被 4 整除。")
        # int，瓶颈主路内部通道数 C_out/4。
        hidden_channels = output_channels // self.expansion
        self.main = nn.Sequential(
            nn.Conv3d(input_channels, hidden_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(hidden_channels, affine=True),
            nn.ReLU(),
            nn.Conv3d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm3d(hidden_channels, affine=True),
            nn.ReLU(),
            nn.Conv3d(hidden_channels, output_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(output_channels, affine=True),
        )
        self.shortcut: nn.Module
        if stride == 1 and input_channels == output_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv3d(
                input_channels,
                output_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
        self.activation = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """相加主路与捷径，并返回激活后的五维体素特征。

        输入参数：
            - inputs: 浮点张量，(B,C_in,Z,Y,X)；通道数必须与构造值一致。

        返回值：
            - outputs: 浮点张量，(B,C_out,ceil(Z/s),ceil(Y/s),ceil(X/s))。
        """

        return self.activation(self.main(inputs) + self.shortcut(inputs))


class Res2NetBlock3d(nn.Module):
    """把扩展通道分组后按层级依赖卷积，再投影回输入宽度。

    构造参数：
        - channels: 正整数 C；输入、每个层级分组和输出的通道数。
        - scale: 整数 S，至少为 2；扩展后的分组数，每组宽度仍为 C。

    前向输入：
        - inputs: 浮点张量，(B,C,Z,Y,X)。

    前向输出：
        - outputs: 浮点张量，(B,C,Z,Y,X)；层级卷积混合结果与 inputs 的残差和经过 ReLU。
    """

    def __init__(self, channels: int, scale: int) -> None:
        """建立通道扩展、S-1 个层级卷积和通道混合投影。

        输入参数：
            - channels: 正整数 C；本方法显式拒绝非正值。
            - scale: 整数 S；API 要求至少为 2，本方法检查数值下界但不单独检查精确 int 类型。

        实例状态：
            - self.channels: 构造输入 channels；保存但前向不单独读取。
            - self.scale: 构造输入 scale；用于把扩展激活切成 S 组。
            - self.expand: Sequential；一乘一乘一卷积把 C 扩展为 C×S，再做 InstanceNorm。
            - self.hierarchical: ModuleList；含 S-1 个保持 C 通道与空间大小的三乘三乘三卷积。
            - self.pre_mix_norm: InstanceNorm3d；作用于重新连接的 C×S 通道。
            - self.mix: Sequential；一乘一乘一卷积把 C×S 压回 C，并做 InstanceNorm。
            - self.activation: ReLU；用于扩展、混合与最终残差激活。
        """

        super().__init__()
        if channels <= 0 or scale < 2:
            raise ValueError("Res2NetBlock3d 要求 channels>0 且 scale>=2。")
        self.channels = channels
        self.scale = scale
        self.expand = nn.Sequential(
            nn.Conv3d(channels, channels * scale, kernel_size=1, bias=False),
            nn.InstanceNorm3d(channels * scale, affine=True),
        )
        self.hierarchical = nn.ModuleList(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
            for _ in range(scale - 1)
        )
        self.pre_mix_norm = nn.InstanceNorm3d(channels * scale, affine=True)
        self.mix = nn.Sequential(
            nn.Conv3d(channels * scale, channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
        )
        self.activation = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """按固定层级依赖处理 S 个通道组。

        输入参数：
            - inputs: 浮点张量，(B,C,Z,Y,X)；C 与构造时 channels 一致。

        返回值：
            - outputs: 浮点张量，(B,C,Z,Y,X)；层级分组混合与输入残差相加后的结果。

        第 0 组直接保留扩展激活；第 1 组直接卷积；从第 2 组开始，当前原始分组先与上一层级卷积输出相加再卷积。
        """

        # 长度 S 的 tuple，每项浮点 (B,C,Z,Y,X)；来自 ReLU 后 C×S 扩展通道的等宽切分。
        chunks = self.activation(self.expand(inputs)).chunk(self.scale, dim=1)
        # list[Tensor]，层级输出；首项是不经过 hierarchical 卷积的第 0 组。
        outputs = [chunks[0]]
        for index, chunk in enumerate(chunks[1:]):
            # 浮点，(B,C,Z,Y,X)；第 1 组直接作为卷积输入，后续组加入上一层级输出。
            source = chunk if index == 0 else chunk + outputs[-1]
            outputs.append(self.hierarchical[index](source))
        # 浮点，(B,C×S,Z,Y,X)，按分组顺序连接并归一化的全部层级输出。
        concatenated = self.pre_mix_norm(torch.cat(outputs, dim=1))
        # 浮点，(B,C,Z,Y,X)，激活层级连接后由一乘一乘一卷积压回输入宽度。
        mixed = self.mix(self.activation(concatenated))
        return self.activation(mixed + inputs)


def _res2_stack(channels: int, scale: int) -> nn.Sequential:
    """构造四个串联且保持形状的 Res2NetBlock3d。

    输入参数：
        - channels: 整数 C；四个块共同的输入与输出通道数。
        - scale: 整数 S；四个块共同的层级分组数。

    返回值：
        - stack: Sequential；固定含四个 Res2NetBlock3d，输入与输出均为浮点 (B,C,Z,Y,X)。

    本函数自身不验证参数；每个 Res2NetBlock3d 构造时执行其已有门禁。
    """

    return nn.Sequential(*(Res2NetBlock3d(channels, scale) for _ in range(4)))


class ConvResidual3d(nn.Module):
    """用两层三乘三乘三卷积和按需通道投影融合同分辨率特征。

    构造参数：
        - input_channels: 整数 C_in；输入宽度。
        - output_channels: 整数 C_out；输出宽度。

    前向输入：
        - inputs: 浮点张量，(B,C_in,Z,Y,X)。

    前向输出：
        - outputs: 浮点张量，(B,C_out,Z,Y,X)；卷积主路与恒等或投影捷径之和经过 ReLU。
    """

    def __init__(self, input_channels: int, output_channels: int) -> None:
        """建立保持空间大小的卷积主路与残差捷径。

        输入参数：
            - input_channels: 整数 C_in；首层卷积和可选捷径投影的输入宽度。
            - output_channels: 整数 C_out；两层主路卷积和输出的宽度。

        实例状态：
            - self.main: Sequential；两层三乘三乘三卷积，各自后接 InstanceNorm，第一层后含 ReLU。
            - self.shortcut: Identity 或一乘一乘一 Conv3d；仅通道变化时执行投影。
            - self.activation: ReLU；作用于主路与捷径之和。

        本构造器不显式检查正通道数，依赖 Conv3d 拒绝不能构造的宽度。
        """

        super().__init__()
        self.main = nn.Sequential(
            nn.Conv3d(input_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(output_channels, affine=True),
            nn.ReLU(),
            nn.Conv3d(output_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(output_channels, affine=True),
        )
        self.shortcut = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv3d(input_channels, output_channels, kernel_size=1)
        )
        self.activation = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """返回卷积主路和捷径的激活残差和。

        输入参数：
            - inputs: 浮点张量，(B,C_in,Z,Y,X)；通道数与构造值一致。

        返回值：
            - outputs: 浮点张量，(B,C_out,Z,Y,X)。
        """

        return self.activation(self.main(inputs) + self.shortcut(inputs))


class AttentionGate3d(nn.Module):
    """用上采样粗特征为细尺度跳跃特征生成逐头空间门。

    构造参数：
        - skip_channels: 整数 C_s；细尺度跳跃特征通道数。
        - coarse_channels: 整数 C_c；粗尺度输入、value 投影和门控后通道数。
        - output_channels: 整数 C_out；融合残差块的输出通道数。

    前向输入：
        - coarse: 浮点张量，(B,C_c,Z_c,Y_c,X_c)；先三线性插值到 skip 的空间大小。
        - skip: 浮点张量，(B,C_s,Z,Y,X)；产生 key 和逐头 value 的细尺度特征。

    前向输出：
        - fused: 浮点张量，(B,C_out,Z,Y,X)；门控 skip value 与上采样 coarse 相加后经 ConvResidual3d 融合。

    门值由 64 通道 query/key 和经过 ReLU、Sigmoid 的一乘一乘一卷积产生，有限精度输出位于 [0,1]。这里每个 head 对应 C_c/K 个 value 通道，同一头内通道共享该头的空间门；不执行 query-key 点积或 softmax。
    """

    def __init__(
        self,
        skip_channels: int,
        coarse_channels: int,
        output_channels: int,
    ) -> None:
        """建立粗尺度 query、细尺度 key/value、逐头 Sigmoid 门和融合块。

        输入参数：
            - skip_channels: 整数 C_s；self.key 与 self.value 的输入宽度。
            - coarse_channels: 整数 C_c；self.query 输入宽度，且用于选择不超过 8 的最大整除头数。
            - output_channels: 整数 C_out；self.fuse 的输出宽度。

        实例状态：
            - self.head_count: 正整数 K；不超过 8 且整除 C_c。
            - self.value_channels_per_head: 正整数 C_c/K；value reshape 后每个门头对应的通道组宽度。
            - self.query: Sequential；把上采样 coarse 投影成 64 通道并做 InstanceNorm。
            - self.key: Sequential；把 skip 投影成 64 通道并做 InstanceNorm。
            - self.value: Sequential；把 skip 投影成 C_c 通道并做 InstanceNorm。
            - self.gate: Sequential；对 query+key 执行 ReLU、一乘一乘一卷积和 Sigmoid，输出 K 个空间门。
            - self.fuse: ConvResidual3d；把 C_c 通道门控和投影到 C_out。

        本构造器不独立检查三种通道为正；正式 RaUnetBackbone 从已验证 planes 构造。
        """

        super().__init__()
        # int，query 与 key 共享的固定中间通道宽度。
        attention_channels = 64
        # int，不超过 8 且整除 coarse_channels 的最大门控头数。
        head_count = _largest_divisor_not_above(coarse_channels, 8)
        self.head_count = head_count
        self.value_channels_per_head = coarse_channels // head_count
        self.query = nn.Sequential(
            nn.Conv3d(coarse_channels, attention_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(attention_channels, affine=True),
        )
        self.key = nn.Sequential(
            nn.Conv3d(skip_channels, attention_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(attention_channels, affine=True),
        )
        self.value = nn.Sequential(
            nn.Conv3d(skip_channels, coarse_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(coarse_channels, affine=True),
        )
        self.gate = nn.Sequential(
            nn.ReLU(),
            nn.Conv3d(attention_channels, head_count, kernel_size=1),
            nn.Sigmoid(),
        )
        self.fuse = ConvResidual3d(coarse_channels, output_channels)

    def forward(self, coarse: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """上采样粗特征，门控细尺度 value，并返回融合结果。

        输入参数：
            - coarse: 浮点张量，(B,C_c,Z_c,Y_c,X_c)；通道数为构造时 coarse_channels。
            - skip: 浮点张量，(B,C_s,Z,Y,X)；通道数为构造时 skip_channels，提供目标空间大小。

        返回值：
            - fused: 浮点张量，(B,C_out,Z,Y,X)；门控 value 与上采样 coarse 相加、ReLU 后由 self.fuse 变换。

        本方法不显式核对批次、通道或空间为正；卷积、reshape 和张量相加负责暴露不相容输入。
        """

        # 浮点，(B,C_c,Z,Y,X)，coarse 以 align_corners=True 三线性插值到 skip 空间大小。
        upsampled = F.interpolate(
            coarse,
            size=skip.shape[-3:],
            mode="trilinear",
            align_corners=True,
        )
        # 浮点，(B,K,Z,Y,X)，由粗细 64 通道特征之和产生的逐头 Sigmoid 空间门。
        gate = self.gate(self.query(upsampled) + self.key(skip))
        # 浮点，(B,C_c,Z,Y,X)，skip 投影后的门控 value，通道将改排成每头组。
        value = self.value(skip)
        # 五个 int，value 的批次、通道占位与细尺度 Z/Y/X；通道值由构造时 C_c 确定。
        batch_size, _, depth, height, width = value.shape
        # 浮点，[B,C_c,Z,Y,X] -> [B,C_c/K,K,Z,Y,X]，让 gate 的 K 轴与相应通道组对齐。
        value = value.reshape(
            batch_size,
            self.value_channels_per_head,
            self.head_count,
            depth,
            height,
            width,
        )
        # 浮点，(B,C_c,Z,Y,X)，逐头空间门乘 value 后恢复上采样粗特征的通道布局。
        gated = (value * gate[:, None]).reshape_as(upsampled)
        return self.fuse(F.relu(gated + upsampled))


class GatedTransition(nn.Module):
    """对展平瓶颈位置执行逐 token 的门控通道前馈与残差归一化。

    构造参数：
        - channels: 整数 C；输入、输出和 LayerNorm 宽度，两个门分支扩展到 3C。

    前向输入：
        - inputs: 浮点张量，(B,L,C)；L 个瓶颈空间位置的 token。

    前向输出：
        - outputs: 浮点张量，(B,L,C)；SiLU 左分支与线性右分支逐元素相乘、投影回 C 后加入输入并做 LayerNorm。
    """

    def __init__(self, channels: int) -> None:
        """建立两个无偏置 3C 门分支、输出投影和 LayerNorm。

        输入参数：
            - channels: 整数 C；本构造器不显式检查正值，依赖 Linear 与 LayerNorm。

        实例状态：
            - self.left: 无偏置 Linear；C 到 3C，输出经过 SiLU。
            - self.right: 无偏置 Linear；C 到 3C，输出作为逐元素门值因子。
            - self.output: 无偏置 Linear；3C 到 C。
            - self.norm: LayerNorm；归一化输入与更新之和的最后一维。
        """

        super().__init__()
        # int，两个门分支相对输入通道的固定扩展倍数。
        expansion = 3
        self.left = nn.Linear(channels, channels * expansion, bias=False)
        self.right = nn.Linear(channels, channels * expansion, bias=False)
        self.output = nn.Linear(channels * expansion, channels, bias=False)
        self.norm = nn.LayerNorm(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """计算逐 token 门控更新并加入输入残差。

        输入参数：
            - inputs: 浮点张量，(B,L,C)；最后一维与构造 channels 一致。

        返回值：
            - outputs: 浮点张量，(B,L,C)；self.norm(inputs+update)。
        """

        # 浮点，(B,L,C)，两个 (B,L,3C) 分支逐元素门控后投影回 C 的更新。
        update = self.output(F.silu(self.left(inputs)) * self.right(inputs))
        return self.norm(inputs + update)


class RotarySelfAttention3d(nn.Module):
    """在最深三维网格的全部空间位置之间执行带三轴旋转位置编码的自注意力。

    构造参数：
        - channels: 正整数 C；输入、输出和残差通道数，必须能被 head_count 整除。
        - head_count: 正整数 K；注意力头数。

    前向输入：
        - inputs: 浮点张量，(B,C,Z,Y,X)；最深尺度规则网格特征。

    前向输出：
        - outputs: 浮点张量，(B,C,Z,Y,X)；全位置自注意力残差和 GatedTransition 后恢复的网格。

    每头查询、键和值宽度 C_h=6C/K，是输入每头基础宽度 C/K 的六倍；C_h 再均分给 Z/Y/X，每轴相邻通道成对旋转。注意力对 key 位置轴 L=Z×Y×X 求和，同时保留每个 query 位置。
    """

    def __init__(self, channels: int, head_count: int) -> None:
        """建立扩宽六倍的多头投影、输出残差归一化与门控前馈。

        输入参数：
            - channels: 正整数 C；必须能被 head_count 整除。
            - head_count: 正整数 K；每个头的 C_h 必须能被 6 整除。

        实例状态：
            - self.channels: 整数 C；前向恢复五维网格时使用。
            - self.head_count: 整数 K；改排查询、键和值时使用。
            - self.head_channels: 整数 C_h=6C/K；每头扩展宽度。
            - self.query: Linear；C 到 K×C_h，含偏置。
            - self.key: 无偏置 Linear；C 到 K×C_h。
            - self.value: 无偏置 Linear；C 到 K×C_h。
            - self.output: 无偏置 Linear；K×C_h 回到 C。
            - self.attention_norm: LayerNorm；归一化输入 token 与注意力输出之和。
            - self.transition: GatedTransition；对注意力残差后的每个位置独立更新通道。
        """

        super().__init__()
        if head_count <= 0 or channels <= 0 or channels % head_count != 0:
            raise ValueError("瓶颈通道数必须能被注意力头数整除。")
        # int，原 RAUNet 为每个头分配 6×C/K 个查询、键和值通道。
        head_channels = channels // head_count * 6
        if head_channels % 6 != 0:
            raise ValueError("三维旋转位置编码要求每个头的通道数能被 6 整除。")
        self.channels = channels
        self.head_count = head_count
        self.head_channels = head_channels
        # int，全部 K 个头连接后的查询、键和值投影宽度，等于 6C。
        projected_channels = head_count * head_channels
        self.query = nn.Linear(channels, projected_channels)
        self.key = nn.Linear(channels, projected_channels, bias=False)
        self.value = nn.Linear(channels, projected_channels, bias=False)
        self.output = nn.Linear(projected_channels, channels, bias=False)
        self.attention_norm = nn.LayerNorm(channels)
        self.transition = GatedTransition(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """展平网格、执行全位置自注意力，再恢复原五维形状。

        输入参数：
            - inputs: 浮点张量，(B,C,Z,Y,X)；通道数必须与构造 channels 一致，本方法不另做显式门禁。

        返回值：
            - outputs: 浮点张量，(B,C,Z,Y,X)；空间大小与输入一致，dtype 由线性层、注意力算子和自动混合精度上下文决定。

        scaled_dot_product_attention 沿 key 位置轴 L 聚合 value。这里没有 padding mask，因为规则瓶颈网格的全部 Z×Y×X 位置都有效。
        """

        # 五个 int，当前瓶颈网格的 B/C/Z/Y/X；L=Z×Y×X。
        batch_size, channels, depth, height, width = inputs.shape
        # 浮点，[B,C,Z,Y,X] -> [B,C,L] -> [B,L,C]，X 位置最先变化的空间 token 序列。
        tokens = inputs.flatten(2).transpose(1, 2)
        # 浮点，(B,K,L,C_h)，线性查询投影并改排头轴后的张量。
        query = self._split_heads(self.query(tokens))
        # 浮点，(B,K,L,C_h)，与 query 同位置轴的键。
        key = self._split_heads(self.key(tokens))
        # 浮点，(B,K,L,C_h)，与 query 同位置轴的值。
        value = self._split_heads(self.value(tokens))
        # 与 inputs 同 device/dtype 的 (L,3) 模型内部 Z/Y/X 位置，X 最先变化。
        positions = _grid_positions(depth, height, width, inputs.device, inputs.dtype)
        query = _apply_three_axis_rope(query, positions)
        key = _apply_three_axis_rope(key, positions)
        # 浮点，(B,K,L,C_h)，每个 query 位置沿全部 key 位置归一化并聚合 value。
        attended = F.scaled_dot_product_attention(query, key, value)
        # 浮点，[B,K,L,C_h] -> [B,L,K,C_h] -> [B,L,K×C_h]，连接注意力头。
        attended = attended.transpose(1, 2).reshape(batch_size, -1, self.head_count * self.head_channels)
        # 浮点，(B,L,C)，输入 token 与投影注意力输出的归一化残差和。
        tokens = self.attention_norm(tokens + self.output(attended))
        # 浮点，(B,L,C)，逐位置门控前馈后的 token。
        tokens = self.transition(tokens)
        return tokens.transpose(1, 2).reshape(batch_size, channels, depth, height, width)

    def _split_heads(self, values: torch.Tensor) -> torch.Tensor:
        """把投影通道拆成头轴，并把头轴移到位置轴之前。

        输入参数：
            - values: 浮点张量，(B,L,K×C_h)；query、key 或 value 的线性投影结果。

        返回值：
            - headed: 浮点张量，[B,L,K×C_h] -> [B,L,K,C_h] -> [B,K,L,C_h]。

        本方法不单独核验最后一维；reshape 在元素数量与构造宽度不相容时失败。
        """

        # 两个 int 与一个未使用通道占位，分别为批次 B、位置数 L 和投影宽度 K×C_h。
        batch_size, token_count, _ = values.shape
        return values.reshape(
            batch_size,
            token_count,
            self.head_count,
            self.head_channels,
        ).transpose(1, 2)


# ================================================================================================


class RaUnetBackbone(nn.Module):
    """复现 Find_0 的多尺度三维 RAUNet，并把最高分辨率解码状态用于回收。

    构造参数：
        - input_channels: 整数 C_in；候选体素输入通道数，CandidateEncoder 固定传 105。
        - feature_channels: 整数 C_v；最终最高分辨率特征通道数。
        - planes: 九项可转换为整数的序列；依次为 enc0、enc1、enc2、enc3、bottleneck、dec3、dec2、dec1、dec0。
        - recycle_iterations: 正整数 R；同一 voxel_input 的前向轮数，前置轮显式 no_grad，第 R-1 轮不受该内部上下文包裹，是否建图由外层梯度模式决定。
        - gradient_checkpointing: 布尔语义值；训练且梯度启用时，决定高开销模块是否使用非重入激活重计算。
        - bottleneck_attention_layers: 整数 T；最深网格 RotarySelfAttention3d 的串联个数，正式 MatcherModelConfig 要求为正，本类直接构造时未显式拒绝非正整数。

    前向输入：
        - voxel_input: 浮点张量，(B,C_in,48,48,48)；空间轴依次为 Z/Y/X。

    前向输出：
        - voxel_features: 浮点张量，(B,C_v,48,48,48)；三种卷积核多尺度分支连接后投影的最高分辨率特征。

    固定 48 立方体调用的编码空间大小依次为 48、24、12、6、3；解码空间大小依次恢复为 6、12、24、48。planes 必须满足 enc3=bottleneck=dec3，使最深上采样可与 level3 相加；还必须满足 dec0=enc0，使 decoded0 可直接送入下一轮 recycle_norm。
    """

    def __init__(
        self,
        input_channels: int,
        feature_channels: int,
        planes: Sequence[int],
        recycle_iterations: int,
        gradient_checkpointing: bool,
        bottleneck_attention_layers: int,
    ) -> None:
        """建立四级编码器、瓶颈注意力、门控解码器和多尺度输出分支。

        输入参数：
            - input_channels: 整数 C_in；input_projection 的输入宽度，本构造器不单独检查正值。
            - feature_channels: 整数 C_v；feature_projection 的输出宽度，本构造器不单独检查正值。
            - planes: 序列；每项先执行 int(value)，随后要求共九项且转换结果为正。
            - recycle_iterations: 数值；显式要求大于零，正式配置保证是精确 int，直接传入正非整数会在 forward 的 range 调用处失败。
            - gradient_checkpointing: 布尔语义值；保存后由 _heavy 与 training、grad_enabled 共同判断。
            - bottleneck_attention_layers: 整数 T；用于构造 T 个瓶颈层，直接传入零或负整数会建立空 Sequential，正式配置层另行要求正值。

        实例状态：
            - self.input_channels: 构造输入 C_in；forward 检查 voxel_input 通道轴。
            - self.feature_channels: 构造输入 C_v；保存但 forward 不单独读取。
            - self.planes: 九项正整数元组；已完成 int 转换和通道关系检查。
            - self.recycle_iterations: 构造输入 R；forward 的循环次数。
            - self.gradient_checkpointing: 布尔语义值；_heavy 的激活重计算开关。
            - self.input_projection: 三乘三乘三 Conv3d；C_in 到 enc0，保持 48 立方体。
            - self.input_norm: 无仿射 InstanceNorm3d；归一化当前 voxel_input 投影。
            - self.recycle_norm: 有仿射 InstanceNorm3d；归一化上一轮 decoded0 或首轮零状态。
            - self.input_activation: ELU；激活当前输入投影与回收状态的归一化和。
            - self.stem: Sequential；enc0 到 4enc0，保持最高分辨率。
            - self.down1: ResidualBottleneck3d；4enc0 到 enc1，空间 48 到 24。
            - self.down2: ResidualBottleneck3d；enc1 到 enc2，空间 24 到 12。
            - self.down3: ResidualBottleneck3d；enc2 到 enc3，空间 12 到 6。
            - self.down4: ResidualBottleneck3d；enc3 到 bottleneck，空间 6 到 3。
            - self.bottleneck_attention: Sequential；T 个保持 bottleneck 通道和 3 立方体的 RotarySelfAttention3d。
            - self.decode3: Sequential；四个保持 dec3 通道和 6 立方体的 Res2NetBlock3d。
            - self.gate2: AttentionGate3d；把 dec3 粗特征与 enc2 跳跃特征融合成 dec2 通道和 12 立方体。
            - self.decode2: Sequential；四个保持 dec2 通道和 12 立方体的 Res2NetBlock3d。
            - self.gate1: AttentionGate3d；把 dec2 粗特征与 enc1 跳跃特征融合成 dec1 通道和 24 立方体。
            - self.decode1: Sequential；四个保持 dec1 通道和 24 立方体的 Res2NetBlock3d。
            - self.gate0: AttentionGate3d；把 dec1 粗特征与 4enc0 跳跃特征融合成 dec0 通道和 48 立方体。
            - self.decode0: Sequential；四个保持 dec0 通道和 48 立方体的 Res2NetBlock3d。
            - self.multiscale3: 三乘三乘三 Conv3d；decoded0 到 branch_channels，保持空间大小。
            - self.multiscale5: 五乘五乘五 Conv3d；decoded0 到 branch_channels，保持空间大小。
            - self.multiscale7: 七乘七乘七 Conv3d；decoded0 到 branch_channels，保持空间大小。
            - self.feature_projection: 三乘三乘三 Conv3d；连接的 3×branch_channels 映射到 C_v。
        """

        super().__init__()
        # 九项 int 元组；直接构造允许可由 int 转换的序列元素，MatcherModelConfig 路径则已保证精确正整数。
        plane_values = tuple(int(value) for value in planes)
        if len(plane_values) != 9 or any(value <= 0 for value in plane_values):
            raise ValueError("planes 必须由九个正整数构成。")
        if recycle_iterations <= 0:
            raise ValueError("recycle_iterations 必须为正整数。")
        # 九个正整数，按编码、瓶颈和解码的实际调用顺序解包。
        enc0, enc1, enc2, enc3, bottleneck, dec3, dec2, dec1, dec0 = plane_values
        if not (enc3 == bottleneck == dec3):
            raise ValueError("当前 RAUNet 契约要求 enc3、bottleneck 和 dec3 通道相同。")
        if dec0 != enc0:
            raise ValueError("原 RAUNet 直接回收 decoded0，因此 dec0 必须等于 enc0。")
        self.input_channels = input_channels
        self.feature_channels = feature_channels
        self.planes = plane_values
        self.recycle_iterations = recycle_iterations
        self.gradient_checkpointing = gradient_checkpointing

        self.input_projection = nn.Conv3d(input_channels, enc0, kernel_size=3, padding=1, bias=False)
        self.input_norm = nn.InstanceNorm3d(enc0, affine=False)
        self.recycle_norm = nn.InstanceNorm3d(enc0, affine=True)
        self.input_activation = nn.ELU()
        self.stem = nn.Sequential(
            nn.Conv3d(enc0, enc0 * 4, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(enc0 * 4, affine=True),
            nn.ReLU(),
        )
        self.down1 = ResidualBottleneck3d(enc0 * 4, enc1, stride=2)
        self.down2 = ResidualBottleneck3d(enc1, enc2, stride=2)
        self.down3 = ResidualBottleneck3d(enc2, enc3, stride=2)
        self.down4 = ResidualBottleneck3d(enc3, bottleneck, stride=2)
        # 正整数 K，不超过 8 且整除 bottleneck；瓶颈全部注意力层共享该头数。
        head_count = _largest_divisor_not_above(bottleneck, 8)
        self.bottleneck_attention = nn.Sequential(
            *(RotarySelfAttention3d(bottleneck, head_count) for _ in range(bottleneck_attention_layers))
        )
        self.decode3 = _res2_stack(dec3, scale=3)
        self.gate2 = AttentionGate3d(enc2, dec3, dec2)
        self.decode2 = _res2_stack(dec2, scale=4)
        self.gate1 = AttentionGate3d(enc1, dec2, dec1)
        self.decode1 = _res2_stack(dec1, scale=4)
        self.gate0 = AttentionGate3d(enc0 * 4, dec1, dec0)
        self.decode0 = _res2_stack(dec0, scale=4)

        # 正整数，三个最高分辨率卷积分支各自的宽度；至少 32，否则取 dec0 的向下取整一半。
        branch_channels = max(32, dec0 // 2)
        self.multiscale3 = nn.Conv3d(dec0, branch_channels, kernel_size=3, padding=1)
        self.multiscale5 = nn.Conv3d(dec0, branch_channels, kernel_size=5, padding=2)
        self.multiscale7 = nn.Conv3d(dec0, branch_channels, kernel_size=7, padding=3)
        self.feature_projection = nn.Conv3d(branch_channels * 3, feature_channels, kernel_size=3, padding=1)

    def forward(self, voxel_input: torch.Tensor) -> torch.Tensor:
        """执行 R 轮回收，并只返回最后一轮最高分辨率特征。

        输入参数：
            - voxel_input: 浮点张量，(B,C_in,48,48,48)；通道数必须等于 self.input_channels，空间轴依次为 Z/Y/X。

        返回值：
            - voxel_features: 浮点张量，(B,C_v,48,48,48)；最后一轮 _forward_once 的 feature_projection 输出，dtype 由参数、设备与自动混合精度上下文共同决定。

        第 0 到 R-2 轮在 torch.no_grad 中运行，只保存已停止梯度的 decoded0；每轮后调用 torch.clear_autocast_cache 清除当前自动混合精度权重转换缓存。第 R-1 轮使用上一轮状态且不受本函数内部 no_grad 包裹；是否建立计算图服从外层梯度模式及参与运算张量的 requires_grad，外层 no_grad 或 inference_mode 下不建图。缓存清理是进程内 PyTorch 自动混合精度状态副作用，不写文件或修改输入张量。
        """

        if voxel_input.ndim != 5 or voxel_input.shape[1] != self.input_channels:
            raise ValueError(
                f"voxel_input 必须为 (B,{self.input_channels},Z,Y,X)，"
                f"实际为 {tuple(voxel_input.shape)}。"
            )
        if voxel_input.shape[-3:] != (48, 48, 48):
            raise ValueError(f"Matcher RAUNet 的空间形状必须为 (48,48,48)，实际为 {tuple(voxel_input.shape[-3:])}。")
        # 浮点，(B,enc0,48,48,48)，首轮回收状态全零，device 与 dtype 跟随 voxel_input。
        recycle = torch.zeros(
            (voxel_input.shape[0], self.planes[0], *voxel_input.shape[-3:]),
            device=voxel_input.device,
            dtype=voxel_input.dtype,
        )
        # None 或浮点 (B,C_v,48,48,48)；只在最后一轮赋值并作为正式返回。
        final_feature: torch.Tensor | None = None
        for iteration in range(self.recycle_iterations):
            # bool，仅最后一轮为真；只控制是否跳过本函数内部 no_grad，不能覆盖调用位置的外层梯度模式。
            compute_gradients = iteration == self.recycle_iterations - 1
            if compute_gradients:
                final_feature, _ = self._forward_once(voxel_input, recycle)
            else:
                with torch.no_grad():
                    _, decoded0 = self._forward_once(voxel_input, recycle)
                # 浮点，(B,enc0,48,48,48)，与前置轮计算图断开的最高分辨率 decoded0，供下一轮 recycle_norm 读取。
                recycle = decoded0.detach()
                # 清空前置无梯度轮留下的自动混合精度权重缓存，避免最后一轮复用无梯度参数转换并改变非重入激活重计算保存的张量序列。
                torch.clear_autocast_cache()
        if final_feature is None:
            raise AssertionError("至少应执行一次 RAUNet recycle。")
        return final_feature

    def _forward_once(
        self,
        voxel_input: torch.Tensor,
        recycle: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """执行一次完整编码、瓶颈注意力、门控解码和多尺度输出。

        输入参数：
            - voxel_input: 浮点张量，(B,C_in,48,48,48)；每轮都重新经过同一 input_projection。
            - recycle: 浮点张量，(B,enc0,48,48,48)；首轮为零，后续轮为上一轮已停止梯度的 decoded0。

        返回值：
            - voxel_features: 浮点张量，(B,C_v,48,48,48)；三个卷积核分支连接并投影后的特征。
            - decoded0: 浮点张量，(B,dec0,48,48,48)；最高分辨率解码状态，dec0=enc0，可直接作为下一轮 recycle。

        固定 48 立方体时，level0 至 level4 的空间边长依次为 48、24、12、6、3，decoded3 至 decoded0 依次为 6、12、24、48。
        """

        # 浮点，(B,enc0,48,48,48)，当前输入投影与回收状态分别归一化后相加并经 ELU。
        fused = self.input_activation(
            self.input_norm(self.input_projection(voxel_input)) + self.recycle_norm(recycle)
        )
        # 浮点，(B,4enc0,48,48,48)，最高分辨率编码跳跃特征。
        level0 = self.stem(fused)
        # 浮点，(B,enc1,24,24,24)，第一次步幅二下采样。
        level1 = self._heavy(self.down1, level0)
        # 浮点，(B,enc2,12,12,12)，第二次步幅二下采样。
        level2 = self._heavy(self.down2, level1)
        # 浮点，(B,enc3,6,6,6)，第三次步幅二下采样。
        level3 = self._heavy(self.down3, level2)
        # 浮点，(B,bottleneck,3,3,3)，最深编码网格。
        level4 = self._heavy(self.down4, level3)
        # 浮点，(B,bottleneck,3,3,3)，经过 T 层全位置三轴 RoPE 自注意力。
        deepest = self._heavy(self.bottleneck_attention, level4)
        # 浮点，(B,dec3,6,6,6)，deepest 上采样并与 level3 相加后经四个 Res2Net 块。
        decoded3 = self._heavy(self.decode3, _upsample_add(deepest, level3))
        # 浮点，(B,dec2,12,12,12)，decoded3 查询并门控 level2 后经四个 Res2Net 块。
        decoded2 = self._heavy(self.decode2, self._heavy(self.gate2, decoded3, level2))
        # 浮点，(B,dec1,24,24,24)，decoded2 查询并门控 level1 后经四个 Res2Net 块。
        decoded1 = self._heavy(self.decode1, self._heavy(self.gate1, decoded2, level1))
        # 浮点，(B,dec0,48,48,48)，decoded1 查询并门控 level0 后经四个 Res2Net 块。
        decoded0 = self._heavy(self.decode0, self._heavy(self.gate0, decoded1, level0))
        # 浮点，(B,3×branch_channels,48,48,48)，三、五、七卷积核分支按通道连接后经 ReLU。
        multiscale = F.relu(
            torch.cat(
                (
                    self.multiscale3(decoded0),
                    self.multiscale5(decoded0),
                    self.multiscale7(decoded0),
                ),
                dim=1,
            )
        )
        return self.feature_projection(multiscale), decoded0

    def _heavy(self, module: nn.Module, *inputs: torch.Tensor) -> torch.Tensor:
        """按运行状态直接调用模块或使用非重入激活重计算。

        输入参数：
            - module: nn.Module；down、瓶颈注意力、decode 或 attention gate 子模块。
            - inputs: 一个或多个浮点张量；按原顺序位置传给 module，具体形状由调用位置决定。

        返回值：
            - outputs: 浮点张量；与 module 直接前向的返回形状相同。

        只有 self.gradient_checkpointing、self.training 和 torch.is_grad_enabled() 同时为真才调用 checkpoint。当前使用 use_reentrant=False 且 preserve_rng_state=False；现有 RAUNet 高开销模块不含 Dropout，若以后加入随机算子，该选项不会为重计算恢复随机状态。
        """

        if not (self.gradient_checkpointing and self.training and torch.is_grad_enabled()):
            return module(*inputs)
        return checkpoint(module, *inputs, use_reentrant=False, preserve_rng_state=False)
