"""集中公开 Matcher 模型层的稳定构造入口与结构化输出类型。

外部训练、验证和测试代码应从本包导入下列名称，避免依赖子模块内的实现辅助类。

- MatcherModelConfig: 模型结构、融合开关、身份门初值和 RAUNet 参数的冻结配置。
- CandidateEncoder: 只编码候选体素与 A、P、V、PP token 的候选预测模块。
- CandidateEncoderOutput: CandidateEncoder 返回的五字段 TypedDict 契约。
- ClosedSetMatcher: 组合候选预测与 PDB 内联合身份打分的完整模型。
- FullMatcherOutput: ClosedSetMatcher 返回的候选字段与身份字段合并契约。
- IdentityScoreComponents: 身份语言基线、图增量和清单增量的诊断字段契约。

GraphMessageLayer、IdentityEncoder、PairScorer 等内部模块仍可由测试从其实现文件读取，但不在
本包 __all__ 中承诺为稳定公共入口。
"""

from .candidate_encoder import CandidateEncoder, CandidateEncoderOutput
from .config import MatcherModelConfig
from .ligand_matcher import ClosedSetMatcher, FullMatcherOutput, IdentityScoreComponents

__all__ = [
    "CandidateEncoder",
    "CandidateEncoderOutput",
    "ClosedSetMatcher",
    "FullMatcherOutput",
    "IdentityScoreComponents",
    "MatcherModelConfig",
]
