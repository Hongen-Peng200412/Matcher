# Matcher 运行支撑工具

`matcher/utils` 保存三类不定义科学口径、但会影响运行资源和产物可靠性的支撑能力：
`ByteBudgetLru` 复用当前 Python 实例已经读取的数组载荷，`json_artifacts.py` 把运行记录写成
严格 UTF-8 JSON，`exclusive_file_lock` 串行遵守同一路径锁协议的进程。类别、联合身份、
模型结构、损失和指标仍分别由 `matcher/data`、`matcher/model`、`matcher/training` 与
`matcher/evaluation` 定义。

本文中的“最近使用缓存”指按最近访问顺序淘汰对象的 LRU（least recently used）缓存；
“合作锁”指所有参与者都主动使用同一路径和相同锁协议时才成立的互斥，不表示锁文件能阻止
绕过协议的外部写者。

## 文件与公开入口

| 文件 | 公开入口 | 职责与不负责事项 |
| --- | --- | --- |
| `matcher/__init__.py` | `CLASS_NAMES`、`MANIFEST_SCHEMA_VERSION` | 只把数据 schema 中的六分类固定顺序和清单版本转发到包命名空间；不转发 Dataset、模型、训练或评估入口 |
| `matcher/utils/__init__.py` | 无 | 标记工具子包；导入该包不会预先导入三个工具模块 |
| `matcher/utils/cache.py` | `ByteBudgetLru`、`array_nbytes` | 维护单个 Python 实例内的对象引用和选定数组字节计数；不限制进程常驻内存，不提供线程锁、进程共享或磁盘缓存 |
| `matcher/utils/json_artifacts.py` | `write_json`、`append_jsonl` | 规范运行记录的 JSON 文本、同步和发布方式；不定义各记录的业务字段，不自动串行并发写者 |
| `matcher/utils/locking.py` | `exclusive_file_lock` | 对同一路径取得阻塞式平台文件锁；不锁定其它文件，不定义 Slurm 四锁资源状态机 |

包级 `CLASS_NAMES` 是长度为 6 的字符串元组，顺序固定为 `background`、`ion`、
`nucleotide_like`、`peptide_like`、`small_molecule`、`sugar`。`MANIFEST_SCHEMA_VERSION`
当前为整数 4，表示清单以 `object_key` 和 PreparedLigand 顶层 `smiles` 共同组成身份键。
二者都是 `matcher.data.schema` 中同名对象的直接绑定，不是另存的副本。

## 字节预算最近使用缓存

### 实例状态

`ByteBudgetLru[Key, Value]` 使用 `OrderedDict` 保存三项状态：

| 状态 | 类型与单位 | 含义 |
| --- | --- | --- |
| `maximum_bytes` | 非负 `int`，byte | 构造参数经 `int` 转换后的记录预算；正式调用传整数，本函数没有严格运行时类型门禁 |
| `current_bytes` | 非负 `int`，byte | 当前各项写入时记录的 `size` 之和；不含键、容器和未被 `array_nbytes` 统计的开销 |
| `_values` | `OrderedDict[Key, tuple[Value, int]]` | 从最久未使用到最近使用排列；每项保存原对象引用与写入时的字节数 |

缓存不会复制 `Value` 或把其中的 NumPy 数组设为只读。调用者在 `put` 后原地改变数组形状或
内容时，`current_bytes` 不会重新计算；`get` 返回的也是同一对象引用。该状态属于单个 Python
实例：在 DataLoader 场景中通常每个 worker 有自己的 Dataset 和缓存副本，`num_workers=0`
时也可以由主进程中的 Dataset 实例持有，不能把预算理解成全进程内存硬上限。

### `array_nbytes` 的计量范围

| 输入节点 | 本节点贡献与递归规则 |
| --- | --- |
| `numpy.ndarray` | 贡献 `int(value.nbytes)`；数组视图按视图自身字节数计算，不追踪共享底层存储 |
| `Mapping` | 只递归全部映射值；映射键、映射容器和哈希表开销均不计 |
| `list`、`tuple` | 按位置递归全部元素；序列容器自身开销不计 |
| dataclass 实例 | 按声明字段递归当前字段值；dataclass 类对象不计 |
| 其它对象 | 贡献 0；包括字符串、路径、NumPy 标量、Torch 张量、集合和普通 Python 对象 |

`object` dtype 的 NumPy 数组只贡献数组缓冲区报告的 `nbytes`，不会继续统计其中引用的
Python 对象。同一数组若在多个映射值、序列位置或 dataclass 字段中重复出现，会被重复累计；
函数不追踪对象身份，也不检测循环引用。由此得到的是缓存采用的载荷估算，不是进程 RSS
（resident set size，进程当前驻留物理内存）或唯一底层存储字节数。

### 查询、写入和淘汰

| 操作与条件 | 顺序和字节状态 | 返回值 |
| --- | --- | --- |
| `get(key)` 未命中 | 状态不变 | `None` |
| `get(key)` 命中 | 该键移动到最近使用端，字节数不变 | 缓存对象的同一引用 |
| `put(key, value)` 替换已有键 | 先删除旧项并减去旧 `size` | 继续判断新值是否接纳 |
| 总预算为 0 | 新值不保存；若是替换，旧值已经删除 | `None` |
| 新值 `size` 大于总预算 | 新值不保存；等于预算时仍可保存 | `None` |
| 新值被接纳 | 放到最近使用端，再从最久未使用端淘汰到预算内 | `None` |
| `keys()` | 返回动态键视图的迭代器，顺序为最久到最近使用 | `Iterator[Key]` |

当预算大于 0 时，`array_nbytes` 为 0 的值可以进入缓存，因此字节预算不限制这类对象的数量。
泛型 `Value` 也没有排除 `None`；若调用者主动缓存 `None`，公开 `get` 返回值无法区分该命中
与未命中。`keys()` 不复制列表，取得迭代器后再修改缓存可能触发 `OrderedDict` 的迭代期间
变更错误。

正式 Dataset 把 `cache_bytes_per_worker` 分给候选 artifact、联合身份、体素来源、PP PDB 和
LigandObject 图五类独立缓存。分账、未统计对象与 worker 序列化边界见
[`matcher/datasets/README.md`](../datasets/README.md)；体素、PP 和图缓存的磁盘层契约见
[`matcher/data/README.md`](../data/README.md)。

## 严格 JSON 与 JSONL

### 共同值转换

`write_json` 与 `append_jsonl` 都先调用 `_json_safe`。递归转换只包含以下规则：

| 内存值 | 传给标准 JSON 编码器的值 |
| --- | --- |
| 满足 `isinstance(value, float)` 的有限值 | 原浮点值 |
| 满足 `isinstance(value, float)` 的 `NaN`、`+Inf`、`-Inf` | `None`，落盘为 JSON `null` |
| `Mapping` | 新 `dict`；所有层级的键必须是 `str` 实例，值继续递归 |
| `list`、`tuple` | 新 `list`；元素按位置递归 |
| 其它对象 | 原样交给 `json.dumps` |

因此，“严格”表示编码器使用 `allow_nan=False`，不会把非标准 `NaN` 或 `Infinity` 字面量写入
成功产物；不表示任意 Python 对象都会自动获得 JSON 表示。`Path`、NumPy 数组、未满足
`isinstance(value, float)` 的 NumPy 标量、Torch 张量、dataclass 和集合等未支持值是否可编码，
仍由标准编码器决定，常见正式输入之外的这些对象会报错。转换不检测循环容器。

训练指标可能用 Python `NaN` 表示零分母，跨批 Python float 累计在理论上也可能得到无穷值；
这些满足上述 float 判断的值会落盘为 `null`。读取者必须结合具体 TypedDict 或记录字段判断
某个 `null` 是允许的非有限指标，还是业务层原本就允许的空值。

### `write_json`

`write_json(path, value)` 接受顶层字符串键映射，成功后在 `path` 保存一个 JSON 对象：

| 属性 | 固定行为 |
| --- | --- |
| 编码和换行 | UTF-8，`ensure_ascii=False`，末尾一个 LF |
| 排版 | 两空格缩进，所有层级映射键排序 |
| 临时路径 | 目标同目录下 `.{目标文件名}.tmp.{当前 PID}` |
| 数据同步 | 临时文件写完后 `flush` 并 `fsync` 文件句柄 |
| 发布 | `os.replace(temporary, path)`；已有目标会在成功调用中被替换 |
| 清理 | `finally` 对临时路径执行 `unlink(missing_ok=True)` |

父目录在序列化前创建，完整 JSON 文本在打开临时文件前生成。序列化失败不会由本调用新建
临时文件，但父目录可能已经出现。临时文件与目标位于同一目录；目录项替换是否原子以及跨
崩溃持久性仍服从操作系统和文件系统，函数只 `fsync` 文件，不 `fsync` 父目录。

PID 命名不区分同一进程内对同一目标的并发调用，本函数也没有锁。并发调用可能争用临时
路径、互相替换目标，或在一方 `finally` 清理时影响另一方。清理操作自身失败可能遮蔽此前
异常并留下临时文件。因此需要串行发布时，调用者应在更高层用同一路径合作锁包住完整的
检查、构造和发布状态转换。

### `append_jsonl`

`append_jsonl(path, value)` 把一个无缩进 JSON 对象和一个 LF 追加到文件尾，使用 UTF-8、
`ensure_ascii=False`、键排序和 `allow_nan=False`。打开方式为追加模式；已有行不会被读取、
验证或重写。每次调用在写入后执行 `flush` 与文件 `fsync`，但不 `fsync` 父目录。

该入口没有临时文件、原子替换、回滚或内部锁。写入或同步失败可能留下不完整末行，多进程或
多线程调用也不获得“整条记录绝不交错”的模块级保证。训练引擎当前由单个阶段进程按 epoch
追加 `history.jsonl`；若未来增加并发写者，必须先在调用层建立共同互斥。

整体 JSON 的正式调用包括实验 `run.json`、`resolved.json`、阶段/运行 `summary.json`、
W&B `tracking.json` 与 calibration artifact/claim。各文件的真实字段分别由
[`matcher/training/README.md`](../training/README.md) 和
[`matcher/evaluation/README.md`](../evaluation/README.md) 定义，通用写入器不检查这些业务 schema。

## 合作文件锁

### 生命周期和平台分支

`exclusive_file_lock(path)` 先递归创建父目录，再以 UTF-8 `a+` 模式打开锁文件。只有 `_lock`
成功后才进入 `with` 正文；正文退出时尝试 `_unlock`，最后关闭句柄。文件不会删除，因此
`path.exists()` 只能说明锁文件曾被建立，不能说明当前有人持锁。

| 平台分支 | 取得锁 | 释放锁 | 文件内容 |
| --- | --- | --- | --- |
| `os.name == "nt"` | 本句柄观察到空文件便写字符 `0` 并 `flush`，再从偏移 0 通过 `msvcrt.LK_LOCK` 请求 1 byte 锁；重试与最终 `OSError` 服从 Windows C runtime | 定位偏移 0，通过 `msvcrt.LK_UNLCK` 释放 1 byte | 并发首次初始化可能追加多个 `0`，但只有首字节参与锁；初始化不执行 `fsync` |
| 其它平台 | `fcntl.flock(handle, LOCK_EX)` 阻塞锁定打开文件 | `fcntl.flock(handle, LOCK_UN)` | 本模块不改写文件内容 |

模块没有调用者可配置的超时、非阻塞入口、取消令牌、所有者 PID 记录或重入承诺；Windows
`LK_LOCK` 的重试和最终失败、非 Windows `LOCK_EX` 的等待与中断均服从平台运行库。不同节点
上的互斥是否成立取决于共享文件系统对对应平台锁的实现。锁只约束使用完全相同 `path` 和兼容协议的合作调用者；
它不自动锁住缓存 NPY/NPZ、manifest JSON 或 calibration 输出，也不能排除绕过锁的外部写者。

当前调用者把状态检查和发布放在持锁区间内：

- `matcher.data.manifest` 用版本清单目录的 `.build.lock` 串行清单建立或扩充；
- `matcher.data.voxel_input` 用候选 NPY 对应的 `.lock` 保护缺失缓存二次检查与构建；
- `matcher.data.pp_input` 用 PDB 级 NPZ 对应的 `.lock` 保护缺失缓存二次检查与构建；
- `matcher.evaluation.calibration` 用数据指纹 claim 对应的 `.lock` 串行合作调用者的 claim 与输出状态转换。

这些文件锁与服务器 allocation 目录中的 `try_lock`、`pre_lock`、`after_lock`、`kill_lock` 是不同
层级的机制；本模块不会读取、创建或删除 Slurm 资源锁。

### 异常优先级

| 失败位置 | 后续行为与对外异常 |
| --- | --- |
| `_lock` | 不进入正文，也不调用 `_unlock`；仍在外层 `finally` 尝试关闭句柄 |
| 正文失败、`_unlock` 成功 | 正文异常继续传播 |
| 正文失败、`_unlock` 也失败 | 保留正文异常；异常对象支持 `add_note` 时附加解锁异常说明，否则解锁异常被抑制 |
| 正文成功、`_unlock` 失败 | 解锁异常传播 |
| `handle.close` 失败 | 位于最外层 `finally`，可能遮蔽此前正在传播的加锁、正文或解锁异常 |

这里的优先级只描述 Python 控制流，不承诺异常退出后任意网络文件系统的远端锁状态。缓存和
calibration 调用者还必须分别保留自己的二次存在性检查、输出路径门禁和失败恢复规则。

## 验证边界

相称行为门禁由现有只读测试覆盖：

- [`matcher/tests/test_data_pipeline.py`](../tests/test_data_pipeline.py) 检查 LRU 命中提升、按数组字节淘汰、加锁失败不误解锁，以及正文异常优先于解锁异常；
- [`matcher/tests/test_metrics_and_config.py`](../tests/test_metrics_and_config.py) 检查非有限 float 写成 `null`、JSONL 可读和非字符串映射键被拒绝；
- 训练、初始化、追踪与 calibration 测试通过真实调用间接覆盖 JSON 产物路径。

这些测试没有把进程 RSS、断电后的目录项持久性、同一进程并发写同一临时路径、跨节点网络
文件锁或绕过合作协议的外部写者纳入保证。测试目录只作为行为门禁，本轮不修改测试代码、
测试注释或测试 Docstring。
