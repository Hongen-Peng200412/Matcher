# 大致的比较想法

注意： test_真实场景(Structure & Emap2lig)\参考材料 & 想法\Li 等 - 2026 - Direct detection and atomic modeling of ligands in cryo-EM maps using deep learning.pdf ， 这篇文章宣称可以实现类似于Match的功能，它就是我们的baseline。我们要使用 测试结果与追溯\experiment-centric\density_only-0 、 测试结果与追溯\experiment-centric\density_only-106 这两个"消融模型"跟他比，注意这两个消融模型本身只需要一个"位点"，不需要stage1推理出特定的blob。

测试可以分成两档： NxN的，以及 mxm的(分别对应于:  测试结果与追溯\测试结果总览\NxN真实测试.md 、 测试结果与追溯\测试结果总览\mxm真实测试.md )。


# NOTE:
- 评估的"身份"自始至终就是我们规定的SMILES(去手性)。但，我们的模型直接输入SMILES就好了，而对于baseline，似乎必须还要提供"SMILES之外的额外信息"，似乎有必须由GT-pose导出的电荷键角等等。对此，我们允许baseline"作弊"，但这仍然有一系列的后续影响，比如：由于baseline的判别方式是"对每个位点都做一次对接"，可能导致NxN的算术复杂度，N为occurrence数目而不只是identity数目(对此，我接受根据N进行的样本过滤)。


# NxN
- 这表示: 已知一个PDB的所有配体身份(统一用相同的SMILES契约来代表；包括小分子、金属、糖、核酸状配体、多肽样配体, 不过可以排除极少量的"other")，且知有N个配体occurrence， 它们对应的实例几何中心都已知("几何中心"应该对应我们密度切块的方式)，测试任务就是将这些 occurrence 与这些提供的 identity 匹配。

这一部分的未过滤样本原则上可以是 test_0 里面的所有occurrence。

# mxm
- 这表示：已知一个PDB中所有类别为"small molecule"的配体身份，且知有n个小分子配体occurrence，它们对应的实例几何中心都已知，测试任务就是将这些小分子 occurrence 与这些提供的小分子 identity 匹配。

这一部分的未过滤样本原则上可以是 test_0 里面的所有小分子occurrence。