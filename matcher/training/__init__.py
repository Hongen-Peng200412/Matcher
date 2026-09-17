"""Matcher 正式实验配置、损失、checkpoint 与单卡训练编排包.

本包没有稳定的顶层重导出; 调用方使用 matcher.training.config、losses、
checkpointing、engine 和 experiment 的具名入口. 训练主链只构造 train 与
validation; 一次性 release、launch 与文件摘要取证不属于本包.
"""
