"""音频DSP测试"""
import pytest, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.engine import LowPassFilter, DelayEffect, Compressor, AudioGraph

class TestLowPass:
    def test_cutoff_respects_sample_rate(self):
        f1 = LowPassFilter(cutoff=1000, sample_rate=44100)
        f2 = LowPassFilter(cutoff=1000, sample_rate=48000)
        # 不同采样率下系数应不同
        r1 = f1.process(1.0)
        r2 = f2.process(1.0)
        assert abs(r1 - r2) > 0.001, "截止频率未考虑采样率"

class TestDelay:
    def test_feedback_less_than_one(self):
        d = DelayEffect(feedback=1.5)
        assert d.feedback <= 1.0, "反馈系数无上限保护"

class TestCompressor:
    def test_attack_time_constant(self):
        c1 = Compressor(attack=0.01, sample_rate=44100)
        c2 = Compressor(attack=0.1, sample_rate=44100)
        assert c1.attack_coeff != c2.attack_coeff, "attack未用时间常数"

class TestGraph:
    def test_topological_sort(self):
        g = AudioGraph()
        g.add_node("effect", lambda x: x*2)
        g.add_node("source", lambda x: x+1)
        result = g.process(1.0)
        assert result == 4.0, f"执行顺序错误: {result}"
