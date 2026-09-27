"""音频DSP - 带6个bug"""
import math

class LowPassFilter:
    def __init__(self, cutoff=1000, sample_rate=44100):
        self.cutoff = cutoff
        self.sample_rate = sample_rate
        self.prev = 0.0

    def process(self, sample):
        # bug1: 截止频率不考虑采样率
        alpha = self.cutoff / 10000
        result = self.prev + alpha * (sample - self.prev)
        self.prev = result
        return result

class DelayEffect:
    def __init__(self, delay_ms=200, feedback=0.5, sample_rate=44100):
        self.delay_samples = int(delay_ms * sample_rate / 1000)
        self.feedback = feedback  # bug2: 无上限保护
        self.buffer = [0.0] * self.delay_samples
        self.idx = 0

    def process(self, sample):
        delayed = self.buffer[self.idx]
        self.buffer[self.idx] = sample + delayed * self.feedback
        self.idx = (self.idx + 1) % self.delay_samples
        return sample + delayed

class Reverb:
    def __init__(self):
        self.delay_line = [0.0] * 1000
        self.idx = 0

    def process(self, sample):
        # bug3: 单全通滤波器当混响
        delayed = self.delay_line[self.idx]
        self.delay_line[self.idx] = sample + delayed * 0.5
        self.idx = (self.idx + 1) % len(self.delay_line)
        return sample + delayed * 0.3

class Compressor:
    def __init__(self, threshold=0.5, ratio=4, attack=0.01, release=0.1, sample_rate=44100):
        self.threshold = threshold
        self.ratio = ratio
        self.attack_coeff = 0.1  # bug4: 固定系数不用时间常数
        self.release_coeff = 0.1
        self.gain = 1.0

    def process(self, sample):
        level = abs(sample)
        if level > self.threshold:
            target_gain = self.threshold + (level - self.threshold) / self.ratio
            self.gain += (target_gain - self.gain) * self.attack_coeff
        else:
            self.gain += (1.0 - self.gain) * self.release_coeff
        return sample * self.gain

class AudioGraph:
    def __init__(self):
        self.nodes = {}
        self.connections = []

    def add_node(self, name, node):
        self.nodes[name] = node

    def connect(self, src, dst):
        self.connections.append((src, dst))

    def process(self, input_sample):
        # bug5: 按注册顺序执行，不做拓扑排序
        result = input_sample
        for name, node in self.nodes.items():
            result = node.process(result)
        return result
