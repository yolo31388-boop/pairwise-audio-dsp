"""音频DSP效果链"""
import math


class LowPassFilter:
    """一阶低通滤波器，系数由双线性变换(BLT)求得，截止频率与采样率相关。"""

    def __init__(self, cutoff=1000, sample_rate=44100):
        self.cutoff = cutoff
        self.sample_rate = sample_rate
        # 双线性变换 + 预畸变：k = tan(pi * fc / fs)，alpha = k / (1 + k)
        k = math.tan(math.pi * cutoff / sample_rate)
        self.alpha = k / (1.0 + k)
        self.prev = 0.0

    def process(self, sample):
        result = self.prev + self.alpha * (sample - self.prev)
        self.prev = result
        return result


class DelayEffect:
    MAX_FEEDBACK = 0.99  # 反馈必须小于1，否则信号指数增长爆音

    def __init__(self, delay_ms=200, feedback=0.5, sample_rate=44100):
        self.delay_samples = max(1, int(delay_ms * sample_rate / 1000))
        self.feedback = max(-self.MAX_FEEDBACK, min(self.MAX_FEEDBACK, feedback))
        self.buffer = [0.0] * self.delay_samples
        self.idx = 0

    def process(self, sample):
        delayed = self.buffer[self.idx]
        self.buffer[self.idx] = sample + delayed * self.feedback
        self.idx = (self.idx + 1) % self.delay_samples
        return sample + delayed


class Reverb:
    """混响 = 早期反射(抽头延迟线) + 晚期混响(反馈延迟网络 FDN)。

    decay:   晚期混响反馈衰减系数 [0, 0.99]，越大尾巴越长
    density: FDN 延迟线数量，决定反射密度
    mix:     干湿比 [0, 1]
    """

    def __init__(self, sample_rate=44100, decay=0.6, density=4, mix=0.3):
        self.sample_rate = sample_rate
        self.decay = min(0.99, max(0.0, decay))
        self.mix = min(1.0, max(0.0, mix))

        # 早期反射：稀疏抽头延迟线（前 60ms 内的离散回声）
        self.early_taps = [(int(0.013 * sample_rate), 0.60),
                           (int(0.027 * sample_rate), 0.45),
                           (int(0.041 * sample_rate), 0.30),
                           (int(0.059 * sample_rate), 0.20)]
        self.early_buf = [0.0] * (self.early_taps[-1][0] + 1)
        self.early_idx = 0

        # 晚期混响：N 条互质长度的延迟线 + Householder 反馈矩阵
        n = max(2, int(density))
        lengths = []
        length = int(0.029 * sample_rate)
        for _ in range(n):
            length += 1
            while any(math.gcd(length, other) != 1 for other in lengths):
                length += 1
            lengths.append(length)
        self.fdn_lines = [[0.0] * length for length in lengths]
        self.fdn_idx = [0] * n
        # 延迟线越长，单次循环的反馈增益越小，保证各线衰减时间一致
        self.fdn_gains = [self.decay ** (length / lengths[0]) for length in lengths]

    def process(self, sample):
        # 早期反射
        early = 0.0
        for tap, gain in self.early_taps:
            early += gain * self.early_buf[(self.early_idx - tap) % len(self.early_buf)]
        self.early_buf[self.early_idx] = sample
        self.early_idx = (self.early_idx + 1) % len(self.early_buf)

        # 晚期混响：FDN
        n = len(self.fdn_lines)
        outputs = [self.fdn_lines[i][self.fdn_idx[i]] for i in range(n)]
        mean = sum(outputs) / n
        fdn_input = sample + early * 0.5
        for i in range(n):
            # Householder 反馈：2*mean - x，保持能量在延迟线间均匀扩散
            feedback = (2.0 * mean - outputs[i]) * self.fdn_gains[i]
            self.fdn_lines[i][self.fdn_idx[i]] = fdn_input + feedback
            self.fdn_idx[i] = (self.fdn_idx[i] + 1) % len(self.fdn_lines[i])
        late = mean

        wet = early * 0.5 + late
        return (1.0 - self.mix) * sample + self.mix * wet


class Compressor:
    """压缩器，attack/release 用时间常数 exp(-1/(t*fs))，与采样率无关。"""

    def __init__(self, threshold=0.5, ratio=4, attack=0.01, release=0.1, sample_rate=44100):
        self.threshold = threshold
        self.ratio = ratio
        self.attack_coeff = math.exp(-1.0 / (attack * sample_rate))
        self.release_coeff = math.exp(-1.0 / (release * sample_rate))
        self.gain = 1.0

    def process(self, sample):
        level = abs(sample)
        if level > self.threshold:
            target_gain = self.threshold + (level - self.threshold) / self.ratio
            self.gain += (target_gain - self.gain) * (1.0 - self.attack_coeff)
        else:
            self.gain += (1.0 - self.gain) * (1.0 - self.release_coeff)
        return sample * self.gain


class _Biquad:
    """RBJ cookbook 二阶节，Q = 1/sqrt(2) 即 Butterworth。"""

    def __init__(self, kind, cutoff, sample_rate):
        w = 2.0 * math.pi * cutoff / sample_rate
        cos_w = math.cos(w)
        alpha = math.sin(w) / math.sqrt(2.0)  # Q = 1/sqrt(2)
        a0 = 1.0 + alpha
        if kind == "lp":
            b0, b1, b2 = (1.0 - cos_w) / 2.0, 1.0 - cos_w, (1.0 - cos_w) / 2.0
        else:
            b0, b1, b2 = (1.0 + cos_w) / 2.0, -(1.0 + cos_w), (1.0 + cos_w) / 2.0
        self.b0, self.b1, self.b2 = b0 / a0, b1 / a0, b2 / a0
        self.a1, self.a2 = -2.0 * cos_w / a0, (1.0 - alpha) / a0
        self.x1 = self.x2 = self.y1 = self.y2 = 0.0

    def process(self, sample):
        y = self.b0 * sample + self.b1 * self.x1 + self.b2 * self.x2 \
            - self.a1 * self.y1 - self.a2 * self.y2
        self.x2, self.x1 = self.x1, sample
        self.y2, self.y1 = self.y1, y
        return y


class LinkwitzRileyEQ:
    """两段 EQ：LR4(四阶 Linkwitz-Riley) 交叉分频。

    低/高通道各由两级 Butterworth 二阶节级联，分频点处各 -6dB，
    两段幅度之和恒为 0dB，提升低频时中高频不被连带抬起。
    """

    def __init__(self, crossover=1000, sample_rate=44100, low_gain=1.0, high_gain=1.0):
        self.low_gain = low_gain
        self.high_gain = high_gain
        self.lp = [_Biquad("lp", crossover, sample_rate) for _ in range(2)]
        self.hp = [_Biquad("hp", crossover, sample_rate) for _ in range(2)]

    def process(self, sample):
        low = high = sample
        for stage in self.lp:
            low = stage.process(low)
        for stage in self.hp:
            high = stage.process(high)
        return low * self.low_gain + high * self.high_gain


class AudioGraph:
    def __init__(self):
        self.nodes = {}
        self.connections = []

    def add_node(self, name, node):
        self.nodes[name] = node

    def connect(self, src, dst):
        self.connections.append((src, dst))

    def _topological_order(self):
        successors = {name: [] for name in self.nodes}
        for src, dst in self.connections:
            successors[src].append(dst)
        order = []
        state = {}  # 1=访问中, 2=已完成

        def visit(name):
            node_state = state.get(name, 0)
            if node_state == 1:
                raise ValueError(f"音频图存在循环: {name}")
            if node_state == 2:
                return
            state[name] = 1
            for nxt in successors[name]:
                visit(nxt)
            state[name] = 2
            order.append(name)

        for name in self.nodes:
            visit(name)
        order.reverse()
        return order

    def process(self, input_sample):
        # 拓扑排序：源节点先执行，效果器后执行；有循环直接报错
        result = input_sample
        for name in self._topological_order():
            node = self.nodes[name]
            process = getattr(node, "process", node)
            result = process(result)
        return result
