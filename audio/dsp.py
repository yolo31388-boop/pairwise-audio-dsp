"""音频 DSP 效果链核心模块。

包含:
- LowPassFilter    一阶低通滤波器, 系数由双线性变换(BLT)求得
- DelayEffect      带上限保护的反馈延迟
- Reverb           早期反射延迟线 + 晚期混响反馈延迟网络(FDN)
- Compressor       attack/release 使用时间常数 exp(-1/(t*fs))
- LinkwitzRileyEQ  Linkwitz-Riley 交叉分频 EQ, 频段和恒为 0dB
- AudioGraph       拓扑排序执行, 含循环检测
"""

import math


class LowPassFilter:
    """一阶低通滤波器。

    系数由双线性变换 (Bilinear Transform) 求得, 同时对截止频率做
    预畸变 (pre-warping), 因此系数同时取决于截止频率和采样率,
    44.1kHz 与 48kHz 下的频响一致。
    """

    def __init__(self, cutoff=1000, sample_rate=44100):
        if sample_rate <= 0:
            raise ValueError("sample_rate 必须为正数")
        if cutoff <= 0:
            raise ValueError("cutoff 必须为正数")
        # 截止频率不能超过 Nyquist 频率, 否则 tan(pi*fc/fs) 无意义
        cutoff = min(cutoff, sample_rate / 2.0 * (1.0 - 1e-9))
        self.cutoff = cutoff
        self.sample_rate = sample_rate

        # BLT + 预畸变: s -> (2*fs)* (z-1)/(z+1), w = 2*fs*tan(pi*fc/fs)
        k = math.tan(math.pi * cutoff / sample_rate)
        # 一阶低通差分方程: y[n] = b0*x[n] + b1*x[n-1] - a1*y[n-1]
        self.b0 = k / (1.0 + k)
        self.b1 = self.b0
        self.a1 = (k - 1.0) / (1.0 + k)
        # 一阶 IIR 也可写成 y = prev + alpha*(x-prev) 的形式
        self.alpha = self.b0
        self._x1 = 0.0
        self._y1 = 0.0

    def process(self, sample):
        result = (self.b0 * sample + self.b1 * self._x1
                  - self.a1 * self._y1)
        self._x1 = sample
        self._y1 = result
        return result

    def reset(self):
        self._x1 = 0.0
        self._y1 = 0.0


class DelayEffect:
    """反馈延迟 (echo)。

    反馈系数必须 |feedback| < 1, 否则延迟回路中的信号指数增长
    导致爆音; 构造时对其做夹紧(clamp)上限保护。
    """

    MAX_FEEDBACK = 0.99

    def __init__(self, delay_ms=200, feedback=0.5, sample_rate=44100):
        if sample_rate <= 0:
            raise ValueError("sample_rate 必须为正数")
        self.sample_rate = sample_rate
        self.delay_samples = max(1, int(delay_ms * sample_rate / 1000.0))
        # 上限保护: 反馈严格小于 1
        self.feedback = max(-self.MAX_FEEDBACK,
                            min(self.MAX_FEEDBACK, feedback))
        self.buffer = [0.0] * self.delay_samples
        self.idx = 0

    def process(self, sample):
        delayed = self.buffer[self.idx]
        self.buffer[self.idx] = sample + delayed * self.feedback
        self.idx = (self.idx + 1) % self.delay_samples
        return sample + delayed

    def reset(self):
        self.buffer = [0.0] * self.delay_samples
        self.idx = 0


class EarlyReflections:
    """早期反射: 抽头延迟线 (tapped delay line)。

    前约 60ms 内的一簇稀疏、幅度递减的离散回声, 模拟墙面
    第一次反射, 抽头数量决定早期反射密度。
    """

    DEFAULT_TAPS = ((0.013, 0.60),
                    (0.027, 0.45),
                    (0.041, 0.30),
                    (0.059, 0.20))

    def __init__(self, sample_rate=44100, taps=None):
        self.sample_rate = sample_rate
        spec = self.DEFAULT_TAPS if taps is None else taps
        self.taps = [(max(1, int(t_sec * sample_rate)), gain)
                     for t_sec, gain in spec]
        size = max(tap for tap, _ in self.taps) + 1
        self.buffer = [0.0] * size
        self.idx = 0

    def process(self, sample):
        out = 0.0
        size = len(self.buffer)
        for tap, gain in self.taps:
            out += gain * self.buffer[(self.idx - tap) % size]
        self.buffer[self.idx] = sample
        self.idx = (self.idx + 1) % size
        return out

    def reset(self):
        self.buffer = [0.0] * len(self.buffer)
        self.idx = 0


class FeedbackDelayNetwork:
    """晚期混响: 反馈延迟网络 (FDN)。

    N 条互质长度的延迟线通过 Householder 反馈矩阵耦合,
    能量在延迟线间均匀扩散形成稠密、平滑的衰减尾音。
    density 配置延迟线数量, decay 配置每条线每轮的衰减。
    """

    MAX_DECAY = 0.99

    def __init__(self, sample_rate=44100, decay=0.6, density=4,
                 base_delay_ms=29.0):
        self.sample_rate = sample_rate
        self.decay = min(self.MAX_DECAY, max(0.0, decay))
        n = max(2, int(density))
        self.n = n

        lengths = []
        length = int(base_delay_ms * sample_rate / 1000.0)
        for _ in range(n):
            length += 1
            while any(math.gcd(length, other) != 1 for other in lengths):
                length += 1
            lengths.append(length)
        self.lengths = lengths
        self.lines = [[0.0] * length for length in lengths]
        self.indices = [0] * n
        # 线越长, 单位时间循环次数越少, 故按长度比例减小增益,
        # 使各条延迟线的混响时间(RT60)保持一致
        self.gains = [self.decay ** (length / lengths[0])
                      for length in lengths]

    def process(self, sample):
        n = self.n
        outputs = [self.lines[i][self.indices[i]] for i in range(n)]
        mean = sum(outputs) / n
        for i in range(n):
            # Householder 矩阵: y_i = (2*mean - x_i) * g_i
            feedback = (2.0 * mean - outputs[i]) * self.gains[i]
            self.lines[i][self.indices[i]] = sample + feedback
            self.indices[i] = (self.indices[i] + 1) % self.lengths[i]
        return mean

    def reset(self):
        for line in self.lines:
            for i in range(len(line)):
                line[i] = 0.0
        self.indices = [0] * self.n


class Reverb:
    """混响效果器: 独立的早期反射 + 晚期混响 FDN。

    参数:
        decay:   晚期混响衰减系数, 越大尾巴越长 (0 ~ 0.99)
        density: FDN 延迟线数量, 决定反射密度 (>=2)
        mix:     干湿混合比, 0 = 全干声, 1 = 全湿声
    """

    def __init__(self, sample_rate=44100, decay=0.6, density=4, mix=0.3):
        self.sample_rate = sample_rate
        self.decay = min(FeedbackDelayNetwork.MAX_DECAY,
                         max(0.0, decay))
        self.density = max(2, int(density))
        self.mix = min(1.0, max(0.0, mix))
        self.early = EarlyReflections(sample_rate)
        self.late = FeedbackDelayNetwork(sample_rate,
                                         decay=self.decay,
                                         density=self.density)

    @property
    def decay_gain(self):
        return self.decay

    def process(self, sample):
        early = self.early.process(sample)
        late = self.late.process(sample + early * 0.5)
        wet = early * 0.5 + late
        return (1.0 - self.mix) * sample + self.mix * wet

    def reset(self):
        self.early.reset()
        self.late.reset()


class Compressor:
    """动态压缩器。

    attack/release 的包络跟随系数使用时间常数:
        coeff = exp(-1 / (time_seconds * sample_rate))
    包络每帧向目标逼近 (1 - coeff), 因此达到目标的时间尺度
    由 time(秒)定义, 与采样率无关, 不同采样率下压缩曲线形状一致。
    """

    def __init__(self, threshold=0.5, ratio=4, attack=0.01, release=0.1,
                 sample_rate=44100):
        if sample_rate <= 0:
            raise ValueError("sample_rate 必须为正数")
        if attack <= 0 or release <= 0:
            raise ValueError("attack/release 必须为正数")
        self.threshold = threshold
        self.ratio = ratio
        self.sample_rate = sample_rate
        self.attack_time = attack
        self.release_time = release
        self.attack_coeff = math.exp(-1.0 / (attack * sample_rate))
        self.release_coeff = math.exp(-1.0 / (release * sample_rate))
        self.gain = 1.0

    def process(self, sample):
        level = abs(sample)
        if level > self.threshold:
            target_gain = self.threshold + (level - self.threshold) / self.ratio
            self.gain += ((target_gain - self.gain)
                          * (1.0 - self.attack_coeff))
        else:
            self.gain += ((1.0 - self.gain)
                          * (1.0 - self.release_coeff))
        return sample * self.gain

    def reset(self):
        self.gain = 1.0


class _Biquad:
    """RBJ cookbook 二阶节, Q = 1/sqrt(2) 时为 Butterworth。"""

    def __init__(self, kind, cutoff, sample_rate):
        if not 0 < cutoff < sample_rate / 2.0:
            raise ValueError("cutoff 必须位于 (0, Nyquist) 之间")
        w = 2.0 * math.pi * cutoff / sample_rate
        cos_w = math.cos(w)
        alpha = math.sin(w) / math.sqrt(2.0)
        a0 = 1.0 + alpha
        if kind == "lp":
            b0 = (1.0 - cos_w) / 2.0
            b1 = 1.0 - cos_w
            b2 = (1.0 - cos_w) / 2.0
        elif kind == "hp":
            b0 = (1.0 + cos_w) / 2.0
            b1 = -(1.0 + cos_w)
            b2 = (1.0 + cos_w) / 2.0
        else:
            raise ValueError("kind 必须是 'lp' 或 'hp'")
        self.b0 = b0 / a0
        self.b1 = b1 / a0
        self.b2 = b2 / a0
        self.a1 = -2.0 * cos_w / a0
        self.a2 = (1.0 - alpha) / a0
        self.x1 = self.x2 = 0.0
        self.y1 = self.y2 = 0.0

    def process(self, sample):
        y = (self.b0 * sample + self.b1 * self.x1 + self.b2 * self.x2
             - self.a1 * self.y1 - self.a2 * self.y2)
        self.x2, self.x1 = self.x1, sample
        self.y2, self.y1 = self.y1, y
        return y

    def reset(self):
        self.x1 = self.x2 = 0.0
        self.y1 = self.y2 = 0.0


class LinkwitzRileyEQ:
    """两段 Linkwitz-Riley 交叉分频 EQ (LR4)。

    低通/高通通道各由两级 Butterworth 二阶节级联(共四阶)。
    分频点处每段 -6dB 且相位相同, 两段幅度和恒为 0dB,
    因此提升某一段时不会连带抬起其它频段。
    支持线性增益(low_gain/high_gain)或分贝增益(low_db/high_db)。
    """

    def __init__(self, crossover=1000, sample_rate=44100,
                 low_gain=1.0, high_gain=1.0,
                 low_db=0.0, high_db=0.0, cutoff=None):
        if cutoff is not None:
            crossover = cutoff
        self.crossover = crossover
        self.sample_rate = sample_rate
        self.low_gain = low_gain * (10.0 ** (low_db / 20.0))
        self.high_gain = high_gain * (10.0 ** (high_db / 20.0))
        self.lp = [_Biquad("lp", crossover, sample_rate) for _ in range(2)]
        self.hp = [_Biquad("hp", crossover, sample_rate) for _ in range(2)]

    def process(self, sample):
        low = high = sample
        for stage in self.lp:
            low = stage.process(low)
        for stage in self.hp:
            high = stage.process(high)
        return low * self.low_gain + high * self.high_gain

    def reset(self):
        for stage in self.lp + self.hp:
            stage.reset()


class GraphCycleError(ValueError):
    """音频图中存在循环依赖。"""


class AudioGraph:
    """音频处理图。

    节点按连接关系(src -> dst)做拓扑排序后执行:
    源节点先执行, 效果器后执行, 保证每帧处理的都是当帧数据;
    无连接信息时按注册的逆序 DFS, 使先注册的源排在前面。
    图中存在循环时抛出 GraphCycleError。
    """

    def __init__(self):
        self.nodes = {}
        self.connections = []

    def add_node(self, name, node):
        self.nodes[name] = node

    def connect(self, src, dst):
        self.connections.append((src, dst))

    def topological_sort(self):
        successors = {name: [] for name in self.nodes}
        for src, dst in self.connections:
            if src in successors and dst in successors:
                successors[src].append(dst)
        order = []
        state = {}  # 0=未访问, 1=访问中, 2=已完成

        def visit(name, path):
            current = state.get(name, 0)
            if current == 1:
                cycle = path[path.index(name):] + [name]
                raise GraphCycleError(
                    "音频图存在循环: " + " -> ".join(cycle))
            if current == 2:
                return
            state[name] = 1
            for nxt in successors[name]:
                visit(nxt, path + [name])
            state[name] = 2
            order.append(name)

        # 按注册顺序 DFS: 先注册的节点(通常为源)排在前面;
        # 有连接(src->dst)时连接关系决定真正的先后
        for name in self.nodes:
            visit(name, [])
        order.reverse()
        return order

    def process(self, input_sample):
        result = input_sample
        for name in self.topological_sort():
            node = self.nodes[name]
            process = getattr(node, "process", node)
            result = process(result)
        return result


# 兼容常见命名
Equalizer = LinkwitzRileyEQ
EQ = LinkwitzRileyEQ
LREq = LinkwitzRileyEQ
LinkwitzRileyCrossover = LinkwitzRileyEQ
FDNReverb = Reverb
__all__ = [
    "LowPassFilter", "DelayEffect", "Reverb", "FDNReverb",
    "EarlyReflections", "FeedbackDelayNetwork", "Compressor",
    "LinkwitzRileyEQ", "Equalizer", "EQ", "LREq", "LinkwitzRileyCrossover",
    "AudioGraph", "GraphCycleError",
]
