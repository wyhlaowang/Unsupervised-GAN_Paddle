import paddle.fluid as fluid
import paddle.fluid.dygraph as dygraph
import paddle.fluid.layers as layers
# from paddle.fluid.layers import relu, tanh, resize_nearest, pad2d, unsqueeze, adaptive_max_pool2d, reshape, concat, reduce_sum, create_parameter, reduce_mean, sqrt
from paddle.fluid.dygraph import Conv2D, InstanceNorm, Linear, Sequential



def var(input, axis=None, keepdim=False, unbiased=True, out=None, name=None):
    dtype = input.dtype
    if dtype not in ["float32", "float64"]:
        raise ValueError("Layer tensor.var() only supports floating-point "
                         "dtypes, but received {}.".format(dtype))
    rank = len(input.shape)
    axes = axis if axis != None and axis != [] else range(rank)
    axes = [e if e >= 0 else e + rank for e in axes]
    inp_shape = input.shape if fluid.in_dygraph_mode() else layers.shape(input)
    mean = layers.reduce_mean(input, dim=axis, keep_dim=True, name=name)
    tmp = layers.reduce_mean(
        (input - mean)**2, dim=axis, keep_dim=keepdim, name=name)

    if unbiased:
        n = 1
        for i in axes:
            n *= inp_shape[i]
        if not fluid.in_dygraph_mode():
            n = layers.cast(n, dtype)
            zero_const = layers.fill_constant(shape=[1], dtype=dtype, value=0.0)
            factor = layers.where(n > 1.0, n / (n - 1.0), zero_const)
        else:
            factor = n / (n - 1.0) if n > 1.0 else 0.0
        tmp *= factor
    if out:
        layers.assign(input=tmp, output=out)
        return out
    else:
        return tmp

class ReLU(dygraph.Layer):
    def __init__(self, inplace=False):
        super(ReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        if self.inplace:
            input.set_value(layers.relu(input))
            return input
        else:
            y = layers.relu(input)
            return y


class LeakyReLU(dygraph.Layer):
    def __init__(self, alpha=0.02, inplace=False):
        super(LeakyReLU, self).__init__()
        self.inplace = inplace
        self.alpha = alpha

    def forward(self, input):
        if self.inplace:
            input.set_value(layers.leaky_relu(input, self.alpha))
            return input
        else:
            y = layers.leaky_relu(input, self.alpha)
            return y


class Tanh(dygraph.Layer):
    def __init__(self, inplace=False):
        super(Tanh, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        if self.inplace:
            input.set_value(layers.tanh(input))
            return input
        else:
            y = layers.tanh(input)
            return y


class Upsample(dygraph.Layer):
    def __init__(self, scale_factor):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, input):
        y = layers.resize_nearest(input, scale=self.scale_factor)
        return y

class ReflectionPad2d(dygraph.Layer):
    def __init__(self, padding):
        super(ReflectionPad2d, self).__init__()
        self.padding = padding

    def forward(self, input):
        y = layers.pad2d(input, paddings=self.padding, mode='reflect')
        return y

class Spectralnorm(dygraph.Layer):
    def __init__(self,
                 layer,
                 dim=0,
                 power_iters=1,
                 eps=1e-12,
                 dtype='float32'):
        super(Spectralnorm, self).__init__()
        self.spectral_norm = dygraph.SpectralNorm(layer.weight.shape, dim, power_iters, eps, dtype)
        self.dim = dim
        self.power_iters = power_iters
        self.eps = eps
        self.layer = layer
        weight = layer._parameters['weight']
        del layer._parameters['weight']
        self.weight_orig = self.create_parameter(weight.shape, dtype=weight.dtype)
        self.weight_orig.set_value(weight)

    def forward(self, x):
        weight = self.spectral_norm(self.weight_orig)
        self.layer.weight = weight
        out = self.layer(x)
        return out

class RhoClipper(object):

    def __init__(self, min, max):
        self.clip_min = min
        self.clip_max = max
        assert min < max

    def __call__(self, module):

        if hasattr(module, 'rho'):
            w = module.rho.data
            w = w.clamp(self.clip_min, self.clip_max)
            module.rho.data = w

def bce_Loss(input, target):
    loss = layers.sigmoid_cross_entropy_with_logits(
    x=input,
    label=target,
    ignore_index=-1,
    normalize=True)
    loss = layers.reduce_sum(loss)
    return loss

class ResnetBlock(dygraph.Layer):
    def __init__(self, dim, use_bias):
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block += [ReflectionPad2d([1,1,1,1]),
                       Conv2D(dim, dim, filter_size=3, stride=1, padding=0, bias_attr=use_bias),
                       InstanceNorm(dim),
                       ReLU(True)]

        #TODO: 这里加一个不带ReLU的有何意义
        conv_block += [ReflectionPad2d([1,1,1,1]),
                       Conv2D(dim, dim, filter_size=3, stride=1, padding=0, bias_attr=use_bias),
                       InstanceNorm(dim)]

        self.conv_block = Sequential(*conv_block)

    def forward(self, x):
        #TODO: 这里相加代表什么
        out = x + self.conv_block(x)
        return out

class adaILN(dygraph.Layer):
    def __init__(self, num_features, eps=1e-5):
        super(adaILN, self).__init__()
        self.eps = eps
        self.rho = layers.create_parameter(shape=[1, num_features, 1, 1], dtype='float32', default_initializer=fluid.initializer.Constant(0.9))

    def forward(self, input, gamma, beta):
        in_mean, in_var = layers.reduce_mean(input, dim=[2, 3], keep_dim=True), var(input, axis=[2, 3], keepdim=True)
        out_in = (input - in_mean) / layers.sqrt(in_var + self.eps)
        ln_mean, ln_var = layers.reduce_mean(input, dim=[1, 2, 3], keep_dim=True), var(input, axis=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / layers.sqrt(ln_var + self.eps)
        out = layers.expand(self.rho, [input.shape[0], -1, -1, -1]) * out_in + (1-layers.expand[input.shape[0], -1, -1, -1]) * out_ln
        out = out * layers.unsqueeze(layers.unsqueeze(gamma, 2), 3) + layers.unsqueeze(layers.unsqueeze(beta, 2), 3)

        return out

class ILN(dygraph.Layer):
    def __init__(self, num_features, eps=1e-5):
        super(ILN, self).__init__()
        self.eps = eps
        self.rho = layers.create_parameter(shape=[1, num_features, 1, 1], dtype='float32', default_initializer=fluid.initializer.Constant(0.0))
        self.gamma = layers.create_parameter(shape=[1, num_features, 1, 1], dtype='float32', default_initializer=fluid.initializer.Constant(1.0))
        self.beta = layers.create_parameter(shape=[1, num_features, 1, 1], dtype='float32', default_initializer=fluid.initializer.Constant(0.0))

    def forward(self, input):
        in_mean, in_var = layers.reduce_mean(input, dim=[2, 3], keep_dim=True), var(input, axis=[2, 3], keepdim=True)
        out_in = (input - in_mean) / layers.sqrt(in_var + self.eps)
        ln_mean, ln_var = layers.reduce_mean(input, dim=[1, 2, 3], keep_dim=True), var(input, axis=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / layers.sqrt(ln_var + self.eps)
        out = layers.expand(self.rho, [input.shape[0], -1, -1, -1]) * out_in + (1-layers.expand[input.shape[0], -1, -1, -1]) * out_ln
        out = out * layers.expand(self.gamma, [input.shape[0], -1, -1, -1]) + layers.expand(self.beta, [input.shape[0], -1, -1, -1])

        return out


class ResnetAdaILNBlock(dygraph.Layer):
    def __init__(self, dim, use_bias):
        super(ResnetAdaILNBlock, self).__init__()
        self.pad1 = ReflectionPad2d([1,1,1,1])
        self.conv1 = Conv2D(dim, dim, filter_size=3, stride=1, padding=0, bias_attr=use_bias)
        self.norm1 = adaILN(dim)
        self.relu1 = ReLU(True)

        self.pad2 = ReflectionPad2d([1,1,1,1])
        self.conv2 = Conv2D(dim, dim, filter_size=3, stride=1, padding=0, bias_attr=use_bias)
        self.norm2 = adaILN(dim)

    def forward(self, x, gamma, beta):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, gamma, beta)
        out = self.relu1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)

        return out + x

class ResnetGenerator(dygraph.Layer):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, img_size=256, light=False):
        super(ResnetGenerator, self).__init__()
        '''
        Args:
            input_cn: 输入通道数
            output_nc: 输出通道数，此处二者都为3
            ngf: base channel number per layer
            n_blocks: The number of resblock
        '''
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.img_size = img_size
        self.light = light

        DownBlock = []
        # 3 ReflectionPad2d 抵消了紧接着的7*7Conv层
        # TODO 此处由于Paddle的pad2d在fluid.layer中，不能作为对象定义，比较麻烦，因此暂时使用普通padding
        DownBlock += [ReflectionPad2d([1, 1, 1, 1]),
                      Conv2D(input_nc, ngf, filter_size=7, stride=1, padding=0, bias_attr=False),
                      InstanceNorm(ngf),
                      # TODO paddle没有单独的ReLU对象，暂时用PReLU代替，后期将const改成0
                      ReLU(True)]

        # Down-Sampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            # 通道以2倍增，stride为2，大小以2减少
            DownBlock += [ReflectionPad2d([1, 1, 1, 1]),
                          Conv2D(ngf * mult, ngf * mult * 2, filter_size=3, stride=2, padding=0, bias_attr=False),
                          InstanceNorm(ngf * mult * 2),
                          ReLU(True)]

        # Down-Sampling Bottleneck
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            DownBlock += [ResnetBlock(ngf * mult, use_bias=False)]

        # Class Activation Map
        self.gap_fc = Linear(ngf * mult, 1, bias_attr=False)
        self.gmp_fc = Linear(ngf * mult, 1, bias_attr=False)
        self.conv1x1 = Conv2D(ngf * mult * 2, ngf * mult, filter_size=1, stride=1, padding=0, bias_attr=True)
        self.relu = ReLU(True)

        # Gamma, Beta block
        if self.light:
            FC = [Linear(ngf * mult, ngf * mult, bias_attr=False),
                  ReLU(True),
                  Linear(ngf * mult, ngf * mult, bias_attr=False),
                  ReLU(True)]
        else:
            # TODO 这里没太理解
            FC = [Linear(img_size // mult * img_size // mult * ngf * mult, ngf * mult, bias_attr=False),
                  ReLU(True),
                  Linear(ngf * mult, ngf * mult, bias_attr=False),
                  ReLU(True)]
        self.gamma = Linear(ngf * mult, ngf * mult, bias_attr=False)
        self.beta = Linear(ngf * mult, ngf * mult, bias_attr=False)

        # Up-Sampling Bottleneck
        for i in range(n_blocks):
            setattr(self, 'UpBlock1_' + str(i + 1), ResnetAdaILNBlock(ngf * mult, use_bias=False))

        # Up-Sampling
        UpBlock2 = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            UpBlock2 += [Upsample(scale_factor=2),
                         ReflectionPad2d([1, 1, 1, 1]),
                         Conv2D(ngf * mult, int(ngf * mult / 2), filter_size=3, stride=1, padding=0, bias_attr=False),
                         ILN(int(ngf * mult / 2)),
                         ReLU(True)]

        UpBlock2 += [ReflectionPad2d([3, 3, 3, 3]),
                     Conv2D(ngf, output_nc, filter_size=7, stride=1, padding=0, bias_attr=False),
                     Tanh(False)]

        self.DownBlock = Sequential(*DownBlock)
        self.FC = Sequential(*FC)
        self.UpBlock2 = Sequential(*UpBlock2)

    def forward(self, input):
        x = self.DownBlock(input)
        print('x: ' + str(x.shape))
        gap = layers.adaptive_pool2d(x, 1, pool_type='avg')
        gap_logit = self.gap_fc(layers.reshape(gap, [x.shape[0], -1]))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = x * layers.unsqueeze(layers.unsqueeze(gap_weight, 2), 3)

        gmp = layers.adaptive_pool2d(x, 1, pool_type='max')
        gmp_logit = self.gmp_fc(layers.reshape(gmp, [x.shape[0], -1]))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = x * layers.unsqueeze(layers.unsqueeze(gmp_weight, 2), 3)

        cam_logit = layers.concat([gap_logit, gmp_logit], 1)
        x = layers.concat([gap, gmp], 1)
        x = self.relu(self.conv1x1(x))

        heatmap = layers.reduce_sum(x, dim=1, keep_dim=True)

        if self.light:
            x_ = layers.adaptive_pool2d(x, 1, pool_type='avg')
            x_ = self.FC(layers.reshape(x_, [x_.shape[0], -1]))
        else:
            x_ = self.FC(layers.reshape(x, [x.shape[0], -1]))
        gamma, beta = self.gamma(x_), self.beta(x_)

        for i in range(self.n_blocks):
            x = getattr(self, 'UpBlock1_' + str(i + 1))(x, gamma, beta)
        out = self.UpBlock2(x)

        return out, cam_logit, heatmap


class Discriminator(dygraph.Layer):
    def __init__(self, input_nc, ndf=64, n_layers=5):
        super(Discriminator, self).__init__()
        model = [ReflectionPad2d([1,1,1,1]),
                # TODO 谱归一化
                 Conv2D(input_nc, ndf, filter_size=4, stride=2, padding=0, bias_attr=True),
                 LeakyReLU(0.2, True)]

        for i in range(1, n_layers - 2):
            mult = 2 ** (i - 1)
            model += [ReflectionPad2d([1,1,1,1]),
                      Conv2D(ndf * mult, ndf * mult * 2, filter_size=4, stride=2, padding=0, bias_attr=True),
                      LeakyReLU(0.2, True)]

        mult = 2 ** (n_layers - 2 - 1)
        model += [ReflectionPad2d([1,1,1,1]),
                  Conv2D(ndf * mult, ndf * mult * 2, filter_size=4, stride=1, padding=0, bias_attr=True),
                  LeakyReLU(0.2, True)]

        # Class Activation Map
        mult = 2 ** (n_layers - 2)
        self.gap_fc = Linear(ndf * mult, 1, bias_attr=False)
        self.gmp_fc = Linear(ndf * mult, 1, bias_attr=False)
        self.conv1x1 = Conv2D(ndf * mult * 2, ndf * mult, filter_size=1, stride=1, bias_attr=True)
        self.leaky_relu = LeakyReLU(0.2, True)

        self.pad = ReflectionPad2d([1,1,1,1])
        self.conv = Conv2D(ndf * mult, 1, filter_size=4, stride=1, padding=0, bias_attr=False)

        self.model = Sequential(*model)

    def forward(self, input):
        x = self.model(input)

        gap = layers.adaptive_pool2d(x, 1, pool_type='avg')
        gap_logit = self.gap_fc(layers.reshape(gap, [x.shape[0], -1]))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = x * layers.unsqueeze(layers.unsqueeze(gap_weight, 2), 3)

        gmp = layers.adaptive_pool2d(x, 1, pool_type='max')
        gmp_logit = self.gmp_fc(layers.reshape(gmp, [x.shape[0], -1]))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = x * layers.unsqueeze(layers.unsqueeze(gmp_weight, 2), 3)

        cam_logit = layers.concat([gap_logit, gmp_logit], 1)
        x = layers.concat([gap, gmp], 1)
        x = self.leaky_relu(self.conv1x1(x))

        heatmap = layers.reduce_sum(x, dim=1, keep_dim=True)

        x = self.pad(x)
        out = self.conv(x)

        return out, cam_logit, heatmap