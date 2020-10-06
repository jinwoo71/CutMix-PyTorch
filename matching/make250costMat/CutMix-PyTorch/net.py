import torch.nn as nn
import torch
import torch.nn.functional as F
from function import adaptive_instance_normalization as adain
from function import calc_mean_std
import numpy as np
decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)
vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)
criterion = nn.CrossEntropyLoss().cuda()
class Jinwoo(nn.Module):
    def __init__(self):
        super(Jinwoo, self).__init__()
    def forward(self, output, target_a, target_b, sr, x, ratio):
        log_preds = F.log_softmax(output, dim=-1)
        a_loss = -log_preds[torch.arange(output.shape[0]),target_a]
        b_loss = -log_preds[torch.arange(output.shape[0]),target_b]
        cr_loss = a_loss.mean() * (x) + b_loss.mean() * (1.0-x)
        sr_loss = a_loss * (1-sr) + b_loss * sr
        return ratio * cr_loss + (1.0-ratio) * sr_loss.mean()


class Net(nn.Module):
    def __init__(self, encoder, decoder):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.decoder = decoder
        self.mse_loss = nn.MSELoss()
        self.mse_loss_none = nn.MSELoss(reduction='none')
        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1], results[2], results[3], results[4]

    # extract relu4_1 from input image
    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    def per_calc_content_loss(self, input, target):
        assert(input.size()==target.size())
        assert(target.requires_grad is False)
        t = self.mse_loss_none(input, target)
        return torch.mean(t,dim=[1,2,3])

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def per_calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        t1 = self.mse_loss_none(input_mean, target_mean)
        t2 = self.mse_loss_none(input_std, target_std)
        return torch.mean(t1, dim=[1,2,3]) + torch.mean(t2, dim=[1,2,3])

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def forward(self, content, style, flag):
        if (flag == 0) :
            x = 1.0
            y = 0.0
            style_f1, style_f2, style_f3, style_f4 = self.encode_with_intermediate(style)
            content_f1, content_f2, content_f3, content_f4 = self.encode_with_intermediate(content)

            t = np.random.uniform(max(0, x+y-1), min(x, y), 1)[0]
            g_t = t * content + (1.0-x-y+t)*style + self.decoder((x-t) * adain(content_f4, style_f4) + (y-t) * adain(style_f4, content_f4))
            g_t_f1, g_t_f2, g_t_f3, g_t_f4 = self.encode_with_intermediate(g_t)
            loss_a_s = (self.per_calc_style_loss(g_t_f1, content_f1) + self.per_calc_style_loss(g_t_f2, content_f2) +
                        self.per_calc_style_loss(g_t_f3, content_f3) + self.per_calc_style_loss(g_t_f4, content_f4))
            return loss_a_s
        else :
            x = 0.0
            y = 1.0
            style_f1, style_f2, style_f3, style_f4 = self.encode_with_intermediate(style)
            content_f1, content_f2, content_f3, content_f4 = self.encode_with_intermediate(content)

            t = np.random.uniform(max(0, x+y-1), min(x, y), 1)[0]
            g_t = t * content + (1.0-x-y+t)*style + self.decoder((x-t) * adain(content_f4, style_f4) + (y-t) * adain(style_f4, content_f4))
            g_t_f1, g_t_f2, g_t_f3, g_t_f4 = self.encode_with_intermediate(g_t)
            loss_b_s = (self.per_calc_style_loss(g_t_f1, style_f1) + self.per_calc_style_loss(g_t_f2, style_f2)
                        + self.per_calc_style_loss(g_t_f3, style_f3) + self.per_calc_style_loss(g_t_f4, style_f4))
            return loss_b_s

