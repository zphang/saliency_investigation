import math
import os
import types

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from casme.ext.pytorch_inpainting_with_partial_conv import PConvUNet, PCBActiv
from casme.ext.torchray import imsmooth
import casme.criterion as criterion
import casme.tasks.imagenet.utils as imagenet_utils
import casme.utils.torch_utils as torch_utils


class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super(Upsample, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(x, self.size, self.scale_factor, self.mode, self.align_corners)


class PConvUNetGEN(nn.Module):
    def __init__(self, layer_size=7, input_channels=3, upsampling_mode='nearest', infoGAN=False,
                 final_activation=None):
        super().__init__()
        self.freeze_enc_bn = False
        self.upsampling_mode = upsampling_mode
        self.layer_size = layer_size
        self.enc_1 = PCBActiv(input_channels, 64, bn=False, sample='down-7')
        self.enc_2 = PCBActiv(64, 128, sample='down-5')
        self.enc_3 = PCBActiv(128, 256, sample='down-5')
        self.enc_4 = PCBActiv(256, 512, sample='down-3')
        for i in range(4, self.layer_size):
            name = 'enc_{:d}'.format(i + 1)
            setattr(self, name, PCBActiv(512, 512, sample='down-3'))

        for i in range(4, self.layer_size):
            name = 'dec_{:d}'.format(i + 1)
            if i == self.layer_size-1:
                if infoGAN:
                    setattr(self, name, PCBActiv(512 + 512 + 4, 512, activ='leaky'))
                else:
                    setattr(self, name, PCBActiv(512 + 512 + 1, 512, activ='leaky'))
            else:
                setattr(self, name, PCBActiv(512 + 512, 512, activ='leaky'))
        self.dec_4 = PCBActiv(512 + 256, 256, activ='leaky')
        self.dec_3 = PCBActiv(256 + 128, 128, activ='leaky')
        self.dec_2 = PCBActiv(128 + 64, 64, activ='leaky')
        self.dec_1 = PCBActiv(64 + input_channels, input_channels,
                              bn=False, activ=None, conv_bias=True)
        
        self.infoGAN = infoGAN
        self.emb = nn.Embedding(3, 3) 
        self.emb.requires_grad = False
        self.emb.weight.data = torch.eye(3)
        self.upsample = nn.Upsample(scale_factor=2**(8 - layer_size + 1), mode='nearest') # TODO: fix, rather than using magic number 8
        self.num_labels = 3 # TODO: fix

        self.final_activation = final_activation

    def forward(self, input, input_mask, labels=None):
        h_dict = {}  # for the output of enc_N
        h_mask_dict = {}  # for the output of enc_N

        h_dict['h_0'], h_mask_dict['h_0'] = input, input_mask
        import collections as col
        zstorage = col.OrderedDict()

        h_key_prev = 'h_0'
        for i in range(1, self.layer_size + 1):
            l_key = 'enc_{:d}'.format(i)
            h_key = 'h_{:d}'.format(i)
            h_dict[h_key], h_mask_dict[h_key] = getattr(self, l_key)(
                h_dict[h_key_prev], h_mask_dict[h_key_prev])
            h_key_prev = h_key
            # print(h_key, tuple(h_dict[h_key].shape))
            zstorage["A_h_"+h_key] = h_dict[h_key]
            zstorage["A_m_" + h_key] = h_mask_dict[h_key]

        h_key = 'h_{:d}'.format(self.layer_size)
        h, h_mask = h_dict[h_key], h_mask_dict[h_key]

        for i in range(self.layer_size, 0, -1):
            enc_h_key = 'h_{:d}'.format(i - 1)
            dec_l_key = 'dec_{:d}'.format(i)

            h = F.interpolate(h, scale_factor=2, mode=self.upsampling_mode)
            h_mask = F.interpolate(h_mask, scale_factor=2, mode='nearest')
            zstorage["B_h_{}".format(i)] = h
            zstorage["B_m_{}".format(i)] = h_mask
            if i == self.layer_size:
                # inserts random channel
                random_input = torch.rand(h.size(0),1,h.size(2),h.size(3)).cuda()
                if self.infoGAN:
                    label_channels = self.emb(labels).view(labels.shape[0], self.num_labels, 1, 1)
                    label_channels = self.upsample(label_channels)
                    h = torch.cat([h, h_dict[enc_h_key], random_input, label_channels], dim=1)
                    h_mask = torch.cat([h_mask, h_mask_dict[enc_h_key], torch.ones_like(random_input), torch.ones_like(label_channels)], dim=1)
                else:
                    
                    h = torch.cat([h, h_dict[enc_h_key], random_input], dim=1)
                    h_mask = torch.cat([h_mask, h_mask_dict[enc_h_key], torch.ones_like(random_input)], dim=1)
                    # print(i, h.shape)
                    zstorage["B_h_{}".format(i)] = h
                    zstorage["B_m_{}".format(i)] = h_mask
                
            else:
                h = torch.cat([h, h_dict[enc_h_key]], dim=1)
                h_mask = torch.cat([h_mask, h_mask_dict[enc_h_key]], dim=1)
                # print(i, h.shape)
                zstorage["C_h_{}".format(i)] = h
                zstorage["C_m_{}".format(i)] = h_mask
            h, h_mask = getattr(self, dec_l_key)(h, h_mask)
            # print(i, h.shape)
            zstorage["D_h_{}".format(i)] = h
            zstorage["D_m_{}".format(i)] = h_mask

        if self.final_activation is None:
            pass
        elif self.final_activation == "tanh":
            h = torch.tanh(h)
        else:
            raise NotImplementedError()

        return h, h_mask

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        if self.freeze_enc_bn:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d) and 'enc' in name:
                    module.eval()


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [
            block(self.inplanes, planes, stride, downsample)
        ]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def forward(self, x, return_intermediate=False):
        l = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        l.append(x)

        x = self.layer1(x)
        l.append(x)
        x = self.layer2(x)
        l.append(x)
        x = self.layer3(x)
        l.append(x)
        x = self.layer4(x)
        l.append(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        if return_intermediate:
            return x, l
        else:
            return x


class ResNetShared(ResNet):
    def forward(self, x, return_intermediate=False):
        l = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        l.append(x)

        x = self.layer1(x)
        l.append(x)
        x = self.layer2(x)
        l.append(x)
        x = self.layer3(x)
        l.append(x)
        x = self.layer4(x)
        l.append(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        if return_intermediate:
            return x, l
        else:
            return x


def monkey_patch_resnet(resnet_model):
    if getattr(resnet_model, "__is_patched", None):
        return
    resnet_model.forward = types.MethodType(ResNetShared.forward, resnet_model)
    resnet_model.__is_patched = True


def replace_prefix(s, prefix):
    if s.startswith(prefix):
        return s[len(prefix):]
    else:
        return s


def resnet50shared(pretrained=False, path=None, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    # Monkey-patch
    #print("Patching ResNet.forward")
    #monkey_patch_resnet(model)
    if pretrained:
        if path is None:
            model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth'))
        else:
            loaded = torch.load(path)
            if "best_acc1" in loaded:
                # From PyTorch script
                modified = {replace_prefix(k, "module."): v for k, v in loaded["state_dict"].items()}
                model.load_state_dict(modified)
            else:
                model.load_state_dict(loaded)
    else:
        assert path is None
    return model


class Discriminator(nn.Module):
    def __init__(self, input_dim, return_logits=False):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 1)
        self.return_logits = return_logits

    def forward(self, h, resnet):
        h = resnet.final_bn(h)
        h = resnet.relu(h)
        pooled_h = h.view(h.shape[0], h.shape[1], -1).mean(dim=2)
        # TODO: detach?
        logits = self.fc1(pooled_h)#.detach())
        #print(logits)
        if self.return_logits:
            output = logits
        else:
            output = F.log_softmax(logits, dim=1)
        return output


class Infiller(nn.Module):

    def __init__(self, model_type, input_channels, num_layers=6):
        super().__init__()
        self.model_type = model_type
        self.input_channels = input_channels
        self.num_layers = num_layers
        if model_type == "ciGAN":
            # do I have a mask for each category, 1 indicating salient region?
            pass
        elif model_type =="pconv":
            self.model = PConvUNet(layer_size=num_layers, input_channels=input_channels)
        elif model_type == "pconv_gan":
            self.model = PConvUNetGEN(layer_size=num_layers, input_channels=input_channels)
        elif model_type == "pconv_infogan":
            self.model = PConvUNetGEN(layer_size=num_layers, input_channels=input_channels, infoGAN=True)
        elif model_type == "pconv_gan2":
            self.model = PConvUNetGEN(layer_size=num_layers, input_channels=input_channels,
                                      final_activation="tanh")
        elif model_type == "none":
            # To stop PyTorch from complaining about not having parameters
            self.model = torch.nn.Linear(1, 1)
        else:
            raise NotImplementedError()

    def forward(self, x, mask, labels=None):
        if self.model_type == "ciGAN":
            pass
        elif self.model_type in ["pconv", "pconv_gan"]:
            return self.model(x, mask)
        elif self.model_type == "pconv_infogan":
            return self.model(x, mask, labels)
        elif self.model_type == "pconv_gan2":
            return self.model(x, mask)
        elif self.model_type == "none":
            return x, mask
        else:
            raise NotImplementedError()


class Film(nn.Module):
    def __init__(self, dim_size, input_dim_size):
        super().__init__()
        self.film_fc = nn.Linear(dim_size, input_dim_size)
        self.film_hc = nn.Linear(dim_size, input_dim_size)

    def forward(self, x, conditioning_info):
        gamma = self.film_fc(conditioning_info).unsqueeze(-1).unsqueeze(-1)
        beta = self.film_fc(conditioning_info).unsqueeze(-1).unsqueeze(-1)
        return x * gamma + beta


class ClassInputModule(nn.Module):
    def __init__(self, num_classes=1000, dim_size=64, input_dim_size=2048):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, dim_size)
        # self.film = Film(dim_size=dim_size, input_dim_size=input_dim_size)

    def forward(self, x, class_ids):
        embedded = self.embedding(class_ids)
        # return self.film(x, embedded)
        return embedded


def binary_gumbel_softmax(logits, tau=0.1, hard=False, epsilon=1e-8):
    # see: F.gumbel_softmax
    #gumbels_0 = -torch.empty_like(logits).exponential_().log()  # ~Gumbel(0,1)
    #gumbels_1 = -torch.empty_like(logits).exponential_().log()  # ~Gumbel(0,1)
    #mod_logits = (logits + gumbels_1) / tau  # ~Gumbel(logits,tau)

    uniform_noise = torch.empty_like(logits).uniform_()
    approx = logits + (uniform_noise + epsilon).log() - (1 - uniform_noise + epsilon).log()

    y_soft = torch.sigmoid(approx / tau)

    if hard:
        y_hard = (y_soft > 0.5).detach().float()
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


class GumbelFinal(nn.Module):
    def __init__(self, in_channels, out_channels, tau=0.1, final_upsample_mode="nearest",
                 gumbel_output_mode="hard",
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(tau, (float, int)):
            self.tau = tau
        elif tau == -1:
            # clean this up
            self.tau = nn.Parameter(torch.tensor([0.1]))
        else:
            raise KeyError(tau)
        self.final_upsample_mode = final_upsample_mode

        self.gumbel_output_mode = gumbel_output_mode

        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=True,
        )
        self.upsample = Upsample(scale_factor=4, mode=final_upsample_mode)

    def forward(self, x):
        h = self.conv(x)
        n_dim, c_dim, h_dim, w_dim = h.shape
        h = h.view(n_dim, h_dim, w_dim)
        if self.gumbel_output_mode in ("hard", "soft"):
            if self.training:
                h = binary_gumbel_softmax(h, tau=self.tau, hard=self.gumbel_output_mode == "hard")
            else:
                h = (h > 0).float()
        elif self.gumbel_output_mode == "sigmoid":
            h = torch.sigmoid(h)
        elif self.gumbel_output_mode == "logits":
            pass
        else:
            raise KeyError(self.gumbel_output_mode)
        h = h.view(n_dim, c_dim, h_dim, w_dim)
        h = self.upsample(h)
        return h


class Masker(nn.Module):

    def __init__(self, in_channels, out_channels,
                 final_upsample_mode='nearest',
                 add_prob_layers=False,
                 add_class_ids=False,
                 apply_gumbel=False,
                 apply_gumbel_tau=0.1,
                 gumbel_output_mode="hard",
                 use_layers=(0, 1, 2, 3, 4),
                 ):
        super().__init__()
        self.final_upsample_mode = final_upsample_mode
        self.add_prob_layers = add_prob_layers
        self.add_class_ids = add_class_ids
        self.apply_gumbel = apply_gumbel
        self.apply_gumbel_tau = apply_gumbel_tau
        self.gumbel_output_mode = gumbel_output_mode
        self.use_layers = use_layers

        more_dims = 0
        if self.add_prob_layers:
            more_dims += 1
        if self.add_class_ids:
            more_dims += 64

        self.conv1x1_0 = None
        self.conv1x1_1 = None
        self.conv1x1_2 = None
        self.conv1x1_3 = None
        self.conv1x1_4 = None
        self._setup_add_conv_layers(
            in_channels=in_channels,
            out_channels=out_channels,
            more_dims=more_dims,
        )

        final_layer_in_channels = self._get_final_layer_in_channels(
            in_channels=in_channels,
            out_channels=out_channels,
            more_dims=more_dims,
        )
        self.final = self._get_final_layer(
            in_channels=final_layer_in_channels,
            out_channels=1,
        )

        if self.add_class_ids:
            self.class_module = ClassInputModule()
        else:
            self.class_module = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _get_final_layer_in_channels(self, in_channels, out_channels, more_dims):
        if self.use_layers == (0,):
            final_layer_in_channels = more_dims
        else:
            final_layer_in_channels = 0
        if 0 in self.use_layers:
            final_layer_in_channels += in_channels[0]
        for i in range(1, 5):
            if i in self.use_layers:
                final_layer_in_channels += out_channels
        return final_layer_in_channels

    def _get_final_layer(self, in_channels, out_channels):
        if self.apply_gumbel:
            print(f"Using gumbel softmax with tau={self.apply_gumbel_tau}")
            final = GumbelFinal(
                in_channels=in_channels,
                out_channels=out_channels,
                tau=self.apply_gumbel_tau,
                final_upsample_mode=self.final_upsample_mode,
                gumbel_output_mode=self.gumbel_output_mode,
            )
        else:
            final = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=3, stride=1, padding=1, bias=True),
                nn.Sigmoid(),
                Upsample(scale_factor=4, mode=self.final_upsample_mode),
            )
        return final

    def _setup_add_conv_layers(self, in_channels, out_channels, more_dims):
        if self.use_layers == (0,):
            self.conv1x1_0 = self._make_conv1x1_upsampled(in_channels[0] + more_dims, out_channels)
        if 1 in self.use_layers:
            self.conv1x1_1 = self._make_conv1x1_upsampled(in_channels[1] + more_dims, out_channels)
        if 2 in self.use_layers:
            self.conv1x1_2 = self._make_conv1x1_upsampled(in_channels[2] + more_dims, out_channels, 2)
        if 3 in self.use_layers:
            self.conv1x1_3 = self._make_conv1x1_upsampled(in_channels[3] + more_dims, out_channels, 4)
        if 4 in self.use_layers:
            self.conv1x1_4 = self._make_conv1x1_upsampled(in_channels[4] + more_dims, out_channels, 8)

    @classmethod
    def _make_conv1x1_upsampled(cls, inplanes, outplanes, scale_factor=None):
        if scale_factor:
            return nn.Sequential(
                nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(outplanes),
                nn.ReLU(inplace=True),
                Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(outplanes),
                nn.ReLU(inplace=True)
            )

    def forward(self, layers, use_p, class_ids, no_sigmoid=False):
        additional_channels = (
            self.maybe_add_prob_layers(layers, use_p)
            + self.maybe_add_class_layers(layers, class_ids)
        )

        layers = self.append_channels(layers, additional_channels)

        k = []
        if 0 in self.use_layers:
            if self.use_layers == (0,):
                k.append(self.conv1x1_0(layers[0]))
            else:
                k.append(layers[0])
        if 1 in self.use_layers:
            k.append(self.conv1x1_1(layers[1]))
        if 2 in self.use_layers:
            k.append(self.conv1x1_2(layers[2]))
        if 3 in self.use_layers:
            k.append(self.conv1x1_3(layers[3]))
        if 4 in self.use_layers:
            k.append(self.conv1x1_4(layers[4]))

        final_input = torch.cat(k, 1)

        if not no_sigmoid:
            return self.final(final_input)
        else:
            assert not self.apply_gumbel
            return self.final[2](self.final[0](final_input))

    def maybe_add_prob_layers(self, layers, use_p):
        if self.add_prob_layers:
            assert use_p is not None
            if not isinstance(use_p, torch.Tensor):
                batch_size = layers[0].shape[0]
                device = layers[0].device
                use_p = torch.Tensor([use_p]).expand(batch_size, -1).to(device)
            elif len(use_p.shape) == 1:
                use_p = use_p.unsqueeze(-1)
            return [use_p]
        else:
            assert use_p is None
            return []

    def maybe_add_class_layers(self, layers, class_ids):
        if self.add_class_ids:
            return [self.class_module(layers[-1], class_ids)]
        else:
            assert class_ids is None
            return []

    @classmethod
    def append_channels(cls, layers, additional_channels):
        if not additional_channels:
            return layers
        elif len(additional_channels) == 1:
            additional_channels_tensor = additional_channels[0]
        else:
            additional_channels_tensor = torch.cat(additional_channels, dim=1)

        new_layers = []
        for layer in layers:
            additional_channels_slice = additional_channels_tensor \
                .expand(layer.shape[2], layer.shape[3], -1, -1) \
                .permute(2, 3, 0, 1)
            new_layers.append(torch.cat([layer, additional_channels_slice], dim=1))
        return new_layers


class InfillerCNN(nn.Module):
    def __init__(self, in_chans, out_chans, intermediate_dim_ls):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.intermediate_dim_ls = intermediate_dim_ls
        self.down_ls = nn.ModuleList()
        self.up_ls = nn.ModuleList()
        self.n_dims = len(intermediate_dim_ls)

        self.down_ls.append(self.get_conv_layer(
            in_chans, intermediate_dim_ls[0],
        ))
        for i in range(0, self.n_dims - 1):
            self.down_ls.append(self.get_conv_layer(
                intermediate_dim_ls[i],
                intermediate_dim_ls[i + 1],
            ))
        self.up_ls.append(self.get_conv_layer(
            intermediate_dim_ls[-1],
            intermediate_dim_ls[-2],
        ))
        for i in list(range(self.n_dims - 2, 0, -1)):
            self.up_ls.append(self.get_conv_layer(
                self.up_ls[-1].out_channels + intermediate_dim_ls[i],
                intermediate_dim_ls[i],
            ))
        self.up_ls.append(self.get_conv_layer(
            intermediate_dim_ls[0] + intermediate_dim_ls[1],
            out_chans,
        ))
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.upsample = Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.init_weights()

    def forward(self, masked_x, mask, x, mask_mode):
        h = torch.cat([masked_x, mask], dim=1)
        down_h_ls = []
        for i in range(self.n_dims):
            h = self.max_pool(self.relu(self.down_ls[i](h)))
            down_h_ls.append(h)
        for i in range(self.n_dims - 1):
            h = self.upsample(self.relu(self.up_ls[i](h)))
            h = torch.cat([h, down_h_ls[-i - 2]], dim=1)
        h = self.upsample(self.up_ls[-1](h))
        return h

    @classmethod
    def get_conv_layer(cls, in_channels, out_channels):
        return nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=1, padding=1,
        )

    def init_weights(self):
        for conv_layer in list(self.down_ls) + list(self.up_ls):
            torch.nn.init.xavier_uniform_(conv_layer.weight)


class CAInfillerWrapper(nn.Module):
    def __init__(self, iproc):
        super().__init__()
        self.iproc = iproc

        from generative_inpainting_pytorch.model.networks import Generator
        self.generator = Generator(
            config={
                "input_dim": 5,
                "ngf": 32,
            },
            use_cuda=True,
            device_ids=[0],
        )
        self.generator.load_state_dict(torch.load(os.environ["CA_MODEL_PATH"]))

    def forward(self, masked_x, mask, x, mask_mode):
        # masked_x: normalize from [0, 1]
        # mask: 1 = selected region

        # generator input: [0, 255] / 127.5 - 1
        input_x = self.iproc.denorm_tensor(masked_x) * 255/127.5 - 1
        input_x = criterion.MaskFunc.static_apply(x=input_x, mask=mask, mask_mode=mask_mode)
        _, raw_result, _ = self.generator(
            x=input_x,
            mask=mask,
        )
        result = self.iproc.norm_tensor((raw_result + 1) * 127.5/255)
        return result


class DFNInfillerWrapper(nn.Module):
    def __init__(self, iproc):
        super().__init__()
        self.iproc = iproc
        from casme.ext.dfnet import DFNet
        self.model = DFNet()
        self.model.load_state_dict(torch.load(
            os.environ["DFNET_MODEL_PATH"],
            map_location=torch.device("cpu"),
        ))
        self.model.eval()

    def forward(self, masked_x, mask, x, mask_mode):
        # masked_x: normalize from [0, 1]
        # mask: 1 = selected region
        # Resize
        masked_x = F.interpolate(masked_x, 256, mode="bilinear", align_corners=True)
        mask = F.interpolate(mask, 256, mode="bilinear", align_corners=True)

        # desired input: [0, 1]
        # desired mask: [0, 1], 0 = masked out
        input_x = self.iproc.denorm_tensor(masked_x)
        imgs_miss = criterion.MaskFunc.static_apply(x=input_x, mask=mask, mask_mode=mask_mode)

        if mask_mode == criterion.MaskFunc.MASK_OUT:
            used_mask = 1 - mask
        elif mask_mode == criterion.MaskFunc.MASK_IN:
            used_mask = mask
        else:
            raise KeyError(mask_mode)
        result, alpha, raw = self.model(imgs_miss, used_mask)
        result, alpha, raw = result[0], alpha[0], raw[0]
        result = imgs_miss + result * mask

        result = F.interpolate(result, 224, mode="bilinear", align_corners=True)
        result = self.iproc.norm_tensor(result)
        return result


class DummyInfiller(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Linear(1, 1)

    def forward(self, masked_x, mask, x, mask_mode):
        return masked_x * 0


class BlurInfiller(nn.Module):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma
        self.dummy = nn.Linear(1, 1)

    def forward(self, masked_x, mask, x, mask_mode):
        return imsmooth(x, sigma=self.sigma)


class ImageProc(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std
        self.mean_tensor = nn.Parameter(
            torch.FloatTensor(self.mean).unsqueeze(0).unsqueeze(-1).unsqueeze(-1),
            requires_grad=False,
        )
        self.std_tensor = nn.Parameter(
            torch.FloatTensor(self.std).unsqueeze(0).unsqueeze(-1).unsqueeze(-1),
            requires_grad=False,
        )

    def forward(self):
        pass

    def norm(self, x):
        return (x - self.mean) / self.std

    def denorm(self, x):
        return x * self.std + self.mean

    def norm_tensor(self, x):
        return (x - self.mean_tensor) / self.std_tensor

    def denorm_tensor(self, x):
        return x * self.std_tensor + self.mean_tensor


def get_infiller(infiller_model):
    if infiller_model == "cnn":
        infiller = InfillerCNN(
            4, 3, [32, 64, 128, 256],
        )
    elif infiller_model == "ca_infiller":
        infiller = CAInfillerWrapper(ImageProc(
            mean=imagenet_utils.NORMALIZATION_MEAN,
            std=imagenet_utils.NORMALIZATION_STD,
        ))
    elif infiller_model == "dfn_infiller":
        infiller = DFNInfillerWrapper(ImageProc(
            mean=imagenet_utils.NORMALIZATION_MEAN,
            std=imagenet_utils.NORMALIZATION_STD,
        ))
    elif infiller_model == "dummy":
        infiller = DummyInfiller()
    elif infiller_model == "blur":
        infiller = BlurInfiller(sigma=20)
    else:
        raise KeyError(infiller_model)

    torch_utils.set_requires_grad(infiller.named_parameters(), False)
    return infiller


def default_masker(**kwargs):
    return Masker([64, 256, 512, 1024, 2048], 64, **kwargs)


def string_to_tuple(string, cast=None):
    tokens = [x.strip() for x in string.split(",")]
    if cast is not None:
        tokens = [cast(x) for x in tokens]
    return tuple(tokens)
