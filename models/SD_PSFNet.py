import torch
import torch.nn as nn
import torch.nn.functional as F

##########################################################################
def dconv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)



class PSFChannelReducer(nn.Module):
    def __init__(self, k_c_in, k_c_out=1, reduction=4, bias=False):
        super().__init__()
        if k_c_in <= k_c_out:
            self.reducer = nn.Conv2d(k_c_in, k_c_out, kernel_size=1, padding=0, bias=bias)
        else:
            self.reducer = nn.Sequential(
                nn.Conv2d(k_c_in, k_c_in // reduction, kernel_size=1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(k_c_in // reduction, k_c_out, kernel_size=1, padding=0, bias=bias)
            )

    def forward(self, psf_multi_channel): # Input: [B, K_c_in, K, K]
        reduced_psf = self.reducer(psf_multi_channel) # Output: [B, K_c_out, K, K]
        reduced_psf_normalized = reduced_psf / (reduced_psf.sum(dim=(2,3), keepdim=True) + 1e-8)
        return reduced_psf_normalized

class MultiScalePSFHead(nn.Module):
    def __init__(self, n_feat, scale_unetfeats, kernel_size, reduction, bias, act,
                 psf_sizes=None, psf_target_channels=40):
        super().__init__()
        feat_dims = [n_feat, n_feat + scale_unetfeats, n_feat + scale_unetfeats * 2]
        self.psf_sizes = psf_sizes if psf_sizes is not None else [3, 5, 7]
        self.psf_target_channels = psf_target_channels # K_c

        self.predictors = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(dim, size ** 2, 1),
                nn.Softmax(dim=1)
            ) for dim, size in zip(feat_dims, self.psf_sizes)
        ])

        k_final_sq = self.psf_sizes[-1] ** 2
        self.fusion = nn.Sequential(
            CAB(sum(s ** 2 for s in self.psf_sizes), kernel_size, reduction, bias, act),
            nn.Conv2d(sum(s ** 2 for s in self.psf_sizes), self.psf_target_channels * k_final_sq, 1)
        )

    def forward(self, features):
        num_features_to_process = min(len(features), len(self.predictors))
        psfs_flat_normalized = []
        current_psf_sizes_for_features = self.psf_sizes[:num_features_to_process]

        for i, (pred, feat) in enumerate(zip(self.predictors, features[:num_features_to_process])):
            b_feat = feat.shape[0]
            k_s = current_psf_sizes_for_features[i]
            psf_i_flat = pred(feat) # [B, k_s**2, 1, 1]
            psfs_flat_normalized.append(psf_i_flat) # Softmax已处理，直接用

        fused_input = torch.cat(psfs_flat_normalized, dim=1) # [B, sum(k_s**2), 1, 1]
        fused = self.fusion(fused_input)  # Output: [B, K_c * K_final^2, 1, 1]

        k_final = self.psf_sizes[-1]
        # Reshape to [B, K_c, K_final, K_final]
        return fused.view(fused.size(0), self.psf_target_channels, k_final, k_final)


class PSFAwareAttention(nn.Module):
    def __init__(self, channels, psf_size=7,
                 psf_input_channels=40, psf_channel_reducer_reduction=4):
        super().__init__()
        self.psf_input_channels = psf_input_channels

        if self.psf_input_channels > 1:
            self.psf_channel_reducer = PSFChannelReducer(
                self.psf_input_channels, 1, psf_channel_reducer_reduction
            )
        else:
            self.psf_channel_reducer = None

        self.psf_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=psf_size, padding=psf_size // 2),
            nn.ReLU(),
            dconv(16, 32, kernel_size=psf_size),
            nn.AdaptiveAvgPool2d(1)
        )

        self.channel_fc = nn.Sequential(
            nn.Linear(32, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels * 2)
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channels + 1, channels // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels // 2, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, psf_kc_input): # psf_kc_input: [B, K_c, K, K]
        B, C, H, W = x.shape

        if self.psf_channel_reducer is not None:
            psf_single_channel = self.psf_channel_reducer(psf_kc_input) # [B, 1, K, K]
        else:
            psf_single_channel = psf_kc_input

        psf_feat = self.psf_encoder(psf_single_channel)
        gamma, beta = self.channel_fc(psf_feat.view(B, 32)).chunk(2, dim=1)

        x_modulated = x * gamma.view(B, C, 1, 1) + beta.view(B, C, 1, 1)

        psf_map = F.interpolate(psf_single_channel, (H, W), mode='bilinear', align_corners=False)
        spatial_weight = self.spatial_att(torch.cat([x_modulated, psf_map], dim=1))

        return x_modulated * spatial_weight

# class PSFAwareAttention(nn.Module):
#     """物理引导的PSF注意力机制"""
#
#     def __init__(self, channels, psf_size=7):
#         super().__init__()
#         # PSF特征编码器
#         self.psf_encoder = nn.Sequential(
#             nn.Conv2d(1, 16, psf_size, padding=psf_size // 2),
#             nn.ReLU(),
#             dconv(16, 32, psf_size),
#             nn.AdaptiveAvgPool2d(1)
#         )
#
#         # 动态特征调制
#         self.channel_fc = nn.Sequential(
#             nn.Linear(32, channels * 2),
#             nn.GELU(),
#             nn.Linear(channels * 2, channels * 2)
#         )
#
#         # 空间注意力
#         self.spatial_att = nn.Sequential(
#             nn.Conv2d(channels + 1, channels // 2, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(channels // 2, 1, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x, psf):
#         B, C, H, W = x.shape
#
#         # PSF特征提取
#         psf_feat = self.psf_encoder(psf)  # [B,32,INet,INet]
#         gamma, beta = self.channel_fc(psf_feat.view(B, 32)).chunk(2, dim=1)
#
#         # 通道调制
#         x = x * gamma.view(B, C, 1, 1) + beta.view(B, C, 1, 1)
#
#         # 空间调制
#         psf_map = F.interpolate(psf, (H, W), mode='bilinear')
#         spatial_weight = self.spatial_att(torch.cat([x, psf_map], dim=1))
#
#         return x * spatial_weight

##########################################################################
## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


##########################################################################
## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(dconv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(dconv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res

class PSFAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(PSFAB, self).__init__()
        modules_body = []
        modules_body.append(dconv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(dconv(n_feat, n_feat, kernel_size, bias=bias))

        self.PSFA = PSFAwareAttention(n_feat)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x, psf):
        res = self.body(x)
        res = self.PSFA(res, psf)
        res += x
        return res



##########################################################################
## Gate
class Gate(nn.Module):
    def __init__(self, n_feat, reduction, act):
        super(Gate, self).__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            dconv(n_feat * 2, n_feat // reduction, 1),
            act,
            dconv(n_feat // reduction, n_feat, 1),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        w = self.gate(torch.cat([x, y], dim=1))
        return w * x + (1.0 - w) * y


##########################################################################
## Shallow Features
class ShallowFeature(nn.Module):
    def __init__(self, in_c, out_c, n_feat, kernel_size, reduction, bias, act, csff=False):
        super(ShallowFeature, self).__init__()
        self.shallow_feat = nn.Sequential(dconv(in_c, n_feat, kernel_size, bias=bias),
                                          CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        if csff:
            self.gate = Gate(n_feat, reduction, act)
        self.conv = dconv(n_feat, out_c, kernel_size, bias=bias)

    def forward(self, x, H=None):
        h = self.shallow_feat(x)
        if H is not None:
            h = self.gate(h, H)
        h = self.conv(h)
        return h


##########################################################################
## Supervised Attention Module
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(SAM, self).__init__()
        self.conv1 = dconv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = dconv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = dconv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1 * x2
        x1 = x1 + x
        return x1, img


##########################################################################
## Cross feature fusion
class CSFF(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False):
        super(CSFF, self).__init__()
        self.csff = csff

        self.update_gates = nn.ModuleList([
            nn.ModuleList([
                Gate(n_feat + scale_unetfeats * i, reduction, act),
                Gate(n_feat + scale_unetfeats * i, reduction, act) if csff else nn.Identity(),
                dconv(n_feat + scale_unetfeats * i, n_feat + scale_unetfeats * i, kernel_size, bias=bias),
                dconv(n_feat + scale_unetfeats * i, n_feat + scale_unetfeats * i, kernel_size, bias=bias),
            ]) for i in range(3)
        ])

    def forward(self, es, ds, Os=None):
        w = []
        Os = Os if Os is not None else [0] * len(es)
        for e, d, O, update_gate in zip(es, ds, Os, self.update_gates):
            ed_gate, oO_gate, Conv1, Conv2 = update_gate
            o = Conv1(ed_gate(e, d))
            if self.csff:
                o = oO_gate(o, O)
            o = Conv2(o)
            w.append(o)
        return w



##########################################################################
## U-Net
class Encoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff):
        super(Encoder, self).__init__()

        self.encoder_level1 = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level2 = [CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]
        self.encoder_level3 = [CAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]

        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)

        if csff:
            self.encoder_gate1 = Gate(n_feat, reduction, act)
            self.encoder_gate2 = Gate(n_feat + scale_unetfeats, reduction, act)
            self.encoder_gate3 = Gate(n_feat + scale_unetfeats * 2, reduction, act)

        self.down12 = DownSample(n_feat, scale_unetfeats)
        self.down23 = DownSample(n_feat + scale_unetfeats, scale_unetfeats)


    def forward(self, x, last_outs=None):
        enc1 = self.encoder_level1(x)
        if last_outs is not None:
            enc1 = self.encoder_gate1(enc1, last_outs[0])

        x = self.down12(enc1)

        enc2 = self.encoder_level2(x)
        if last_outs is not None:
            enc2 = self.encoder_gate2(enc2, last_outs[1])

        x = self.down23(enc2)

        enc3 = self.encoder_level3(x)
        if last_outs is not None:
            enc3 = self.encoder_gate3(enc3, last_outs[2])

        return [enc1, enc2, enc3]


class Decoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats):
        super(Decoder, self).__init__()

        self.decoder_level1 = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level2 = [CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]
        self.decoder_level3 = [CAB(n_feat + scale_unetfeats * 2, kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]

        self.decoer_psf1 = PSFAB(n_feat, kernel_size, reduction, bias, act)
        self.decoder_level1 = nn.Sequential(*self.decoder_level1)

        self.decoer_psf2 = PSFAB(n_feat + scale_unetfeats, kernel_size, reduction, bias, act)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)

        self.decoer_psf3 = PSFAB(n_feat + scale_unetfeats * 2, kernel_size, reduction, bias, act)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)

        self.skip_attn1 = CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.skip_attn2 = CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act)

        self.up21 = SkipUpSample(n_feat, scale_unetfeats)
        self.up32 = SkipUpSample(n_feat + scale_unetfeats, scale_unetfeats)

    def forward(self, outs, psf):
        enc1, enc2, enc3 = outs
        dec3 = self.decoder_level3(self.decoer_psf3(enc3, psf))

        x = self.up32(dec3, self.skip_attn2(enc2))
        dec2 = self.decoder_level2(self.decoer_psf2(x, psf))

        x = self.up21(dec2, self.skip_attn1(enc1))
        dec1 = self.decoder_level1(self.decoer_psf1(x, psf))

        return [dec1, dec2, dec3]


##########################################################################
##---------- Resizing Modules ----------
class DownSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, stride=2, padding=0, bias=False),
                                  nn.Conv2d(in_channels, in_channels + s_factor, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.ConvTranspose2d(in_channels + s_factor, in_channels + s_factor, kernel_size=2, stride=2, bias=True),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x


class SkipUpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.ConvTranspose2d(in_channels + s_factor, in_channels + s_factor, kernel_size=2, stride=2, bias=True),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x


##########################################################################
## Original Resolution Block (ORB)
class ORB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_cab):
        super(ORB, self).__init__()
        modules_body = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(num_cab)]
        modules_body.append(dconv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


##########################################################################
class ORSNet(nn.Module):
    def __init__(self, n_feat, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats, num_cab):
        super(ORSNet, self).__init__()

        self.orb1 = ORB(n_feat + scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)
        self.orb2 = ORB(n_feat + scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)
        self.orb3 = ORB(n_feat + scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)

        self.up_enc1 = UpSample(n_feat, scale_unetfeats)
        self.up_dec1 = UpSample(n_feat, scale_unetfeats)

        self.up_enc2 = nn.Sequential(UpSample(n_feat + scale_unetfeats, scale_unetfeats),
                                     UpSample(n_feat, scale_unetfeats))
        self.up_dec2 = nn.Sequential(UpSample(n_feat + scale_unetfeats, scale_unetfeats),
                                     UpSample(n_feat, scale_unetfeats))

        self.conv1 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv2 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv3 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)

    def forward(self, x, last_outs):
        x = self.orb1(x)
        x = x + self.conv1(last_outs[0])

        x = self.orb2(x)
        x = x + self.conv2(self.up_enc1(last_outs[1]))

        x = self.orb3(x)
        x = x + self.conv3(self.up_enc2(last_outs[2]))

        return x


class UStage(nn.Module):
    def __init__(self, in_c, n_feat, scale_unetfeats, act, kernel_size, reduction, bias, csff=False):
        super(UStage, self).__init__()
        self.csff = csff
        self.shallow_feat = ShallowFeature(in_c, n_feat, n_feat, kernel_size, reduction, bias, act, csff)

        self.encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=csff)

        self.psf = MultiScalePSFHead(n_feat, scale_unetfeats, kernel_size, reduction, bias, act)

        self.decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)
        self.CSFF = CSFF(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=csff)
        self.sam = SAM(n_feat, kernel_size=1, bias=bias)

        if csff:
            self.gate = Gate(n_feat, reduction, act)

    def forward(self, x, H=None, O=None):
        h = self.shallow_feat(x, H)
        feat = self.encoder(h, O)
        psf = self.psf(feat)
        res = self.decoder(feat, psf)
        h, out = self.sam(res[0], x)
        if self.csff:
            h = self.gate(h, H)
        O = self.CSFF(feat, res, O)

        return out, h, O

class ORStage(nn.Module):
    def __init__(self, in_c, out_c, n_feat, scale_unetfeats, scale_orsnetfeats, num_cab,act, kernel_size, reduction, bias):
        super(ORStage, self).__init__()
        self.shallow_feat = ShallowFeature(in_c, n_feat + scale_orsnetfeats, n_feat, kernel_size, reduction, bias, act, csff=True)
        self.osrnet = ORSNet(n_feat, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats, num_cab)
        self.tail = dconv(n_feat + scale_orsnetfeats, out_c, kernel_size, bias=bias)

    def forward(self, x, H, O):
        h = self.shallow_feat(x, H)
        h = self.osrnet(h, O)
        out = self.tail(h) + x
        return out

class SD_PSFNet(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=40, scale_unetfeats=20, scale_orsnetfeats=16, num_cab=8, tau=3 ,kernel_size=3, reduction=4, bias=False):
        super(SD_PSFNet, self).__init__()
        act = nn.PReLU()
        self.stage_in = UStage(in_c, n_feat, scale_unetfeats, act, kernel_size, reduction, bias, csff=False)
        self.stage_mid = nn.ModuleList([UStage(in_c, n_feat, scale_unetfeats, act, kernel_size, reduction, bias, csff=True) for _ in range(tau)])
        self.stage_osr = ORStage(in_c, out_c, n_feat, scale_unetfeats, scale_orsnetfeats, num_cab, act, kernel_size, reduction, bias)

    def forward(self, x):
        outs = []
        out, H, O = self.stage_in(x)
        outs.append(out)
        for stage in self.stage_mid:
            out, H, O = stage(x, H, O)
            outs.append(out)
        out = self.stage_osr(x, H, O)
        outs.append(out)
        return outs