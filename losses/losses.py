import torch
import torch.nn as nn
import torchvision


EPS = 1e-7


class VGGPerceptualLoss(nn.Module):
    """https://github.com/elliottwu/unsup3d/blob/master/unsup3d/networks.py"""
    def __init__(self, requires_grad=False):
        super(VGGPerceptualLoss, self).__init__()
        mean_rgb = torch.FloatTensor([0.485, 0.456, 0.406])
        std_rgb = torch.FloatTensor([0.229, 0.224, 0.225])
        self.register_buffer('mean_rgb', mean_rgb)
        self.register_buffer('std_rgb', std_rgb)

        vgg_pretrained_features = torchvision.models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def normalize(self, x):
        out = x/2 + 0.5
        out = (out - self.mean_rgb.view(1,3,1,1)) / self.std_rgb.view(1,3,1,1)
        return out

    def __call__(self, im1, im2, mask=None, conf_sigma=None):
        im = torch.cat([im1,im2], 0)
        im = self.normalize(im)  # normalize input

        ## compute features
        feats = []
        f = self.slice1(im)
        feats += [torch.chunk(f, 2, dim=0)]
        f = self.slice2(f)
        feats += [torch.chunk(f, 2, dim=0)]
        f = self.slice3(f)
        feats += [torch.chunk(f, 2, dim=0)]
        f = self.slice4(f)
        feats += [torch.chunk(f, 2, dim=0)]

        losses = []
        for f1, f2 in feats[2:3]:  # use relu3_3 features only
            loss = (f1-f2)**2
            if conf_sigma is not None:
                loss = loss / (2*conf_sigma**2 +EPS) + (conf_sigma +EPS).log()
            if mask is not None:
                b, c, h, w = loss.shape
                _, _, hm, wm = mask.shape
                sh, sw = hm//h, wm//w
                mask0 = nn.functional.avg_pool2d(mask, kernel_size=(sh,sw), stride=(sh,sw)).expand_as(loss)
                loss = (loss * mask0).sum() / mask0.sum()
            else:
                loss = loss.mean()
            losses += [loss]
        return sum(losses)


def kld_loss(mu, var):
    log_var = torch.log(var)
    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 0), dim = 0)
    return kld_loss