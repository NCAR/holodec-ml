import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")



def get_input_optimizer(input_img, lr = 1):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img], lr=lr)
    #optimizer = optim.Adam([input_img], lr = lr)
    return optimizer


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag
        
        
def tv_loss(Y_hat):
    return 0.5 * (torch.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() +
                  torch.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean())


class ContentLoss(nn.Module):
    def __init__(
        self,
        target,
    ):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)
    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
    G = torch.mm(features, features.t())  # compute the gram product
    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


def rename_vgg_layers(model):
    """Renames VGG model layers to match those in the paper."""
    block, number = 1, 1
    renamed = nn.Sequential()
    for i, layer in enumerate(model.children()):
        if i == 0:
            layer = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            name = f"conv{block}_{number}"
        elif isinstance(layer, nn.Conv2d):
            name = f"conv{block}_{number}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu{block}_{number}"
            # The inplace ReLU version doesn't play nicely with NST.
            layer = nn.ReLU(inplace=False)
            number += 1
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool_{block}"
            # Average pooling was found to generate images of higher quality than
            # max pooling by Gatys et al.
            layer = nn.AvgPool2d(layer.kernel_size, layer.stride)
            block += 1
            number = 1
        else:
            raise RuntimeError(f'Unrecognized layer "{layer.__class__.__name__}""')
        renamed.add_module(name, layer)
    return renamed


def get_style_model_and_losses(
    cnn, style_img, content_img, content_layers, style_layers
):
    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []
    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(*[])

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = "conv_{}".format(i)
        elif isinstance(layer, nn.ReLU):
            name = "relu_{}".format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.AvgPool2d):
            name = "pool_{}".format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = "bn_{}".format(i)
        else:
            raise RuntimeError(
                "Unrecognized layer: {}".format(layer.__class__.__name__)
            )

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for ii in range(len(model) - 1, -1, -1):
        if isinstance(model[ii], ContentLoss) or isinstance(model[ii], StyleLoss):
            break

    model = model[: (ii + 1)]

    return model, style_losses, content_losses