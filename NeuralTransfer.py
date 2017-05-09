import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
import copy
dtype = torch.cuda.FloatTensor
cnn = models.vgg19(pretrained=True).features.cuda()
imsize = 512

loader = transforms.Compose([transforms.Scale(imsize),
                             transforms.ToTensor()])
unloader = transforms.ToPILImage()

def image_loader(image):
    image = Image.open(image)
    image = image.resize((512,512))
    image = Variable(loader(image))
    return image.unsqueeze(0)

def imshow(img, title):
    image = img.clone().cpu()
    image = image.squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    plt.title(title)
    plt.pause(0.1)
    
style_image = image_loader('/home/kevin/Downloads/style.jpg').type(dtype)
content_image = image_loader('/home/kevin/Downloads/content.jpg').type(dtype)
assert style_image.size() == content_image.size()

plt.figure()
imshow(style_image.data, 'style')
plt.figure()
imshow(content_image.data, 'content')

class ContentLoss(nn.Module):
    def __init__(self, target, weight):
        super(ContentLoss, self).__init__()
        self.target = weight * Variable(target.data, volatile=False)
        self.weight = weight
        self.loss_func = nn.MSELoss()

    def forward(self, input):
        self.loss = self.loss_func(self.weight * input, self.target)
        self.output = input
        return self.output

    def backward(self):
        self.loss.backward(retain_variables=True)
        return self.loss

class GramMatrix(nn.Module):
    def forward(self, input):
        a,b,c,d = input.size()
        features = input.view(a*b, c*d)
        G = torch.mm(features, features.t())
        return G.div(c*d)

class StyleLoss(nn.Module):
    def __init__(self, target, weight):
        super(StyleLoss, self).__init__()
        self.target = weight * Variable(target.data, volatile=False)
        self.weight = weight
        self.gram = GramMatrix()
        self.loss_func = nn.MSELoss()

    def forward(self, input):
        self.output = input.clone()
        self.G = self.gram(input)
        self.G.mul_(self.weight)
        self.loss = self.loss_func(self.G, self.target)
        return self.output

    def backward(self):
        self.loss.backward(retain_variables=True)
        return self.loss

content_layers = ['conv_8']
style_layers = ['conv_1', 'conv_3', 'conv_6', 'conv_8', 'conv_11']

def get_model_and_losses(cnn, style_image, content_image,
                         style_weight=1000, content_weight=1,
                         content_layers=content_layers,
                         style_layers=style_layers):
    cnn = copy.deepcopy(cnn)
    content_losses = []
    style_losses = []
    model = nn.Sequential().cuda()
    gram = GramMatrix().cuda()
    i = 1
    
    for layer in list(cnn):
        if isinstance(layer, nn.Conv2d):
            name = 'conv_'+str(i)
            print(name)
            model.add_module(name, layer)

            if name in content_layers:
                target = model(content_image).clone()
                content_loss = ContentLoss(target, content_weight)
                model.add_module('content_loss_'+str(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                target_feature = model(style_image).clone()
                target_gram = gram(target_feature)
                style_loss = StyleLoss(target_gram, style_weight)
                model.add_module('style_loss_'+str(i), style_loss)
                style_losses.append(style_loss)

        if isinstance(layer, nn.ReLU):
            name = 'relu_'+str(i)
            model.add_module(name, layer)

            if name in content_layers:
                print 'check'
                target = model(content_image).clone()
                content_loss = ContentLoss(target, content_weight)
                model.add_module("content_loss_" + str(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                target_feature = model(style_image).clone()
                target_feature_gram = gram(target_feature)
                style_loss = StyleLoss(target_feature_gram, style_weight)
                model.add_module("style_loss_" + str(i), style_loss)
                style_losses.append(style_loss)

        if isinstance(layer, nn.MaxPool2d):
            name = 'pool_'+str(i)
            newPool = nn.AvgPool2d(kernel_size=layer.kernel_size,
                                   stride=layer.stride, padding = layer.padding)
            model.add_module(name, newPool)

        i += 1
        
    return model, style_losses, content_losses    

input_image = content_image.clone()
#plt.figure()
#imshow(input_image.data, 'input')

def get_input_optimizer(input_image):
    input_param = nn.Parameter(input_image.data)
    optimizer = optim.LBFGS([input_param])
    return input_param, optimizer

def run(cnn, content_image, style_image, input_image, steps=300,
        style_weight=150, content_weight=1):

    print '-----Building the style transfer model-----'
    model, style_losses, content_losses = get_model_and_losses(
        cnn, style_image, content_image, style_weight, content_weight)
    input_param, optimizer = get_input_optimizer(input_image)

    print '-----------------Running-------------------'
    it = [0]
    while it[0] < steps:
        def calc():
            input_param.data.clamp_(0,1)
            optimizer.zero_grad()
            model(input_param)
            style_score = 0
            content_score = 0
            for sl in style_losses:
                style_score += sl.backward()
            for cl in content_losses:
                content_score += cl.backward()

            it[0] += 1
            if it[0] % 20 == 0:
                print 'iteration ', it[0]
                print 'style loss: ', style_score.data[0], '    content loss: ', content_score.data[0]
                print '-------------------------------------------'
            return style_score + style_score

        optimizer.step(calc)
        
    return input_param.data.clamp_(0,1)

output = run(cnn, content_image, style_image, input_image)
plt.figure()
imshow(output, 'output')
plt.savefig('output.png')
Image.open('output.png').save('output.jpg','JPEG')
plt.ioff()
plt.show()
