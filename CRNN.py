
class CRNN(nn.Module):
    def __init__(self, characters_classes, hidden=256, pretrain=True):
        super(CRNN, self).__init__()
        self.characters_class = characters_classes
        self.body = VGG()
        self.stage5 = nn.Conv2d(512, 512, kernel_size=(3, 2), padding=(1, 0))
        self.hidden = hidden
        self.rnn = nn.Sequential(BidirectionalLSTM(512, self.hidden, self.hidden),
                                 BidirectionalLSTM(self.hidden, self.hidden, self.characters_class))

        self.pretrain = pretrain
        if self.pretrain:
            import torchvision.models.vgg as vgg
            pre_net = vgg.vgg16(pretrained=True)
            pretrained_dict = pre_net.state_dict()
            model_dict = self.body.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.body.load_state_dict(model_dict)

            for param in self.body.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.body(x)
        x = self.stage5(x)
        x = x.squeeze(3)
        x = x.permute(2, 0, 1).contiguous()
        x = self.rnn(x)
        x = F.log_softmax(x, dim=2)
        return x
