with open('Colab_Code_GAN_Only.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Replace classes
old_classes = """class ClassifierModel(nn.Module):
    def __init__(self, num_classes=1):
        super(ClassifierModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4), 
            nn.Linear(128, 64), nn.ReLU(True),
            nn.Dropout(0.4), 
            nn.Linear(64, num_classes), nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.features(x).view(x.size(0), -1)
        return self.classifier(x)

class GeneratorModule(nn.Module):
    def __init__(self, latent_dim=128):
        super(GeneratorModule, self).__init__()
        self.fc = nn.Linear(latent_dim, 128 * 8 * 8)
        self.convs = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1), nn.Tanh()
        )
        
    def forward(self, z):
        h = self.fc(z).view(-1, 128, 8, 8)
        return self.convs(h)

class DiscriminatorModule(nn.Module):
    def __init__(self):
        super(DiscriminatorModule, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
        )
        self.out = nn.Sequential(nn.Linear(128 * 8 * 8, 1), nn.Sigmoid())
        
    def forward(self, x):
        h = self.convs(x).view(-1, 128 * 8 * 8)
        return self.out(h)

class GANOnly(nn.Module):
    def __init__(self, latent_dim=128):
        super(GANOnly, self).__init__()
        self.generator = GeneratorModule(latent_dim)
        self.discriminator = DiscriminatorModule()
        self.classifier = ClassifierModel()

    def forward(self, z):
        recon = self.generator(z)
        pred = self.classifier(recon)
        validity = self.discriminator(recon)
        return recon, pred, validity"""

new_classes = """class ClassifierModel(nn.Module):
    def __init__(self, input_dim=256, num_classes=1):
        super(ClassifierModel, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.classifier(x)

class GeneratorModule(nn.Module):
    def __init__(self, latent_dim=128):
        super(GeneratorModule, self).__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.convs = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1), nn.Tanh()
        )
        
    def forward(self, z):
        h = self.fc(z).view(-1, 256, 4, 4)
        return self.convs(h)

class DiscriminatorModule(nn.Module):
    def __init__(self):
        super(DiscriminatorModule, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
        )
        self.out = nn.Sequential(nn.Linear(256 * 4 * 4, 1), nn.Sigmoid())
        
    def forward(self, x):
        conv_out = self.convs(x)
        h = conv_out.view(-1, 256 * 4 * 4)
        return self.out(h), conv_out

class GANOnly(nn.Module):
    def __init__(self, latent_dim=128):
        super(GANOnly, self).__init__()
        self.generator = GeneratorModule(latent_dim)
        self.discriminator = DiscriminatorModule()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = ClassifierModel(256)

    def forward(self, x=None, z=None):
        if z is not None:
            recon = self.generator(z)
            validity, conv_out = self.discriminator(recon)
            pooled = self.pool(conv_out).view(conv_out.size(0), -1)
            pred = self.classifier(pooled)
            return recon, pred, validity
        elif x is not None:
            validity, conv_out = self.discriminator(x)
            pooled = self.pool(conv_out).view(conv_out.size(0), -1)
            pred = self.classifier(pooled)
            return pred, validity
        return None"""

code = code.replace(old_classes, new_classes)

old_train1 = """        opt_gan_d.zero_grad()
        with torch.no_grad():
            fakes, _, _ = gan_model(z)
        
        d_loss = (bce_loss(gan_model.discriminator(imgs), real_labels) + 
                  bce_loss(gan_model.discriminator(fakes.detach()), fake_labels)) / 2
        d_loss.backward()
        opt_gan_d.step()

        opt_gan_g.zero_grad()
        fakes, _, _ = gan_model(z)
        g_adv_loss = bce_loss(gan_model.discriminator(fakes), real_labels)
        
        # Train classifier on REAL images so it learns actual features
        real_preds = gan_model.classifier(imgs)
        cls_loss = bce_loss(real_preds, labels)"""

new_train1 = """        opt_gan_d.zero_grad()
        with torch.no_grad():
            fakes, _, _ = gan_model(z=z)
        
        real_validity, _ = gan_model.discriminator(imgs)
        fake_validity, _ = gan_model.discriminator(fakes.detach())
        d_loss = (bce_loss(real_validity, real_labels) + 
                  bce_loss(fake_validity, fake_labels)) / 2
        d_loss.backward()
        opt_gan_d.step()

        opt_gan_g.zero_grad()
        fakes, _, _ = gan_model(z=z)
        fake_validity_for_g, _ = gan_model.discriminator(fakes)
        g_adv_loss = bce_loss(fake_validity_for_g, real_labels)
        
        # Train classifier on REAL images so it learns actual features
        real_preds, _ = gan_model(x=imgs)
        cls_loss = bce_loss(real_preds, labels)"""

code = code.replace(old_train1, new_train1)

old_train2 = """            real_preds = gan_model.classifier(imgs)
            t_loss += bce_loss(real_preds, labels).item()"""
new_train2 = """            real_preds, _ = gan_model(x=imgs)
            t_loss += bce_loss(real_preds, labels).item()"""

code = code.replace(old_train2, new_train2)

with open('Colab_Code_GAN_Only.py', 'w', encoding='utf-8') as f:
    f.write(code)
