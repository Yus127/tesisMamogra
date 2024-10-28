import lightning as L
from torchmetrics.text import BLEUScore
import pytorch_lightning as pl
import torch
from torch import optim
#def my_custom_loss()

class LightningBiomedCLIP(pl.LightningModule):
    def __init__(self, model):
        super(LightningBiomedCLIP, self).__init__()
        self.model = model
        #self.loss = loss
        self.bleu = BLEUScore()
        #self.optimizer = optimizer
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        #print("batch")
        #print(batch)
        x, y = batch['image'], batch['text']

        #print("x")
        #print(x)
        """
        y_hat = self.model(x)
        #scores = self.forward(x)
        #loss = self.loss_fn(scores, y)
        loss = self.loss(y_hat, y)
        bleu_score= self.bleu(y_hat,y)
        # Logging to TensorBoard by default
        self.log({"train_loss": loss, "bleu_score": bleu_score})
        return loss
        """
        image_features, text_features, logit_scale = self.model(x, y)
        
        # Calculate similarity
        logits = image_features @ text_features.T
        
        # Labels for contrastive learning (diagonal is positive pairs)
        labels = torch.arange(len(x), device=self.device)
        #loss_idk = self.bleu(logits,labels)
        #print(loss_idk)
        # Calculate loss both ways (image->text and text->image)
        loss_i2t = self.criterion(logits, labels)
        loss_t2i = self.criterion(logits.T, labels)
        loss = (loss_i2t + loss_t2i) / 2
        print("Loss")
        print(loss)
        
        self.log('train_loss', loss)
        return loss
    
    

    def configure_optimizers(self):
        #return self.optimizer
        optimizer = optim.AdamW(self.parameters(), lr=1e-4)
        return optimizer
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch['image'], val_batch['text']
        
