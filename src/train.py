import torch

"""
This file contains all the function used to train or validate the model for one epoch
"""

alpha = 0.5     #Normalization parameter for the Dice Loss

#Training function
def train_one_epoch(model, dataloader, optimizer, criterion_ce, criterion_dice, device):

    model.train()       #Put the model in training mode
    running_loss = 0.0  #Total loss equal to 0
    running_ce = 0.0    #CE loss
    running_dice = 0.0  #DL loss

    #Training loop
    for images, masks in dataloader:

        images = images.to(device)                      #Setup the image on the definend device
        masks = masks.to(device).squeeze(1).long()      #[B, H, W]

        outputs = model(images)                         #Create a prediction
        loss_ce = criterion_ce(outputs, masks)          #Obtain the CE loss
        loss_dice = criterion_dice(outputs, masks)      #Obtain the DL loss

        total_loss = loss_ce + (alpha * loss_dice)      #Combination of CE loss and DL loss

        optimizer.zero_grad()                           #Set to zero the gradient
        total_loss.backward()                           #Backpropagation
        optimizer.step()                                #One step for the optimizer

        #Incrementing the loss
        running_loss += total_loss.item()   
        running_ce += loss_ce.item()
        running_dice += loss_dice.item()

    #Calculate the mean loss
    epoch_loss = running_loss / len(dataloader)
    epoch_ce = running_ce / len(dataloader)
    epoch_dice = running_dice / len(dataloader)

    return epoch_loss, epoch_ce, epoch_dice

#Validation function
def validate(model, dataloader, criterion_ce, criterion_dice, device):

    model.eval()   #Put the model in evaluation mode

    running_loss = 0.0
    running_ce = 0.0
    running_dice = 0.0

    with torch.no_grad():   #Power off the gradient

        #Validation loop
        for images, masks in dataloader:

            images = images.to(device)                      #Setup the image on the definend device
            masks = masks.to(device).squeeze(1).long()      #[B, H, W]

            outputs = model(images)                         #Create a prediction

            loss_ce = criterion_ce(outputs, masks)          #Obtain the CE loss
            loss_dice = criterion_dice(outputs, masks)      #Obtain the DL loss

            total_loss = loss_ce + (alpha * loss_dice)      #Combination of CE loss and DL loss

            #Incrementing the loss
            running_loss += total_loss.item()
            running_ce += loss_ce.item()
            running_dice += loss_dice.item()

    #Calculate the mean loss
    epoch_loss = running_loss / len(dataloader)
    epoch_ce = running_ce / len(dataloader)
    epoch_dice = running_dice / len(dataloader)

    return epoch_loss, epoch_ce, epoch_dice



