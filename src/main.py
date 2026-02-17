import torch
from .utils.train_utils import get_loss, get_optimizer, save_checkpoint, DiceLoss
from .utils.dataloaders import get_Train_Val_loader_split
from .models.Unet import U_Net  # importing the U Net model
from .train import train_one_epoch, validate
from tqdm.auto import tqdm  # import tqdm for progress bar
import time # import library for time counting
import os
"""
ATTENZIONE PROVA A INTRODURRE
Se vuoi spingere oltre, 
puoi provare learning rate scheduler, 
così la rete continua a migliorare senza overfittare
"""


def main():

    #Agnostic code
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    #Start timer
    start = time.time()

    NUM_CLASS = 21  #Total class
    NUM_EPOCHS = 1  #Numbers of training epochs 
    BATCH_SIZE = 8  #Size of the batch

    #Dataloaders definition
    train_loader, val_loader = get_Train_Val_loader_split(BATCH_SIZE=BATCH_SIZE)

    #U-Net model definition
    model = U_Net(NUM_CLASS).to(device)

    #Loss e Optimizer
    criterion_ce = get_loss()       #CrossEntropy loss
    criterion_dice = DiceLoss()     #DiceLoss
    optimizer = get_optimizer(model.parameters())   #AdamW

    #Used when the model starts from zero epochs
    #start_epoch = 0

    base_dir = os.path.dirname(os.path.abspath(__file__))
    CHECKPOINT_PATH = os.path.join(base_dir,"evaluation","metrics", "unet_checkpoint50.pth")

    #"""
    #This snippet is used to load a pretrained model, optimizer and the epochs of training
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"]  # riparti dall’epoca successiva
    #"""

    #Training loop
    for epoch in tqdm(range(start_epoch, start_epoch + NUM_EPOCHS)):

        print(f"Epoch: {epoch}\n-------")
        print("TRAINING")
        #Call a function that train the model for one epoch and return the loss
        train_loss, loss_ce_train, loss_dice_train = train_one_epoch(model, train_loader, optimizer, criterion_ce, criterion_dice, device)
        
        print("VALIDATING")
        #Call a function that validate the model for one epoch 
        val_loss, loss_ce_val, loss_dice_val = validate(model, val_loader,criterion_ce, criterion_dice, device) #Validation dataset

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
              f"Train Loss: {train_loss:.4f} "
              f"Cross Entropy loss: {loss_ce_train:.4f} "
              f"Dice loss: {loss_dice_train:.4f}\n")
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
              f"Validation Loss: {val_loss:.4f} "
              f"Cross Entropy loss: {loss_ce_val:.4f} "
              f"Dice loss: {loss_dice_val:.4f}\n")
        
        #Call a function to save the state of the model
        save_checkpoint(model, optimizer, epoch + 1, path="evaluation/metrics/unet_checkpoint55.pth")

    #Stop timer
    end = time.time()

    #Print the total time for epoch
    print("Time for epoch:", end - start)

if __name__ == "__main__":
    print("Benvenuto nel progetto U-Net!")
    print("Scegli un'opzione:")
    print("1 - Allenamento")
    print("2 - Valutazione modello")
    print("3 - Predizione su immagine singola")

    scelta = input("Inserisci il numero dell'opzione: ").strip()
    if scelta == "1":
        main()  # qui richiami la tua funzione di training
    elif scelta == "2":
        from .evaluation.evaluate import evaluation
        evaluation()  # funzione di evaluation che hai già scritto
    elif scelta == "3":
        from .evaluation.testing import testing
        testing()  # funzione per predire singole immagini
    else:
        print("Opzione non valida. Chiudo il programma.")
    