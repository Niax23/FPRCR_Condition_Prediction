import torch
import torch.nn as nn
import torch.nn.functional as F


class FPRCR_Model(nn.Module):
    def __init__(self, catalyst_classes, solvent1_classes, solvent2_classes, reagent1_classes, reagent2_classes):
        super(FPRCR_Model, self).__init__()

        self.fp_layers = nn.Sequential(
            nn.Linear(16384 * 2, 1000),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1000, 1000),
            nn.ReLU()
        )

        self.catalyst_layers = nn.Sequential(
            nn.Linear(1000, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.Tanh(),
            nn.Linear(300, catalyst_classes)
        )

        self.cat2sov = nn.Sequential(
            nn.Linear(catalyst_classes, 100),
            nn.ReLU()
        )

        self.solvent1_layers = nn.Sequential(
            nn.Linear(1100, 300), 
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.Tanh(),
            nn.Linear(300, solvent1_classes)
        )

        self.sov2sov = nn.Sequential(
            nn.Linear(solvent1_classes, 100),
            nn.ReLU()
        )

        self.solvent2_layers = nn.Sequential(
            nn.Linear(1200, 300),  
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.Tanh(),
            nn.Linear(300, solvent2_classes)
        )

        self.sov2reg = nn.Sequential(
            nn.Linear(solvent2_classes, 100),
            nn.ReLU()
        )

        self.reagent1_layers = nn.Sequential(
            nn.Linear(1300, 300),  
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.Tanh(),
            nn.Linear(300, reagent1_classes)
        )

        self.reg2reg = nn.Sequential(
            nn.Linear(reagent1_classes, 100),
            nn.ReLU()
        )

        self.reagent2_layers = nn.Sequential(
            nn.Linear(1400, 300),  
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.Tanh(),
            nn.Linear(300, reagent2_classes)
        )

    def forward(self, fp):
        # Concatenate product and reaction fingerprints
        # print(reaction_fp.shape)
        #print("here")
        #print(fp.dtype)

        dense_fp = self.fp_layers(fp)

        catalyst_pred = self.catalyst_layers(dense_fp)

        cat_one_hot = F.one_hot(torch.argmax(
            catalyst_pred, dim=1), num_classes=catalyst_pred.shape[1]).float()
        #print(catalyst_pred)
        #print(cat_one_hot)

        # Solvent 1 prediction
        solvent1_input = torch.cat(
            (dense_fp, self.cat2sov(cat_one_hot)), dim=1)

        solvent1_pred = self.solvent1_layers(solvent1_input)

        sov1_one_hot = F.one_hot(torch.argmax(
            solvent1_pred, dim=1), num_classes=solvent1_pred.shape[1]).float()

        # Solvent 2 prediction
        solvent2_input = torch.cat(
            (solvent1_input, self.sov2sov(sov1_one_hot)), dim=1)
        solvent2_pred = self.solvent2_layers(solvent2_input)

        sov2_one_hot = F.one_hot(torch.argmax(
            solvent2_pred, dim=1), num_classes=solvent2_pred.shape[1]).float()

        # Reagent 1 prediction
        reagent1_input = torch.cat(
            (solvent2_input, self.sov2reg(sov2_one_hot)), dim=1)
        reagent1_pred = self.reagent1_layers(reagent1_input)

        reg1_one_hot = F.one_hot(torch.argmax(
            reagent1_pred, dim=1), num_classes=reagent1_pred.shape[1]).float()

        # Reagent 2 prediction
        reagent2_input = torch.cat(
            (reagent1_input, self.reg2reg(reg1_one_hot)), dim=1)
        reagent2_pred = self.reagent2_layers(reagent2_input)

        return catalyst_pred, solvent1_pred, solvent2_pred, reagent1_pred, reagent2_pred
