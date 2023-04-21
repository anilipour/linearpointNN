import numpy as np
from astropy.table import Table
from astropy.io import ascii


import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

from readTables import txt_to_table, txt_to_table_fiducial


def dataPullMask(inputStart, inputEnd, oBoMcutoff = 0.105, nsCutoff=1.11, binning=True, nBins=3):
    scaleTableFile = ascii.read(f'scaleTable.dat')  
    scaleTable = Table(scaleTableFile)
    
    
    simulation = 'latinHypercubeParameters.txt'
    param_list = []
    for line in open(simulation, 'r'): # get each line
        item = line.rstrip() # strip off newline and any other trailing whitespace
        param_list.append(item)

    del param_list[0]

    omegaMlist, omegaBlist, hList, nsList, s8list, cfList = [], [], [], [], [], []
    i = 0
    for item in param_list: # get radius and correlation from each line
        omegaM, omegaB, h, ns, s8 = item.split() # each line has both radius and correlation, so split
        omegaMlist.append(float(omegaM))
        omegaBlist.append(float(omegaB))
        hList.append(float(h))
        nsList.append(float(ns))
        s8list.append(float(s8))
        cfList.append(i)
        i += 1

        param_table = Table([cfList, omegaMlist, omegaBlist, hList, nsList, s8list],
                            names=('CF', 'omegaM', 'omegaB', 'h', 'ns', 's8')) # astropy table

    # Keep the two "bad" CFs separate, then delete them
    param244 = Table(param_table[244])
    param763 = Table(param_table[763])

    param_table.remove_rows([244, 763])
    
    
    
    inputs_xiTable = []
    inputs_rTable = []
    lpsTable = []
    
    # get data
    for row in param_table: # all simulations with scales
        x, oBoM, ns, s8, h = row['CF'], row['omegaB']/row['omegaM'], row['ns'], row['s8'], row['h']
        lpMask = scaleTable['CF'] == x
        lp = scaleTable['lp'][lpMask][0]
        
        #if oBoM > oBoMcutoff or s8 > 0.95 or ns < nsCutoff or h < 0.55:   
        if lp > 80 and lp < 115:
            inputTable = txt_to_table(x, 0)
            if binning:
                inputTable = binCorr(inputTable, nBins)

            inputTable = inputTable[(inputTable['r'] <= inputEnd) & (inputTable['r'] >= inputStart)]
            inputs_xiTable.append(inputTable['xi'])                        
            inputs_rTable.append(inputTable['r'])

            # get linear point
            lpsTable.append(lp)
    
    inputs_xi, inputs_r, lps = np.array(inputs_xiTable), np.array(inputs_rTable), np.array(lpsTable)
    return inputs_r, inputs_xi, lps        


def binCorr(corrTable, binSize):
    rad = corrTable['r']
    xi = corrTable['xi']
    
    rSpacing = np.diff(rad)
    
    npoints = binSize + 1 # binSize is the width of each bin, and there is one more point than bins
    
    radBins = rad[0:-1:binSize]
    originalBinWidths = 0.5*np.diff(rad)[0:-1:binSize]
    if len(radBins) > len(originalBinWidths):
        radBinBounds = radBins[:-1] + 0.5*np.diff(rad)[0:-1:binSize] # new bin boundaries
    else:
        radBinBounds = radBins + 0.5*np.diff(rad)[0:-1:binSize] # new bin boundaries

    newRad = radBinBounds[:-1] + np.diff(radBinBounds)/2. # new bin centers
    binWidths = np.diff(radBinBounds)/2.

    newCorrTable = np.zeros(len(newRad))
    
    for i in range(len(newRad)):
        newCorr = 0
        for j in range(binSize):
            point = binSize*i + j + 1
            spacePoint = binSize*i + j
            pointCorr = ((rad[point] + (rSpacing[spacePoint+1]/2))**3 - (rad[point] - (rSpacing[spacePoint]/2))**3)*xi[point]
            newCorr += pointCorr
        

        newCorr /= ((newRad[i] + binWidths[i])**3 - (newRad[i] - binWidths[i])**3)
        
        newCorrTable[i] = newCorr
    

        
    
    binnedTable = Table([newRad, newCorrTable], names=('r', 'xi')) # astropy table

    return binnedTable


def dataPull(inputStart, inputEnd, binning=True, nBins=3, z=0):
    scaleTableFile = ascii.read(f'scaleTable.dat')  
    scaleTable = Table(scaleTableFile)

    inputs_xiTable = []
    inputs_rTable = []
    lpsTable = []
    
    # get data
    for x in scaleTable['CF']: # all simulations with scales
        # get r, xi
        inputTable = txt_to_table(x, z)
        if binning:
            inputTable = binCorr(inputTable, nBins)
        
        inputTable = inputTable[(inputTable['r'] <= inputEnd) & (inputTable['r'] >= inputStart)]
        inputs_xiTable.append(inputTable['xi'])                        
        inputs_rTable.append(inputTable['r'])
        
        # get linear point
        lp = scaleTable[scaleTable['CF'] == x]['lp'][0]        
        lpsTable.append(lp)
    
    inputs_xi, inputs_r, lps = np.array(inputs_xiTable), np.array(inputs_rTable), np.array(lpsTable)
    return inputs_r, inputs_xi, lps


def meanVar(inputs_xi, lps):
    mean_inputs = np.mean(inputs_xi, 0)
    mean_lps = np.mean(lps)

    var_inputs = np.sqrt(np.var(inputs_xi, 0))
    var_lps = np.sqrt(np.var(lps))
    
    return mean_inputs, mean_lps, var_inputs, var_lps

def setDataloaders(inputs_xi, lps, trainTestSplit=1000, inputNormalize=True):
    # which simulations to train on (rest are for testing)
    train_sample_max = trainTestSplit

    # split into training and testing first
    train_inputs = np.copy(inputs_xi[:train_sample_max])
    train_lps = np.copy(lps[:train_sample_max])
    test_inputs = np.copy(inputs_xi[train_sample_max:])
    test_lps = np.copy(lps[train_sample_max:])
    
    
    # divide all correlation functions by mean across training simulations
    # i.e. for each r, the correlations are divided by the average correlation for that r
    mean_xi, mean_lps, var_xi, var_lps = meanVar(train_inputs, train_lps)

    if inputNormalize:
        train_inputs -= mean_xi
        test_inputs -= mean_xi

        train_inputs /= var_xi
        test_inputs /= var_xi
        
    train_lps -= mean_lps
    test_lps -= mean_lps

    train_lps /= var_lps
    test_lps /= var_lps
    

    tensor_train_x = torch.Tensor(train_inputs) # transform to torch tensor
    tensor_train_y = torch.Tensor(train_lps)

    tensor_test_x = torch.Tensor(test_inputs) # transform to torch tensor
    tensor_test_y = torch.Tensor(test_lps)


    train_dataset = TensorDataset(tensor_train_x,tensor_train_y) # create datset
    train_dataloader = DataLoader(train_dataset, shuffle=True) # create dataloader

    test_dataset = TensorDataset(tensor_test_x, tensor_test_y) # create datset
    test_dataloader = DataLoader(test_dataset, shuffle=True) # create dataloader
    
    return train_dataloader, test_dataloader

def trainBAOReconNN(model, train_dataloader, test_dataloader, lr=1e-5, epochs=1000):
    
    # mean-squared error loss
    criterion = nn.MSELoss()
    
    # create an optimizer object
    # Adam optimizer with learning rate + decay
    optimizer = optim.Adam(model.parameters(), lr=lr)
    decayRate = 0.999
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

    #### Training ####
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        train_loss_tot, test_loss_tot = 0, 0

        #################
        # Training Loop #
        #################
        model.train(True)
        
        # train on random small batches #
        
        r = np.random.randint(1, 10)
        i = 1
        
        train_loss = 0
        optimizer.zero_grad()

        for batch_features, batch_labels in train_dataloader:                
            # compute reconstructions
            model_outputs = model(batch_features)

            # compute training reconstruction loss
            train_loss += criterion(model_outputs, batch_labels)
            
            if np.floor(i/r) == i/r:
                # compute accumulated gradients
                train_loss.backward()

                # perform parameter update based on current gradients
                optimizer.step()
                
                # reset train loss
                train_loss = 0
                
                # reset the gradients back to zero
                optimizer.zero_grad()

            
            i += 1



        
        ################
        # Testing Loop #
        ################
        model.train(False)
        for batch_features, batch_labels in test_dataloader:
            # compute reconstructions
            model_outputs = model(batch_features)

            # compute training reconstruction loss
            test_loss = criterion(model_outputs, batch_labels)

            # add the mini-batch training loss to epoch loss
            test_loss_tot += test_loss.item()
            
        for batch_features, batch_labels in train_dataloader:
            # compute reconstructions
            model_outputs = model(batch_features)

            # compute training reconstruction loss
            train_loss = criterion(model_outputs, batch_labels)

            # add the mini-batch training loss to epoch loss
            train_loss_tot += train_loss.item()

        # compute the epoch training loss
        train_loss_tot = train_loss_tot# / len(train_dataloader)
        test_loss_tot = test_loss_tot# / len(test_dataloader)

        # record losses
        train_losses.append(train_loss_tot)
        test_losses.append(test_loss_tot)

        # learning rate decay step
        lr_scheduler.step()

        # display the epoch losses
        print("epoch : {}/{}, train loss = {:.8f}, test loss = {:.8f}".format(epoch + 1, epochs, train_loss_tot, test_loss_tot))
        
    return model, train_losses, test_losses

class encoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(in_features=kwargs["input_shape"], out_features=kwargs['features']),
            torch.nn.ReLU(),
            torch.nn.Linear(kwargs["features"], 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 1))


    def forward(self, features):
        encoded = self.encoder(features)
        return encoded[0]