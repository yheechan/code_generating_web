import torch
import os

def graphModel(dataloader, model, writer, device):
    # Again, grab a single mini-batch of images
    dataiter = iter(dataloader)
    prefix, postfix, label,\
    label_prefix, label_postfix, case = dataiter.next()

    model = model.to(device)

    prefix = prefix.to(device)
    postfix = postfix.to(device)
    label = label.to(device)
    label_prefix = label_prefix.to(device)
    label_postfix = label_postfix.to(device)
    case = case.to(device)

    # add_graph() will trace the sample input through your model,
    # and render it as a graph.
    writer.add_graph(model, (prefix, postfix, label))
    writer.flush()

    print('uploaded model graph to tensorboard!')

def saveModel(fn, project_nm, model):

    # os.makedirs('../model/'+fn+'/')

    model.cpu()
    torch.save(model, '../../model/'+project_nm+'.pt')


def getModel(fn, project_nm):
    model = torch.load('../../model/'+project_nm+'.pt')

    return model
