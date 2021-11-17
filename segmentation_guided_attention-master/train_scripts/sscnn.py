# coding: utf-8
### FIXME: DEPRECATED FILE
## dependencies

import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torch
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy,Loss, RunningAverage
from ignite.contrib.handlers import ProgressBar
from ignite.handlers import ModelCheckpoint

def train(device, net, dataloader, val_loader, args, logger, experiment):
    device = device
    net = net.to(device)

    def update(engine, data):
        input_left, input_right, label = data['left_image'], data['right_image'], data['winner']
        input_left, input_right, label = input_left.to(device), input_right.to(device), label.to(device)
        inverse_label = label.clone()
        label[label==-1] = 0
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(input_left,input_right)

        loss = criterion(outputs, label)

        loss.backward()
        optimizer.step()
        out_loss = loss.item()
        # reverse example
        inverse_label*=-1 #swap label
        inverse_label[inverse_label==-1] = 0
        inverse_outputs = net(input_right,input_left) #pass swapped input
        inverse_loss = criterion(inverse_outputs, inverse_label)
        inverse_loss.backward()
        optimizer.step()

        return  { 'loss':loss.item(), 
                'y':label,
                'y_pred': outputs
                }

    def inference(engine,data):
        with torch.no_grad():
            input_left, input_right, label = data['left_image'], data['right_image'], data['winner']
            input_left, input_right, label = input_left.to(device), input_right.to(device), label.to(device)
            
            label[label==-1] = 0
            
            # forward
            outputs = net(input_left,input_right)
            loss = criterion(outputs, label)
        return  { 'loss':loss.item(), 
                'y':label,
                'y_pred': outputs
                }

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)

    trainer = Engine(update)
    evaluator = Engine(inference)

    RunningAverage(output_transform=lambda x: x['loss']).attach(trainer, 'loss')
    RunningAverage(Accuracy(output_transform=lambda x: (x['y_pred'],x['y']))).attach(trainer,'avg_acc')

    RunningAverage(output_transform=lambda x: x['loss']).attach(evaluator, 'loss')
    RunningAverage(Accuracy(output_transform=lambda x: (x['y_pred'],x['y']))).attach(evaluator,'avg_acc')


    # pbar = ProgressBar(persist=False)
    # pbar.attach(trainer,['loss','avg_acc'])

    # pbar = ProgressBar(persist=False)
    # pbar.attach(evaluator,['loss','avg_acc'])

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        trainer.state.metrics['val_acc'] = metrics['avg_acc']
        print("Training Results - Epoch: {}  Avg Train accuracy: {:.5f} Avg Train loss: {:.5f}".format(trainer.state.epoch, trainer.state.metrics['avg_acc'], trainer.state.metrics['loss']))
        print("Training Results - Epoch: {}  Avg Val accuracy: {:.5f} Avg Val loss: {:.5f}".format(trainer.state.epoch, metrics['avg_acc'], metrics['loss']))

    handler = ModelCheckpoint(args.model_dir, '{}_{}_{}'.format(args.model, args.premodel, args.attribute),
                                n_saved=1,
                                create_dir=True,
                                save_as_state_dict=True,
                                require_empty=False,
                                score_function=lambda engine: engine.state.metrics['val_acc'])
    trainer.add_event_handler(Events.EPOCH_COMPLETED, handler, {
                'model': net
                })

    if (args.resume):

        def start_epoch(engine):
            engine.state.epoch = args.epoch
        trainer.add_event_handler(Events.STARTED, start_epoch)
        evaluator.add_event_handler(Events.STARTED, start_epoch)
        
    trainer.run(dataloader,max_epochs=args.max_epochs)

def test(device, net, dataloader, args):
    device = device
    net = net.to(device)

    def inference(engine,data):
        with torch.no_grad():
            input_left, input_right, label = data['left_image'], data['right_image'], data['winner']
            input_left, input_right, label = input_left.to(device), input_right.to(device), label.to(device)
            
            label[label==-1] = 0
            
            # forward
            outputs = net(input_left,input_right)
        return  {
                'y':label,
                'y_pred': outputs
                }
    tester = Engine(inference)

    RunningAverage(Accuracy(output_transform=lambda x: (x['y_pred'],x['y']))).attach(tester,'avg_acc')

    pbar = ProgressBar(persist=False)
    pbar.attach(tester,['avg_acc'])

    @tester.on(Events.EPOCH_COMPLETED)
    def log_validation_results(tester):
        metrics = tester.state.metrics
        print("Test Results - Epoch: {}  Avg Val accuracy: {:.5f}".format(args.epoch, metrics['avg_acc']))
        
    tester.run(dataloader,max_epochs=1)

if __name__ == '__main__':
    net = SsCnn(models.alexnet)
    x = torch.randn([3,224,224]).unsqueeze(0)
    fwd =  net(x,x)
    print(fwd.size())