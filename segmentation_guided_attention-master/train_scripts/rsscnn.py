import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
from torch.autograd import Variable
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy,Loss, RunningAverage
from ignite.contrib.handlers import ProgressBar
from ignite.handlers import ModelCheckpoint
from tensorboardX import SummaryWriter
from timeit import default_timer as timer
from utils.ranking import compute_ranking_loss, compute_ranking_accuracy
from utils.log import tb_log

def train(device, net, dataloader, val_loader, args, logger, experiment):
    def update(engine, data):
        input_left, input_right, label = data['left_image'], data['right_image'], data['winner']
        input_left, input_right, label = input_left.to(device), input_right.to(device), label.to(device)
        rank_label = label.clone()
        inverse_label = label.clone()
        label[label==-1] = 0
        # zero the parameter gradients
        optimizer.zero_grad()
        rank_label = rank_label.float()

        start = timer()
        output_clf,output_rank_left, output_rank_right = net(input_left,input_right)
        end = timer()
        logger.info(f'FORWARD,{end-start:.4f}')

        #compute clf loss
        start = timer()
        loss_clf = clf_crit(output_clf,label)

        #compute ranking loss
        loss_rank = compute_ranking_loss(output_rank_left, output_rank_right, label, rank_crit)
        loss = loss_clf + loss_rank

        end = timer()
        logger.info(f'LOSS,{end-start:.4f}')

        #compute ranking accuracy
        start = timer()
        rank_acc = compute_ranking_accuracy(output_rank_left, output_rank_right, label)
        end = timer()
        logger.info(f'RANK-ACC,{end-start:.4f}')

        # backward step
        start = timer()
        loss.backward()
        optimizer.step()
        end = timer()
        logger.info(f'BACKWARD,{end-start:.4f}')

        #swapped forward
        start = timer()
        inverse_label*=-1 #swap label
        inverse_rank_label = inverse_label.clone()
        inverse_rank_label = inverse_rank_label.float()
        inverse_label[inverse_label==-1] = 0
        end = timer()
        logger.info(f'SWAPPED-SETUP,{end-start:.4f}')
        start = timer()
        outputs, output_rank_left, output_rank_right = net(input_right,input_left) #pass swapped input
        end = timer()
        logger.info(f'SWAPPED-FORWARD,{end-start:.4f}')
        start = timer()
        inverse_loss_clf = clf_crit(outputs, inverse_label)
        #compute ranking loss
        inverse_loss_rank = compute_ranking_loss(output_rank_left, output_rank_right, label, rank_crit)
        #swapped backward
        inverse_loss = inverse_loss_clf + inverse_loss_rank
        end = timer()
        logger.info(f'SWAPPED-LOSS,{end-start:.4f}')
        start = timer()
        inverse_loss.backward()
        optimizer.step()
        end = timer()
        logger.info(f'SWAPPED-BACKWARD,{end-start:.4f}')

        return  { 'loss':loss.item(),
                'loss_clf':loss_clf.item(),
                'loss_rank':loss_rank.item(),
                'y':label,
                'y_pred': output_clf,
                'rank_acc': rank_acc
                }

    def inference(engine,data):
        with torch.no_grad():
            start = timer()
            input_left, input_right, label = data['left_image'], data['right_image'], data['winner']
            input_left, input_right, label = input_left.to(device), input_right.to(device), label.to(device)
            rank_label = label.clone()
            label[label==-1] = 0
            rank_label = rank_label.float()
            # forward
            output_clf,output_rank_left, output_rank_right = net(input_left,input_right)
            loss_clf = clf_crit(output_clf,label)
            loss_rank = compute_ranking_loss(output_rank_left, output_rank_right, label, rank_crit)
            rank_acc = compute_ranking_accuracy(output_rank_left, output_rank_right, label)
            loss = loss_clf + loss_rank
            end = timer()
            logger.info(f'INFERENCE,{end-start:.4f}')
            return  { 'loss':loss.item(),
                'loss_clf':loss_clf.item(),
                'loss_rank':loss_rank.item(),
                'y':label,
                'y_pred': output_clf,
                'rank_acc': rank_acc
                }
    net = net.to(device)

    clf_crit = nn.NLLLoss()
    rank_crit = nn.MarginRankingLoss(reduction='mean', margin=1)
    optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)
    lamb = Variable(torch.FloatTensor([1]),requires_grad = False).cuda()[0]

    trainer = Engine(update)
    evaluator = Engine(inference)

    writer = SummaryWriter()
    RunningAverage(output_transform=lambda x: x['loss']).attach(trainer, 'loss')
    RunningAverage(output_transform=lambda x: x['loss_clf']).attach(trainer, 'loss_clf')
    RunningAverage(output_transform=lambda x: x['loss_rank']).attach(trainer, 'loss_rank')
    RunningAverage(output_transform=lambda x: x['rank_acc']).attach(trainer, 'rank_acc')
    RunningAverage(Accuracy(output_transform=lambda x: (x['y_pred'],x['y']))).attach(trainer,'avg_acc')

    RunningAverage(output_transform=lambda x: x['loss']).attach(evaluator, 'loss')
    RunningAverage(output_transform=lambda x: x['loss_clf']).attach(evaluator, 'loss_clf')
    RunningAverage(output_transform=lambda x: x['loss_rank']).attach(evaluator, 'loss_rank')
    RunningAverage(output_transform=lambda x: x['rank_acc']).attach(evaluator, 'rank_acc')
    RunningAverage(Accuracy(output_transform=lambda x: (x['y_pred'],x['y']))).attach(evaluator,'avg_acc')

    if args.pbar:
        pbar = ProgressBar(persist=False)
        pbar.attach(trainer,['loss','avg_acc', 'rank_acc'])

        pbar = ProgressBar(persist=False)
        pbar.attach(evaluator,['loss','loss_clf', 'loss_rank','avg_acc'])

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        net.eval()
        evaluator.run(val_loader)
        trainer.state.metrics['val_acc'] = evaluator.state.metrics['rank_acc']
        net.train()

        tb_log(
            {
                "accuracy":{
                    'accuracy':trainer.state.metrics['avg_acc'],
                    'rank_accuracy':trainer.state.metrics['rank_acc']
                },
                "loss": {
                    'total':trainer.state.metrics['loss'],
                    'clf':trainer.state.metrics['loss_clf'],
                    'rank':trainer.state.metrics['loss_rank']
                }
            },
            {
                "accuracy":{
                    'accuracy':evaluator.state.metrics['avg_acc'],
                    'rank_accuracy':evaluator.state.metrics['rank_acc']
                },
                "loss": {
                    'total':evaluator.state.metrics['loss'],
                    'clf':evaluator.state.metrics['loss_clf'],
                    'rank':evaluator.state.metrics['loss_rank']
                }
            },
            writer,
            args.attribute,
            trainer.state.epoch
        )

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

if __name__ == '__main__':
    from torchviz import make_dot
    net = RSsCnn(models.alexnet)
    x = torch.randn([3,244,244]).unsqueeze(0)
    y = torch.randn([3,244,244]).unsqueeze(0)
    fwd =  net(x,y)
    print(fwd)