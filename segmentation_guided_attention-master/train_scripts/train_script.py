import torch.nn as nn
import torch.optim as optim

from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from ignite.handlers import ModelCheckpoint
from random import randint

from utils.ranking import *
from utils.log import console_log, comet_log
from utils.accuracy import RankAccuracy

def train(device, net, dataloader, val_loader, args, logger, experiment):
    def update(engine, data):
        # Load training sample
        input_left, input_right, label, left_original = data['left_image'], data['right_image'], data['winner'], data['left_image_original']
        input_left, input_right, label = input_left.to(device), input_right.to(device), label.to(device)
        attribute = data['attribute'].to(device)
        label = label.float()

        # zero the parameter gradients
        optimizer.zero_grad()

        # Forward the training sample
        forward_dict = net(input_left,input_right)
        output_rank_left, output_rank_right =  forward_dict['left']['output'], forward_dict['right']['output']

        if args.attribute == 'all':
            loss = compute_multiple_ranking_loss(output_rank_left, output_rank_right, label, rank_crit, attribute)
        else:
            loss = compute_ranking_loss(output_rank_left, output_rank_right, label, rank_crit)

        # Backward step
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        return  { 'loss':loss.item(),
                'rank_left': output_rank_left,
                'rank_right': output_rank_right,
                'label': label
                }

    def inference(engine,data):
        with torch.no_grad():
            # Load training sample
            input_left, input_right, label, left_original = data['left_image'], data['right_image'], data['winner'], data['left_image_original']
            input_left, input_right, label = input_left.to(device), input_right.to(device), label.to(device)
            attribute = data['attribute'].to(device)
            label = label.float()

            # Forward the training sample
            forward_dict = net(input_left,input_right)
            output_rank_left, output_rank_right =  forward_dict['left']['output'], forward_dict['right']['output']
            if args.attribute == 'all':
                loss = compute_multiple_ranking_loss(output_rank_left, output_rank_right, label, rank_crit, attribute)
            else:
                loss = compute_ranking_loss(output_rank_left, output_rank_right, label, rank_crit)

            return  { 'loss':loss.item(),
                'rank_left': output_rank_left,
                'rank_right': output_rank_right,
                'label': label
                }

    # Define model, loss, optimizer and scheduler
    net = net.to(device)
    rank_crit = nn.MarginRankingLoss(reduction='mean', margin=1)
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0, betas=(0.9, 0.98), eps=1e-09)
    if args.lr_decay:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.995, last_epoch=-1)
    else:
        scheduler = None

    # Engine specific parameters
    trainer = Engine(update)
    evaluator = Engine(inference)

    RunningAverage(output_transform=lambda x: x['loss'], device=device).attach(trainer, 'loss')
    RankAccuracy(output_transform=lambda x: (x['rank_left'], x['rank_right'], x['label']), device=device).attach(trainer,'acc')

    RunningAverage(output_transform=lambda x: x['loss'], device=device).attach(evaluator, 'loss')
    RankAccuracy(output_transform=lambda x: (x['rank_left'], x['rank_right'], x['label']),device=device).attach(evaluator,'acc')

    # Log training parameters after every epoch
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        net.eval()
        evaluator.run(val_loader)
        trainer.state.metrics['val_acc'] = evaluator.state.metrics['acc']
        net.train()
        if hasattr(net,'partial_eval'): net.partial_eval()
        metrics = {
                'train_loss':trainer.state.metrics['loss'],
                'acc': trainer.state.metrics['acc'],
                'val_acc': evaluator.state.metrics['acc'],
                'val_loss':evaluator.state.metrics['loss']
            }
        comet_log(
            metrics,
            experiment,
            epoch=trainer.state.epoch,
            step=trainer.state.epoch,
        )
        console_log(metrics,{},trainer.state.epoch)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_results(trainer):
        # Log training every 100th iteration
        if trainer.state.iteration %100 == 0:
            metrics = {
                    'train_loss':trainer.state.metrics['loss'],
                }
            comet_log(
                metrics,
                experiment,
                step=trainer.state.iteration,
                epoch=trainer.state.epoch
            )
            console_log(
                metrics,
                {},
                trainer.state.epoch,
                step=trainer.state.iteration,
            )

    model_name = '{}_{}_{}'.format(args.model, args.premodel, args.attribute)
    handler = ModelCheckpoint(args.model_dir, model_name,
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

    trainer.run(dataloader,max_epochs=args.max_epochs, seed=randint(1,15))

if __name__ == '__main__':
    net = SegRank(image_size=(244,244))
    x = torch.randn([3,244,244]).unsqueeze(0)
    y = torch.randn([3,244,244]).unsqueeze(0)
    fwd =  net(x,y)
    print(fwd)
