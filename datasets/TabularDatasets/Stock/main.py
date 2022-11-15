from argparse import Namespace
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error
import torch.optim as optim
from StockDataset import *
from tqdm import tqdm_notebook
from utils import *
from models import *



def main():
    with open("config.json") as f:
        configs = json.load(f)
    args = Namespace(**configs)
        
    if not torch.cuda.is_available():
        args.cuda = False

    print("Using CUDA: {}".format(args.cuda))

    args.device = torch.device("cuda" if args.cuda else "cpu")

    set_seed_everywhere(args.seed, args.cuda)

    handle_dirs(args.save_dir)
    
    train_dataset = StockDataset(dtype='train')   
    val_dataset = StockDataset(dtype='validation')

    if not args.evaluate:
        model = Model(input_features=1)
        model = model.to(args.device)

        loss_func = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=1)

        train_state = make_train_state(args)

        epoch_bar = tqdm_notebook(desc='training routine',  total=args.num_epochs, position=0)
        train_bar = tqdm_notebook(desc='split=train', total=train_dataset.get_num_batches(args.batch_size), position=1, leave=True)
        val_bar = tqdm_notebook(desc='split=val', total=val_dataset.get_num_batches(args.batch_size), position=1, leave=True)

        try:
            for epoch_index in range(args.num_epochs):
                train_state['epoch_index'] = epoch_index
                train_loader = DLCJobDataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=args.shuffle, 
                                                drop_last=args.drop_last, num_workers=args.workers, pin_memory=True)
                batch_generator = generate_batches(train_loader, device=args.device)
                
                running_loss = 0.0
                running_mse = 0.0
                model.train()

                for batch_index, batch_dict in enumerate(batch_generator):
                    optimizer.zero_grad()
                    y_pred = model(x_in=batch_dict['x_data'].float())

                    loss = loss_func(y_pred, batch_dict['y_target'].float())
                    loss_t = loss.item()
                    running_loss += (loss_t - running_loss) / (batch_index + 1)

                    loss.backward()

                    optimizer.step()
                    mse_t = mean_absolute_error(y_pred, batch_dict['y_target'])
                    running_mse += (mse_t - running_mse) / (batch_index + 1)

                    train_bar.set_postfix(loss=running_loss, mse=running_mse, epoch=epoch_index)
                    train_bar.update()

                train_state['train_loss'].append(running_loss)
                train_state['train_mse'].append(running_mse)

                val_loader = DLCJobDataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=args.shuffle, 
                                              drop_last=args.drop_last, num_workers=args.workers, pin_memory=True)
                batch_generator = generate_batches(val_loader, device=args.device)
                running_loss = 0.
                running_mse = 0.
                model.eval()

                for batch_index, batch_dict in enumerate(batch_generator):

                    y_pred = model(x_in=batch_dict['x_data'].float())

                    loss = loss_func(y_pred, batch_dict['y_target'].float())
                    loss_v = loss.item()
                    running_loss += (loss_v - running_loss) / (batch_index + 1)

                    mse_v = mean_absolute_error(y_pred, batch_dict['y_target'])
                    running_mse += (mse_v - running_mse) / (batch_index + 1)
                    
                    val_bar.set_postfix(loss=running_loss, mse=running_mse, epoch=epoch_index)
                    val_bar.update()

                train_state['val_loss'].append(running_loss)
                train_state['val_mse'].append(running_mse)

                train_state = update_train_state(args=args, model=model, train_state=train_state)

                scheduler.step(train_state['val_loss'][-1])

                train_bar.n = 0
                val_bar.n = 0
                epoch_bar.update()

                if train_state['stop_early']:
                    break

                train_bar.n = 0
                val_bar.n = 0
                epoch_bar.update()
        except KeyboardInterrupt:
            print("Exiting loop")
    else:
        model.load_state_dict(torch.load(train_state['model_filename']))
        model = model.to(args.device)

        test_dataset = StockDataset(dtype='test') 
        test_loader = DLCJobDataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=args.shuffle, 
                                       drop_last=args.drop_last, num_workers=args.workers, pin_memory=True)
        batch_generator = generate_batches(test_loader, device=args.device)
        running_loss = 0.
        running_mse = 0.
        model.eval()

        for batch_index, batch_dict in enumerate(batch_generator):
            y_pred = model(x_in=batch_dict['x_data'].float())

            loss = loss_func(y_pred, batch_dict['y_target'].float())
            loss_t = loss.item()
            running_loss += (loss_t - running_loss) / (batch_index + 1)

            mse_t = mean_absolute_error(y_pred, batch_dict['y_target'])
            running_mse += (mse_t - running_mse) / (batch_index + 1)

        train_state['test_loss'] = running_loss
        train_state['test_mse'] = running_mse

        print("Test loss: {:.3f}".format(train_state['test_loss']))
        print("Test MSE: {:.2f}".format(train_state['test_mse']))
        

if __name__ == "__main__":
    main()