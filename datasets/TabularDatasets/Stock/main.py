from argparse import Namespace
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error
import torch.optim as optim
from StockDataset import *
from tqdm.notebook import tqdm
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
    
    train_dataset = StockDataset(dtype='train', steps=args.steps)   
    val_dataset = StockDataset(dtype='validation', steps=args.steps)

    try:
        if not args.evaluate:
            model = Model(input_features=args.n_features, 
                          hidden_size=args.hidden_size, 
                          num_layers=args.num_layers, 
                          dropout_rate=args.dropout_rate)
            model = model.to(args.device)

            loss_func = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=1)

            train_state = make_train_state(args)
            with tqdm(total=args.num_epochs, file=sys.stdout, desc='Epoch') as epoch_bar:
                for epoch_index in range(args.num_epochs):
                    train_state['epoch_index'] = epoch_index
                    train_loader = DLCJobDataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False, 
                                                    drop_last=args.drop_last, num_workers=args.workers, pin_memory=True)
                    batch_generator = generate_batches(train_loader, device=args.device)
                    
                    running_loss = 0.0
                    running_mse = 0.0
                    model.train()

                    with tqdm(total=train_dataset.get_num_batches(args.batch_size), file=sys.stdout, desc='Train Batch') as train_bar:
                        for batch_index, batch_dict in enumerate(batch_generator):
                            optimizer.zero_grad()
                            y_pred = model(batch_dict['x_data'])

                            loss = loss_func(y_pred, batch_dict['y_target'])
                            loss_t = loss.item()
                            running_loss += (loss_t - running_loss) / (batch_index + 1)

                            loss.backward()

                            optimizer.step()
                            mse_t = mean_absolute_error(y_pred.flatten().detach().numpy(), batch_dict['y_target'].flatten().detach().numpy())
                            running_mse += (mse_t - running_mse) / (batch_index + 1)

                            train_bar.set_postfix(loss=running_loss, mse=running_mse, epoch=epoch_index)
                            train_bar.update()

                        train_state['train_loss'].append(running_loss)
                        train_state['train_mse'].append(running_mse)

                    val_loader = DLCJobDataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, 
                                                    drop_last=args.drop_last, num_workers=args.workers, pin_memory=True)
                    batch_generator = generate_batches(val_loader, device=args.device)
                    running_loss = 0.
                    running_mse = 0.
                    model.eval()

                    with tqdm(total=val_dataset.get_num_batches(args.batch_size), file=sys.stdout, desc="Val Batch") as val_bar:
                        for batch_index, batch_dict in enumerate(batch_generator):

                            y_pred = model(batch_dict['x_data'])

                            loss = loss_func(y_pred, batch_dict['y_target'])
                            loss_v = loss.item()
                            running_loss += (loss_v - running_loss) / (batch_index + 1)

                            mse_v = mean_absolute_error(y_pred.flatten().detach().numpy(), batch_dict['y_target'].flatten().detach().numpy())
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
        else:
            model.load_state_dict(torch.load(train_state['model_filename']))
            model = model.to(args.device)

            test_dataset = StockDataset(dtype='test', steps=args.steps) 
            test_loader = DLCJobDataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, 
                                           drop_last=args.drop_last, num_workers=args.workers, pin_memory=True)
            batch_generator = generate_batches(test_loader, device=args.device)
            running_loss = 0.
            running_mse = 0.
            model.eval()

            for batch_index, batch_dict in enumerate(batch_generator):
                y_pred = model(batch_dict['x_data'])

                loss = loss_func(y_pred, batch_dict['y_target'])
                loss_t = loss.item()
                running_loss += (loss_t - running_loss) / (batch_index + 1)

                mse_t = mean_absolute_error(y_pred, batch_dict['y_target'])
                running_mse += (mse_t - running_mse) / (batch_index + 1)

            train_state['test_loss'] = running_loss
            train_state['test_mse'] = running_mse

            print("Test loss: {:.3f}".format(train_state['test_loss']))
            print("Test MSE: {:.2f}".format(train_state['test_mse']))
    except KeyboardInterrupt:
        print("Exiting loop")
            

if __name__ == "__main__":
    main()