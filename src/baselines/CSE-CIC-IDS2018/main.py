import torch
import json
from argparse import Namespace
import torch.optim as optim
from models import *
from tqdm.notebook import tqdm
from DDoSDataset import *
from torch.utils.data import DataLoader
from utils import *


def main():
    with open("config.json") as f:
        configs = json.load(f)
    args = Namespace(**configs)
    
    if not torch.cuda.is_available():
        args.cuda = False

    print("Using CUDA: {}".format(args.cuda))

    args.device = torch.device("cuda" if args.cuda else "cpu")

    set_seed_everywhere(args.seed, args.cuda)
    
    classifier = Model(input_features=args.input_features, 
                       hiddens=args.hiddens, 
                       dropout_rate=args.dropout_rate)
    loss_func = nn.BCELoss()
    optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=1)
    
    if not args.evaluate:
        train_dataset = DDoSDataset(data_path=args.train_data_path)
        val_dataset = DDoSDataset(data_path=args.val_data_path)
        train_state = make_train_state(args)
        
        with tqdm(desc='training routine',  total=args.epochs, position=0) as epoch_bar:
            for epoch_index in range(args.epochs):
                train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=args.shuffle, 
                                          drop_last=args.drop_last, num_workers=args.num_workers, pin_memory=args.pin_memory)
                
                batch_generator = generate_batches(train_loader, device=args.device)
                optimizer.zero_grad()
                running_loss = 0.0
                running_acc = 0.0
                
                classifier.train()
                
                with tqdm(desc='split=train', total=train_dataset.get_num_batches(args.batch_size), position=1, leave=True) as train_bar:
                    for index, input in enumerate(batch_generator):
                        optimizer.zero_grad()
                        X_input, y_input = input
                        y_pred = classifier(X_input)

                        loss = loss_func(y_pred, y_input)
                        loss_t = loss.item()
                        running_loss += (loss_t - running_loss) / (index + 1)

                        loss.backward()

                        optimizer.step()
                        acc_t = compute_accuracy(y_pred, y_input)
                        running_acc += (acc_t - running_acc) / (index + 1)

                        train_bar.set_postfix(loss=running_loss, acc=running_acc, epoch=epoch_index)
                        train_bar.update()
                    
                    train_state['train_loss'].append(running_loss)
                    train_state['train_acc'].append(running_acc)

                with tqdm(desc='split=val', total=val_dataset.get_num_batches(args.batch_size), position=1, leave=True) as val_bar:
                    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=args.shuffle, 
                                            drop_last=args.drop_last, num_workers=args.workers, pin_memory=True)
                    batch_generator = generate_batches(val_loader, device=args.device)
                    running_loss = 0.
                    running_acc = 0.
                    classifier.eval()

                    for index, input in enumerate(batch_generator):
                        X_input, y_input = input
                        y_pred = classifier(X_input)

                        loss = loss_func(y_pred, y_input)
                        loss_t = loss.item()
                        running_loss += (loss_t - running_loss) / (index + 1)

                        acc_t = compute_accuracy(y_pred, y_input)
                        running_acc += (acc_t - running_acc) / (index + 1)
                        
                        val_bar.set_postfix(loss=running_loss, acc=running_acc, epoch=index)
                        val_bar.update()

                    train_state['val_loss'].append(running_loss)
                    train_state['val_acc'].append(running_acc)

                    train_state = update_train_state(args=args, model=classifier, train_state=train_state)

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
        classifier.load_state_dict(torch.load(train_state['model_filename']))
        classifier = classifier.to(args.device)

        test_dataset = DDoSDataset(data_path=args.test_data_path) 
        test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=args.shuffle, 
                                 drop_last=args.drop_last, num_workers=args.workers, pin_memory=True)
        batch_generator = generate_batches(test_loader, device=args.device)
        running_loss = 0.
        running_acc = 0.
        classifier.eval()

        for index, input in enumerate(batch_generator):
            X_input, y_input = input
            y_pred = classifier(X_input)

            loss = loss_func(y_pred, y_input)
            loss_t = loss.item()
            running_loss += (loss_t - running_loss) / (index + 1)

            acc_t = compute_accuracy(y_pred, y_input)
            running_acc += (acc_t - running_acc) / (index + 1)

        train_state['test_loss'] = running_loss
        train_state['test_acc'] = running_acc

        print("Test loss: {:.3f}".format(train_state['test_loss']))
        print("Test Accuracy: {:.2f}".format(train_state['test_acc']))
    

if __name__ == "__main__":
    main()