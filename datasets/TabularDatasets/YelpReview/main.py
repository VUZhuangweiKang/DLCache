from argparse import Namespace
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ReviewDataset import *
from tqdm.notebook import tqdm
from utils import *
from models import *
        

def predict_rating(review, classifier, vectorizer, decision_threshold=0.5):
    review = preprocess_text(review)
    
    vectorized_review = torch.tensor(vectorizer.vectorize(review))
    result = classifier(vectorized_review.view(1, -1))
    
    probability_value = F.sigmoid(result).item()
    index = 1
    if probability_value < decision_threshold:
        index = 0

    return vectorizer.rating_vocab.lookup_index(index)
        
        
def main():
    with open("config.json") as f:
        configs = json.load(f)
    args = Namespace(**configs)

    if args.expand_filepaths_to_save_dir:
        args.vectorizer_file = os.path.join(args.save_dir, args.vectorizer_file)
        args.model_state_file = os.path.join(args.save_dir, args.model_state_file)
        
        print("Expanded filepaths: ")
        print("\t{}".format(args.vectorizer_file))
        print("\t{}".format(args.model_state_file))
        
    if not torch.cuda.is_available():
        args.cuda = False

    print("Using CUDA: {}".format(args.cuda))

    args.device = torch.device("cuda" if args.cuda else "cpu")

    set_seed_everywhere(args.seed, args.cuda)

    handle_dirs(args.save_dir)

    print("Loading dataset and creating vectorizer")
    train_dataset = ReviewDataset(dtype='train')
    vectorizer = train_dataset.get_vectorizer()    
    val_dataset = ReviewDataset(dtype='validation')

    try:
        classifier = ReviewClassifier(num_features=len(vectorizer.review_vocab))
        classifier = classifier.to(args.device)

        loss_func = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=1)

        train_state = make_train_state(args)
        
        if not args.evaluate:
            with tqdm(desc='training routine',  total=args.num_epochs, position=0) as epoch_bar:
                for epoch_index in range(args.num_epochs):
                    train_state['epoch_index'] = epoch_index
                    train_loader = DLCJobDataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=args.shuffle, 
                                                    drop_last=args.drop_last, num_workers=args.workers, pin_memory=True)
                    batch_generator = generate_batches(train_loader, device=args.device)
                    
                    running_loss = 0.0
                    running_acc = 0.0
                    classifier.train()

                    with tqdm(desc='split=train', total=train_dataset.get_num_batches(args.batch_size), position=1, leave=True) as train_bar:
                        for batch_index, batch_dict in enumerate(batch_generator):
                            optimizer.zero_grad()
                            y_pred = classifier(x_in=batch_dict['x_data'].float())

                            loss = loss_func(y_pred, batch_dict['y_target'].float())
                            loss_t = loss.item()
                            running_loss += (loss_t - running_loss) / (batch_index + 1)

                            loss.backward()

                            optimizer.step()
                            acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
                            running_acc += (acc_t - running_acc) / (batch_index + 1)

                            train_bar.set_postfix(loss=running_loss, acc=running_acc, epoch=epoch_index)
                            train_bar.update()

                        train_state['train_loss'].append(running_loss)
                        train_state['train_acc'].append(running_acc)

                    val_loader = DLCJobDataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=args.shuffle, 
                                                    drop_last=args.drop_last, num_workers=args.workers, pin_memory=True)
                    batch_generator = generate_batches(val_loader, device=args.device)
                    running_loss = 0.
                    running_acc = 0.
                    classifier.eval()

                    with tqdm(desc='split=val', total=val_dataset.get_num_batches(args.batch_size), position=1, leave=True) as val_bar:
                        for batch_index, batch_dict in enumerate(batch_generator):

                            y_pred = classifier(x_in=batch_dict['x_data'].float())

                            loss = loss_func(y_pred, batch_dict['y_target'].float())
                            loss_t = loss.item()
                            running_loss += (loss_t - running_loss) / (batch_index + 1)

                            acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
                            running_acc += (acc_t - running_acc) / (batch_index + 1)
                            
                            val_bar.set_postfix(loss=running_loss, acc=running_acc, epoch=epoch_index)
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

            test_dataset = ReviewDataset(dtype='test') 
            test_loader = DLCJobDataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=args.shuffle, 
                                        drop_last=args.drop_last, num_workers=args.workers, pin_memory=True)
            batch_generator = generate_batches(test_loader, device=args.device)
            running_loss = 0.
            running_acc = 0.
            classifier.eval()

            for batch_index, batch_dict in enumerate(batch_generator):
                y_pred = classifier(x_in=batch_dict['x_data'].float())

                loss = loss_func(y_pred, batch_dict['y_target'].float())
                loss_t = loss.item()
                running_loss += (loss_t - running_loss) / (batch_index + 1)

                acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
                running_acc += (acc_t - running_acc) / (batch_index + 1)

            train_state['test_loss'] = running_loss
            train_state['test_acc'] = running_acc

            print("Test loss: {:.3f}".format(train_state['test_loss']))
            print("Test Accuracy: {:.2f}".format(train_state['test_acc']))
    except KeyboardInterrupt:
        print("Exiting loop")
        

if __name__ == "__main__":
    main()