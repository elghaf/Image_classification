# train.py
import argparse
from model_utils import build_model, train_model, save_checkpoint

def main():
    parser = argparse.ArgumentParser(description='Train a new network on a dataset and save the model as a checkpoint.')
    parser.add_argument('data_directory', type=str, help='Path to the data directory')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg16', help='Choose architecture')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Set learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Set number of hidden units')
    parser.add_argument('--epochs', type=int, default=20, help='Set number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')

    args = parser.parse_args()

    # Build the model
    model, criterion, optimizer = build_model(arch=args.arch, learning_rate=args.learning_rate, hidden_units=args.hidden_units, gpu=args.gpu)
    print('its passe')
    # Train the model
    train_model(model, criterion, optimizer, data_dir=args.data_directory, epochs=args.epochs, gpu=args.gpu)

    # Save the checkpoint
    save_checkpoint(model, optimizer, args.save_dir)

if __name__ == '__main__':
    main()
