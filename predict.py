# predict.py
import argparse
from model_utils import load_checkpoint, predict
from data_utils import process_image

def main():
    parser = argparse.ArgumentParser(description='Predict flower name from an image with the probability of that name.')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('checkpoint', type=str, help='Path to the checkpoint file')
    parser.add_argument('--top_k', type=int, default=3, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Path to the category names mapping file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')

    args = parser.parse_args()

    # Load the checkpoint
    model, optimizer = load_checkpoint(args.checkpoint)

    # Process the image
    image = process_image(args.image_path)

    # Predict the class
    probs, classes = predict(model, image, topk=args.top_k, gpu=args.gpu)

    print("Top K Classes:")
    for prob, class_idx in zip(probs, classes):
        print(f"Class: {class_idx}, Probability: {prob:.4f}")

if __name__ == '__main__':
    main()
