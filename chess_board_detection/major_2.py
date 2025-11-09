import os
import cv2
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import pickle
from collections import Counter
from nn_models import *
import random


def load_xiangqi_dataset(dataset_dir, target_size, resize=False, aug_per_image=0):
    """
    Loads all Xiangqi piece images from a structured dataset and assigns class labels.
    Reshape all images to target_size.
    """

    # Step 1: Define all valid Xiangqi pieces and colors
    piece_names = ["bing", "che", "ma", "pao", "xiang", "shi", "jiang"]
    colors = ["r", "g"]  # r = red, g = green/black

    # Build full class names like bing_r, ma_g, etc.
    valid_classes = [f"{piece}_{color}" for piece in piece_names for color in colors]

    valid_classes.append("empty_0")

    class_dict = {class_name: idx for idx, class_name in enumerate(valid_classes)}

    images = []
    labels = []

    for class_name in valid_classes:
        print(f'Loading {class_name}...')
        class_id = class_dict[class_name]

        # check for empty which has no colour
        piece, color = class_name.split("_")
        if piece == 'empty':
            piece_folder = os.path.join(dataset_dir, piece)
        else:
            piece_folder = os.path.join(dataset_dir, class_name)

        if not os.path.isdir(piece_folder):
            raise FileNotFoundError(f"Piece folder {piece_folder} not found.")

        # loop thru angles
        for angle in ["big_angle", "small_angle"]:
            angle_folder = os.path.join(piece_folder, angle)
            if not os.path.isdir(angle_folder):
                raise FileNotFoundError(f"Angle folder {angle_folder} not found.")

            # loop thru files
            for fname in os.listdir(angle_folder):
                # read image
                img_path = os.path.join(angle_folder, fname)
                img = cv2.imread(img_path)

                if img is None:
                    raise Exception(f"Unreadable image: {img_path}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # resize image
                if resize:
                    img = resize_and_pad(img, target_size)
                images.append(img)
                labels.append({
                    "piece": piece,
                    "color": color,
                    "angle": angle,
                    "class_name": class_name,
                    "class_id": class_id,
                    'fname': fname,
                })

                # augmentation
                for i in range(aug_per_image):
                    images.append(data_augmentation(img))
                    labels.append({
                        "piece": piece,
                        "color": color,
                        "angle": angle,
                        "class_name": class_name,
                        "class_id": class_id,
                        'fname': f'{fname[:-4]}_aug{i}.jpg',
                    })

    images = np.array(images, dtype=np.float32)
    return images, labels, class_dict


def resize_and_pad(image, target_size=(256, 256), pad_value=0):
    """
    Resize an image to fit within `target_size` while preserving aspect ratio,
    then pad to reach the exact target size.

    Parameters:
        image (np.array): Input image
        target_size (tuple): Final output size (width, height)
        pad_value (int or tuple): Padding color (0=black)

    Returns:
        padded_image (np.array): Output image with exact `target_size`
    """
    target_w, target_h = target_size
    h, w = image.shape[:2]

    # Compute scaling factor (preserving aspect ratio)
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Compute padding
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top

    # Choose padding method based on image shape
    if len(image.shape) == 3:  # Color image
        padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                                    borderType=cv2.BORDER_CONSTANT, value=(pad_value,)*3)
    else:  # Grayscale
        padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                                    borderType=cv2.BORDER_CONSTANT, value=pad_value)

    return padded


def data_augmentation(image):
    # Random rotation
    angle = random.uniform(-20, 20)
    M = cv2.getRotationMatrix2D((128, 128), angle, 1)
    image = cv2.warpAffine(image, M, (256, 256), borderMode=cv2.BORDER_REFLECT)

    # Random saturation/brightness adjustment
    hsv = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * np.random.uniform(0.8, 1.2), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * np.random.uniform(0.8, 1.2), 0, 255)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # Small random crop
    crop = random.randint(0, 16)
    image = image[crop:256-crop, crop:256-crop, :]
    image = cv2.resize(image, (256, 256))

    return image


def plot_dataset_histogram(labels, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # plot histogram
    class_counts = Counter(label['class_name'] for label in labels)
    class_names = sorted(class_counts.keys())
    counts = [class_counts[name] for name in class_names]

    plt.figure(figsize=(10, 5))
    plt.bar(class_names, counts, color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.title("Number of Images per Class")
    plt.xlabel("Class Name")
    plt.ylabel("Image Count")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/class_count.png")


def save_image(images, labels, output_dir, file_ext):
    # saves image
    images = images.astype(np.uint8)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for idx, image in enumerate(images):
        # create folders
        if labels[idx]["piece"] == 'empty':
            piece_folder = f'{output_dir}/{labels[idx]["piece"]}'
        else:
            piece_folder = f'{output_dir}/{labels[idx]["piece"]}_{labels[idx]["color"]}'
        if not os.path.exists(piece_folder):
            os.makedirs(piece_folder)
        angle_folder = f'{piece_folder}/{labels[idx]["angle"]}'
        if not os.path.exists(angle_folder):
            os.makedirs(angle_folder)

        plt.imsave(f'{angle_folder}/{labels[idx]["fname"][:-4]}{file_ext}.jpg', image)
        plt.close()


def sample_data(labels, n_train_groups, num_test):
    labels_id = np.array([label['class_id'] for label in labels])

    # calculate number of images per class per fold
    num_class = len(set(labels_id))
    train_groups_idx = [[] for _ in range(n_train_groups)]
    test_idx = []

    num_train = len(labels_id) - num_test
    test_per_class = num_test // num_class
    train_per_class = num_train // num_class
    train_per_group_per_class = train_per_class // n_train_groups

    # debug
    # print(f"Total samples: {len(labels_id)}")
    # print(f"Test samples: {num_test} ({test_per_class} per class)")
    # print(f"Train samples: {num_train} ({train_per_class} per class)")

    # fix random generator for reproducible results
    rng = np.random.default_rng(seed=42)

    for digit in range(num_class):
        digit_indices = np.where(labels_id == digit)[0]
        rng.shuffle(digit_indices)

        # Get test indices
        digit_test_idx = digit_indices[:test_per_class]
        test_idx.extend(digit_test_idx)

        # Remaining for training
        digit_train_idx = digit_indices[test_per_class:]

        for i in range(n_train_groups):
            start = i * train_per_group_per_class
            end = (i + 1) * train_per_group_per_class
            group_slice = digit_train_idx[start:end]
            train_groups_idx[i].extend(group_slice)

    # debug
    # for i, group in enumerate(train_groups_idx):
    #     group_labels = labels_id[group]
    #     print(f"\nTrain Group {i}: {len(group)} samples")
    #     for cls in range(num_class):
    #         count = np.sum(group_labels == cls)
    #         print(f"  Class {cls}: {count} samples")

    # analyze test set
    # test_labels = labels_id[test_idx]
    # print(f"\nTest Set: {len(test_idx)} samples")
    # for cls in range(num_class):
    #     count = np.sum(test_labels == cls)
    #     print(f"  Class {cls}: {count} samples")

    return train_groups_idx, test_idx, labels_id, num_class


def extract_rgb_histogram(images, bins=(6, 6, 6), n_regions=1):
    f_hist = []
    for image in images:
        blocks = []

        # split image into regions for some positional information
        block_size = image.shape[0] // n_regions
        for i in range(n_regions):
            for j in range(n_regions):
                blocks.append(image[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size])

        # Compute the histogram and normalize it
        hist = [cv2.calcHist([block], [0, 1, 2], None, bins,
                             [0, 256, 0, 256, 0, 256]) for block in blocks]
        f_hist.append(hist)

    f_hist = np.array(f_hist, dtype=np.float32).reshape((len(images), -1))

    # normalise the histogram
    f_hist /= np.square(block_size)

    return f_hist


def extract_HOG(images, block_size=(16, 16), cell_size=(8, 8), n_bins=9):
    # pre calc win ize
    winSize = (images[0].shape[1], images[0].shape[0])
    hog = cv2.HOGDescriptor(winSize, block_size, cell_size, cell_size, n_bins)
    images_int = images.astype('uint8')
    f_hog = []

    # loop thru all images
    for img in images_int:
        hog_feat = hog.compute(img)
        f_hog.append(hog_feat.flatten())

    f_hog = np.array(f_hog)

    return f_hog


def performance_metrics(y_pred, y_true, y_prob, images, model_name, num_classes=10, top_k=2, plot_name='', plot=True):
    # confusion matrix
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        confusion[t, p] += 1

    # accuracy, precision, recall, f1 score
    accuracy = np.trace(confusion) / confusion.sum()

    TP = np.diag(confusion).astype(float)

    pred_pos = confusion.sum(axis=1).astype(float)
    actual_pos = confusion.sum(axis=0).astype(float)

    precision = np.divide(TP, pred_pos, out=np.zeros_like(TP), where=pred_pos != 0)
    recall = np.divide(TP, actual_pos, out=np.zeros_like(TP), where=actual_pos != 0)

    denom = precision + recall
    f1_score = np.divide(2 * precision * recall, denom, out=np.zeros_like(precision), where=denom != 0)

    # top k accuracy
    top_k_preds = np.argsort(y_prob, axis=1)[:, -top_k:]  # top k predicted classes per sample
    # Check if true label is in top-k predictions for each sample
    matches = [y_true[i] in top_k_preds[i] for i in range(len(y_true))]
    top_k_accuracy = np.mean(matches)

    # debug
    # print("Confusion matrix:\n", confusion)
    # print(f"Overall accuracy: {accuracy:.4f}\n")
    # print(f"\nTop-{top_k} accuracy: {top_k_accuracy:.4f}")

    # debug
    # for i in range(num_classes):
    #     print(f"Class {i}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1-score={f1_score[i]:.4f}")

    # plot
    if plot:
        incorrect_indices = np.where(y_pred != y_true)[0]
        for idx in incorrect_indices:
            # print(f'True {y_true[idx]} but predicted {y_pred[idx]}')
            img_float = images[idx]
            plt.imshow(img_float.astype(np.uint8).clip(0, 255))
            plt.axis('off')
            plt.savefig(f'{model_output_dir}/{model_name}/{plot_name}_{idx}.jpg')
            plt.close()
    return {'confusion': confusion, 'accuracy': accuracy, 'precision': precision, 'recall': recall,
            'f1_score': f1_score, 'top_k_accuracy': top_k_accuracy}


def plot_metrics(model_output_dir):
    """
    Plot metrics by looping over metric keys, then over models to plot each feature.
    Each metric produces len(features) separate plots.
    For cross-validation metrics, average the data before plotting.
    """
    models = [
        "SVM (C=1)",
        "SVM (C=3)",
        "KNN (k=3)",
        "KNN (k=5)",
        "KNN (k=5, weighted)",
        'CNN 1',
        'CNN 2',
        'CNN 3',
    ]
    features = [
        'rgb_b1_n3',
        'rgb_b1_n6',
        'rgb_b4_n3',
        'hog_b128_c64_n6',
        'hog_b64_c32_n6',
        'hog_b128_c64_n9',
        'raw_image'
    ]

    metrics_dict = {}
    if not os.path.exists(f'{model_output_dir}'):
        os.makedirs(f'{model_output_dir}')

    # loop thru all models trained
    for model_name in models:
        for feature in features:
            # CNN only takes raw image
            if (feature == 'raw_image' and 'CNN' in model_name) or (feature != 'raw_image' and 'CNN' not in model_name):
                with open(f'{model_output_dir}/{model_name}_{feature}/{model_name}_{feature}_metrics.pkl', 'rb') as f:
                    temp_dict = pickle.load(f)
                if not metrics_dict:
                    for key in temp_dict:
                        metrics_dict[key] = {f'{model_name} {feature}': temp_dict[key]}
                else:
                    for key in temp_dict:
                        metrics_dict[key][f'{model_name} {feature}'] = temp_dict[key]

    colors = ['dodgerblue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    # extract values and compute mean for cross folds, and plot top 15
    for metric in metrics_dict:
        if 'confusion' in metric:
            continue
        values = []
        model_names = list(metrics_dict[metric].keys())
        plt.figure(figsize=(16, 8))
        for model_name in model_names:
                vals = metrics_dict[metric][model_name]

                # cross validation contains numbers for all folds
                val = np.mean(vals)
                values.append(val)

        # plot top 15
        x = np.arange(15) #len(model_names))
        bar_colors = [colors[i % len(colors)] for i in range(15)] #len(model_names))]

        # sort bars from highest to lowest
        sorted_pairs = sorted(zip(values, model_names), reverse=True)[:15]
        values, model_names = zip(*sorted_pairs)

        # plot
        bars = plt.bar(x, values, color=bar_colors)

        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + 0.02,  # slightly above the bar
                     f'{value:.2f}', ha='center', va='bottom', fontsize=18)

        plt.xticks(x, model_names, rotation=45, ha='right', fontsize=18)
        plt.ylim(0, 1)
        plt.ylabel(metric.upper(), fontsize=18)
        plt.title(f'{metric.upper()}', fontsize=18)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        plt.savefig(f'{model_output_dir}/{metric}.png')
        plt.close()


def train_model(model_name, model_params, x, y):
    # train model
    model = model_params['constructor'](**model_params['args'])

    # CNN needs to be built and compiled
    if 'CNN' in model_name:
        model.build(input_shape=[(None,) + x.shape[1:]])
        model.call(np.zeros((1,) + x.shape[1:]))
        model.summary()
        model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.CategoricalCrossentropy())
        model.fit(x, y, epochs=50, batch_size=32)
    else:
        model.fit(x, y)

    return model


def evaluate_model(model_name, model, x):
    # Predict test labels
    if 'CNN' in model_name:
        # CNN gives prob by default
        y_prob = model.predict(x)
        y_pred = np.argmax(y_prob, axis=-1)
    else:
        y_pred = model.predict(x)
        y_prob = model.predict_proba(x)
    return y_pred, y_prob


def train(dataset_dir, model_output_dir, target_size=(256, 256), n_train_groups=5, num_test=100):
    # prepare data
    images, labels, class_dict = load_xiangqi_dataset(dataset_dir, target_size)
    train_groups_idx, test_idx, labels_id, num_class = sample_data(labels, n_train_groups, num_test)
    labels_one_hot = tf.one_hot(labels_id, num_class).numpy()

    # model list
    models = {
        # "SVM (C=1)": {'constructor': sklearn.svm.SVC, 'args': {'C': 1, 'probability': True}},
        # "SVM (C=3)": {'constructor': sklearn.svm.SVC, 'args': {'C': 3, 'probability': True}},
        # "KNN (k=3)": {'constructor': sklearn.neighbors.KNeighborsClassifier, 'args': {'n_neighbors': 3,}},
        # "KNN (k=5)": {'constructor': sklearn.neighbors.KNeighborsClassifier, 'args': {'n_neighbors': 5,}},
        # "KNN (k=5, weighted)": {'constructor': sklearn.neighbors.KNeighborsClassifier, 'args': {'n_neighbors': 5, 'weights': 'distance',}},
        # "CNN 1": {'constructor': CNN,
        #         'args': {
        #             'params': {
        #                 'cnn': [{'filters': 16, 'kernel_size': (7, 7), 'strides': (1, 1), 'padding': 'same',
        #                          'activation': 'relu'},
        #                         {'filters': 64, 'kernel_size': (5, 5), 'strides': (1, 1), 'padding': 'same',
        #                          'activation': 'relu'},
        #                         {'filters': 128, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same',
        #                          'activation': 'relu'},
        #                         ],
        #                 'pool': [{'pool_size': (4, 4)},  # (64, 64, 16)
        #                          {'pool_size': (4, 4), },  # (16, 16, 64)
        #                          {'pool_size': (4, 4)},  # (4, 4, 128)
        #                          ],
        #                 'output_dense': {'units': num_class},
        #                 'output_act': {'activation': 'softmax'},
        #             }},
        #         },
        "CNN 2": {'constructor': CNN,
                  'args': {
                      'params': {
                          'cnn': [{'filters': 8, 'kernel_size': (7, 7), 'strides': (1, 1), 'padding': 'same',
                                   'activation': 'relu'},
                                  {'filters': 32, 'kernel_size': (5, 5), 'strides': (1, 1), 'padding': 'same',
                                   'activation': 'relu'},
                                  {'filters': 64, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same',
                                   'activation': 'relu'},
                                  ],
                          'pool': [{'pool_size': (4, 4)},  # (64, 64, 8)
                                   {'pool_size': (4, 4), },  # (16, 16, 32)
                                   {'pool_size': (4, 4)},  # (4, 4, 64)
                                   ],
                          'output_dense': {'units': num_class},
                          'output_act': {'activation': 'softmax'},
                      }},
                  },
        "CNN 3": {'constructor': CNN,
                  'args': {
                      'params': {
                          'cnn': [{'filters': 16, 'kernel_size': (11, 11), 'strides': (1, 1), 'padding': 'same',
                                   'activation': 'relu'},
                                  {'filters': 64, 'kernel_size': (9, 9), 'strides': (1, 1), 'padding': 'same',
                                   'activation': 'relu'},
                                  {'filters': 128, 'kernel_size': (7, 7), 'strides': (1, 1), 'padding': 'same',
                                   'activation': 'relu'},
                                  ],
                          'pool': [{'pool_size': (4, 4)},  # (64, 64, 16)
                                   {'pool_size': (4, 4), },  # (16, 16, 64)
                                   {'pool_size': (4, 4)},  # (4, 4, 128)
                                   ],
                          'output_dense': {'units': num_class},
                          'output_act': {'activation': 'softmax'},
                      }},
                  },
    }

    # feature list
    features = {
        # 'rgb_b1_n3': extract_rgb_histogram(images, n_regions=1, bins=[3, 3, 3]),
        # 'rgb_b1_n6': extract_rgb_histogram(images, n_regions=1),
        # 'rgb_b4_n3': extract_rgb_histogram(images, n_regions=2, bins=[3, 3, 3]),
        # 'hog_b128_c64_n6': extract_HOG(images, block_size=[128, 128], cell_size=[64, 64], n_bins=6),
        # 'hog_b64_c32_n6': extract_HOG(images, block_size=[64, 64], cell_size=[32, 32], n_bins=6),
        # 'hog_b128_c64_n9': extract_HOG(images, block_size=[128, 128], cell_size=[64, 64], n_bins=9),
        'raw_image': images,
    }

    # debug
    # for key in features:
    #     print(f'Feature {key} shape: {features[key].shape}')

    # metrics list
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'top_k_accuracy', 'confusion',
               'cross_accuracy', 'cross_precision', 'cross_recall', 'cross_f1_score', 'cross_top_k_accuracy',
               'cross_confusion']

    # train loop for all models, features
    for model_name, model_params in models.items():
        print(f'Training {model_name} model...')
        for key in features:
            metrics_dict = {metric: [] for metric in metrics}
            print(f'Feature {key}')
            feature = features[key]
            if 'CNN' not in model_name and key == 'raw_image':
                continue

            # Create model folder
            if not os.path.exists(f'{model_output_dir}'):
                os.makedirs(f'{model_output_dir}')
            if not os.path.exists(f'{model_output_dir}/{model_name}_{key}'):
                os.makedirs(f'{model_output_dir}/{model_name}_{key}')
                print(f"Folder '{model_name}_{key}' created.")
            else:
                print(f"Folder '{model_name}_{key}' already exists.")

            # cross validation training runs
            print('Cross Validation...')
            for i in range(len(train_groups_idx)):
                train_idx = train_groups_idx[:i] + train_groups_idx[i + 1:]
                train_idx = np.concatenate(train_idx)

                # CNN needs one hot encoding as y
                if 'CNN' in model_name:
                    model = train_model(model_name, model_params, feature[train_idx], labels_one_hot[train_idx])
                else:
                    model = train_model(model_name, model_params, feature[train_idx], labels_id[train_idx])

                # Predict test labels
                x_test = feature[train_groups_idx[i]]
                y_pred, y_prob = evaluate_model(model_name, model, x_test)
                y_true = labels_id[train_groups_idx[i]]

                cross_metrics = performance_metrics(y_pred, y_true, y_prob, images[train_groups_idx[i]],
                                                    model_name + f'_{key}', num_classes=len(set(labels_id)), plot=False)

                for cross_metric in cross_metrics:
                    # if cross_metric != 'confusion':
                        metrics_dict[f'cross_{cross_metric}'].append(cross_metrics[cross_metric])

            # final train
            print('Final train and test...')
            train_idx = np.concatenate(train_groups_idx)

            # save model
            if 'CNN' in model_name:
                model = train_model(model_name, model_params, feature[train_idx], labels_one_hot[train_idx])
                model.save(f'{model_output_dir}/{model_name}_{key}/{model_name}_{key}.keras')
            else:
                model = train_model(model_name, model_params, feature[train_idx], labels_id[train_idx])
                with open(f'{model_output_dir}/{model_name}_{key}/{model_name}_{key}.pkl', 'wb') as f:
                    pickle.dump(model, f)

            # Predict test labels
            x_test = feature[test_idx]
            y_pred, y_prob = evaluate_model(model_name, model, x_test)
            y_true = labels_id[test_idx]

            # save metrics
            test_metrics = performance_metrics(y_pred, y_true, y_prob, images[test_idx],
                                               model_name + f'_{key}', num_classes=len(set(labels_id)), plot_name='test')

            for test_metric in test_metrics:
                metrics_dict[test_metric].append(test_metrics[test_metric])

            with open(f'{model_output_dir}/{model_name}_{key}/{model_name}_{key}_metrics.pkl', 'wb') as f:
                pickle.dump(metrics_dict, f)
    return


def load_model(model_file):
    # load model in
    if 'CNN' in model_file:
        cnn_1 =  {'params': {
                        'cnn': [{'filters': 16, 'kernel_size': (7, 7), 'strides': (1, 1), 'padding': 'same',
                                 'activation': 'relu'},
                                {'filters': 64, 'kernel_size': (5, 5), 'strides': (1, 1), 'padding': 'same',
                                 'activation': 'relu'},
                                {'filters': 128, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same',
                                 'activation': 'relu'},
                                ],
                        'pool': [{'pool_size': (4, 4)},  # (64, 64, 16)
                                 {'pool_size': (4, 4), },  # (16, 16, 64)
                                 {'pool_size': (4, 4)},  # (4, 4, 128)
                                 ],
                        'output_dense': {'units': 15},
                        'output_act': {'activation': 'softmax'},
                    }}
        cnn_2 = {'params': {
                          'cnn': [{'filters': 8, 'kernel_size': (7, 7), 'strides': (1, 1), 'padding': 'same',
                                   'activation': 'relu'},
                                  {'filters': 32, 'kernel_size': (5, 5), 'strides': (1, 1), 'padding': 'same',
                                   'activation': 'relu'},
                                  {'filters': 64, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same',
                                   'activation': 'relu'},
                                  ],
                          'pool': [{'pool_size': (4, 4)},  # (64, 64, 8)
                                   {'pool_size': (4, 4), },  # (16, 16, 32)
                                   {'pool_size': (4, 4)},  # (4, 4, 64)
                                   ],
                          'output_dense': {'units': 15},
                          'output_act': {'activation': 'softmax'},
                      }}
        cnn_3 = {'params': {
                          'cnn': [{'filters': 16, 'kernel_size': (11, 11), 'strides': (1, 1), 'padding': 'same',
                                   'activation': 'relu'},
                                  {'filters': 64, 'kernel_size': (9, 9), 'strides': (1, 1), 'padding': 'same',
                                   'activation': 'relu'},
                                  {'filters': 128, 'kernel_size': (7, 7), 'strides': (1, 1), 'padding': 'same',
                                   'activation': 'relu'},
                                  ],
                          'pool': [{'pool_size': (4, 4)},  # (64, 64, 16)
                                   {'pool_size': (4, 4), },  # (16, 16, 64)
                                   {'pool_size': (4, 4)},  # (4, 4, 128)
                                   ],
                          'output_dense': {'units': 15},
                          'output_act': {'activation': 'softmax'},
                      }}
        if 'CNN 1' in model_file:
            model = CNN(**cnn_1)
        elif 'CNN 2' in model_file:
            model = CNN(**cnn_2)
        elif 'CNN 3' in model_file:
            model = CNN(**cnn_3)
        model.build(input_shape=[(1, 256, 256, 3), ])
        model.load_weights(model_file)
    else:
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
    return model


if __name__ == "__main__":
    # hyperparams
    # dataset_dir = "./xiangqi_dataset"
    dataset_dir = "./processed_xiangqi"
    output_dir = "./processed_xiangqi"
    model_output_dir = './model'
    taget_size = [256, 256]

    # used for first run to resize image and augmentation
    images, labels, class_dict = load_xiangqi_dataset(dataset_dir, taget_size, resize=False, aug_per_image=0)
    # save_image(images, labels, output_dir, "")
    # plot_dataset_histogram(labels, output_dir)

    # train(dataset_dir, model_output_dir, taget_size, n_train_groups=2, num_test=720)
    # plot_metrics(model_output_dir)
    model = load_model('./model/CNN 1_raw_image/CNN 1_raw_image.keras')
    y_pred, y_prob = evaluate_model('CNN 1', model, images)
    print(y_pred)