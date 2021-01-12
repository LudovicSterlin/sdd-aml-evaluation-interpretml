import numpy as np
from typing import Callable

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt

import ignite.engine
import ignite.handlers
import ignite.metrics
import ignite.utils
from ignite.engine import Events

import seaborn as sns
from cutout import Cutout


def model_fn():
    model = nn.Sequential(                                                         #entry image size is of shape (28, 28, 1)
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, bias=False),      # shape (26*26*32)
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),                                                           # shape (13*13*32)
        nn.Conv2d(32, 64, 3, bias=False),                                          # shape (11*11*64)
        nn.BatchNorm2d(64),                                                        
        nn.ReLU(),
        nn.MaxPool2d(2),                                                           # shape (5*5*64)
        nn.Flatten(),
        nn.Linear(5 * 5 * 64, 600),
        nn.Dropout2d(0.25),
        nn.Linear(600, 120),
        nn.Linear(120, 10),                                                        # We have 10 labels to predict !
        nn.LogSoftmax(dim=-1),                                                     
    )
    return model


def iterate_on_model(epochs, DEVICE, trainloader, validloader):
    model = model_fn()

    # moving model to gpu if available
    model.to(DEVICE)

    # declare optimizers and loss
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    criterion = nn.NLLLoss()

    # creating trainer
    trainer = ignite.engine.create_supervised_trainer(model=model, optimizer=optimizer, loss_fn=criterion, device=DEVICE)

    # create metrics
    metrics = {
        "accuracy": ignite.metrics.Accuracy(),
        "nll": ignite.metrics.Loss(criterion),
        "cm": ignite.metrics.ConfusionMatrix(num_classes=10),
    }

    ignite.metrics.RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")

    # Evaluators
    train_evaluator = ignite.engine.create_supervised_evaluator(model, metrics=metrics, device=DEVICE)
    val_evaluator = ignite.engine.create_supervised_evaluator(model, metrics=metrics, device=DEVICE)

    # Logging
    #train_evaluator.logger = ignite.utils.setup_logger("train")
    #val_evaluator.logger = ignite.utils.setup_logger("val")

    # init variables for logging
    training_history = {"accuracy": [], "loss": []}
    validation_history = {"accuracy": [], "loss": []}
    last_epoch = []


    model_name = "basic_cnn"
    dataset_name = "Fashion_MNIST"

    checkpointer = ignite.handlers.ModelCheckpoint(
        "./saved_models",
        filename_prefix=dataset_name,
        n_saved=2,
        create_dir=True,
        save_as_state_dict=True,
        require_empty=False,
    )

    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {model_name: model})

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        train_evaluator.run(trainloader)
        metrics = train_evaluator.state.metrics
        accuracy = metrics["accuracy"] * 100
        loss = metrics["nll"]
        last_epoch.append(0)
        training_history["accuracy"].append(accuracy)
        training_history["loss"].append(loss)
        print('*', end='')


    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        val_evaluator.run(validloader)
        metrics = val_evaluator.state.metrics
        accuracy = metrics["accuracy"] * 100
        loss = metrics["nll"]
        validation_history["accuracy"].append(accuracy)
        validation_history["loss"].append(loss)

    
    trainer.run(trainloader, max_epochs=epochs)
    
    return(training_history["accuracy"][-1], validation_history["accuracy"][-1], training_history["loss"][-1], validation_history["loss"][-1])


def find_best_cutout_size(range_to_test, nb_evaluation_per_size, epochs):
    result = dict()
    seed = 12
    for size in range_to_test:
        print('\n', 'size : ', size, '\n')
        n_holes = 1
        hole_length = size

        labels_text = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

        transform_training = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.2867], std=[0.3205]),
            Cutout(n_holes, hole_length)
        ])

        transform_validation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.2867], std=[0.3205]),
        ])


        full_trainset = torchvision.datasets.FashionMNIST(root='data', train=True, download=True, transform=transform_training)
        trainset, _ = torch.utils.data.random_split(full_trainset, (10000, 50000), generator=torch.Generator().manual_seed(seed))

        full_validset = torchvision.datasets.FashionMNIST(root='data', train=False, download=True, transform=transform_training)                                           
        validset, _ = torch.utils.data.random_split(full_validset, (1000, 9000), generator=torch.Generator().manual_seed(seed))
        
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
        validloader = torch.utils.data.DataLoader(validset, batch_size=64, shuffle=False, num_workers=2)
        
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        train_stats = np.zeros((nb_evaluation_per_size, 4))
        for i in range(nb_evaluation_per_size):
            print(f"try : {i +1}")
            train_stats[i] = iterate_on_model(epochs, DEVICE, trainloader, validloader)
            print('*')

        avg_training_accuracy = np.mean(train_stats[:,0])
        avg_validation_accuracy = np.mean(train_stats[:,1])
        avg_training_loss = np.mean(train_stats[:,2])
        avg_validation_loss = np.mean(train_stats[:,3])
        
        std_training_accuracy = np.std(train_stats[:,0])
        std_validation_accuracy = np.std(train_stats[:,1])
        std_training_loss = np.std(train_stats[:,2])
        std_validation_loss = np.std(train_stats[:,3])
        result[size] = [avg_training_accuracy, avg_validation_accuracy, avg_training_loss, avg_validation_loss, std_training_accuracy, std_validation_accuracy,
        std_training_loss, std_validation_loss]
        
    return result
