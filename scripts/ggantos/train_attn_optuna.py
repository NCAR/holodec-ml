import keras
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from holodecml.data import load_scaled_datasets, make_random_valid_outputs
from holodecml.models import ParticleAttentionNet
from holodecml.losses import attention_net_loss, attention_net_validation_loss

from aimlutils.hyper_opt.utils import trial_suggest_loader

import optuna
from optuna.integration import KerasPruningCallback


def create_model(trial, config):

    # Get list of hyperparameters from the config
    hyperparameters = config["optuna"]["parameters"]

    # Now update some hyperparameters via custom rules
    attention_neurons = trial_suggest_loader(trial, hyperparameters["attention_neurons"])
    hidden_layers = trial_suggest_loader(trial, hyperparameters["hidden_layers"])
    hidden_neurons = trial_suggest_loader(trial, hyperparameters["hidden_neurons"])
    min_filters = trial_suggest_loader(trial, hyperparameters["min_filters"])
    lr = trial_suggest_loader(trial, hyperparameters['learning_rate'])

    # We define our MLP.
    net = ParticleAttentionNet(attention_neurons=attention_neurons,
                               hidden_layers=hidden_layers,
                               hidden_neurons=hidden_neurons,
                               min_filters=min_filters,
                               **config["attention_network"])

    # We compile our model with a sampled learning rate.
    net.compile(optimizer=Adam(lr=lr), loss=attention_net_loss,
               metrics=[attention_net_validation_loss])
    return net


def objective(trial, config):
    # Clear clutter from previous session graphs.
    keras.backend.clear_session()

    path_data = config["path_data"]
    path_save = config["path_save"]
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    num_particles = config["num_particles"]
    output_cols = config["output_cols"]
    seed = config["random_seed"]
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

    # load data
    print("Loading data...")
    scaler_out = scalers[config["scaler_out"]]()
    train_inputs, \
    train_outputs, \
    valid_inputs, \
    valid_outputs = load_scaled_datasets(path_data,
                                         num_particles,
                                         output_cols,
                                         scaler_out,
                                         config["subset"],
                                         config["num_z_bins"],
                                         config["mass"])

    # add noise to the outputs
    train_outputs_noisy = train_outputs * (1 + np.random.normal(0, config['noisy_sd'], train_outputs.shape))
    valid_outputs_noisy = make_random_valid_outputs(path_data, num_particles,
                                                    valid_inputs.shape[0],
                                                    train_outputs.shape[1])
    valid_outputs_noisy = valid_outputs_noisy * (1 + np.random.normal(0, config['noisy_sd'], valid_outputs_noisy.shape))

    # Generate our trial model.
    model_start = datetime.now()
    net = create_model(trial, config)

    # Fit the model on the training data.
    # The KerasPruningCallback checks for pruning condition every epoch.
    hist = net.fit([train_outputs_noisy[:,:,:-1], train_inputs], train_outputs,
                   validation_data=([valid_outputs_noisy[:,:,:-1], valid_inputs], valid_outputs),
                   epochs=config["train"]['epochs'],
                   batch_size=config["train"]['batch_size'],
                   callbacks=[KerasPruningCallback(trial, "val_accuracy")],
                   verbose=config["train"]['verbose'])
    print(f"Running model took {datetime.now() - model_start} time")

    # predict outputs
    print("Predicting outputs..")
    valid_outputs_pred = net.predict([valid_outputs_noisy[:,:,:-1], valid_inputs],
                                     batch_size=config['train']["batch_size"])

    # Evaluate the model accuracy on the validation set.
    score = attention_net_validation_loss(valid_outputs, valid_outputs_pred)

    return score


if __name__ == "__main__":

    print("Starting script...")

    # parse arguments from config/yaml file
    parser = argparse.ArgumentParser(description='Describe a Conv2D nn')
    parser.add_argument("config", help="Path to config file")
    args = parser.parse_args()
    with open(args.config) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    study = optuna.create_study(direction=config["direction"],
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, config, n_trials=config["n_trials"])
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
