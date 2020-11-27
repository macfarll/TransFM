# A class used to save all the arguments used for a model, with a function to log arguments used
class DummyArgs(object):

    def __init__(self):

        # Core args
        self.filename = ml_short_df #name of pandas df (preformatted)
        self.model = "PRME" # Currently supports "PRME", "TransFM"
        self.features = "full_features" # Currently only supports "full_features", TODO: add "time"
        self.item_df = ml_item_feat # The item pandas df (already formatted)
        self.user_df = ml_user_feat # Already formatted user data 
        self.user_region_df = "none" # A data frame that restricts items and users to specific regions

        # Training args
        self.min_epoch = 30 # Minimum number of epochs, model will not stop even if accuracy is decreasing
        self.eval_freq = 10 # Frequency at which to evaluate model
        self.max_iters = 100 # Epoch to end training, even if model is still improving
        self.quit_delta = 40 # Number of iterations at which to quit if no improvement
        self.num_train_samples = 3 # Number of negative samples to evaluate against to calc auc
        self.num_val_samples = 10 # Number of negative samples to compare against positive sample in val
        self.num_test_samples = 10 # Recommended to be same as val
        self.val_set = 1 # 1 to include validation set, 0 to only use train/test (skipping val set is very prone to overfitting)
        self.weighted_sampling = 0 #set negative samples to be weighted according to observations in train

        # Model args
        self.num_dims = 3 # Model dimensionality
        self.linear_reg = 6.236 #L2 regularization: linear regularization
        self.emb_reg = 10.54 #L2 regularization: embedding regularization
        self.trans_reg = 3.135 #from first pass #L2 regularization: translation regularization
        self.init_mean = 0.133 
        self.starting_lr = 0.05 # Learning rate of model at epoch 1
        self.lr_decay_factor = 1.33 # Decay factor for learning rate
        self.lr_decay_freq = 700 # Frequency at which to decay learning rate
        self.user_min = 2 # Minimum number of interactions for a user to be included in model
        self.item_min = 3 # Minimum number of interactions for an item to be included in model
        self.secondary_reg_scale = 1 # Scale loss inside sigmoid function, does nothing if set to 1

        # Debug args
        self.verbosity = 1 # 2 for maximum verbosity, 0 supressess all
        self.random_seed = 1 # Set the random seed to the model, useful for debugging
        self.log_cache = list() # Init empty log 

        # Deployment args
        self.return_k_preds = 100 # Number of predictions to return per user
        self.deploy_preds = 1 # 0 disables the generation of deployment predictions for better performance, 1 to enable
         
    # A function to store input text to log_cache as well as print if the input verbosity_max is greater than the 
    # verbosity of the 
    def logger(self, input_text, verbosity_max = 1):
        self.log_cache.append(input_text)
        if self.verbosity >= verbosity_max:
            print(input_text)
            
# Default args, set as something other than a command line implementation
args = DummyArgs()