import pandas as pd
import scipy.sparse as sp
import json
import random
import numpy as np
import json
import sys
import tensorflow as tf
from tensorflow.keras import layers

# DEBUG - for time stamps
import time

# A layer object, where one layer represents the entire PRME algorithm
class PRME(layers.Layer):

    # When initialized, create the two lower-dimenstional matrices to approximate data
    def __init__(self, input_dim, factor_dim, seed):

        # Set random seed for reproducability
        tf.random.set_seed(seed)

        # Both matrices init as random, then the model learns how they can better represent data
        super(PRME, self).__init__()
        lin_init = tf.random_normal_initializer()

        # Var_linear can be thought of as the linear bias to any comparisons
        self.var_linear = tf.Variable(initial_value=lin_init(shape=(input_dim, 1),
                                                  dtype='float32'), trainable=True)

        # Var_factors can optionally have multiple dimenstions and represent a more complex space
        factor_init = tf.random_normal_initializer()
        self.var_factors = tf.Variable(initial_value=factor_init(shape=(input_dim, factor_dim),
                                                  dtype='float32'), trainable=True)

    # When the model is called, expects features as input and returns scores for each item, user pair
    def call(self, sparse_feats):
        linear_bias = tf.sparse.sparse_dense_matmul(sparse_feats, self.var_linear)
        var_emb_product = tf.reduce_sum(tf.square(self.var_factors), axis=1, keepdims = True)

        feats_sum = tf.sparse.reduce_sum(sparse_feats, axis=1, keepdims = True)
        emb_mul = tf.sparse.sparse_dense_matmul(sparse_feats, self.var_factors)
        # Term 1
        prod_term = tf.sparse.sparse_dense_matmul(sparse_feats, var_emb_product)
        term_1 = prod_term * feats_sum
        # Term 2
        term_2 = 2 * tf.reduce_sum(tf.square(emb_mul), axis=1, keepdims = True)
        # Term 3
        term_3 = term_1
        # Predictions
        preds = linear_bias + 0.5 * (term_1 - term_2 + term_3)
        return(preds)

# A layer object, where one layer represents the entire PRME algorithm
class TransFM(layers.Layer):

    # When initialized, create the two lower-dimenstional matrices to approximate data
    def __init__(self, input_dim, factor_dim, seed):

        # Set random seed for reproducability
        tf.random.set_seed(seed)

        # Both matrices init as random, then the model learns how they can better represent data
        super(TransFM, self).__init__()
        lin_init = tf.random_normal_initializer()

        # Var_linear can be thought of as the linear bias to any comparisons
        self.var_linear = tf.Variable(initial_value=lin_init(shape=(input_dim, 1),
                                                  dtype='float32'), trainable=True)

        # Var_emb_factors and var_trans_factors can optionally have multiple dimenstions
        factor_init = tf.random_normal_initializer()
        self.var_emb_factors = tf.Variable(initial_value=factor_init(shape=(input_dim, factor_dim),
                                                  dtype='float32'), trainable=True)

        self.var_trans_factors = tf.Variable(initial_value=factor_init(shape=(input_dim, factor_dim),
                                                  dtype='float32'), trainable=True)

    # When the model is called, expects features as input and returns scores for each item, user pair
    def call(self, sparse_feats):
        linear_bias = tf.sparse.sparse_dense_matmul(sparse_feats, self.var_linear)
        var_emb_product = tf.reduce_sum(tf.square(self.var_emb_factors), axis=1, keepdims = True)

        var_trans_product = tf.reduce_sum(tf.square(self.var_trans_factors), axis=1, keepdims = True)
        var_emb_trans_product = tf.reduce_sum(tf.math.multiply(self.var_emb_factors, self.var_trans_factors),
                axis=1, keepdims=True)

        feats_sum = tf.sparse.reduce_sum(sparse_feats, axis=1, keepdims = True)
        emb_mul = tf.sparse.sparse_dense_matmul(sparse_feats, self.var_emb_factors)
        trans_mul = tf.sparse.sparse_dense_matmul(sparse_feats, self.var_trans_factors)

        # Term 1
        prod_term = tf.sparse.sparse_dense_matmul(sparse_feats, var_emb_product)
        term_1 = prod_term * feats_sum

        # Term 2
        prod_term = tf.sparse.sparse_dense_matmul(sparse_feats, var_trans_product)
        term_2 = prod_term * feats_sum

        # Term 3
        term_3 = term_1

        # Term 4
        prod_term = tf.sparse.sparse_dense_matmul(sparse_feats, var_emb_trans_product)
        term_4 = 2 * prod_term * feats_sum

        # Term 5
        term_5 = 2 * tf.reduce_sum(tf.square(emb_mul), axis=1, keepdims=True)

        # Term 6
        term_6 = 2 * tf.reduce_sum(trans_mul * emb_mul, axis=1, keepdims=True)

        # Diag term
        diag_term = tf.reduce_sum(tf.square(trans_mul), axis=1, keepdims=True)

        # Predictions
        preds = linear_bias + 0.5 * (term_1 + term_2 + term_3
                + term_4 - term_5 - term_6) - 0.5 * diag_term
        return(preds)

# The class representing the recommender, contains methods for training and recommending
class Recommender:
    def __init__(self, dataset, args, ckpt_path="./tf_ckpts"):
        self.dataset = dataset
        self.args = args
        self.ckpt_path = ckpt_path

        # Use a training batch to figure out feature dimensionality and init inputs
        pos_users, pos_feats = self.dataset.generate_pos_train_batch_sp(ith_seed = 1)
        neg_users, neg_feats = self.dataset.generate_neg_train_batch_sp(ith_seed = 1)

        # Format training data to create predictions, including sparse features
        pos_users, sparse_pos_feats = self.feed_dict(pos_users, pos_feats)
        neg_users, sparse_neg_feats = self.feed_dict(neg_users, neg_feats)

        self.feature_dim = pos_feats.shape[1]
        self.args.logger('Feature dimension = ' + str(self.feature_dim) + "x" + str(self.args.num_dims), 1)

        # Init an empty keras model with Adam optimizer
        self.model = tf.keras.Sequential()
        self.opt = tf.keras.optimizers.Adam(learning_rate=self.args.starting_lr)

        if self.args.model == "PRME":
            # The only 'layer' is PRME, this declaration will initialize new random matricies
            self.model.add(PRME(self.feature_dim, self.args.num_dims, self.args.random_seed))

            # Increases as model quality increases, multiplied by -1 for minimization
            prereg_loss = lambda: tf.reduce_sum(tf.math.log(1e-6 + tf.math.sigmoid(
                    ((self.model(sparse_pos_feats) - self.model(sparse_neg_feats)) * args.secondary_reg_scale)))) * -1

            # L2 regularization, using scaling passed in from args
            l2_reg = lambda: tf.add_n([
                tf.reduce_sum(tf.math.square(self.model.layers[0].var_linear)) * self.args.linear_reg,
                tf.reduce_sum(tf.math.square(self.model.layers[0].var_factors)) * self.args.emb_reg
            ])

            # Total loss expressed as sum of model loss and l2 regularization
            self.loss_fn = lambda: tf.add_n([prereg_loss(), l2_reg()])

            # Save model and low-dimenstional features as attributes for later reference
            self.var_linear = self.model.layers[0].var_linear
            self.var_factors = self.model.layers[0].var_factors

            # Declare variables to be included in checkpoint and save checkpoint and checkpoint manager as model attributes
            self.ckpt = tf.train.Checkpoint(step = tf.Variable(1), optimizer = self.opt, var_linear = self.var_linear,
                                            var_factors = self.var_factors)
            self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.ckpt_path, max_to_keep=3)

        if self.args.model == "TransFM":
            # The only 'layer' is PRME, this declaration will initialize new random matricies
            self.model.add(TransFM(self.feature_dim, self.args.num_dims, self.args.random_seed))

            # Increases as model quality increases, multiplied by -1 for minimization
            prereg_loss = lambda: tf.reduce_sum(tf.math.log(1e-6 + tf.math.sigmoid(
                    ((self.model(sparse_pos_feats) - self.model(sparse_neg_feats)) * args.secondary_reg_scale)))) * -1

            # L2 regularization, using scaling passed in from args
            l2_reg = lambda: tf.add_n([
                tf.reduce_sum(tf.math.square(self.model.layers[0].var_linear)) * self.args.linear_reg,
                tf.reduce_sum(tf.math.square(self.model.layers[0].var_emb_factors)) * self.args.emb_reg,
                tf.reduce_sum(tf.math.square(self.model.layers[0].var_trans_factors)) * self.args.trans_reg
            ])

            # Total loss expressed as sum of model loss and l2 regularization
            self.loss_fn = lambda: tf.add_n([prereg_loss(), l2_reg()])

            # Save model and low-dimenstional features as attributes for later reference
            self.var_linear = self.model.layers[0].var_linear
            self.var_emb_factors = self.model.layers[0].var_emb_factors
            self.var_trans_factors = self.model.layers[0].var_trans_factors

            # Declare variables to be included in checkpoint and save checkpoint and checkpoint manager as model attributes
            self.ckpt = tf.train.Checkpoint(step = tf.Variable(1), optimizer = self.opt, var_linear = self.var_linear,
                                            var_emb_factors = self.var_emb_factors, var_trans_factors = self.var_trans_factors)
            self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.ckpt_path, max_to_keep=3)

        # Declare the minimization should occur by modifying all trainable weights
        self.var_list_fn = lambda: self.model.trainable_weights

    # Structure user and object features into sparse inputs to model
    def feed_dict(self, user_obj, feat_obj):
        pl_users = user_obj.nonzero()[1]
        pl_indices = np.hstack((feat_obj.nonzero()[0][:, None], feat_obj.nonzero()[1][:, None]))
        pl_values = feat_obj.data.astype('float32')
        pl_shape = feat_obj.shape
        sparse_feats = tf.SparseTensor(pl_indices, pl_values, pl_shape)
        return pl_users, sparse_feats

    def format_output(self, user_list, item_list, score_list, json_out = False):

        out_df = pd.DataFrame()
        out_df['user_id'] = user_list
        out_df['item_id'] = item_list
        out_df['score'] = score_list

        if json_out:
            final_json = "{"
            unique_ids = out_df['user_id'].unique()
            for pos, user_id in enumerate(unique_ids):

                final_json += "\"{}\": {{\"".format(user_id)
                user_sample = out_df[out_df['user_id'] == user_id]
                user_sample = user_sample.drop(columns = ['user_id'])
                user_sample = user_sample.set_index('item_id')
                user_json = user_sample.to_json(orient = 'columns')
                # if items and scores are empty strings, replace with empty dictionary
                if user_json[11] == "\"":
                    final_json = final_json[:-1]
                    final_json = final_json + "}"
                else:
                    final_json += user_json[11:-1]

                if pos != len(unique_ids) - 1:
                    final_json += ", "
                else:
                    final_json += "}"
            final_json = json.loads(final_json)

            return final_json
        else:
            return out_df
    # Restore the best model and generate predictions for all users and items
    def user_preds(self, user_idx, item_whitelist=None):

        # Find starting index in deploy_set
        this_idx = range(self.dataset.deploy_user_start_idx[user_idx], self.dataset.deploy_user_start_idx[user_idx + 1])

        # Generate user list and features for this sample
        deploy_users, deploy_feats = self.dataset.generate_deploy_batch_sp(idx_sample = this_idx, one_pass = 0)
        sparse_deploy_users, sparse_deploy_feats = self.feed_dict(deploy_users, deploy_feats)

        # Run recommender and find scores for user, item pairs
        deploy_preds = self.model(sparse_deploy_feats)

        # Convert user index to user_id
        user_id = self.dataset.idx_to_user[user_idx]

        # List of user ids same length of input index
        #user_list = [user_id for i in this_idx]

        # Use the block of user, item pairs to create a list of item ids for that user
        item_ids = [self.dataset.deploy_cols[i] for i in this_idx]

        # List of all item_ids used as input
        item_list = np.array([self.dataset.idx_to_item[i] for i in item_ids])

        # List of all scores output by the model, formatted as floats
        score_list = deploy_preds.numpy()

        # Make sure top_k isn't larger than all items in region
        adjusted_k = min(len(this_idx), self.args.return_k_preds)

        # Find list of indices representing top k scores, sorted for largest values
        top_k_indices = score_list.argsort(axis=0)[::-1][:adjusted_k]

        # Find scores / items corresponding to highest indices
        top_k_scores = [score_list[i][0][0] for i in top_k_indices]
        top_k_items = [item_list[i][0] for i in top_k_indices]

        # List of user ids same length of output
        user_list = [user_id for i in top_k_indices]

        return user_list, top_k_items, top_k_scores

    def get_all_user_preds(self, final_user_list, final_item_list, final_score_list, this_chunk=0, num_chunks=1, user_id=None, json_out=False):
        # iterate over all users (or partition of users) and produce predictions

        # One more chunk then evenly divisible to ensure all users are represented
        users_per_chunk = (self.dataset.num_users // num_chunks) + 1

        user_start = users_per_chunk * this_chunk

        user_end = users_per_chunk * (this_chunk + 1)

        # Make sure end is at most total number of users
        user_end = min(user_end, self.dataset.num_users)

        # Create a prediction chunk for each iteration and merge into output
        for this_user_idx in range(user_start, user_end):

            # Pass in this user's index and return three lists of same length, to be merged into df
            user_list, item_list, score_list = self.user_preds(user_idx = this_user_idx)

            final_user_list.extend(user_list)
            final_item_list.extend(item_list)
            final_score_list.extend(score_list)

        # Structure lists into either pd.Dataframe or json, depending on json_out val
        deploy_out = self.format_output(final_user_list, final_item_list, final_score_list, json_out)

        return deploy_out

    # Restore the best model and generate predictions for all users and items
    def deploy_preds(self, this_chunk = 0, num_chunks = 1, user_id=None, json_out=False):

        # Restore best model weights
        final_ckpt = self.ckpt_manager.latest_checkpoint
        self.ckpt.restore(final_ckpt)

        # Init lists used in case more than 1 user predictions requested
        final_user_list = []
        final_item_list = []
        final_score_list = []

        if type(user_id) == list:
            if 0 in user_id:
                deploy_out = self.get_all_user_preds(final_user_list, final_item_list, final_score_list, this_chunk, num_chunks, user_id, json_out)
                return deploy_out
            # Iterate over users and only include predictions for included users
            for this_user_id in user_id:

                # Convert ids to indices
                this_user_idx = self.dataset.user_to_idx.get(this_user_id)

                if this_user_idx is not None:
                    # Pass in this user's index and return three lists of same length, to be merged into df
                    user_list, item_list, score_list = self.user_preds(user_idx = this_user_idx)
                else:
                    user_list = [this_user_id]
                    item_list = [""]
                    score_list = [""]
                final_user_list.extend(user_list)
                final_item_list.extend(item_list)
                final_score_list.extend(score_list)

            # Structure lists into either pd.Dataframe or json, depending on json_out val
            deploy_out = self.format_output(final_user_list, final_item_list, final_score_list, json_out)

            return(deploy_out)

        # If no user_id, or list of user_ids, iterate over all users (or partition of users)
        else:
            deploy_out = self.get_all_user_preds(final_user_list, final_item_list, final_score_list, this_chunk, num_chunks, user_id, json_out)
            return deploy_out

    # Function to train, test, and save best model checkpoint
    def train(self):

        # Setting tf random seeds must be done inside each graph
        np.random.seed(self.args.random_seed)
        tf.random.set_seed(self.args.random_seed)

        # Initialize bests, starting with 0/-1 so that first score becomes the 'best' after first iteration
        best_epoch = 0
        best_val_acc = -1
        best_test_acc = -1

        # Iterate through epoch until at most max_iters, but can auto-stop if model overfits
        for epoch in range(self.args.max_iters):

            # Iterate seed
            #ith_seed = self.args.random_seed * epoch

            # Generate training data
            pos_users, pos_feats = self.dataset.generate_pos_train_batch_sp(items_per_user = self.args.num_train_samples)
            neg_users, neg_feats = self.dataset.generate_neg_train_batch_sp(items_per_user = self.args.num_train_samples)

            # Format training data to create predictions, including sparse features
            pos_users, sparse_pos_feats = self.feed_dict(pos_users, pos_feats)
            neg_users, sparse_neg_feats = self.feed_dict(neg_users, neg_feats)

            # Train model by minimizing lost by modifying the trainable variables
            self.opt.minimize(self.loss_fn, self.var_list_fn)

            # If this epoch should be an evaluation step, calculate accuracy and check for new best model
            if epoch % self.args.eval_freq == 0:

                # Check if val_set is being used
                if self.args.val_set == 1:
 
                    # Generate list of users and features from val dataset
                    pos_users, pos_feats = self.dataset.generate_pos_val_batch_sp()
                    neg_users, neg_feats = self.dataset.generate_neg_val_batch_sp(items_per_user = self.args.num_val_samples)

                else:

                    # Generate list of users and features from test dataset
                    pos_users, pos_feats = self.dataset.generate_pos_test_batch_sp()
                    neg_users, neg_feats = self.dataset.generate_neg_test_batch_sp(items_per_user = self.args.num_val_samples)

                # Format training data to create predictions, including sparse features
                pos_users, sparse_pos_feats = self.feed_dict(pos_users, pos_feats)
                neg_users, sparse_neg_feats = self.feed_dict(neg_users, neg_feats)

                pos_preds = self.model(sparse_pos_feats)

                # Get the correct number of pos rows to compare with the neg rows
                pos_preds = tf.tile(pos_preds, tf.constant([1, self.args.num_val_samples]))

                # Reshape the above to match neg_preds shape
                pos_preds = tf.reshape(pos_preds, [-1, 1])

                # Model scores random negative samples, multiple per user
                neg_preds = self.model(sparse_neg_feats)

                # Validation Accuracy - how often the correct answer is predicted over the random one
                val_acc = np.mean(tf.dtypes.cast(((pos_preds - neg_preds) > 0), tf.float32))
                self.args.logger("Epoch: " + str(epoch) + ", Current val ACC: " + str(val_acc), 1)

                # If val_acc is a new best, save this iteration and check test_val
                if val_acc > best_val_acc:
                    best_epoch = epoch
                    best_val_acc = val_acc
                      
                    # If using a seperate test set, check the test_acc as well
                    if self.args.val_set == 1:

                        # Generate list of users and features from test dataset
                        pos_users, pos_feats = self.dataset.generate_pos_test_batch_sp()
                        neg_users, neg_feats = self.dataset.generate_neg_test_batch_sp(items_per_user = self.args.num_val_samples)

                        # Format training data to create predictions, including sparse features
                        pos_users, sparse_pos_feats = self.feed_dict(pos_users, pos_feats)
                        neg_users, sparse_neg_feats = self.feed_dict(neg_users, neg_feats)

                        pos_preds = self.model(sparse_pos_feats)

                        # Get the correct number of pos rows to compare with the neg rows
                        pos_preds = tf.tile(pos_preds, tf.constant([1, self.args.num_val_samples]))

                        # Reshape the above to match neg_preds shape
                        pos_preds = tf.reshape(pos_preds, [-1, 1])

                        # Model scores random negative samples, multiple per user
                        neg_preds = self.model(sparse_neg_feats)

                        # Validation Accuracy - how often the correct answer is predicted over the random one
                        best_test_acc = np.mean(tf.dtypes.cast(((pos_preds - neg_preds) > 0), tf.float32))
                        self.args.logger('New best epoch: ' + str(best_epoch) + ', Test ACC: ' +str(best_test_acc), 1)

                    # If option to generate deployment predictions is enabled, save checkpoints
                    if self.args.deploy_preds == 1:

                        #dynamically create the name of the current checkpoint using epoch
                        self.most_recent_checkpoint = 'checkpoint_' + str(epoch)
                        self.ckpt_manager.save()

                # Early stopping criteria
                else:

                    # Make sure not quitting too early
                    if epoch < self.args.min_epoch:
                        pass

                    # If no improvement in quit_delta rounds, print final results, create preds
                    elif epoch >= (best_epoch + self.args.quit_delta):

                        self.args.logger('Early stopping, best epoch: ' + str(best_epoch) + ', Val ACC: ' +
                                    str(best_val_acc), 1)

                        break
        # Completed all iterations through max iters, end train
        self.args.logger('Finished, best epoch: ' + str(best_epoch) + ', Val ACC: ' +
                                    str(best_val_acc) + ', Test ACC: ' +
                                    str(best_test_acc), 1)