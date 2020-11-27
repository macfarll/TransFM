import scipy.sparse as sp
import numpy as np
import pandas as pd
from collections import defaultdict
import copy
import random

# Class to represent a dataset
class Dataset:
    def __init__(self, args):
        
        df = args.filename
        self.args = args
        
        # Use random seed so that future runs with the same params are deterministic 
        np.random.seed(self.args.random_seed)
        
        print('First pass')
        print('\tnum_users = ' + str(len(df['user_id'].unique())))
        print('\tnum_items = ' + str(len(df['item_id'].unique())))
        print('\tdf_shape  = ' + str(df.shape))
        
        user_counts = df['user_id'].value_counts()
        print('Collected user counts...')
        item_counts = df['item_id'].value_counts()
        print('Collected item counts...')
        
        # Filter based on user and item counts
        df = df[df.apply(lambda x: user_counts[x['user_id']] >= self.args.user_min, axis=1)]
        print('User filtering done...')
        df = df[df.apply(lambda x: item_counts[x['item_id']] >= self.args.item_min, axis=1)]
        print('Item filtering done...')
        
        print('Second pass')
        self.args.logger('\tnum_users = ' + str(len(df['user_id'].unique())), 1)
        self.args.logger('\tnum_items = ' + str(len(df['item_id'].unique())), 1)
        self.args.logger('\tdf_shape  = ' + str(df.shape), 1)
        
        # If restricting users to certain regions, assign regional mapping dataframe to dataset
        if type(self.args.user_region_df) == pd.DataFrame:
            
            # Assign both dataframes as attributes of dataset
            self.user_region_df = self.args.user_region_df
            self.item_region_df = self.args.item_df[['item_id', 'region_id']]
            
        # Either way, make sure 'region_id' is no longer in item_df and set as dataset attribute
        self.item_df = self.args.item_df.drop(columns = ['region_id'], errors='ignore')
        
        # Original code normalized temporal values here
        
        print('Constructing datasets...')
        training_set = defaultdict(list)
        # Start counting users and items at 1 to facilitate sparse matrix computation
        # NOTE: this means these dictionaries will NOT be 0 indexed, careful tracking 
        #    of 0-indexed series is needed for development  
        num_users = 0
        num_items = 0
        item_to_idx = {}
        user_to_idx = {}
        idx_to_item = {}
        idx_to_user = {}
        
        # Iterate through items, creating dense dicts for item and users
        for row in df.itertuples():
            
            # New item
            if row.item_id not in item_to_idx:
                item_to_idx[row.item_id] = num_items
                idx_to_item[num_items] = row.item_id
                num_items += 1
                
            # New user
            if row.user_id not in user_to_idx:
                user_to_idx[row.user_id] = num_users
                idx_to_user[num_users] = row.user_id
                num_users += 1
                
            # Converts all ratings to positive implicit feedback
            training_set[user_to_idx[row.user_id]].append(
                    (item_to_idx[row.item_id], row.time))
        
        # Save item and use count as attributes of dataset
        self.num_users = num_users
        self.num_items = num_items
        
        # Sort training_set by time, so each users' observations are in order
        for user in training_set:
            training_set[user].sort(key=lambda x: x[1])
        
        # Create dictionaries for user to region and region to user
        if type(self.args.user_region_df) == pd.DataFrame:
            
            # Define default values to fall back on for users without regions                
            default_item_list = range(self.num_items)
            default_num_items = self.num_items
            
            # Only include users with enough data to be included after prefiltering
            active_user_mask = self.args.user_region_df['user_id'].isin(df['user_id'].unique())
            user_region_df = self.args.user_region_df.loc[active_user_mask]
            
            # Start with empty dictionaries for region lookups
            item_to_region = {}
            region_to_item = {}
            user_to_region_item_idx = {}
            user_to_num_items = {}
            region_str_to_items = {}
            
            # Iterate over tall input data where each row is a single item, region pair
            for row in self.item_region_df.itertuples():
                
                # Building dict with item_id as key, and value is that item's region
                item_to_region[row.item_id] = [int(row.region_id)]
                
                # Check if this item is being used in dataset (has adequate data)
                if item_to_idx.get(row.item_id):
                
                    # Building dict with region_id as key, and value is list of items in region
                    if row.region_id not in region_to_item:

                        # If this row is a new region then add entry to dict, where value is list with 1 value
                        region_to_item[row.region_id] = [item_to_idx.get(row.item_id)] 
                    else:

                        # If region already in dict, append new item_id to the value
                        region_to_item[row.region_id].append(item_to_idx.get(row.item_id))  
            
            # Iterate through user region pairs, and find all items in those regions
            for row in user_region_df.itertuples():
                
                # Building dict with cached_region_ids as key and all items for that region as value
                if row.cached_region_ids not in region_str_to_items:
                    
                    # Unpack string into list of region_ids
                    this_region_list = row.cached_region_ids.split(",")
                    
                    # Create empty list of all items in list of regions 
                    this_region_list_items = []
                    
                    # Iterate over regions and add all items to list
                    for this_region in this_region_list:
                        
                        # Adds all items associated with this_region, adds nothing if key not found
                        this_region_list_items.extend(region_to_item.get(int(this_region), []))
                        
                    # If all regions in list collectively have 0 items, replace with default list 
                    if this_region_list_items == []:
                        this_region_list_items = default_item_list
                    
                    # Add the full list to the dictionary
                    region_str_to_items[row.cached_region_ids] = this_region_list_items
                
                # Translate the user_id to the user_idx
                this_user_idx = user_to_idx[row.user_id] # Should this be a .get? what to default?
                
                # Building dict with user_id as key and all items associated with their regions as value
                user_to_region_item_idx[this_user_idx] = region_str_to_items.get(row.cached_region_ids, default_item_list)
                
                # Building a dict with user_id as key and number of items associated with their regions as value
                user_to_num_items[this_user_idx] = len(region_str_to_items.get(row.cached_region_ids, default_num_items))

            # Populate all users without region data with all items
            for user_idx in training_set:
                
                if user_idx not in user_to_region_item_idx:
                    user_to_region_item_idx[user_idx] = default_item_list
                    user_to_num_items[user_idx] = default_num_items
        
        # Create deep copy of training set before removing test and val for deployment
        deploy_set = copy.deepcopy(training_set)
        
        # Init lists of datasets
        training_times = {}
        
        # Only init val lists if they'll be used
        if self.args.val_set == 1:
            val_set = {} 
            val_times = {}
        
        test_set = {}
        test_times = {}
        # Map from user to set of items for easy lookup
        item_set_per_user = {}
        
        if self.args.val_set == 1:
          print("Trying to structure dataset with validation set, this isn't fully supported. ")

        for user in training_set:
            
            # If user has inadequate data for train/test split, use dummy values. (if val_set == 0, only 1 train, 1 test needed)
            if len(list(training_set[user])) < (2 + self.args.val_set):
                
                # Reviewed < 3 items, insert dummy values
                test_set[user] = (-1, -1)
                test_times[user] = (-1, -1)
                
                if self.args.val_set == 1:
                    val_set[user] = (-1, -1)
                    val_times[user] = (-1, -1)
                
            # User has adequate data, populate train, val, and test data
            else:
                
                # No validation set needed, only populate train/test
                if self.args.val_set == 0:
                    
                    # Remove last item from train to serve as test
                    test_item, test_time = training_set[user].pop() 
                    
                    # This lookback is what requires the starting at 1 indexing
                    last_item, last_time = training_set[user][-1] 
                    
                    # Test item is the most recent item by user, last item is the one previous
                    test_set[user] = (test_item, last_item) 
                    
                    # Time functionality currently not supported
                    test_times[user] = (test_time, last_time)
                
                #note: currently not well maintained, will need adjustments to work with some new functionality 
                else:
                    test_item, test_time = training_set[user].pop() #remove last item from train to serve as test
                    val_item, val_time = training_set[user].pop() #remove second to last item from train to serve as val
                    last_item, last_time = training_set[user][-1] #this lookback is what requires the starting at 1 indexing 
                    test_set[user] = (test_item, val_item) #test item is the most recent item by user, val item is the one previous
                    test_times[user] = (test_time, val_time)
                    val_set[user] = (val_item, last_item) #val item is second to last item by usaer, 'last item' is third to last (last in training set) 
                    val_times[user] = (val_time, last_time)
                    
            # Separate timestamps and create item set
            training_times[user] = copy.deepcopy(training_set[user])
            training_set[user] = [x[0] for x in training_set[user]]
            item_set_per_user[user] = set(training_set[user])
            
        # Iterate over users to get total count of training items
        num_train_items = 0
        for user in training_set:
            num_train_items += len(list(training_set[user]))

        # Set newly created datasets, dictionaries, and counts as dataset attributes
        self.deploy_set = deploy_set
        # self.deploy_times = deploy_times
        
        self.training_set = training_set
        self.training_times = training_times
        
        if self.args.val_set == 1:
            self.val_set = val_set
            self.val_times = val_times
        
        self.test_set = test_set
        self.test_times = test_times
        self.item_set_per_user = item_set_per_user

        self.item_to_idx = item_to_idx
        self.user_to_idx = user_to_idx
        self.idx_to_item = idx_to_item
        self.idx_to_user = idx_to_user
        
        if type(self.args.user_region_df) == pd.DataFrame:
            self.item_to_region = item_to_region
            self.region_to_item = region_to_item
            self.user_to_region_item_idx = user_to_region_item_idx
            self.region_str_to_items = region_str_to_items
            self.user_to_num_items = user_to_num_items
            self.user_region_df = user_region_df

        
        self.num_train_items = num_train_items


        # Full_features replaced 'content'
        if self.args.features == 'full_features':
          
            # Place index on users df and set as attribute of dataset
            print('Reading user demographics...')
            user_df = self.args.user_df
            user_df = user_df.set_index('user_id')
            self.user_df = user_df

            # Create dictionary to build sparse user feature matrix
            self.orig_indices = []
            for i in range(1, self.num_users):
                self.orig_indices.append(self.idx_to_user[i])
            self.user_feats = sp.csr_matrix(user_df.loc[self.orig_indices].values)
          
            # Repeat above for items instead of users
            print('Reading item demographics...')
            self.item_df = self.item_df.set_index('item_id')
            self.orig_item_indices = []
            for i in range(1, self.num_items):
                self.orig_item_indices.append(self.idx_to_item[i])
            self.item_feats = sp.csr_matrix(self.item_df.loc[self.orig_item_indices].values)
        
        else:
            self.user_feats = None
            self.item_feats = None
            
        # Create scipy.sparse matrices #NOTE: This is where indexing fix matters
        self.user_one_hot = sp.identity(self.num_users - 0).tocsr()
        self.item_one_hot = sp.identity(self.num_items - 0).tocsr()
        
        for user in deploy_set:

            # Separate timestamps and create item set
            deploy_set[user] = [x[0] for x in deploy_set[user]]
        
        # Sparse training matrices
        train_rows = []
        train_cols = []
        train_vals = []
        train_prev_vals = []
        train_times = []
        train_prev_times = []
        
        # Init list for generating negative samples #note - must happen after .pop() to avoid leakage
        weighted_item_list = []
        
        # Restructure data into training rows with previous items as feature
        for user in self.training_set:

            # Start with 1st item instead of 0th item of training data to allow for prev item reference
            for i in range(1, len(list(self.training_set[user]))):
                
                item = self.training_set[user][i]
                item_prev = self.training_set[user][i-1]
                item_time = self.training_times[user][i]
                item_prev_time = self.training_times[user][i-1]
                train_rows.append(user)
                train_cols.append(item)
                train_vals.append(1)
                train_prev_vals.append(item_prev)
                train_times.append(item_time[1])
                train_prev_times.append(item_prev_time[1])
                
                # Add one observation to weighted item list
                weighted_item_list.append(item) 
                
        # Normalize values then set weights as attribute for negative sampling
        self.weighted_item_list = weighted_item_list
        
        # Determine mean and std to normalize timestamps
        self.train_mean = np.mean(train_times)
        self.train_std  = np.std(train_times)
        self.ONE_YEAR = (60 * 60 * 24 * 365) / self.train_mean
        self.ONE_DAY = (60 * 60 * 24) / self.train_mean
        train_times = (train_times - self.train_mean) / self.train_std

        self.sp_train = sp.coo_matrix((train_vals, (train_rows, train_cols)),
                shape=(self.num_users, self.num_items))
        self.sp_train_prev = sp.coo_matrix((train_prev_vals, (train_rows, train_cols)),
                shape=(self.num_users, self.num_items))
        self.sp_train_times = sp.coo_matrix((train_times, (train_rows, train_cols)),
                shape=(self.num_users, self.num_items))
        self.sp_train_prev_times = sp.coo_matrix((train_prev_times, (train_rows, train_cols)),
                shape=(self.num_users, self.num_items))
        
        # Repeat training processing for validation set
        if self.args.val_set == 1:
            
            # Sparse validation matrices
            val_rows = []
            val_cols = []
            val_vals = []
            val_prev_vals = []
            val_times = []
            val_prev_times = []
            for user in self.val_set:
                item = self.val_set[user][0]
                item_prev = self.val_set[user][1]
                item_time = self.val_times[user][0]
                item_prev_time = self.val_times[user][1]
                if item == -1 or item_prev == -1:
                    continue

                val_rows.append(user)
                val_cols.append(item)
                val_vals.append(1)
                val_prev_vals.append(item_prev)
                val_times.append(item_time)
                val_prev_times.append(item_prev_time)

            #normalize val timestamps with train mean/std (avoid leakage)    
            val_times = (val_times - self.train_mean) / self.train_std
                
            self.sp_val = sp.coo_matrix((val_vals, (val_rows, val_cols)),
                    shape=(self.num_users, self.num_items))
            self.sp_val_prev = sp.coo_matrix((val_prev_vals, (val_rows, val_cols)),
                    shape=(self.num_users, self.num_items))
            self.sp_val_times = sp.coo_matrix((val_times, (val_rows, val_cols)),
                    shape=(self.num_users, self.num_items))
            self.sp_val_prev_times = sp.coo_matrix((val_prev_times, (val_rows, val_cols)),
                    shape=(self.num_users, self.num_items))

        # Repeat processing for test set
        test_rows = []
        test_cols = []
        test_vals = []
        test_prev_vals = []
        test_times = []
        test_prev_times = []
        for user in self.test_set:
            item = self.test_set[user][0] #for test and val set, this_item is 0
            item_prev = self.test_set[user][1] #prev_item is 1
            item_time = self.test_times[user][0]
            item_prev_time = self.test_times[user][1]
            if item == -1 or item_prev == -1:
                continue

            test_rows.append(user)
            test_cols.append(item)
            test_vals.append(1)
            test_prev_vals.append(item_prev)
            test_times.append(item_time)
            test_prev_times.append(item_prev_time)

        # Normalize test timestamps with train mean/std (avoid leakage)    
        test_times = (test_times - self.train_mean) / self.train_std
            
        self.sp_test = sp.coo_matrix((test_vals, (test_rows, test_cols)),
                shape=(self.num_users, self.num_items))
        self.sp_test_prev = sp.coo_matrix((test_prev_vals, (test_rows, test_cols)),
                shape=(self.num_users, self.num_items))
        self.sp_test_times = sp.coo_matrix((test_times, (test_rows, test_cols)),
                shape=(self.num_users, self.num_items))
        self.sp_test_prev_times = sp.coo_matrix((test_prev_times, (test_rows, test_cols)),
                shape=(self.num_users, self.num_items))

        # Sparse training matrices for deploy set #old deploy method
        deploy_rows = []
        deploy_cols = []
        deploy_vals = []
        deploy_prev_vals = []
        
        #deploy_times = []
        #deploy_prev_times = []
        
        # Check if region data should be used
        if type(self.args.user_region_df) == pd.DataFrame:
            
            # Create empty dict to find index user starts at for efficient user lookups
            deploy_user_start_idx = {}
            
            # Start dict with user 0's starting point, index 0
            deploy_user_start_idx[0] = 0
            
            # Iterate over the copy of the training set, for previous items and user feats
            for user in self.deploy_set:
                
                # Find index of last item, to populate 'prev_item'
                last_item_idx = len(list(self.deploy_set[user])) - 1
                
                item_prev = self.deploy_set[user][last_item_idx - 1]
                
                this_user_items = self.user_to_region_item_idx.get(user)
                
                # The next user will start after this one, so add this user's num rows to this user's start idx
                deploy_user_start_idx[user + 1] = len(this_user_items) + deploy_user_start_idx[user]
                
                for item in this_user_items:
                    
                    # For each item add a row for this user, their prev item, a positive, and each item
                    deploy_rows.append(user)
                    deploy_cols.append(item)
                    deploy_vals.append(1)
                    deploy_prev_vals.append(item_prev)
                
        # If region data isn't being used, build exhaustive list of all user, item pairs
        else:
            
            # Create empty dict to find index user starts at for efficient user lookups
            deploy_user_start_idx = {}
            
            # Start dict with user 0's starting point, index 0
            deploy_user_start_idx[0] = 0

            # Iterate over the copy of the training set, for previous items and user feats
            for user in self.deploy_set:

                # Find index of last item, to populate 'prev_item'
                last_item_idx = len(list(self.deploy_set[user])) - 1

                item_prev = self.deploy_set[user][last_item_idx - 1]

                # The next user will start after this one, so add this user's num rows to this user's start idx
                deploy_user_start_idx[user + 1] = self.num_items + deploy_user_start_idx[user]

                for item in range(self.num_items):

                    # For each item add a row for this user, their prev item, a positive, and each item
                    deploy_rows.append(user)
                    deploy_cols.append(item)
                    deploy_vals.append(1)
                    deploy_prev_vals.append(item_prev)

        self.deploy_user_start_idx = deploy_user_start_idx
        self.deploy_rows = deploy_rows
        self.deploy_cols = deploy_cols
        self.deploy_vals = deploy_vals
        self.deploy_prev_vals = deploy_prev_vals
        
        # Assign the totla number of rows in deploy set as model attribute, to determine batch size
        self.deploy_num_rows = len(deploy_rows)
        
    # Function to generate model input training data, actual values
    def generate_pos_train_batch_sp(self, ith_seed = 1, items_per_user = 3):
        
        np.random.seed(ith_seed)

        # Subtract 1 to account for missing 0 index
        user_indices = np.repeat(self.sp_train.row, items_per_user) - 1
        prev_indices = np.repeat(self.sp_train_prev.data, items_per_user) - 1
        pos_indices = np.repeat(self.sp_train.col, items_per_user) - 1
        
        # Convert from indices to one hot matrices
        pos_users = self.user_one_hot[user_indices]
        prev_items = self.item_one_hot[prev_indices]
        pos_items = self.item_one_hot[pos_indices]

        # Horizontally stack sparse matrices to get single positive
        pos_feats = sp.hstack([pos_users, prev_items, pos_items])
            
        # Full_features replaced 'content', adds both user and item features
        if self.args.features == 'full_features':
            # Join with content data            
            user_content = self.user_feats[user_indices]
            pos_item_content = self.item_feats[pos_indices]
            pos_feats = sp.hstack([pos_feats, user_content, pos_item_content])

        return(pos_users, pos_feats)
      
    # Generate random observations of the same size as actual training input
    def generate_neg_train_batch_sp(self, ith_seed = 1, items_per_user = 3):
        
        np.random.seed(ith_seed)
        
        # Subtract 1 to account for missing 0 index
        user_indices = np.repeat(self.sp_train.row, items_per_user) - 1
        prev_indices = np.repeat(self.sp_train_prev.data, items_per_user) - 1
        
        # Check if region data should be used
        if type(self.args.user_region_df) == pd.DataFrame:
            neg_indices = []
            
            # Iterate over each user index to build training data for only this region
            for user_idx in user_indices:
                
                # Unpacked list of lists generated above into a single list of elligible items for user
                #     If no regions associated with user, defdault to all items
                all_user_items = self.user_to_region_item_idx.get(user_idx + 1, [])
                
                # Randomly sample a single element from the list
                rand_idx = random.randint(0, len(all_user_items) - 1)
                neg_indices.append(all_user_items[rand_idx] - 1)
                
        
        elif self.args.weighted_sampling == 1:
            neg_indices = np.random.choice(range(len(self.weighted_item_list)), 
                          size = len(self.sp_train.row) * items_per_user) 
        
            neg_indices = [(self.weighted_item_list[i] - 1) for i in neg_indices] 
            
        else:
            neg_indices = np.random.randint(1, self.sp_train.shape[1],
                          size=len(self.sp_train.row)*items_per_user, dtype=np.int32) - 1
        
        # Convert from indices to one hot matrices
        neg_users = self.user_one_hot[user_indices]
        prev_items = self.item_one_hot[prev_indices]
        neg_items = self.item_one_hot[neg_indices]

        # Horizontally stack sparse matrices to get negative feature matrices
        neg_feats = sp.hstack([neg_users, prev_items, neg_items])
            
        # Full_features replaced 'content', adds both user and item features
        if self.args.features == 'full_features':
            # Join with content data            
            user_content = self.user_feats[user_indices]
            neg_item_content = self.item_feats[neg_indices]
            neg_feats = sp.hstack([neg_feats, user_content, neg_item_content])

        return(neg_users, neg_feats)
    
    # Dataset containing only correct inputs, model should how to score these well
    def generate_pos_val_batch_sp(self, ith_seed = 1): 
        
        np.random.seed(ith_seed)
        
        user_indices = self.sp_val.row - 1
        prev_indices = self.sp_val_prev.data - 1
        pos_indices = self.sp_val.col - 1

        # Convert from indices to one-hot matrices
        pos_users = self.user_one_hot[user_indices]
        prev_items = self.item_one_hot[prev_indices]
        pos_items = self.item_one_hot[pos_indices]

        # Horizontally stack sparse matrices to get single positive feats
        pos_feats = sp.hstack([pos_users, prev_items, pos_items])

        # Full_features replaced 'content', adds both user and item features
        if self.args.features == 'full_features':
            # Join with content data
            user_content = self.user_feats[user_indices]
            pos_item_content = self.item_feats[pos_indices]
            pos_feats = sp.hstack([pos_feats, user_content, pos_item_content])

        return(pos_users, pos_feats)
      
    # Dataset containing random samples, model should learn these are less likely than the above
    def generate_neg_val_batch_sp(self, ith_seed = 1, items_per_user = 10): 
        
        np.random.seed(ith_seed)
        
        user_indices = np.repeat(self.sp_val.row, items_per_user) - 1
        prev_indices = np.repeat(self.sp_val_prev.data, items_per_user) - 1
        
        # Check if region data should be used
        if type(self.args.user_region_df) == pd.DataFrame:
            neg_indices = []
            
            # Iterate over each user index to build training data for only this region
            for user_idx in user_indices:
                
                # Unpacked list of lists generated above into a single list of elligible items for user
                #     If no regions associated with user, defdault to all items
                all_user_items = self.user_to_region_item_idx.get(user_idx + 1, [])
                
                # Randomly sample a single element from the list
                rand_idx = random.randint(0, len(all_user_items) - 1)
                neg_indices.append(all_user_items[rand_idx] - 1)
                
        
        elif self.args.weighted_sampling == 1:
            neg_indices = np.random.choice(range(len(self.weighted_item_list)), 
                size = len(self.sp_val.row)*items_per_user) #- 1
        
            neg_indices = [(self.weighted_item_list[i] - 1) for i in neg_indices] 
        
        else:
            neg_indices = np.random.randint(1, self.sp_val.shape[1],
                          size=len(self.sp_val.row)*items_per_user, dtype=np.int32) - 1
        
        self.neg_ind = neg_indices
        
        # Convert from indices to one-hot matrices
        neg_users = self.user_one_hot[user_indices]
        prev_items = self.item_one_hot[prev_indices]
        neg_items = self.item_one_hot[neg_indices]
        
        # Horizontally stack sparse matrices to get negative feature matrices
        neg_feats = sp.hstack([neg_users, prev_items, neg_items])
        
        # Full_features replaced 'content', adds both user and item features
        if self.args.features == 'full_features':
            # Join with content data
            user_content = self.user_feats[user_indices]
            neg_item_content = self.item_feats[neg_indices]
            neg_feats = sp.hstack([neg_feats, user_content, neg_item_content])
            
        return (neg_users, neg_feats)

    # Dataset containing only correct inputs, model should how to score these well
    def generate_pos_test_batch_sp(self, ith_seed = 1): 
        
        np.random.seed(ith_seed)
        
        user_indices = self.sp_test.row - 1
        prev_indices = self.sp_test_prev.data - 1
        pos_indices = self.sp_test.col - 1

        # Convert from indices to one-hot matrices
        pos_users = self.user_one_hot[user_indices]
        prev_items = self.item_one_hot[prev_indices]
        pos_items = self.item_one_hot[pos_indices]

        # Horizontally stack sparse matrices to get single positive feats
        pos_feats = sp.hstack([pos_users, prev_items, pos_items])

        # Full_features replaced 'content', adds both user and item features
        if self.args.features == 'full_features':
            # Join with content data
            user_content = self.user_feats[user_indices]
            pos_item_content = self.item_feats[pos_indices]
            pos_feats = sp.hstack([pos_feats, user_content, pos_item_content])

        return(pos_users, pos_feats)
      
    # Dataset containing random samples, model should learn these are less likely than the above
    def generate_neg_test_batch_sp(self, ith_seed = 1, items_per_user = 10): 
        
        np.random.seed(ith_seed)
        
        user_indices = np.repeat(self.sp_test.row, items_per_user) - 1
        prev_indices = np.repeat(self.sp_test_prev.data, items_per_user) - 1
        
        # Check if region data should be used
        if type(self.args.user_region_df) == pd.DataFrame:
            neg_indices = []
            
            # Iterate over each user index to build training data for only this region
            for user_idx in user_indices:
                
                # Unpacked list of lists generated above into a single list of elligible items for user
                #     If no regions associated with user, defdault to all items
                all_user_items = self.user_to_region_item_idx.get(user_idx + 1, [])
                
                # Randomly sample a single element from the list
                rand_idx = random.randint(0, len(all_user_items) - 1)
                neg_indices.append(all_user_items[rand_idx] - 1)
                
        
        elif self.args.weighted_sampling == 1:
            neg_indices = np.random.choice(range(len(self.weighted_item_list)), 
                size = len(self.sp_test.row)*items_per_user) #- 1
        
            neg_indices = [(self.weighted_item_list[i] - 1) for i in neg_indices] 
        
        else:
            neg_indices = np.random.randint(1, self.sp_test.shape[1],
                          size=len(self.sp_test.row)*items_per_user, dtype=np.int32) - 1
        
        self.neg_ind = neg_indices
        
        # Convert from indices to one-hot matrices
        neg_users = self.user_one_hot[user_indices]
        prev_items = self.item_one_hot[prev_indices]
        neg_items = self.item_one_hot[neg_indices]
        
        # Horizontally stack sparse matrices to get negative feature matrices
        neg_feats = sp.hstack([neg_users, prev_items, neg_items])
        
        # Full_features replaced 'content', adds both user and item features
        if self.args.features == 'full_features':
            # Join with content data
            user_content = self.user_feats[user_indices]
            neg_item_content = self.item_feats[neg_indices]
            neg_feats = sp.hstack([neg_feats, user_content, neg_item_content])
            
        return (neg_users, neg_feats)
        
    # All user, item pairs, to be evaluated in chunks defined by idx_sample
    def generate_deploy_batch_sp(self, idx_sample, one_pass = 1): 
   
        this_deploy_rows = [self.deploy_rows[i] for i in idx_sample]
        this_deploy_cols = [self.deploy_cols[i] for i in idx_sample]
        this_deploy_vals = [self.deploy_vals[i] for i in idx_sample]
        this_deploy_prev_vals = [self.deploy_prev_vals[i] for i in idx_sample]
        
        this_sp_deploy = sp.coo_matrix((this_deploy_vals, (this_deploy_rows, this_deploy_cols)),
                shape=(self.num_users, self.num_items))
        this_sp_deploy_prev = sp.coo_matrix((this_deploy_prev_vals, (this_deploy_rows, this_deploy_cols)),
                shape=(self.num_users, self.num_items))
        
        # Subtract 1 to account for missing 0 index
        user_indices = this_sp_deploy.row - 1
        prev_indices = this_sp_deploy_prev.data - 1
        deploy_indices = this_sp_deploy.col - 1

        # Convert from indices to one hot matrices
        pos_users = self.user_one_hot[user_indices]
        prev_items = self.item_one_hot[prev_indices]
        pos_items = self.item_one_hot[deploy_indices]

        # Horizontally stack sparse matrices to get single positive feature matrices
        pos_feats = sp.hstack([pos_users, prev_items, pos_items])

        # Full_features replaced 'content', adds both user and item features
        if self.args.features == 'full_features':
            # Join with content data            
            user_content = self.user_feats[user_indices]
            pos_item_content = self.item_feats[deploy_indices]
            pos_feats = sp.hstack([pos_feats, user_content, pos_item_content])

        return(pos_users, pos_feats)