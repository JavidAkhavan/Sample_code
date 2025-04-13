import cv2
import scipy.io as sio
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras import layers as tf_layer
import keras as k
from os.path import exists
from sklearn.metrics import r2_score
from termcolor import colored
import random
from scipy.signal import savgol_filter
import pandas as pd
import tqdm
import heapq

##region DATA

def load_test_data_CK(directory, aug=1):
    """
    Load test data from the given directory, process the images, and return the processed image arrays.

    Parameters:
    - directory (str): Path to the directory containing the images.
    - aug (int): Flag for data augmentation. Default is 1 (enabled).

    Returns:
    - master_arr_raw_1 (numpy.ndarray): Processed first part of the image.
    - master_arr_raw_2 (numpy.ndarray): Processed second part of the image.
    """
    old_max = 0
    list_png = glob.glob(os.path.join(directory, "**/*.png"), recursive=True)
    master_list_raw = {}
    
    for img_path in list_png:
        image_raw = cv2.imread(img_path) / 255.0
        if image_raw.shape[1] == 128:
            image_raw = image_raw[40:88, :, :]
        if old_max < np.max(image_raw):
            print(np.max(image_raw))
            old_max = np.max(image_raw)
        master_list_raw[os.path.basename(img_path).split('.png')[0]] = image_raw

    master_arr_raw = np.array(list(master_list_raw.values()))
    master_arr_raw_1, master_arr_raw_2 = master_arr_raw[:, :, :64], master_arr_raw[:, :, 64:]

    # Normalize images
    imr1m, imr2m = np.max(master_arr_raw_1), np.max(master_arr_raw_2)
    master_arr_raw_1 = master_arr_raw_1 / imr1m
    master_arr_raw_2 = master_arr_raw_2 / imr2m

    # Reshape arrays
    master_arr_raw_1 = master_arr_raw_1.reshape(-1, 48, 64, 3)
    master_arr_raw_2 = master_arr_raw_2.reshape(-1, 48, 64, 3)

    return master_arr_raw_1, master_arr_raw_2


class DataGenerator(tf.keras.utils.Sequence):
    """
    Data generator for loading and augmenting data in batches.
    
    Parameters:
    - directory (str): Path to the directory containing the images.
    - batch_size (int): Number of samples per batch. Default is 32.
    - dim1 (int), dim2 (int): Dimensions of the input images. Default is (48, 64).
    - n_channels (int): Number of image channels (3 for RGB). Default is 3.
    - reverse (int): Flag to reverse the image channels. Default is 0 (no reversal).
    - aug (int): Flag for enabling/disabling data augmentation. Default is 1 (enabled).
    """
    
    def __init__(self, directory, batch_size=32, dim1=48, dim2=64, n_channels=3, reverse=0, aug=1):
        self.batch_size = batch_size
        self.list_png = glob.glob(os.path.join(directory, "**/*.png"), recursive=True)
        self.indexes = np.arange(len(self.list_png))
        self.dim1 = dim1
        self.dim2 = dim2
        self.n_channels = n_channels
        self.reverse = reverse
        self.aug = aug

    def __len__(self):
        """Returns the number of batches per epoch."""
        return int(np.floor(len(self.list_png) / self.batch_size))

    def __getitem__(self, index):
        """Generates one batch of data."""
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        list_IDs_temp = [self.list_png[k] for k in indexes]
        X = self.__data_generation(list_IDs_temp)
        return [X]

    def __data_generation(self, list_IDs_temp):
        """
        Generates data for the given batch of image paths.

        Parameters:
        - list_IDs_temp (list): List of image paths for the current batch.

        Returns:
        - master_arr_raw_1 (numpy.ndarray): Processed batch of images (part 1).
        - master_arr_raw_2 (numpy.ndarray): Processed batch of images (part 2).
        """
        master_list_raw = {}
        master_arr_raw = []

        for ID in list_IDs_temp:
            image_raw = cv2.imread(ID) / 255.0
            if image_raw.shape[0] == 128:
                image_raw = image_raw[40:88, :, :]
            master_list_raw[os.path.basename(ID).split('.png')[0]] = image_raw
        
        for name in master_list_raw:
            master_arr_raw.append(master_list_raw[name])

        master_arr_raw = np.array(master_arr_raw)
        try:
            master_arr_raw_1 = master_arr_raw[:, :, :64]
            master_arr_raw_2 = master_arr_raw[:, :, 64:]
        except:
            print("Error splitting image channels.")

        # Normalize images
        master_arr_raw_1 = np.divide(master_arr_raw_1, 0.8705882352941177)
        master_arr_raw_2 = np.divide(master_arr_raw_2, 1.0)

        # Augment data if flag is set
        if self.aug:
            master_arr_raw_1, master_arr_raw_2 = augment_data(np.concatenate([master_arr_raw_1, master_arr_raw_2], axis=-1), shift=5, step=3, test=1)

        master_arr_raw_1 = master_arr_raw_1.reshape(-1, self.dim1, self.dim2, self.n_channels)
        master_arr_raw_2 = master_arr_raw_2.reshape(-1, self.dim1, self.dim2, self.n_channels)

        if self.reverse:
            return [master_arr_raw_2, master_arr_raw_1]
        else:
            return [master_arr_raw_1, master_arr_raw_2]


def test1(directory, ground_truth_check=0, reversed_=0):
    """
    Run the test function to evaluate the model.

    Parameters:
    - directory (str): Path to the test data directory.
    - ground_truth_check (int): Flag to include ground truth comparison. Default is 0 (disabled).
    - reversed_ (int): Flag to reverse image channels. Default is 0 (no reversal).
    """
    name_tag = os.path.basename(directory).split('\\')[-2]
    
    if ground_truth_check:
        master_list_Tc, master_list_im_1, master_list_im_2, master_list_raw_1, master_list_raw_2 = load_data_CK(directory=directory, aug=1)
        results_mean_TC = np.array([np.mean(i_) if np.mean(i_) > -1 else 0 for i_ in master_list_Tc])
        
        df = pd.DataFrame(dict(x=results_mean_TC))
        x_filtered_TC_mean = df[["x"]].apply(savgol_filter, window_length=62, polyorder=5)
        
        results_max_TC = np.array([np.mean(heapq.nlargest(3, i_.reshape([-1]))) for i_ in master_list_Tc])
        df = pd.DataFrame(dict(x=results_max_TC))
        x_filtered_TC_max = df[["x"]].apply(savgol_filter, window_length=62, polyorder=5)

        results1 = []
        batch = 25
        for i in tqdm.tqdm(range(0, len(master_list_raw_1), batch)):
            results1.append(model_1([master_list_raw_1[i:i+batch], master_list_raw_2[i:i+batch]]))
        results1 = np.concatenate(results1, axis=0)
        results1[results1 < 0] = 0

        print("R2_test: ", r2_score(np.reshape(master_list_Tc, [-1]), np.reshape(results1, [-1])))

    if True:
        params = {
            'directory': directory,
            'batch_size': 512,
            'dim1': 48,
            'dim2': 64,
            'n_channels': 3,
            'reverse': reversed_,
            'aug': 0
        }

        training_generator = DataGenerator(**params)
        results1 = model_1.predict(training_generator)
        results1 = np.multiply(results1, 5500)
        results1[results1 < 0] = 0

        plt.figure()
        results_max = np.array([np.mean(heapq.nlargest(3, i_.reshape([-1]))) for i_ in results1])
        plt.plot(results_max)

        df = pd.DataFrame(dict(x=results_max))
        x_filtered_max = df[["x"]].apply(savgol_filter, window_length=62, polyorder=5)
        plt.plot(x_filtered_max)

        if ground_truth_check:
            plt.plot(x_filtered_TC_max)

        plt.title(f"{name_tag} Max")
        plt.xlabel("Frames")
        plt.ylabel("Frame Temperature Max Value")
        plt.legend(["Single Frame Data", "Gaussian Smoothed Data"])

        plt.figure()
        w, N = 20, 60
        results_mean = argmax_2d(results1, w=w, under_ther=N, method=1)
        plt.plot(results_mean)

        df = pd.DataFrame(dict(x=results_mean))
        x_filtered_mean = df[["x"]].apply(savgol_filter, window_length=30, polyorder=5)
        plt.plot(x_filtered_mean)

        if ground_truth_check:
            plt.plot(x_filtered_TC_mean)

        plt.title(f"{name_tag} Mean \n using w={w} N={N}")
        plt.xlabel("Frames")
        plt.ylabel("Frame Temperature Mean Value")
        plt.legend(["Single Frame Data", "Gaussian Smoothed Data"])

        out = np.array([results_max, x_filtered_max.values[:, 0], results_mean, x_filtered_mean.values[:, 0]])
        a = pd.DataFrame(data=np.transpose(out))
        a.columns = ['all_max', 'all_max_smooth', 'all_mean', 'all_mean_smooth']

        with pd.ExcelWriter(r'D:\Javid\InternShip\CK_DUAL_images\CK_out.xlsx', engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            a.to_excel(writer, sheet_name=f"{name_tag}")

def argmax_2d(results1, w=5, under_ther=40, method=0):
    """
    Finds the coordinates of the maximum value within a specified window and applies background removal.

    Args:
        results1 (numpy.ndarray): Input 3D array of results.
        w (int, optional): The window size for local averaging. Default is 5.
        under_ther (int, optional): Threshold for background removal. Default is 40.
        method (int, optional): Method for background removal:
            0 - No background removal.
            1 - Remove background using the N-th minimum value.
            2 - Remove background using the user-defined value.
    
    Returns:
        numpy.ndarray: Processed frames with background removed and maxima computed.
    """
    s1, s2 = results1.shape[1], results1.shape[2]
    max_index = [np.argmax(frame) for frame in results1]

    # Determine threshold for background removal
    if method == 0:
        ther = np.zeros([results1.shape[0]]) - 1  # No background removal
    elif method == 1:
        ther = np.array([sorted(frame.flatten())[-under_ther] for frame in results1])  # Using N-th min value
    elif method == 2:
        ther = np.zeros([results1.shape[0]]) + under_ther  # Using user-defined value

    # Convert 1D indices to 2D coordinates (i.e., row and column)
    out = np.stack([np.unravel_index(max_index_, results1.shape[1:3]) for max_index_ in max_index])
    frames = np.array([my_mean(results1[x, max(i-w, 0):min(i+w, s1), max(j-w, 0):min(j+w, s2)], ther[x])
                       for x, [i, j] in enumerate(out)])

    return frames

def my_mean(arr, ther):
    """
    Computes the mean of the array elements greater than a specified threshold.

    Args:
        arr (numpy.ndarray): Input array.
        ther (float): The threshold for filtering elements.

    Returns:
        float: Mean of elements greater than the threshold.
    """
    arr = arr[arr > ther]
    return np.mean(arr)

def rank(tensor):
    """
    Returns the rank (number of dimensions) of the input tensor.

    Args:
        tensor (tensorflow.Tensor): The input tensor.

    Returns:
        int: Rank of the tensor.
    """
    return len(tensor.get_shape())

def load_data_CK(aug=0, dir='C:\\Users\\User\\OneDrive - stevens.edu\\CK project\\Data for Temperature Maps2\\'):
    """
    Loads and preprocesses the data for the CK project. Includes reading MAT files and associated PNG images.

    Args:
        aug (int, optional): Whether to augment data (1 for true, 0 for false). Default is 0.
        dir (str, optional): Directory where the data is stored. Default is a predefined path.

    Returns:
        tuple: Tuple containing processed temperature, image, and raw data arrays.
    """
    old_max = 0  # For monitoring the maximum value in images
    max_Tc = 5500
    list = glob.glob(dir + "*.mat")
    list_png = glob.glob(dir + "**/*.png", recursive=True)
    
    master_list_Tc = {}
    master_list_im_1 = {}
    master_list_im_2 = {}
    master_list_raw = {}
    
    master_arr_Tc = []
    master_arr_im_1 = []
    master_arr_im_2 = []
    master_arr_raw = []
    
    for f_name in list:
        temp = sio.loadmat(f_name)
        key_list = temp.keys()
        key_list_tc = [i for i in key_list if "Tc" in i]
        key_list_img_1 = [i for i in key_list if "_1_" in i]
        key_list_img_2 = [i for i in key_list if "_2_" in i]

        # Process temperature and images
        for i in key_list_tc:
            temp[i][np.isnan(temp[i])] = 0
            temp[i][temp[i] < 0] = 0
            temp[i][temp[i] > max_Tc + 1] = max_Tc
            master_list_Tc[i.split('_')[-1]] = np.float64(temp[i])

            png_dir = [j for j in list_png if i.split('_')[-1] in j]
            if len(png_dir) == 1:
                image_raw = cv2.imread(png_dir[0]) / 255.0
                if old_max < np.max(image_raw):
                    print(np.max(image_raw))
                    old_max = np.max(image_raw)
                master_list_raw[i.split('_')[-1]] = image_raw

        # Process image 1 and 2
        for i in key_list_img_1:
            temp[i][np.isnan(temp[i])] = 0
            temp[i][temp[i] < 0] = 0
            master_list_im_1[i.split('_')[0].split('im')[-1]] = np.float64(temp[i])
        
        for i in key_list_img_2:
            temp[i][np.isnan(temp[i])] = 0
            temp[i][temp[i] < 0] = 0
            master_list_im_2[i.split('_')[0].split('im')[-1]] = np.float64(temp[i])

    # Organize and prepare final arrays
    for name in master_list_Tc:
        if name in master_list_im_1 and name in master_list_im_2:
            master_arr_Tc.append(master_list_Tc[name])
            master_arr_im_1.append(master_list_im_1[name])
            master_arr_im_2.append(master_list_im_2[name])
            master_arr_raw.append(master_list_raw[name])
        else:
            print(f"Error: Missing name for {name}")
    
    master_arr_Tc = np.array(master_arr_Tc)
    master_arr_im_1 = np.array(master_arr_im_1)
    master_arr_im_2 = np.array(master_arr_im_2)
    master_arr_raw = np.array(master_arr_raw)

    # Resize arrays
    master_arr_Tc = master_arr_Tc[:, :, :64]
    master_arr_im_1 = master_arr_im_1[:, :, :64]
    master_arr_im_2 = master_arr_im_2[:, :, 64:]
    master_arr_raw_1 = master_arr_raw[:, :, :64]
    master_arr_raw_2 = master_arr_raw[:, :, 64:]

    # Normalize data
    am = np.max(master_arr_Tc, axis=-1).max()
    im1m = np.max(master_arr_im_1, axis=-1).max()
    im2m = np.max(master_arr_im_2, axis=-1).max()
    imr1m = np.max(master_arr_raw_1, axis=-1).max()
    imr2m = np.max(master_arr_raw_2, axis=-1).max()

    master_arr_Tc = np.divide(master_arr_Tc, am)
    master_arr_im_1 = np.divide(master_arr_im_1, im1m)
    master_arr_im_2 = np.divide(master_arr_im_2, im2m)
    master_arr_raw_1 = np.divide(master_arr_raw_1, imr1m)
    master_arr_raw_2 = np.divide(master_arr_raw_2, imr2m)

    # Data augmentation (if requested)
    if aug:
        master_arr_Tc, master_arr_im_1, master_arr_im_2, master_arr_raw_1, master_arr_raw_2 = \
            augment_data(np.concatenate([np.stack([master_arr_Tc, master_arr_im_1, master_arr_im_2], axis=-1),
                                         master_arr_raw_1, master_arr_raw_2], axis=-1))

    # Reshape the arrays for model input
    master_arr_Tc = np.array(master_arr_Tc).reshape([-1, 48, 64, 1])
    master_arr_im_1 = np.array(master_arr_im_1).reshape([-1, 48, 64, 1])
    master_arr_im_2 = np.array(master_arr_im_2).reshape([-1, 48, 64, 1])
    master_arr_raw_1 = np.array(master_arr_raw_1).reshape([-1, 48, 64, 3])
    master_arr_raw_2 = np.array(master_arr_raw_2).reshape([-1, 48, 64, 3])

    return master_arr_Tc, master_arr_im_1, master_arr_im_2, master_arr_raw_1, master_arr_raw_2

def augment_data(data, shift=5, step=3, test=0):
    """
    Augments the dataset by applying random shifts to the images.

    Args:
        data (numpy.ndarray): Input data to be augmented.
        shift (int, optional): Maximum shift value. Default is 5.
        step (int, optional): Step size for the shifts. Default is 3.
        test (int, optional): Whether this is for test data (1) or not (0). Default is 0.

    Returns:
        tuple: Augmented dataset arrays for temperature, images, and raw data.
    """
    big_data = np.stack(data)

    # Apply random shifts in both x and y directions
    for dy in range(-shift, shift, step):
        for dx in range(-shift, shift, step):
            if dy == 0 and dx == 0:
                continue
            dy = np.int32(dy)
            dx = np.int32(dx)
            X = np.roll(data, dy, axis=1)
            X = np.roll(X, dx, axis=1)
            if dy > 0:
                X[:, :dy, :, :] = 0
            elif dy < 0:
                X[:, dy:, :, :] = 0
            if dx > 0:
                X[:, :, :dx, :] = 0
            elif dx < 0:
                X[:, :, dx:, :] = 0
            big_data = np.concatenate([X, big_data], axis=0)

    if test:
        master_arr_raw_1 = big_data[:, :, :, :3]
        master_arr_raw_2 = big_data[:, :, :, 3:]
        return master_arr_raw_1, master_arr_raw_2
    else:
        master_arr_Tc = big_data[:, :, :, 0]
        master_arr_im_1 = big_data[:, :, :, 1]
        master_arr_im_2 = big_data[:, :, :, 2]
        master_arr_raw_1 = big_data[:, :, :, 3:6]
        master_arr_raw_2 = big_data[:, :, :, 6:]
        return master_arr_Tc, master_arr_im_1, master_arr_im_2, master_arr_raw_1, master_arr_raw_2
##endregion
##region Deep Model

class Same_net(tf.keras.Model):
    """
    Defines the 'Same_net' model which uses various convolutions (1x1, 3x3, 5x5) and pooling layers 
    to build a deep learning model. It uses 'gelu' activation and batch normalization.
    """

    def __init__(self, filters_1x1=16):
        """
        Initializes the 'Same_net' model with a set of convolutional layers and activation functions.
        
        Args:
        - filters_1x1 (int): Number of filters for the 1x1 convolutional layers.
        """
        super(Same_net, self).__init__()

        filters_3x3_reduce = filters_1x1
        filters_3x3 = filters_1x1
        filters_5x5_reduce = filters_1x1
        filters_5x5 = filters_1x1
        init = 'RandomUniform'
        
        # Convolution layers initialization
        self.conv1x1 = tf_layer.Conv2D(filters_1x1, (1, 1), padding='same', kernel_initializer=init)
        self.conv3x3_reduce = tf_layer.Conv2D(filters_3x3_reduce, (1, 1), padding='same', kernel_initializer=init)
        self.conv3x3 = tf_layer.Conv2D(filters_3x3, (3, 3), padding='same', kernel_initializer=init)
        self.conv5x5_reduce = tf_layer.Conv2D(filters_5x5_reduce, (1, 1), padding='same', kernel_initializer=init)
        self.conv5x5 = tf_layer.Conv2D(filters_5x5, (5, 5), padding='same', kernel_initializer=init)
        self.convpool = tf_layer.Conv2D(filters_1x1, (1, 1), padding='same', kernel_initializer=init)

        # Batch Normalization and Activation Layer
        self.bn = tf_layer.BatchNormalization()
        self.act = tf_layer.Activation('gelu')

        # Max Pooling layer
        self.max_pool = tf_layer.MaxPool2D((1, 1), strides=(1, 1), padding='same')

        # Concatenate operation for combining different layers
        self.Concat = tf_layer.Concatenate()

    def call(self, inputs):
        """
        Defines the forward pass through the network.

        Args:
        - inputs (tensor): The input tensor for the model.
        
        Returns:
        - output (tensor): The output tensor after applying convolutions, activations, and concatenation.
        """
        b1 = self.conv1x1(inputs)
        b1_act = self.act(b1)

        b2 = self.conv3x3_reduce(inputs)
        b2_act = self.act(b2)
        b2_2 = self.conv3x3(b2_act)
        b2_2_act = self.act(b2_2)

        b3 = self.conv5x5_reduce(inputs)
        b3_act = self.act(b3)
        b3_2 = self.conv5x5(b3_act)
        b3_2_act = self.act(b3_2)

        b4 = self.max_pool(inputs)
        b4_2 = self.convpool(b4)
        b4_2_act = self.act(b4_2)

        # Concatenate all blocks together to form the output
        output = self.Concat([b1_act, b2_2_act, b3_2_act, b4_2_act, inputs])
        return output


class Temp_deep(tf.keras.Model):
    """
    Defines the 'Temp_deep' model, which consists of multiple 'Same_net' instances followed by convolutional 
    layers, subtraction, addition, and concatenation operations.
    """

    def __init__(self, F_N=64, filt_size=5):
        """
        Initializes the 'Temp_deep' model by creating layers using 'Same_net' and convolutional layers.
        
        Args:
        - F_N (int): Number of filters used in the network.
        - filt_size (int): Filter size for the convolution layers (currently not used).
        """
        super(Temp_deep, self).__init__()

        # 'Same_net' instances for different parts of the network
        self.a_1 = Same_net(filters_1x1=F_N)
        self.b_1 = Same_net(filters_1x1=F_N)
        self.c_1 = Same_net(filters_1x1=F_N)
        self.f_1 = Same_net(filters_1x1=F_N)

        # Convolutional layers for the network
        self.a_2 = tf_layer.Conv2D(F_N, (3, 3), padding='same', activation='gelu', kernel_initializer='RandomUniform')
        self.b_2 = tf_layer.Conv2D(F_N, (3, 3), padding='same', activation='gelu', kernel_initializer='RandomUniform')
        self.c_2 = tf_layer.Conv2D(F_N, (3, 3), padding='same', activation='gelu', kernel_initializer='RandomUniform')
        self.f_2 = tf_layer.Conv2D(F_N, (3, 3), padding='same', activation='gelu', kernel_initializer='RandomUniform')

        # Additional 'Same_net' instances and convolutional layers
        self.a_3 = Same_net(filters_1x1=F_N)
        self.b_3 = Same_net(filters_1x1=F_N)
        self.c_3 = Same_net(filters_1x1=F_N)
        self.f_3 = Same_net(filters_1x1=F_N)

        self.a_4 = tf_layer.Conv2D(F_N, (3, 3), padding='same', activation='gelu', kernel_initializer='RandomUniform')
        self.b_4 = tf_layer.Conv2D(F_N, (3, 3), padding='same', activation='gelu', kernel_initializer='RandomUniform')
        self.c_4 = tf_layer.Conv2D(F_N, (3, 3), padding='same', activation='gelu', kernel_initializer='RandomUniform')
        self.f_4 = tf_layer.Conv2D(1, (3, 3), padding='same', activation='gelu', kernel_initializer='RandomUniform')

        self.a_5 = Same_net(filters_1x1=F_N)
        self.b_5 = Same_net(filters_1x1=F_N)
        self.c_5 = Same_net(filters_1x1=F_N)

        self.a_6 = tf_layer.Conv2D(F_N, (3, 3), padding='same', activation='gelu', kernel_initializer='RandomUniform')
        self.b_6 = tf_layer.Conv2D(F_N, (3, 3), padding='same', activation='gelu', kernel_initializer='RandomUniform')
        self.c_6 = tf_layer.Conv2D(F_N, (3, 3), padding='same', activation='gelu', kernel_initializer='RandomUniform')

        # Inner connection layers for AE (Autoencoder) style operation
        self.Concat1 = tf_layer.Concatenate()
        self.Concat2 = tf_layer.Concatenate()
        self.Concat3 = tf_layer.Concatenate()
        self.Concatf1 = tf_layer.Concatenate()
        self.Concatf2 = tf_layer.Concatenate()

        self.subtracts1 = tf_layer.Subtract()
        self.subtracts2 = tf_layer.Subtract()
        self.subtracts3 = tf_layer.Subtract()

        self.add1 = tf_layer.Add()
        self.add2 = tf_layer.Add()
        self.add3 = tf_layer.Add()

    def call(self, inputs):
        """
        Defines the forward pass for the 'Temp_deep' model. The network takes two inputs and 
        processes them through multiple convolutional and aggregation layers.
        
        Args:
        - inputs (list of tensors): The list contains two input tensors.
        
        Returns:
        - f_out (tensor): The final output tensor after processing through all layers.
        """
        # First set of operations on inputs
        a1 = self.a_1(inputs[0])
        a2 = self.a_2(a1)
        a3 = self.a_3(a2)
        a4 = self.a_4(a3)
        a5 = self.a_5(a4)
        a_out = self.a_6(a5)

        b1 = self.b_1(inputs[1])
        b2 = self.b_2(b1)
        b3 = self.b_3(b2)
        b4 = self.b_4(b3)
        b5 = self.b_5(b4)
        b_out = self.b_6(b5)

        # Compute intermediate connections
        cinp1 = self.add1([inputs[0], inputs[1]])
        cinp2 = self.subtracts1([inputs[0], inputs[1]])
        c1 = self.Concat1([cinp1, cinp2])
        c2 = self.c_1(c1)
        c3 = self.c_2(c2)

        # Further operations on intermediate connections
        c3_1 = self.add2([a2, b2])
        c3_2 = self.subtracts2([a2, b2])
        c3_conc = self.Concat2([c3_1, c3_2, c3])
        c4 = self.c_3(c3_conc)
        c5 = self.c_4(c4)

        c5_1 = self.add3([a4, b4])
        c5_2 = self.subtracts3([a4, b4])
        c5_conc = self.Concat3([c5_1, c5_2, c5])
        c6 = self.c_3(c5_conc)
        c_out = self.c_4(c6)

        # Final concatenation and output
        f1 = self.Concatf1([a_out, b_out, c_out])
        f2 = self.f_1(f1)
        f2_1 = self.Concatf2([f2, a_out, b_out, c_out])
        f3 = self.f_2(f2_1)
        f4 = self.f_3(f3)
        f_out = self.f_4(f4)

        return f_out


def load_model(DB_DEEP_loc, load_weights="0"):
    """
    Loads the deep learning model from a specified directory. If the model weights are available, they will 
    be loaded into the model.

    Args:
    - DB_DEEP_loc (str): The directory location where the model weights are stored.
    - load_weights (str): The name of the weight file to be loaded.

    Returns:
    - model_1 (Temp_deep): The deep learning model with the loaded weights.
    """
    model_1 = Temp_deep()
    if exists(DB_DEEP_loc + load_weights + ".index"):
        print(colored("Model weight is located", 'green'))
        model_1.load_weights(DB_DEEP_loc + load_weights)
        print(colored("**********************loading weights DONE************************", 'green'))
    else:
        print(colored("**********************Loading Weights FAILED**********************", 'magenta'))
    return model_1


def do_plot2(master_list_Tc, master_list_im_1, master_list_im_2, model_1, n=3, title='', raw=1):
    """
    Plots the ground truth, predicted images, and their difference from a model's output.

    Args:
    - master_list_Tc (list): List of temperature maps (ground truth).
    - master_list_im_1 (list): List of first image data.
    - master_list_im_2 (list): List of second image data.
    - model_1 (Temp_deep): The trained model for predictions.
    - n (int): Number of plots to generate.
    - title (str): Title for the plot.
    - raw (int): Flag to indicate if raw data should be used.
    """
    col = 3
    fig, ax = plt.subplots(n, col, figsize=(18, 5))
    fig.suptitle(title)
    if raw:
        dim = 3
    else:
        dim = 1

    for iter_ in range(col):
        iter = random.choice(range(0, len(master_list_Tc)))
        image = np.reshape(master_list_im_1[iter:iter + 2], [-1, 48, 64, dim])
        image_gt = np.reshape(master_list_im_2[iter:iter + 2], [-1, 48, 64, dim])
        label = np.reshape(master_list_Tc[iter:iter + 2], [-1, 48, 64, 1])
        pred_image = model_1([image, image_gt])
        img0 = np.reshape(label[0], [48, 64])
        img1 = np.reshape(pred_image[0], [48, 64])

        ax[0, iter_].imshow(img0, vmin=0, vmax=1)
        ax[0, iter_].set_title('Ground Truth')
        ax[1, iter_].imshow(img1, vmin=0, vmax=1)
        ax[1, iter_].set_title('Predicted')
        ax[2, iter_].imshow(np.abs(img0 - img1), vmin=0, vmax=0.3)
        ax[2, iter_].set_title('Difference')

    plt.tight_layout()
    plt.show()
def my_train(model_1, inp1, inp2, temps, DB_DEEP_loc, epoch=30, learning_rate_=0.001, batch=100, history=[], raw=1):
    model_1.compile(optimizer=k.optimizers.Adam(learning_rate=learning_rate_),
                    loss='mean_squared_error')

    history.append(model_1.fit(x=[inp1, inp2],
                               y=temps, callbacks=[],
                               batch_size=batch,
                               epochs=epoch,
                               verbose=1,
                               shuffle=True))

    model_1.save_weights(DB_DEEP_loc + "CK_model_going_DEEP_backup_lr_{}_btch_{}".format(learning_rate_, batch))
    hist = []
    try:
        for i in history:
            hist.append(i.history['loss'])
    except:
        hist = history[0].history['loss']
    hist = np.concatenate(hist, axis=0)

    fig, ax = plt.subplots(1, 2, figsize=(18, 5))
    ax[0].plot(hist)
    ax[1].plot(history[-1].history['loss'])

    # plt.title('output_1_loss loss')
    # plt.ylabel('val_output_2_accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    if raw:
        do_plot2(master_list_Tc, master_list_raw_1, master_list_raw_2, model_1, n=3,
                title="lr_{}_btch_{}".format(lr, batch),raw=raw)
    else:
        do_plot2(master_list_Tc, master_list_im_1, master_list_im_2, model_1, n=3,
                title="lr_{}_btch_{}".format(lr, batch), raw=raw)
    return model_1, history
##endregion
##region Main

if __name__ == "__main__":
    test = 0
    train_dir = 'C:\\Users\\User\\OneDrive - stevens.edu\\CK project\\Data for Temperature Maps2\\'
    new_loc_1 = 'C:\\Users\\User\\OneDrive - stevens.edu\\CK project\\0224_ST_Tests_NoPowder_DFTracks_2_S1\\'
    new_loc_2 = 'C:\\Users\\User\\OneDrive - stevens.edu\\CK project\\0224_ST_Tests_NoPowder_KHTracks_2_S1\\'
    new_loc_3 = 'C:\\Users\\User\\OneDrive - stevens.edu\\CK project\\0224_ST_Tests_NoPowder_LOFTracks_4_S1\\'

    DB_DEEP_loc = "C:\\Users\\User\\OneDrive - stevens.edu\\CK project\\Deep_Model_outs\\"
    load_tag = "CK_model_raw_going_DEEP"
    epch = 20
    model_1 = load_model(DB_DEEP_loc, load_weights=load_tag)
    for locs in [ new_loc_2, new_loc_1, new_loc_3]:
        test1(dir=locs,ground_trouth_check=1)
    if test:
        master_arr_raw_1, master_arr_raw_2 = load_test_data_CK(dir=new_loc_1, aug=0)
        results = model_1.predict([master_arr_raw_1, master_arr_raw_2],batch_size=5)
    else:
        master_list_Tc, master_list_im_1, master_list_im_2, master_list_raw_1, master_list_raw_2 = load_data_CK(dir=train_dir)
        hist = []
        for lr in [0.0001, 0.00001, 0.000001, 0.0000001]:
            for btch in [20, 5, 2]:
                model_1, hist = my_train(model_1, master_list_raw_1, master_list_raw_2, master_list_Tc, DB_DEEP_loc, epoch=epch,
                         learning_rate_=lr, batch=btch,history=hist)

        model_1.save_weights(DB_DEEP_loc + "CK_model_raw_going_DEEP")
        do_plot2(master_list_Tc, master_list_raw_1, master_list_raw_2, model_1, n=3, title="Final Results")



    pasue = 1
#endregion
