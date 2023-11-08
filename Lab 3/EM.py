import nibabel as nib
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
from tqdm import tqdm

from loguru import logger
import pandas as pd

class FileManager:
    def __init__(self) -> None:
        pass

    def check_file_existence(self, file, description):
        if file is None:
            raise ValueError(f"Please check if the {description} file passed exists in the specified directory")
class NiftiManager:
    def __init__(self) -> None:
        pass

    def load_nifti(self, file_path):
        '''Loads the NIfTI image and access the image data as a Numpy array.'''
        nii_image = nib.load(file_path)
        data_array = nii_image.get_fdata()

        return data_array, nii_image

    def show_nifti(self, file_data, title, slice=25):
        '''Displays a single slice from the nifti volume (change the slice index as needed).'''
        plt.imshow(file_data[:, :, slice], cmap='gray')
        plt.title(title)
        plt.colorbar()
        plt.show()

    def show_label_seg_nifti(self, label, seg, subject_id, slice=25):
        '''Displays both segmentation and ground truth labels as passed to the function.'''
        plt.figure(figsize=(20, 7))
        
        plt.subplot(1, 2, 1)
        plt.imshow(label[:, :, slice], cmap='gray') 
        plt.title(f'Label Image (Subject ID={subject_id})')
        plt.colorbar()

        plt.subplot(1, 2, 2)
        plt.imshow(seg[:, :, slice], cmap='gray') 
        plt.title(f'Segmentation Image (Subject ID={subject_id})')
        plt.colorbar()
        plt.show()

    def show_mean_volumes(self, mean_csf, mean_wm, mean_gm, slices=[128], export=False, filename=None):
        '''Displays the mean volumes for CSF, WM, and GM for a list of slices.'''
        num_slices = len(slices)
        
        plt.figure(figsize=(20, 7 * num_slices))

        for i, slice in enumerate(slices):
            plt.subplot(num_slices, 3, i * 3 + 1)
            plt.imshow(mean_csf[:, :, slice], cmap='gray')
            plt.title(f'Average CSF Volume - Slice {slice}')
            # plt.colorbar()
            plt.axis('off')

            plt.subplot(num_slices, 3, i * 3 + 2)
            plt.imshow(mean_wm[:, :, slice], cmap='gray')
            plt.title(f'Average WM Volume - Slice {slice}')
            # plt.colorbar()
            plt.axis('off')

            plt.subplot(num_slices, 3, i * 3 + 3)
            plt.imshow(mean_gm[:, :, slice], cmap='gray')
            plt.title(f'Average GM Volume - Slice {slice}')
            # plt.colorbar()
            plt.axis('off')

        if export and filename:
            plt.savefig(filename)
            
        plt.show()

    def show_combined_mean_volumes(self, mean_csf, mean_wm, mean_gm, slice_to_display=128, export=False, filename=None):
        # Stack the mean volumes along the fourth axis to create a single 4D array
        combined_mean_volumes = np.stack((mean_csf, mean_wm, mean_gm), axis=3)
    
        # Choose the channel you want to display (0 for CSF, 1 for WM, 2 for GM)
        channel_to_display = 0  # Adjust as needed
    
        # Display the selected channel
        plt.imshow(combined_mean_volumes[:, :, :, :][:, :, slice_to_display]) # [:, :, :, channel_to_display]
        plt.axis('off')  # Turn off axis labels
        plt.title(f'Combined Averaged Volumes at Slice {slice_to_display}')  # Add a title

        if export and filename:
            plt.savefig(filename)
            
        plt.show()

    def min_max_normalization(self, image, max_value):
        # Ensure the image is a NumPy array for efficient calculations
        image = np.array(image)
        
        # Calculate the minimum and maximum pixel values
        min_value = np.min(image)
        max_actual = np.max(image)
        
        # Perform min-max normalization
        normalized_image = (image - min_value) / (max_actual - min_value) * max_value
        
        return normalized_image

    def export_nifti(self, volume, export_path):
        '''Exports nifti volume to a given path.
        '''
        
        # Create a NIfTI image from the NumPy array
        # np.eye(4): Identity affine transformation matrix, it essentially assumes that the images are in the same orientation and position 
        # as the original images
        img = nib.Nifti1Image(volume, np.eye(4))

        # Save the NIfTI image
        nib.save(img, str(export_path))

class Evaluate:
    def __init__(self) -> None:
        pass

    def calc_dice_coefficient(self, mask1, mask2):
        # Ensure the masks have the same shape
        if mask1.shape != mask2.shape:
            raise ValueError("Input masks must have the same shape.")

        # Compute the intersection and union of the masks
        intersection = np.sum(mask1 * mask2)
        union = np.sum(mask1) + np.sum(mask2)

        # Calculate the Dice coefficient
        dice = (2.0 * intersection) / (union + 1e-8)  # Add a small epsilon to avoid division by zero

        return dice
    
class BrainAtlasManager:
    def __init__(self) -> None:
        pass

    def segment_using_tissue_models(self, image, tissue_map_csv):
        '''
        Task (1.1) Tissue models: segmentation using just intensity information.

        Args:
            - image (np.array): a normalized [0, 255] test intensity image to segment.
            - tissue_map_csv (csv): a csf file name that contains the tissue maps probabilities 
        '''
        # read the tissues moodels
        tissue_map_df = pd.read_csv(tissue_map_csv, header=None)
        tissue_map_array = tissue_map_df.values

        # obtain the argmax to know to which cluster each row (histogram pin - 0:255) falls into
        tissue_map_array_argmax = np.argmax(tissue_map_array, axis=1) + 1
        
        # convert bg pixels above 100 to wm
        # the threshold of 100 is observed from the probabilistic tissue map
        bg_mask = np.arange(len(tissue_map_array_argmax)) > 100
        tissue_map_array_argmax[bg_mask] = 2

        # create a black image as a template for the segmentation to fill
        segmentation_result =  np.zeros_like(image)

        # loop through the argmax values of the tissue map
        # index represent the pixel value we want to map to its corrosponding segmentation result of the argmax
        # the value is the new label of that pixel
        for index, value in enumerate(tissue_map_array_argmax):
            # we add a condition to select the pixels that are equal to the index, and not the background as we want to preserve the background
            condition = np.logical_and(image == index, image != 0)
            
            # we update the zeros template with the label value that we obtained from argmax 
            segmentation_result[condition] = value

        return segmentation_result
    
    def segment_using_tissue_atlas(self, image, *atlases):
        '''
        Task (1.2) Label propagation: segmentation using just position information using atlases

        Args:
            - image (np.array): a normalized [0, 255] and skull stripped intensity image to segment.
            - atlases (np.arrays): atlases for CSF, WM, and GM as in order.
        '''

        # get the atlases
        atlas_csf = atlases[0]
        atlas_wm  = atlases[1]
        atlas_gm  = atlases[2]

        # flatten the input image
        registered_volume_test = image.flatten()

        # concatenate the flatenned atlases to form a NxK shaped array of arrays
        concatenated_atlas = np.column_stack((atlas_csf.flatten(), atlas_wm.flatten(), atlas_gm.flatten()))

        # get the argmax for each row to find which cluster does each sample refers to
        atlases_argmax = np.argmax(concatenated_atlas, axis=1) + 1

        # Create a mask to identify non-zero pixels
        non_zero_mask = registered_volume_test != 0

        # Replace non-zero pixels with their equivalent values from atlas argmax
        segmented_image = np.where(non_zero_mask, atlases_argmax, 0)

        # Reshape the segmented image to its original shape if needed
        segmented_image = segmented_image.reshape(image.shape)

        return segmented_image
    
    def segment_using_tissue_models_and_atlas(self, image, tissue_map_csv, *atlases):
        '''(1.3) Tissue models & label propagation: multiplying both results: segmentation using intensity & position information

        Args:
            - image (np.array): a normalized [0, 255] and skull stripped intensity image to segment.
            - tissue_map_csv (csv): a csf file name that contains the tissue maps probabilities 
            - atlases (np.arrays): atlases for CSF, WM, and GM as in order.
        '''
        # read the tissues moodels
        tissue_map_df = pd.read_csv(tissue_map_csv, header=None)
        tissue_map_array = tissue_map_df.values

        # get the atlases
        atlas_csf = atlases[0]
        atlas_wm  = atlases[1]
        atlas_gm  = atlases[2]

        # concatenate the flatenned atlases to form a NxK shaped array of arrays
        concatenated_atlas = np.column_stack((atlas_csf.flatten(), atlas_wm.flatten(), atlas_gm.flatten())) # (18808832, 3)

        # Perform Bayesian segmentation
        registered_volume_test = image.flatten()

        # using registered_volume_test as an index to extract specific rows from tissue_map_array
        tissue_map_array = tissue_map_array[registered_volume_test, :]

        # multiply the probabilities
        posteriors = tissue_map_array * concatenated_atlas

        # ger the argmax for each sample to know for which cluster does it belongs, +1 to avoid 0 value
        posteriors_argmax = np.argmax(posteriors, axis=1) + 1

        # Create a mask to identify non-zero pixels
        non_zero_mask = registered_volume_test != 0

        # Replace non-zero pixels with their equivalent values from atlas argmax
        segmented_image = np.where(non_zero_mask, posteriors_argmax, 0)

        # Reshape the segmented image to its original shape if needed
        segmented_image = segmented_image.reshape(image.shape)

        return segmented_image

class EM:
    def __init__(self, K=3, params_init_type='random', modality='multi'):
        self.K                  = K
        self.params_init_type   = params_init_type
        self.modality           = modality

        self.labels_gt_file, self.t1_path, self.t2_path = None, None, None
        self.labels_nifti, self.t1_volume = None, None

        self.sum_tolerance          = 1e-3
        self.convergence_tolerance  = 200
        self.seed                   = 42

        # Setting a seed
        np.random.seed(self.seed)

        # Helper classes
        self.NM         = NiftiManager()
        self.FM         = FileManager()
        self.BrainAtlas = BrainAtlasManager()

        self.tissue_data, self.gt_binary, self.img_shape = None, None, None      # (N, d) for tissue data

        self.n_samples      = None      # N samples
        self.n_features     = None      # d = number of features 2 or 1 (dimension), 
                                        # based on the number of modalities we pass

        # create parameters objects
        self.clusters_means = None     # (K, d)
        self.clusters_covar = None     # (K, d, d)
        self.alpha_k        = None     # prior probabilities, (K,)

        self.posteriors     = None     # (N, K)
        self.pred_labels    = None     # (N,)
        self.loglikelihood  = [-np.inf]

    def initialize_for_fit(self, labels_gt_file, t1_path, t2_path, tissue_model_csv_dir, *atlases):
        '''Initialize variables only when fitting the algorithm.'''
                
        # get the atlases
        atlas_csf = atlases[0]
        atlas_wm  = atlases[1]
        atlas_gm  = atlases[2]

        # initializing skull stripping variables
        self.labels_gt_file, self.t1_path, self.t2_path \
            = labels_gt_file, t1_path, t2_path
        
        # Removing the background for the data
        self.tissue_data, self.gt_binary, self.img_shape \
                            = self.get_tissue_data(
                                self.labels_gt_file,
                                t1_path=self.t1_path,
                                t2_path=self.t2_path
                            )    # (N, d) for tissue data
        
        self.n_samples      = self.tissue_data.shape[0] # 456532 samples
        self.n_features     = self.tissue_data.shape[1] # number of features 2 or 1 (dimension), based on the number of modalities we pass

        self.clusters_means = np.zeros((self.K, self.n_features))                       # (3, 2)
        self.clusters_covar = np.zeros(((self.K, self.n_features, self.n_features)))    # (3, 2, 2)
        self.alpha_k        = np.ones(self.K)                                           # prior probabilities, (3,)

        self.posteriors     = np.zeros((self.n_samples, self.K), dtype=np.float64)      # (456532, 3)
        self.pred_labels    = np.zeros((self.n_samples,))                               # (456532,)

        if self.modality not in ['single', 'multi']:
            raise ValueError('Wronge modality type passed. Only supports "single" or "multi" options.')
        
        if tissue_model_csv_dir is None and self.params_init_type == 'tissue_models':
            raise ValueError('Missing tissue_model_csv_dir argument.')
        
        if (atlas_csf is None or atlas_wm  is None or atlas_gm is None) and self.params_init_type == 'atlas':
            raise ValueError('Missing atlases argument.')

        # assign model parameters their initial values
        # self.initialize_parameters(data=self.tissue_data)
        self.initialize_parameters(self.tissue_data, tissue_model_csv_dir, atlas_csf, atlas_wm, atlas_gm)

    def skull_stripping(self, image, label):

        # convert the labels to binary form, all tissues to 1, else is 0
        labels_binary   = np.where(label == 0, 0, 1)

        # multiply the image to get only the tissues
        return np.multiply(image, labels_binary)
    
    def get_tissue_data(self, labels_gt_file, t1_path, t2_path):
        '''Removes the black background from the skull stripped volume and returns a 1D array of the voxel intensities \
            of the pixels that falls in the True region of the mask, for GM, WM, and CSF.'''
        
        # Check if files passed
        self.FM.check_file_existence(labels_gt_file, "labels")
        self.FM.check_file_existence(t1_path, "T1")

        # load the nifti files & creating a binary mask from the gt labels
        self.labels_nifti, _ = self.NM.load_nifti(labels_gt_file)
        labels_binary   = np.where(self.labels_nifti == 0, 0, 1)

        # loading the volumes and performing skull stripping 
        self.t1_volume  = self.NM.min_max_normalization(
            self.NM.load_nifti(t1_path)[0], 255).astype('uint8')
        
        t1_selected_tissue = self.t1_volume[labels_binary == 1].flatten()

        # The true mask labels count must equal to the number of voxels we segmented
        # np.count_nonzero(labels_binary) returns the sum of pixel values that are True, the count should be equal to the number
        # of pixels in the selected tissue array
        assert np.count_nonzero(labels_binary) == t1_selected_tissue.shape[0], 'Error while removing T1 black background.'

        # put both tissues into the d-dimensional data vector [[feature_1, feature_2]]
        if self.modality == 'multi':
            self.FM.check_file_existence(t2_path, "T2")

            # loading the volumes and performing skull stripping 
            t2_volume  = self.NM.min_max_normalization(
                self.NM.load_nifti(t2_path)[0], 255).astype('uint8')
            
            t2_selected_tissue = t2_volume[labels_binary == 1].flatten()

            assert np.count_nonzero(labels_binary) == t2_selected_tissue.shape[0], 'Error while removing T2_FLAIR black background.'

            tissue_data =  np.array([t1_selected_tissue, t2_selected_tissue]).T     # multi-modality
        else:
            tissue_data =  np.array([t1_selected_tissue]).T                       # single modality

        return tissue_data, labels_binary, self.t1_volume.shape
    
    def initialize_parameters(self, data, tissue_model_csv_dir, *atlases):
        '''Initializes the model parameters and the weights at the beginning of EM algorithm. It returns the initialized parameters.

        Arguments:
            - data (numpy.ndarray): The intensity image (tissue data) in its original shape.
            - tissue_map_csv_dir (str): path to the tissue model csv file.

        '''

        if self.params_init_type not in ['kmeans', 'random', 'tissue_models', 'atlas', 'tissue_models_atlas']:
            raise ValueError(f"Invalid initialization type {self.params_init_type}. Both 'random' and 'kmeans' initializations are available.")
        

        logger.info(f"Initializing model parameters using '{self.params_init_type}'.")

        if self.params_init_type == 'kmeans':
            kmeans              = KMeans(n_clusters=self.K, random_state=self.seed, n_init='auto', init='k-means++').fit(data)
            cluster_labels      = kmeans.labels_                # labels : ndarray of shape (456532,)
            centroids           = kmeans.cluster_centers_       # (3, 2)
            self.alpha_k        = np.array([np.sum([cluster_labels == i]) / len(cluster_labels) for i in range(self.K)]) # ratio for each cluster

        elif self.params_init_type == 'random':  # 'random' initialization
            random_centroids    = np.random.randint(np.min(data), np.max(data), size=(self.K, self.n_features)) # shape (3,2)
            random_label        = np.random.randint(low=0, high=self.K, size=self.n_samples) # (456532,)
            self.alpha_k        = np.ones(self.K, dtype=np.float64) / self.K 
        
        elif self.params_init_type in ['tissue_models', 'atlas', 'tissue_models_atlas']:  # 'tissue_models' or 'atlas' initialization
            # print(data[:,0].reshape(self.img_shape).shape)

            # the problem here is that the data is no longer in its original shape, it is (Nxd), and we can't reshape it as it is skull stripped
            # we have to re-form the image, or pass it here in a way that data is the skull stripped image
            # segment_using_tissue_models receives the images normalized, we normalized in an earlier step
            data_volume         = self.skull_stripping(image=self.t1_volume, label=self.labels_nifti)

            # self.NM.show_nifti(self.t1_volume, title="self.t1_volume init segmentation", slice=128)

            # get the segmentation labels
            segmentation = None 
            if self.params_init_type == 'tissue_models': # tissue models
                segmentation = self.BrainAtlas.segment_using_tissue_models(image=data_volume, tissue_map_csv=tissue_model_csv_dir)
            elif self.params_init_type == 'atlas': # atlas
                segmentation = self.BrainAtlas.segment_using_tissue_atlas(data_volume, *atlases)
            else: # using both atlas and tissue models
                segmentation = self.BrainAtlas.segment_using_tissue_models_and_atlas(data_volume, tissue_model_csv_dir, *atlases)

            # self.NM.show_nifti(segmentation == 1, title="intermediate parameter init segmentation", slice=137)

            # we have to substract 1 as inside the function `segmentation_tissue_model`, we add 1 to the segmentation argmax predictions
            # we also have to remove the 0 background 
            segmentation_labels = segmentation.flatten()

            cluster_labels      = segmentation_labels[segmentation_labels!=0] - 1
            self.alpha_k        = np.array([np.sum([cluster_labels == i]) / len(cluster_labels) for i in range(self.K)]) # ratio for each cluster

            # compute the new means using the segmentation results and the data
            # adding 1 to K and starting from 1 is important, this is because label 0 is background, and we add 1 to the segmentation argmax predictions inside `segmentation_tissue_model`
            clusters_masks  = [segmentation_labels == k for k in range(1, self.K+1)]
            centroids       = np.array([np.mean(data_volume.flatten()[cluster_mask]) for cluster_mask in clusters_masks])[:, np.newaxis]

        cluster_data            = [data[cluster_labels == i] for i in range(self.K)] if self.params_init_type in ['kmeans', 'tissue_models', 'atlas', 'tissue_models_atlas']  \
                                    else [data[random_label == i] for i in range(self.K)]
        
        # update model parameters (mean and covar)
        self.clusters_means     = centroids if self.params_init_type in ['kmeans', 'tissue_models', 'atlas', 'tissue_models_atlas'] else random_centroids
        self.clusters_covar     = np.array([np.cov(cluster_data[i], rowvar=False) for i in range(self.K)]) # (K, d, d)

        # validating alpha condition
        assert np.isclose(np.sum(self.alpha_k), 1.0, atol=self.sum_tolerance), 'Error in self.alpha_k calculation in "initialize_parameters". Sum of all self.alpha_k elements has to be equal to 1.'

    # def __initialize_parameters(self, data):
    #     '''Initializes the model parameters and the weights at the beginning of EM algorithm. It returns the initialized parameters.

    #     Args:
    #         data (numpy.ndarray): The data points.
    #     '''

    #     if self.params_init_type not in ['kmeans', 'random']:
    #         raise ValueError(f"Invalid initialization type {self.params_init_type}. Both 'random' and 'kmeans' initializations are available.")
        
    #     if self.params_init_type == 'kmeans':
    #         kmeans              = KMeans(n_clusters=self.K, random_state=self.seed, n_init='auto', init='k-means++').fit(data)
    #         cluster_labels      = kmeans.labels_                # labels : ndarray of shape (456532,)
    #         centroids           = kmeans.cluster_centers_       # (3, 2)
    #         self.alpha_k        = np.array([np.sum([cluster_labels == i]) / len(cluster_labels) for i in range(self.K)]) # ratio for each cluster

    #     else:  # 'random' initialization
    #         random_centroids    = np.random.randint(np.min(data), np.max(data), size=(self.K, self.n_features)) # shape (3,2)
    #         random_label        = np.random.randint(low=0, high=self.K, size=self.n_samples) # (456532,)
    #         self.alpha_k        = np.ones(self.K, dtype=np.float64) / self.K 

    #     cluster_data            = [data[cluster_labels == i] for i in range(self.K)] if self.params_init_type == 'kmeans' \
    #                                 else [data[random_label == i] for i in range(self.K)]
        
    #     # update model parameters (mean and covar)
    #     self.clusters_means     = centroids if self.params_init_type == 'kmeans' else random_centroids
    #     self.clusters_covar     = np.array([np.cov(cluster_data[i], rowvar=False) for i in range(self.K)]) # (3, 2, 2)

    #     # validating alpha condition
    #     assert np.isclose(np.sum(self.alpha_k), 1.0, atol=self.sum_tolerance), 'Error in self.alpha_k calculation in "initialize_parameters". Sum of all self.alpha_k elements has to be equal to 1.'

    #     logger.info(f"Successfully initialized model parameters using '{self.params_init_type}'.")

    def multivariate_gaussian_probability(self, x, mean_k, cov_k, regularization=1e-4):
            '''
            Compute the multivariate and single variate gaussian probability density function (PDF) for a given data data.
            The function can handle single or multi-modality (dimensions) and computes the probability on all of the 
            data without a complex iteratitve matrix multiplication.
    
            Args:
                x (numpy.ndarray): The data points.
                mean_k (numpy.ndarray): The mean vector for cluster K.
                cov_k (numpy.ndarray): The covariance matrix for cluster K.
    
            Returns:
                float: The probability density at the given data point.
            '''
    
            dim = self.n_features
            x_min_mean = x - mean_k.T # Nxd
            
            if dim == 1 and cov_k.shape == (): # single modality

                # to handle nan cov_k and inversion in certain cases (mainly when randomly initializing)     
                # we add a small regularisation term to enable the inverse and not to have nan in the final
                # matrix           
                cov_k +=  regularization
                
                # the covariance matrix is a scalar value, thus the inverse is 1 / scalar value
                inv_cov_k = 1 / cov_k

                # to not change the multiplication formula below, we convert it to a (1,1) matrix
                inv_cov_k = np.array([[inv_cov_k.copy()]])
                
                # the determinant is only used for square matrices, for a scalar value, det(a) = a
                determinant = cov_k

            else: # multi-modality

                # to handle nan cov_k and inversion in certain cases (mainly when randomly initializing)     
                # we add a small regularisation term to enable the inverse and not to have nan in the final
                # matrix           
                cov_k += np.eye(cov_k.shape[0]) * regularization

                try:
                    inv_cov_k = np.linalg.inv(cov_k)
                except np.linalg.LinAlgError:
                    inv_cov_k = np.linalg.pinv(cov_k) # Handle singularity by using the pseudo-inverse

                determinant = np.linalg.det(cov_k)
    
            exponent = -0.5 * np.sum((x_min_mean @ inv_cov_k) * x_min_mean, axis=1)
            denominator = (2 * np.pi) ** (dim / 2) * np.sqrt(determinant)
    
            return (1 / denominator) * np.exp(exponent)

    def expectation(self):
        '''Expectation step of EM algorithm. The function initializes the probability placeholder on every iteration, then computes \ 
        the cluster multivariate gaussian probability for every cluster. The final normalised posterior probabilities are normalised \ 
        to ensure the sum of every voxel probabilities for the three clusters is equal to 1.'''

        # initialize membership weights probabilities
        # has to be reset to empty placeholder in every iteration to avoid accumulating the values, the assert below will validate
        posteriors     = np.zeros((self.n_samples, self.K), dtype=np.float64) # posterior probabilities, (456532, 3)

        # calculating the normalised posterior probability for every k cluster using multivariate_gaussian_probability
        for k in range(self.K):

            cluster_prob = self.multivariate_gaussian_probability(
                 x=self.tissue_data, 
                 mean_k=self.clusters_means[k], 
                 cov_k=self.clusters_covar[k]) 
            
            # cluster_prob = multivariate_normal.pdf(
            #    x=self.tissue_data, 
            #    mean=self.clusters_means[k], 
            #    cov=self.clusters_covar[k],
            #    allow_singular=True)
                                    
            # updates every k cluster column 
            posteriors[:,k] = cluster_prob * self.alpha_k[k] 
        
        # normalize the posteriors "membership weights" row by row separately by dividing by the total sum of each row
        posteriors /= np.sum(posteriors, axis=1)[:, np.newaxis]

        # the sum of the 3 clusters probabilities should be equal to 1
        assert np.isclose(np.sum(posteriors[0,]), 1.0, atol=self.sum_tolerance), 'Error with calculating the posterior probabilities "membership weights" for each voxel.'

        return posteriors
    
    def maximization(self, w_ik, tissue_data):
        '''Maximization M-Step of EM algorithm. The function updates the model parameters (mean and covariance matrix) as well as updates the \
            weights (alphas) for every cluster.'''

        # Computing the new means and covariance matrix
        covariance_matrix = np.zeros(((self.K, self.n_features, self.n_features)))
        mu_k = np.zeros((self.K, self.n_features))
        alpha_k = np.ones(self.K)

        for k in range(self.K):
            # sum of weights for every k
            N_k = np.sum(w_ik[:, k])

            # Mean 
            mu_k[k] = np.array([np.sum(w_ik[:, k] * tissue_data[:, i]) / N_k for i in range(self.n_features)])
            
            # covariance 
            x_min_mean = tissue_data-mu_k[k]
            weighted_diff = w_ik[:, k][:, np.newaxis] * x_min_mean
            covariance_matrix[k] = np.dot(weighted_diff.T, x_min_mean) / N_k

            # alpha priors
            alpha_k[k] = N_k / self.n_samples

        # validating alpha condition
        assert np.isclose(np.sum(alpha_k), 1.0, atol=self.sum_tolerance), 'Error in self.alpha_k calculation in "maximization". Sum of all self.alpha_k elements has to be equal to 1.'

        return alpha_k, mu_k, covariance_matrix
    
    def log_likelihood(self, alpha, w):
        # return np.sum(np.log(np.sum(alpha[k] * w[i, k] for i in range(self.n_samples) for k in range(self.K))))
        return np.sum(np.log(np.sum(alpha[k] * w[:, k] for k in range(self.K))))

    def generate_segmentation(self, posteriors, gt_binary):
        predictions = np.argmax(posteriors, axis=1) + 1
        gt = gt_binary
        gt[gt == 1] = predictions
        
        return gt.reshape(self.img_shape)
    
    def correct_pred_labels(self, segmentation_result, gt_binary):
        '''Maps the pixel values of the prediction volume that matches the prediction labels to the gt labels values. This is based on prior knowledge 
        of the correct labl for each cluster, known from a ground truth label volume.
        
        Arguments:
            - segmentation_result (np.array): segmentation volume resulted from the algorithm
            - gt_binary (np.array): binarized ans flattened volume for the segmented volume, the label/ground truth.

        Returns:
            - corrected_segmentation (np.array): segmentation volume with the corrected label for each cluster.
        '''

        logger.info("Finished segmentation. Correcting prediction labels...")

        means = np.mean(self.clusters_means, axis=1)

        # this is based on prior knowledge, the RHS are the corrected labels
        # assuming that CSF=1 (lowest mean), GM=2, and WM=3(highest mean)
        
        # labels for lab 1, where ing gt: CSF=1, GM=2, and CSF=3
        # highest_mean = np.argmax(means) + 1
        # lowest_mean  = np.argmin(means) + 1
        # middle_mean  = len(means) - highest_mean - lowest_mean + 3
        # labels = {
        #     np.argmax(means) + 1: 3, 
        #     len(means) - np.argmax(means) - np.argmin(means) + 1: 2, 
        #     np.argmin(means) + 1: 1}
        
        # labels for lab 3, where ing gt: CSF=1, WM=2, and GM=3
        # the max is wm k=2, we correct it to gm
        labels = {
            np.argmax(means) + 1: 2,  # highest mean is csf in the new dataset of lab 3
            len(means) - np.argmax(means) - np.argmin(means) + 1: 3,  # middle mean is wm, label 2
            np.argmin(means) + 1: 1}
                
        # Modify the labels based on means
        flattened_result = segmentation_result.flatten()
        for mean_index, label_corrected in labels.items():
            gt_binary[flattened_result == mean_index] = label_corrected

        corrected_segmentation = gt_binary.reshape(self.img_shape)

        return corrected_segmentation

    def fit(self, n_iterations, labels_gt_file, t1_path, t2_path = None ,correct_labels=True, tissue_model_csv_dir=None, atlas_csf = None, atlas_wm = None, atlas_gm= None):
        '''Main function that fits the EM algorithm'''

        logger.info(f"Fitting the algorithm with {n_iterations} iterations.")

        # Initialize parameters for fitting
        self.initialize_for_fit(labels_gt_file, t1_path, t2_path, tissue_model_csv_dir, atlas_csf, atlas_wm, atlas_gm)
        
        current_idx         = 0

        while (current_idx <= n_iterations):
            
            # E-Step
            self.posteriors = self.expectation()
                        
            # Log-likelihood convergance check
            current_likelihood = self.log_likelihood(self.alpha_k, self.posteriors)  

            if (np.abs(current_likelihood - self.loglikelihood[-1]) < self.convergence_tolerance):
                break

            self.loglikelihood.append(current_likelihood)
            
            # M Step
            self.alpha_k, self.clusters_means, self.clusters_covar = self.maximization(self.posteriors, self.tissue_data)

            current_idx += 1

        logger.info(f"Iterations performed: {current_idx-1}. Displaying the segmentation result..")

        # creating a segmentation result with the predictions
        segmentation_result = self.generate_segmentation(
            posteriors=self.posteriors, 
            gt_binary=self.gt_binary.flatten())
        
        return self.correct_pred_labels(segmentation_result, self.gt_binary.flatten()) if correct_labels else segmentation_result
