import nibabel as nib
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
from tqdm import tqdm

from loguru import logger
import pandas as pd

class NiftiManager:
    def __init__(self) -> None:
        pass

    def load_nifti(self, file_path):
        '''Loads the NIfTI image and access the image data as a Numpy array.'''
        nii_image = nib.load(file_path)
        data_array = nii_image.get_fdata()

        return data_array, nii_image

    def show_nifti(self, file_data, slice=20):
        '''Displays a single slice from the nifti volume (change the slice index as needed).'''
        plt.imshow(file_data[:, :, slice], cmap='gray')
        plt.title('NIfTI Image')
        plt.colorbar()
        plt.show()

class EM:
    def __init__(self, subject_id=1, K=3, params_init_type='random'):
        self.subject_id         = subject_id
        self.K                  = K
        self.params_init_type   = params_init_type

        # Nifti Manager class
        self.NM = NiftiManager()

        # Removing the background for the data
        self.tissue_data, self.gt_binary, self.img_shape \
                            = self.remove_black_bg()    # (456532, 2) for tissue data
        self.n_samples      = self.tissue_data.shape[0] # 456532 samples
        self.n_features     = self.tissue_data.shape[1] # number of features 2 or 1 (dimension), based on the number of modalities we pass

        # create parameters objects
        self.clusters_means = np.zeros((self.K, self.n_features))                       # (3, 2)
        self.clusters_covar = np.zeros(((self.K, self.n_features, self.n_features)))    # (3, 2, 2)
        self.alpha_k        = np.ones(self.K)                                           # prior probabilities, (3,)

        self.sum_tolerance  = 0.00001

        self.posteriors     = np.zeros((self.n_samples, self.K), dtype=np.float64)      # (456532, 3)
        self.pred_labels    = np.zeros((self.n_samples,))                               # (456532,)

        # assign the parameters their initial values
        self.initialize_parameters(data=self.tissue_data)
    
    def remove_black_bg(self, labels_gt_file='LabelsForTesting.nii', t1_file='T1.nii', t2_file='T2_FLAIR.nii'):
        '''Removes the black background from the skull stripped volume and returns a 1D array of the voxel intensities \
            of the pixels that falls in the True region of the mask, for GM, WM, and CSF.'''
        
        # selecting paths
        data_folder_path    = os.path.join(os.getcwd(), f'../Lab 1/P2_data/{self.subject_id}')

        # load the nifti files
        labels_nifti, _ = self.NM.load_nifti(os.path.join(data_folder_path, labels_gt_file))
        t1_volume, _    = self.NM.load_nifti(os.path.join(data_folder_path, t1_file))
        t2_volume, _    = self.NM.load_nifti(os.path.join(data_folder_path, t2_file))

        # creating a binary mask from the gt labels
        labels_binary   = np.where(labels_nifti == 0, 0, 1)
 
        # selecting the voxel values from the skull stripped
        t1_selected_tissue = t1_volume[labels_binary == 1].flatten()
        t2_selected_tissue = t2_volume[labels_binary == 1].flatten()

        # put both tissues into the d-dimensional data vector [[feature_1, feature_2]]
        # tissue_data =  np.array([t1_selected_tissue, t2_selected_tissue]).T     # multi-modality
        tissue_data =  np.array([t1_selected_tissue]).T                       # single modality

        # The true mask labels count must equal to the number of voxels we segmented
        # np.count_nonzero(labels_binary) returns the sum of pixel values that are True, the count should be equal to the number
        # of pixels in the selected tissue array
        assert np.count_nonzero(labels_binary) == t1_selected_tissue.shape[0], 'Error while removing T1 black background.'
        assert np.count_nonzero(labels_binary) == t2_selected_tissue.shape[0], 'Error while removing T2_FLAIR black background.'

        return tissue_data, labels_binary, t2_volume.shape

    def initialize_parameters(self, data):
        '''Initializes the model parameters and the weights at the beginning of EM algorithm. It returns the initialized parameters.
        '''

        if self.params_init_type not in ['kmeans', 'random']:
            raise ValueError(f"Invalid initialization type {self.params_init_type}. Both 'random' and 'kmeans' initializations are available.")
        
        if self.params_init_type == 'kmeans':
            kmeans              = KMeans(n_clusters=self.K, random_state=0, n_init='auto', init="k-means++")
            cluster_labels      = kmeans.fit_predict(data)  # labels : ndarray of shape (456532,) Index of the cluster each sample belongs to.
            centroids           = kmeans.cluster_centers_   # (3, 2)
        else:  # 'random' initialization
            random_centroids    = np.random.randint(np.min(data), np.max(data), size=(self.K, self.n_features)) # shape (3,2)
            random_label        = np.random.randint(low=0, high=self.K, size=self.n_samples) # (456532,)

        cluster_data            = [data[cluster_labels == i] for i in range(self.K)] if self.params_init_type == 'kmeans' \
                                    else [data[random_label == i] for i in range(self.K)]
        
        # update model parameters (mean and covar)
        self.clusters_means     = centroids if self.params_init_type == 'kmeans' else random_centroids
        self.clusters_covar     = np.array([np.cov(cluster_data[i], rowvar=False) for i in range(self.K)]) # (3, 2, 2)
        self.alpha_k            = np.ones(self.K, dtype=np.float64) / self.K 

        # validating alpha condition
        assert np.isclose(np.sum(self.alpha_k), 1.0, atol=self.sum_tolerance), 'Error in self.alpha_k calculation in "initialize_parameters". Sum of all self.alpha_k elements has to be equal to 1.'

        logger.info(f"Successfully initialized model parameters using '{self.params_init_type}'.")

    def multivariate_gaussian_probability(self, x, mean_k, cov_k):
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
                # the covariance matrix is a scalar value, thus the inverse is 1 / scalar value
                inv_cov_k = 1 / cov_k

                # to not change the multiplication formula below, we convert it to a (1,1) matrix
                inv_cov_k = np.array([[inv_cov_k.copy()]])
                
                # the determinant is only used for square matrices, for a scalar value, det(a) = a
                determinant = cov_k

            else: # multi-modality
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
            mu_k[k] = np.array([np.sum(w_ik[:, k] * tissue_data[:, i]) / N_k for i in range(self.n_features)]) #TODO: CONFIRM ABOUT THE TRANSPOSE
            
            # covariance 
            x_min_mean = tissue_data-mu_k[k]
            weighted_diff = w_ik[:, k][:, np.newaxis] * x_min_mean
            covariance_matrix[k] = np.dot(weighted_diff.T, weighted_diff) / N_k

            # alpha priors
            alpha_k[k] = N_k / self.n_samples

        # validating alpha condition
        assert np.isclose(np.sum(alpha_k), 1.0, atol=self.sum_tolerance), 'Error in self.alpha_k calculation in "maximization". Sum of all self.alpha_k elements has to be equal to 1.'

        return alpha_k, mu_k, covariance_matrix
    
    def log_likelihood(self, alpha, w):
        #return np.log(np.sum(alpha[k] * w[:, k] for k in range(self.K)))
        return np.sum(np.log(np.sum(alpha[k] * w[:, k] for k in range(self.K))))
        

    def fit(self, n_iterations):
        '''Main function that fits the EM algorithm'''

        logger.info(f"Fitting the algorithm with {n_iterations} iterations.")

        # iterating as many as n_iterations indicates
        for i in tqdm(range(n_iterations)):
            
            # E-Step
            self.posteriors = self.expectation()
            
            # likelihood = self.log_likelihood(self.alpha_k, self.posteriors)
            # print(likelihood)
            
            # M Step
            self.alpha_k, self.clusters_means, self.clusters_covar = self.maximization(self.posteriors, self.tissue_data)

        # creating a segmentation result with the predictions
        predictions = np.argmax(self.posteriors,axis=1) + 1
        gt = self.gt_binary.flatten()
        gt[gt == 1] = predictions
        segmentation_result = gt.reshape(self.img_shape)
        
        # displaying the segmentation results
        self.NM.show_nifti(segmentation_result, 24)


if __name__ == '__main__':

    algorithm = EM(
        subject_id=1,
        params_init_type='kmeans'
    )

    algorithm.fit(n_iterations=10)