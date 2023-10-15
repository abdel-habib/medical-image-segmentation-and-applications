import nibabel as nib
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
from tqdm import tqdm

from loguru import logger

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
        self.tissue_data    = self.remove_black_bg()    # (456532, 2)
        self.n_samples      = self.tissue_data.shape[0] # 456532 samples
        self.n_features     = self.tissue_data.shape[1] # number of features 2 or 1 (dimension), based on the number of modalities we pass

        # create parameters objects
        self.clusters_means = np.zeros((self.K, self.n_features))                       # (3, 2)
        self.clusters_covar = np.zeros(((self.K, self.n_features, self.n_features)))    # (3, 2, 2)
        self.alpha_k        = np.ones(self.K)                                           # prior probabilities, (3,)

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

        # Selecting the tissues based on their labels
        labels_nifti_csf = labels_nifti == 1
        labels_nifti_gm  = labels_nifti == 2
        labels_nifti_wm  = labels_nifti == 3

        # Adding the tissues togather to make a single file for the 3 tissues
        tissue_labels    = labels_nifti_csf + labels_nifti_gm + labels_nifti_wm

        # selecting the voxel values from the skull stripped
        t1_selected_tissue = t1_volume[tissue_labels].flatten()
        t2_selected_tissue = t2_volume[tissue_labels].flatten()

        # put both tissues into the d-dimensional data vector [[feature_1, feature_2]]
        tissue_data =  np.array([t1_selected_tissue, t2_selected_tissue]).T

        # The true mask labels count must equal to the number of voxels we segmented
        # np.count_nonzero(tissue_labels) returns the sum of pixel values that are True, the count should be equal to the number
        # of pixels in the selected tissue array
        assert np.count_nonzero(tissue_labels) == t1_selected_tissue.shape[0], 'Error while removing T1 black background.'
        assert np.count_nonzero(tissue_labels) == t2_selected_tissue.shape[0], 'Error while removing T2_FLAIR black background.'

        return tissue_data

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
            random_centroids    = np.random.randint(np.min(data), np.max(data), size=(self.K, data.shape[1])) # shape (3,2)
            random_label        = np.random.randint(low=0, high=self.K, size=data.shape[0]) # (456532,)

        cluster_data            = [data[cluster_labels == i] for i in range(self.K)] if self.params_init_type == 'kmeans' \
                                    else [data[random_label == i] for i in range(self.K)]
        
        # update model parameters (mean and covar)
        self.clusters_means     = centroids if self.params_init_type == 'kmeans' else random_centroids
        self.clusters_covar     = np.array([np.cov(cluster_data[i], rowvar=False) for i in range(self.K)]) # (3, 2, 2)

        # Update self.alpha_k
        if self.params_init_type == 'kmeans': # has to calculate the values based on the each cluser and its value to the label computed, can't randomly init
            self.alpha_k        = np.array([np.mean(cluster_labels == i) for i in range(self.K)])
        else: # 'random' initialization, init with [1/3, 1/3, 1/3]
            self.alpha_k        = np.ones(self.K, dtype=np.float64) / self.K 

        # validating alpha condition
        assert np.sum(self.alpha_k) == 1.0, 'Error in self.alpha_k calculation. Sum of all self.alpha_k elements has to be equal to 1.'
        
        logger.info(f"Successfully initialized model parameters using '{self.params_init_type}'.")


    def multivariate_gaussian_probability(self, x, mean_k, cov_k, reg=1e-6):
        '''
        Compute the multivariate Gaussian probability density function (PDF) for a given data point.

        Args:
            x (numpy.ndarray): The vector of data points.
            mean_k (numpy.ndarray): The mean vector.
            cov_k (numpy.ndarray): The covariance matrix.
            reg (float): Regularization term to prevent singularity.

        Returns:
            float: The probability density at the given data point.
        '''

        # if x.shape[0] != mean_k.shape[0] or x.shape[0] != cov_k.shape[0] or x.shape[0] != cov_k.shape[1]:
        #     raise ValueError("Dimensions of x, mean, and cov must match.")

        dim = x.shape[1]
        x_min_mean = x - mean_k

        # Add regularization to the covariance matrix
        # cov_k += np.eye(cov_k.shape[0]) * reg

        try:
            inv_cov_k = np.linalg.inv(cov_k)
        except np.linalg.LinAlgError:
            # Handle singularity by using the pseudo-inverse
            inv_cov_k = np.linalg.pinv(cov_k)

        exponent = -0.5 * x_min_mean.T @ inv_cov_k @ x_min_mean
        denominator = (2 * np.pi) ** (dim / 2) * np.sqrt(np.linalg.det(cov_k))

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
            # TODO: ask for the error here
            # cluster_prob = self.multivariate_gaussian_probability(
            #     x=self.tissue_data, 
            #     mean_k=self.clusters_means[k], 
            #     cov_k=self.clusters_covar[k]) 
            
            cluster_prob = multivariate_normal.pdf(
                x=self.tissue_data, 
                mean=self.clusters_means[k], 
                cov=self.clusters_covar[k],
                allow_singular=True)
            
            # updates every k cluster column 
            posteriors[:,k] = cluster_prob * self.alpha_k[k] 
            
        # normalize the posteriors "membership weights" row by row separately by dividing by the total sum of each row
        posteriors /= np.sum(posteriors, axis=1)[:, np.newaxis]

        # the sum of the 3 clusters probabilities should be equal to 1
        assert np.sum(posteriors[0,]) == 1.0, 'Error with calculating the posterior probabilities "membership weights" for each voxel.'

        return posteriors

    def fit(self, n_iterations):
        '''Main function that excutes the EM algorithm steps'''

        logger.info(f"Fitting the algorithm with {n_iterations} iterations.")

        # iterating as many as n_iterations indicates
        # TODO: add the convergence condition to stop the iteration
        for i in tqdm(range(n_iterations)):
            
            # E-Step
            posteriors = self.expectation()


if __name__ == '__main__':

    algorithm = EM(
        subject_id=1,
        params_init_type='kmeans'
    )

    algorithm.fit(n_iterations=1)
