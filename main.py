import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from PIL import Image
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

from keras import Sequential
from keras._tf_keras.keras.layers import Conv2D

def load_nifti_image(file_path):
    img = nib.load(file_path).get_fdata()
    img = img / np.max(img)  # Normalize the image if necessary
    img = np.expand_dims(img, axis=-1)  # Add channel dimension if needed
    img = np.rot90(img, k=3)  # Rotate image 90 degrees counterclockwise
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img


def create_model(input_shape=(256, 256, 1)):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        Conv2D(1, (3, 3), activation='sigmoid', padding='same')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

model = create_model()
model.load_weights('models/srcnn_trained_weights_2.h5')

# Load the image
test_image_path = 'test/blur/sub-OAS30026_ses-d0048_T1w.nii.gz_axial_slice_low_field.nii/sub-OAS30026_ses-d0048_T1w.nii.gz_axial_slice_low_field.nii'
test_image = load_nifti_image(test_image_path)

high_res_path = 'test/high/sub-OAS30026_ses-d0048_T1w.nii.gz_axial_slice_high_field.nii\sub-OAS30026_ses-d0048_T1w.nii.gz_axial_slice_high_field.nii'
high_res = load_nifti_image(high_res_path)

# Predict the super-resolved image
predicted_image = model.predict(test_image)

# unet_model = sm.Unet('vgg19', input_shape=(256,256,1), encoder_weights=None, decoder_block_type='transpose')

# loss = sm.losses.DiceLoss()
# metric = sm.metrics.FScore()
# model.compile('Adam', loss=loss, metrics=[metric])


# preds = unet_model.predict(predicted_image)

# Function to display images
def plot_images(original, predicted, unet_preds):
    plt.figure(figsize=(18, 6))  # Adjust the figure size as needed

    # Plot High-field MRI
    plt.subplot(1, 3, 1)
    plt.title('High-field')
    plt.imshow(high_res.squeeze(), cmap='gray')
    plt.axis('off')

    # Plot Low-field MRI
    plt.subplot(1, 3, 2)
    plt.title('Low-field')
    plt.imshow(original.squeeze(), cmap='gray')
    plt.axis('off')

    # Plot SRCNN Result
    plt.subplot(1, 3, 3)
    plt.title('SRCNN Result')
    plt.imshow(predicted.squeeze(), cmap='gray')
    plt.axis('off')

    plt.show()

def save_predicted_image(predicted, save_path):
    plt.figure(figsize=(6, 6))  # Adjust the figure size as needed

    # Plot only the predicted image
    plt.imshow(predicted.squeeze(), cmap='gray')
    plt.axis('off')

    # Save the figure
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0)  # Adjust the saving options as needed

    # Close the plot to free up memory
    plt.close()

# Display the results
plot_images(test_image, predicted_image, None)
save_predicted_image(predicted_image, "srcnn_output")