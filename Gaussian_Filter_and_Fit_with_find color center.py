import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from matplotlib.colors import rgb_to_hsv
import os

from skimage import io, color, filters, measure

def find_color_center(image_path, target_hex_color):
    # Load the image
    image = io.imread(image_path)
    
    # Remove alpha channel if present
    if image.shape[-1] == 4:
        image = image[..., :3]
    
    # Convert the target color from hex to normalized RGB and then to HSV
    target_rgb = np.array([int(target_hex_color[i:i+2], 16) for i in (0, 2, 4)]) / 255.0
    target_hsv = rgb_to_hsv(target_rgb.reshape(1, 1, 3)).squeeze()
    
    # Convert the image to HSV color space
    image_hsv = color.rgb2hsv(image)
    
    # Define thresholds for color matching
    hue_threshold = 0.05
    sat_threshold = 0.2
    val_threshold = 0.2
    
    # Identify pixels within the specified HSV thresholds
    color_match = (
        (np.abs(image_hsv[..., 0] - target_hsv[0]) < hue_threshold) &
        (np.abs(image_hsv[..., 1] - target_hsv[1]) < sat_threshold) &
        (np.abs(image_hsv[..., 2] - target_hsv[2]) < val_threshold)
    )
    
    # Label the regions in the color match mask
    labeled_color_match = measure.label(color_match)
    color_regions = measure.regionprops(labeled_color_match)
    
    # Find the largest region with the matching color and get its centroid
    if color_regions:
        largest_color_region = max(color_regions, key=lambda r: r.area)
        center_y, center_x = largest_color_region.centroid
        minr, minc, maxr, maxc = largest_color_region.bbox
        return center_x,center_y
    else:
        return 0,0
        #print("No matching region found.")
        #return

def gaussian_2d(coords, x0, y0, x_sigma, y_sigma, amplitude):
    """2D Gaussian function for fitting."""
    x, y = coords
    return amplitude * np.exp(-((x - x0) ** 2 / (2 * x_sigma ** 2) + (y - y0) ** 2 / (2 * y_sigma ** 2))).ravel()

def plot_gaussian_2d(x, y, x0, y0, xalpha, yalpha, A):
    """Calculate Amplitude of Gaussian Point."""
    return A * np.exp(-((x-x0)**2/(2*xalpha**2) + (y-y0)**2/(2*yalpha**2)))

def g_surface(filtered_image, gaussian_param):
    """Calculate Gaussian Surface."""
    x0, y0, xalpha, yalpha, A = gaussian_param
    x = np.linspace(0, filtered_image.shape[1]-1, filtered_image.shape[1])
    y = np.linspace(0, filtered_image.shape[0]-1, filtered_image.shape[0])
    x, y = np.meshgrid(x, y)
    gaussian_surface = plot_gaussian_2d(x, y, x0, y0, xalpha, yalpha, A)
    return gaussian_surface

def load_image(path):
    """Load the images and manually convert them to grayscale."""
    image = Image.open(path)
    image_array = np.array(image)
    # Manually converting to grayscale using the luminosity method
    grayscale_array = 0.2989 * image_array[:,:,0] + 0.5870 * image_array[:,:,1] + 0.1140 * image_array[:,:,2]
    return grayscale_array

def average_images(image1_gray,image2_gray,image3_gray):
    """Average the images."""
    average_image_gray = (image1_gray + image2_gray + image3_gray) / 3
    return average_image_gray

def filter_image(gray_array):
    """Apply Gaussian Filter."""
    filtered_image_gray = gaussian_filter(gray_array, sigma=20)
    return filtered_image_gray

def filter_noise(data, threshold):
    """Filter out noise by setting values below a certain threshold to zero."""
    return np.where(data < threshold, 0, data)

def fit_gaussian(data,x_i,y_i):
    """Fit a 2D Gaussian to the image data."""
    # Grid of x, y points
    x = np.linspace(0, data.shape[1]-1, data.shape[1])
    y = np.linspace(0, data.shape[0]-1, data.shape[0])
    x, y = np.meshgrid(x, y)
    # Initial guess for the parameters: [x0, y0, sigma_x, sigma_y, amplitude]
    initial_guess = (x_i,y_i, 50, 50, np.max(data))
    # Perform the curve fitting
    params, pcov = curve_fit(gaussian_2d, (x, y), data.ravel(), p0=initial_guess)
    # Predict the data using the fitted parameters
    fitted_data = gaussian_2d((x, y), *params)
    # Degrees of freedom (number of data points minus number of parameters)
    dof = len(y)*len(x) - len(params)    
    # Calculate the residual sum of squares (RSS)
    residuals = data.ravel() - fitted_data
    rss = np.sum(residuals**2)
    #r_squared = 1 - (np.sum(residuals**2) / np.sum((ydata - np.mean(ydata))**2))
    chi_squared_red = rss / dof
    return params,chi_squared_red

def plot_grid(ax):
    """Plots Circle and Lines."""
    ax.add_artist(Circle((632, 362),250,color='green',fill =False)) 
    ax.add_artist(Circle((632, 362),200,color='green',fill =False)) 
    ax.add_artist(Circle((632, 362),150,color='green',fill =False)) 
    ax.add_artist(Circle((632, 362),100,color='green',fill =False)) 
    ax.add_artist(Circle((632, 362),50,color='green',fill =False))
    ax.add_artist(Line2D((379,885),(362,362))) 
    ax.add_artist(Line2D((632,632),(109,615)))
    for i in range(0,45):
        ax.add_artist(Line2D((382+5*i,382+5*i),(355,370),color='green')) 
    for i in range(0,45):
        ax.add_artist(Line2D((657+5*i,657+5*i),(355,370),color='green')) 
    for i in range(0,45):
        ax.add_artist(Line2D((625,639),(387+5*i,387+5*i),color='green')) 
    for i in range(0,45):
       ax.add_artist(Line2D((625,639),(337-5*i,337-5*i),color='green'))
    for i in range(0,10):
        ax.add_artist(Line2D((382+25*i,382+25*i),(350,375),color='green')) 
    for i in range(0,10):
        ax.add_artist(Line2D((657+25*i,657+25*i),(350,375),color='green'))
    for i in range(0,10):
        ax.add_artist(Line2D((620,644),(337-25*i,337-25*i),color='green')) 
    for i in range(0,10):
        ax.add_artist(Line2D((620,644),(387+25*i,387+25*i),color='green')) 

def plot_result(img,data, x0, y0,i,chi):
    """Plot the result with the Gaussian center marked."""
    fig, ax = plt.subplots(figsize=(10,8))
    ax.imshow(data, cmap='gray',alpha = 0.5)
    ax.imshow(img, cmap='Reds',alpha = 0.5)
    ax.set_title('Filtered Laser Center (%3.3f, %3.3f) at Position %d mm with Chi %d' %((x0-632)/5,(-y0+362)/5,50*i/3,chi))
    ax.scatter(x0, y0, color='red', s=5, label='Gaussian Center') 
    plot_grid(ax)
    plt.savefig('Y-Achse %d.png'%i)
    plt.close(fig)
    #plt.show()

def prep_plot(filtered_mean, gaussian_surface,x0,y0,A):
    """Prepare Plot."""
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(filtered_mean.shape[1])
    y = np.arange(filtered_mean.shape[0])
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, filtered_mean, cmap='turbo', alpha=0.5)
    ax.plot_surface(X, Y, gaussian_surface, cmap='viridis')
    ax.scatter(x0,y0,A, color = 'green', s = 50)
 
    print(x0,y0)

def main():
    Data = os.listdir('Data Y-Achse 1')
    y_result = []
    z_result = []
    for i in range(0,len(Data),3):
    #for i in range(0,9,3):
        file_path1 = os.path.join("Data Y-Achse 1",Data[i])
        file_path2 = os.path.join("Data Y-Achse 1",Data[i+1])
        file_path3 = os.path.join("Data Y-Achse 1",Data[i+2])
      
        gray_array1 = load_image(file_path1)
        gray_array2 = load_image(file_path2)
        gray_array3 = load_image(file_path3)

        init_guess = find_color_center(file_path1,'DFFFF2')

        mean_gray_array = average_images(gray_array1,gray_array2,gray_array3)
        filtered_mean = filter_image(mean_gray_array)

        plt.imsave('Intermediat.png',filtered_mean)
        img = Image.open('Intermediat.png')
        arr =  np.array(img.convert('L'))

        noise_threshold = 200  # Define the noise threshold
        filtered_image = filter_noise(arr, noise_threshold)
        gaussian_param,chi = fit_gaussian(filtered_image,init_guess[0],init_guess[1])
        gaussian_surface = g_surface(filtered_image, gaussian_param)

    #prep_plot(filtered_mean, gaussian_surface,gaussian_param[0],gaussian_param[1],gaussian_param[-1])
    #plt.show()
        y_result.append(gaussian_param[0])
        z_result.append(gaussian_param[1])
        plot_result(img,filtered_image,gaussian_param[0],gaussian_param[1],i,chi)
    print(y_result,z_result)

    with open('Y-Achse X-Data','w') as file:
        for item in y_result:
            file.write(f"{item}\n")

    with open('Y-Achse Z-Data','w') as file:
        for item in z_result:
            file.write(f"{item}\n")

if __name__ == "__main__":
    main()