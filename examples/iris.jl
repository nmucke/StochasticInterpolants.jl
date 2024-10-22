### Load necessary libraries ### 
using Images
using FileIO
using Plots 
using Random
using Lux
using Zygote
using Optimisers
using ComponentArrays
using Statistics
using MLDatasets

### Prepare MNIST dataset ###
function reshape_mnist_data(data::Vector{UInt8}, num_images::Int, num_rows::Int, num_cols::Int)
    # Skip the first 16 bytes (header)
    data = data[17:end]
    
    # Reshape the remaining data vector into a 3D array (num_images, num_rows, num_cols)
    images = reshape(data, (num_cols, num_rows, num_images))
    
    # Permute dimensions to (num_images, num_rows, num_cols)
    images = permutedims(images, (3, 2, 1))
    
    # Normalize the images by dividing by 255.0
    images = Float32.(images) ./ 255.0
    
    return images
end

### Load MNIST dataset ###
function load_mnist_data(train_path::String, num_images_train::Int, num_rows::Int, num_cols::Int)
    # Load raw data
    x_train = read(train_path)

    # Reshape and normalize the training and test images
    train_images = reshape_mnist_data(x_train, num_images_train, num_rows, num_cols)

    return train_images
end

train_images = load_mnist_data("env/archive/train-images.idx3-ubyte", 60000, 28, 28)

train_images, train_y = MNIST.traindata()

### Plotting MNIST Images ###
function plot_images(images, num_images_to_show=4)
    # Calculate the layout dimensions (rows and columns)
    num_cols = ceil(Int, sqrt(num_images_to_show))
    num_rows = ceil(Int, num_images_to_show / num_cols)
    
    # Create a list of heatmaps to display the images
    p = plot(layout=(num_rows, num_cols), size=(800, 800))
    
    for i in 1:num_images_to_show
        img = Gray.(images[i, :, :])  # Convert to grayscale
        heatmap!(img, color=:grays, axis=false, legend=false, subplot=i)
    end
    
    display(p)
end

# Plot the first 4 images to check if the process went correctly
plot_images(train_images, 9)

### Create Initial distribution - Gaussian ###
function generate_gaussian_images(num_images::Int, num_rows::Int, num_cols::Int, mean::Float64=0.0, stddev::Float64=1.0)
    # Create a 3D array to hold the images (num_images, num_rows, num_cols)
    images = randn(Float32, num_images, num_rows, num_cols) .* stddev .+ mean
    
    # Since MNIST images are grayscale and in the range [0, 1], we might want to clip or normalize them
    # If desired, you can apply any clipping or normalization here:
    images = clamp.(images, 0.0, 1.0)
    
    return images
end

train_gaussian_images = generate_gaussian_images(60000, 28, 28)

# Plot the first 4 images to check if the process went correctly 
plot_images(train_gaussian_images, 12)

### Stochastic Interpolant ###
# Define the stochastic interpolant function
function stochastic_interpolant(image1, image2, λ) 
    return cos.(π/2 .*λ) .* image1 .+ sin.(π/2 .*λ).* image2
end

# Example of the stochastic interpolant. 
img1 = train_gaussian_images[1, :, :, 1]  # Extract the 2D image
img2 = train_images[1, :, :, 1]           # Extract the 2D image
interpolated_img = stochastic_interpolant(img1, img2, rand())

# Plot the original images and the interpolated image
p1 = heatmap(img1, color=:grays, axis=false, legend=false, title="Gaussian Image")
p2 = heatmap(img2, color=:grays, axis=false, legend=false, title="Target Image")
p3 = heatmap(interpolated_img, color=:grays, axis=false, legend=false, title="Interpolated Image")

# Display the plots side by side
plot(p1, p2, p3, layout=(1, 3), size=(900, 300))

### Time derivative stochastic interpolant ###
function time_derivative_stochastic_interpolant(image1, image2, λ)
    return -π/2 .*sin.(π/2 .*λ) .* image1 .+ π/2 .*cos.(π/2 .*λ).* image2
end

# Example to check
time_derivative_interpolated_img = time_derivative_stochastic_interpolant(img1, img2, rand())

### Convolutional Neural Network ###
function build_NN()
    return @compact(
        # Define the convolutional layers as a chain
        conv_layers = Chain(
            Conv((3, 3), 2 => 16, tanh, pad=1),  # Input channels: 2, Output channels: 16
            MaxPool((2, 2)),                     # Downsample to 14x14
            Conv((3, 3), 16 => 32, tanh, pad=1), # Keep 14x14
            Conv((3, 3), 32 => 64, tanh, pad=1), # Keep 14x14
            Upsample((2, 2)),                    # Upsample back to 28x28
            Conv((3, 3), 64 => 32, tanh, pad=1), # Keep 28x28
            Conv((3, 3), 32 => 16, tanh, pad=1), # Keep 28x28
            Conv((3, 3), 16 => 1, pad=1),        # Keep 28x28, final velocity field
            x -> 1.0 ./ (1.0 .+ exp.(-x))
        )
    ) do x
        # Extract the input image and time
        I_sample, t_sample = x

        # Reshape t_sample to match the spatial dimensions of I_sample (28, 28, 1, batchsize)
        t_sample_reshaped = reshape_t_sample(t_sample, size(I_sample, 4))  # Expand t_sample to (28, 28, 1, batchsize)
        
        # Concatenate the time t along the channel dimension
        x = cat(I_sample, t_sample_reshaped, dims=3)

        # Pass through the convolutional layers
        @return conv_layers(x)
    end
end

## Example usage of the build_NN function ##
# Define the network
velocity_cnn = build_NN()

# Example input tensors
batch_size = 32  # Example batch size
example_input_image = rand(Float32, 28, 28, 1, batch_size)               # (28, 28, 1, B) for image
example_t = rand(Float32, 1, batch_size)                                 # (1, 1, 1, B) for time

# Initialize network parameters and state
ps, st = Lux.setup(Random.default_rng(), velocity_cnn)
ps = ComponentArray(ps)
length(ps)

# Forward pass through the network with example inputs
output, st = velocity_cnn((example_input_image, example_t), ps, st)

# Print the output shape to verify
println("Output shape: ", size(output))  # Should be (28, 28, 1, B)

### Optimizer ###
# Define the Adam optimizer with a learning rate (optional: you can tweak the lr and betas)
opt = Optimisers.setup(Adam(0.0001),ps)  # 0.001 is the default learning rate

function time_derivative_stochastic_interpolant(image1::AbstractArray{T, 2}, image2::AbstractArray{T, 2}, λ::T) where T<:AbstractFloat
    return -π/2*sin(π/2*λ) .* image1 .+ π/2*cos(π/2*λ).* image2
end

# Example to check
time_derivative_interpolated_img = time_derivative_stochastic_interpolant(img1, img2, rand())

### Loss function ###
function loss_fn(velocity, dI_dt_sample)
    println("Max/Min of velocity: ", maximum(velocity), minimum(velocity))
    println("Max/Min of dI_dt_sample: ", maximum(dI_dt_sample), minimum(dI_dt_sample))

    # Compute the loss
    loss = velocity .^ 2 - 2 .* (dI_dt_sample .* velocity)

    # Check for NaN or Inf in the loss using broadcasting
    if any(isnan.(loss)) || any(isinf.(loss))
        println("Loss contains NaN or Inf")
    end

    mean_loss = mean(loss)
    return mean_loss
end

### Select mini-batch ###
function get_minibatch(images, batch_size, batch_index)
    start_index = (batch_index - 1) * batch_size + 1
    end_index = min(batch_index * batch_size, size(images,1))
    minibatch = images[start_index:end_index,:,:,:]
    minibatch = permutedims(minibatch, (2,3,4,1))
    return minibatch # Shape: (28 , 28, 1, batch_size) 
end

# Example: How many batches do we need?
num_samples = size(train_gaussian_images, 1)
num_batches = ceil(Int, num_samples / batch_size)

### Training ###
function train!(velocity_cnn, ps, st, opt, num_epochs, batch_size, train_gaussian_images, train_images)
    for epoch in 1:num_epochs
        println("Epoch $epoch")

        for batch_index in 1:num_batches
            # Sample a batch from the gaussian distribution (z) and target distribution (MNIST data)
            z_sample = Float32.(get_minibatch(train_gaussian_images, batch_size, batch_index))  # shape: (28, 28, 1, N_b)
            target_sample = Float32.(get_minibatch(train_images, batch_size, batch_index))  # shape: (28, 28, 1, N_b)

            # Sample time t from a uniform distribution between 0 and 1
            t_sample = Float32.(reshape(rand(Float32, batch_size), 1, 1, 1, batch_size))  # shape: (1, 1, 1, N_b)

            # Define the loss function closure for gradient calculation
            loss_fn_closure = (ps_) -> begin
                # Compute the interpolant I_t and its time derivative ∂t I_t
                I_sample = Float32.(stochastic_interpolant(z_sample, target_sample, t_sample)) # shape: (28, 28, 1, N_b)
                dI_dt_sample = Float32.(time_derivative_stochastic_interpolant(z_sample, target_sample, t_sample)) # shape: (28, 28, 1, N_b)

                # Compute velocity using the neural network
                velocity, _ = Lux.apply(velocity_cnn, (I_sample, t_sample), ps_, st) # shape: (28, 28, 1, N_b)

                # Compute the loss for the mini-batch
                return loss_fn(velocity, dI_dt_sample)  # Mean scalar loss
            end

            # Compute gradients using the loss function closure
            gs_tuple = gradient(loss_fn_closure, ps)

            # Unpack the tuple to get the actual gradient
            gs = gs_tuple[1]  # Since the gradient is returned as a Tuple
            println("Gradient norms: ", norm(gs))  # Check if gradients are very small or NaN

            # Debugging: Compare structure of ps and gs
            # println("ps structure: ", typeof(ps))
            # println("Gradient structure: ", typeof(gs))

            # Update the parameters using the optimizer
            # Clip gradients before updating the parameters
            gradient_clip_value = 1.0
            clipped_grads = clamp.(gs, -gradient_clip_value, gradient_clip_value)  # Set gradient_clip_value to a small number, e.g., 1.0
            opt, ps = Optimisers.update!(opt, ps, clipped_grads)

            # Calculate and display the mean loss for the batch
            mean_loss = loss_fn_closure(ps)
            println("Batch loss: $mean_loss")
        end
    end

    return ps, st
end

# Start training
num_epochs = 20
batch_size = 32
train!(velocity_cnn, ps, st, opt, num_epochs, batch_size, train_gaussian_images, train_images)

### Generate a Hand-written digit from a Gaussian image ###
# Function to generate a digit from Gaussian image using velocity field
function generate_digit(velocity_cnn, ps, st, initial_gaussian_image, num_steps, step_size)
    # Initialize with the Gaussian noise image
    image = initial_gaussian_image
    
    # Simulate forward evolution over time from t = 0 to t = 1
    for i in 1:num_steps
        # Compute the current time t in the interval [0, 1]
        t = i / num_steps
        
        # Reshape t_sample to match the batch size
        t_sample = Float32.(reshape([t],1,1,1,batch_size))
        
        # Predict the velocity field using the neural network
        velocity, st = Lux.apply(velocity_cnn, (image, t_sample), ps, st)
        
        # Update the image based on the velocity field
        image = image + step_size * velocity
    end
    
    return image
end

# Example usage of generating a digit
batch_size = 1  # Single image generation
gaussian_image = randn(Float32, 28, 28, 1, batch_size)  # Generate a Gaussian noise image
gaussian_image = clamp.(gaussian_image, 0.0, 1.0) # Pixel values as grey values in [0,1]


# Set the number of time steps and step size for the integration
num_steps = 1000  # Number of steps to evolve the image
step_size = 1.0 / num_steps  # Step size (proportional to time step)

# Generate the digit image
generated_digit = generate_digit(velocity_cnn, ps, st, gaussian_image, num_steps, step_size)

# Visualize the generated image
heatmap(reshape(generated_digit[:, :, 1, 1], (28, 28)), color=:grays, title="Generated Handwritten Digit")