using Plots

function heatgif(A)
    p = heatmap(A[:,:,1])
    anim = @animate for i=1:size(A,3)
        heatmap!(p[1], A[:,:, i])
    end
    return anim
end

"""
    create_gif(data, filename)

Create a gif from a 3D array of images
"""
function create_gif(data, filename)
    anim = heatgif(data)
    gif(anim, filename, fps = 15)
end

