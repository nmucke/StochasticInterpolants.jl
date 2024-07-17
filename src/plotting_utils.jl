using Plots

function create_gif(A, filename, plot_titles)
    
    p_list = [];
    for i = 1:length(A)
        p = heatmap(
            A[i][:,:,1], legend=false, xticks=false, yticks=false, 
            clim=(minimum(A[1]), maximum(A[1])), aspect_ratio=:equal, 
            colorbar=true, title=plot_titles[i]
        )
        push!(p_list, p) 
    end

    if length(A) > 2
        p = plot(p_list..., layout=(2, length(A) ÷ 2), size=(1600, 800))

    elseif length(A) == 1
        p = plot(p_list..., layout=(1, 1), size=(800, 400))
    else
        p = plot(p_list..., layout=(1, length(A)), size=(800, 400))
    end

    anim = @animate for i=1:size(A[1], 3)
        for j = 1:length(A)
            p[j][1][:z] = transpose(A[j][:,:,i])
            
            # p[j][1][:title] = plot_titles[j] + " - time step: " + string(i)
        end
    end
    gif(anim, filename, fps = 15)
    
end

# function heatgif(A, B)
#     p1 = heatmap(
#         A[:,:,1], legend=false, xticks=false, yticks=false, 
#         clim=(minimum(A), maximum(A)), aspect_ratio=:equal, 
#         colorbar=true
#     )
#     p2 = heatmap(
#         B[:,:,1], legend=false, xticks=false, yticks=false, 
#         clim=(minimum(B), maximum(B)), aspect_ratio=:equal, 
#         colorbar=true
#     )
#     p = plot(p1, p2, layout=(1,2))

#     anim = @animate for i=1:size(A, 3)
#         p[1][1][:z] = A[:,:,i]
#         p[2][1][:z] = B[:,:,i]
#     end

#     gif(anim, filename, fps = 5)
# end
# function heatgif(A, B, C)
#     p1 = heatmap(
#         A[:,:,1], legend=false, xticks=false, yticks=false, 
#         clim=(minimum(A), maximum(A)), aspect_ratio=:equal, 
#         colorbar=true
#     )
#     p2 = heatmap(
#         B[:,:,1], legend=false, xticks=false, yticks=false, 
#         clim=(minimum(A), maximum(A)), aspect_ratio=:equal, 
#         colorbar=true
#     )
#     p3 = heatmap(
#         C[:,:,1], legend=false, xticks=false, yticks=false, 
#         clim=(minimum(C), maximum(C)), aspect_ratio=:equal, 
#         colorbar=true
#     )
#     p = plot(p1, p2, p3, layout=(1,3))

#     anim = @animate for i=1:size(A, 3)
#         p[1][1][:z] = A[:,:,i]
#         p[2][1][:z] = B[:,:,i]
#         p[3][1][:z] = C[:,:,i]
#     end

#     gif(anim, filename, fps = 5)
# end

# """
#     create_gif(data, filename)

# Create a gif from a 3D array of images
# """
# function create_gif(data, filename)
#     anim = heatgif(data)
#     gif(anim, filename, fps = 5)
# end

# function create_gif(data, data2, filename)
#     anim = heatgif(data, data2)
#     gif(anim, filename, fps = 5)
# end

# function create_gif(data, data2, data3, filename)
#     anim = heatgif(data, data2, data3)
#     gif(anim, filename, fps = 5)
# end