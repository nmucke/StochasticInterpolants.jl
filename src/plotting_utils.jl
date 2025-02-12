using Plots

function create_gif(A, filename, plot_titles)
    
    p_list = [];
    for i = 1:length(A)
        p = heatmap(
            A[i][:,:,1], legend=false, xticks=false, yticks=false, 
            # clim=(minimum(A[1]), maximum(A[1])), #clim=(0.15, 0.75),
            aspect_ratio=:equal,  title=plot_titles[i], color=cgrad(:Spectral_11, rev=true),
        )
        push!(p_list, p)
    end

    if length(A) > 2 && length(A) % 2 == 0
        p = plot(p_list..., layout=(2, length(A) ÷ 2), size=(1600, 800))

    elseif length(A) == 1
        p = plot(p_list..., layout=(1, 1), size=(800, 400))
    else
        p = plot(p_list..., layout=(1, length(A)), size=(length(A)*400, 400))
    end

    anim = @animate for i=1:size(A[1], 3)
        for j = 1:length(A)
            p[j][1][:z] = transpose(A[j][:,:,i])
        end
    end
    gif(anim, filename, fps = 15)
    
end
