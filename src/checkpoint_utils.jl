using BSON

"""
    save_checkpoint(ps, st, opt_st, output_dir, epoch)
"""
function save_checkpoint(;
    ps::NamedTuple, 
    st::NamedTuple,
    opt_st::NamedTuple,
    output_dir::String,
    epoch::Int
)

    # If path does not exist, create it
    if !isdir(output_dir)
        mkdir(output_dir)
    end

    path = joinpath(output_dir, "checkpoint_epoch_$epoch.bson")

    return bson(
        path,
        Dict(
            :ps => cpu(ps),
            :st => cpu(st),
            :opt_st => cpu(opt_st)
        )
    )
end

"""
    load_checkpoint(path::String)
"""
function load_checkpoint(
    path::String
)

    checkpoint = BSON.load(path)

    return checkpoint[:ps], checkpoint[:st], checkpoint[:opt_st]
end



