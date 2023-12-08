function GridInterpolations.interpolate(grid::AbstractGrid, data::DenseArray, x::AbstractVector)
    index, weight = interpolants(grid, x)
    dot(data[index], weight)
end


function my_interpolate(grid::AbstractGrid, data::DenseArray, x::AbstractVector)
    # indices, weight = interpolants(grid, x)
    # # println("HG")
    # # dot(data[index], weight)
    # v = 0.0
    # # for (index,val) in enumerate(indices)
    # #     v += data[val]*weight[index]
    # # end
    # return v

    if any(isnan, x)
        throw(DomainError("Input contains NaN!"))
    end
    cut_counts = grid.cut_counts
    cuts = grid.cuts

    # Reset the values in index and weight:
    fill!(grid.index,0)
    fill!(grid.index2,0)
    fill!(grid.weight,0)
    fill!(grid.weight2,0)
    grid.index[1] = 1
    grid.index2[1] = 1
    grid.weight[1] = 1.
    grid.weight2[1] = 1.

    l = 1
    subblock_size = 1
    cut_i = 1
    n = 1
    for d = 1:length(x)
        coord = x[d]
        lasti = cut_counts[d]+cut_i-1
        ii = cut_i

        if coord <= cuts[ii]
            i_lo, i_hi = ii, ii
        elseif coord >= cuts[lasti]
            i_lo, i_hi = lasti, lasti
        else
            while cuts[ii] < coord
                ii = ii + 1
            end
            if cuts[ii] == coord
                i_lo, i_hi = ii, ii
            else
                i_lo, i_hi = (ii-1), ii
            end
        end

        if i_lo == i_hi
            for i = 1:l
                grid.index[i] += (i_lo - cut_i)*subblock_size
            end
        else
            low = (1 - (coord - cuts[i_lo])/(cuts[i_hi]-cuts[i_lo]))
            for i = 1:l
                grid.index2[i  ] = grid.index[i] + (i_lo-cut_i)*subblock_size
                grid.index2[i+l] = grid.index[i] + (i_hi-cut_i)*subblock_size
            end
            copyto!(grid.index,grid.index2)
            for i = 1:l
                grid.weight2[i  ] = grid.weight[i]*low
                grid.weight2[i+l] = grid.weight[i]*(1-low)
            end
            copyto!(grid.weight,grid.weight2)
            l = l*2
            n = n*2
        end
        cut_i = cut_i + cut_counts[d]
        subblock_size = subblock_size*(cut_counts[d])
    end

    v = 0.0
    if l<length(grid.index)
        # This is true if we don't need to interpolate all dimensions because we're on a boundary:
        # return grid.index[1:l]::Vector{Int}, grid.weight[1:l]::Vector{Float64}
        # return  grid.index[1:l]
        # return @views grid.index[1:l], grid.weight[1:l]
        for i in 1:l
            data_ind = grid.index[i]
            v += data[data_ind]*grid.weight[i]
        end
    else
        for i in 1:length(grid.index)
            data_ind = grid.index[i]
            v += data[data_ind]*grid.weight[i]
        end
    end
    return v
    # @views grid.index, grid.weight
end

function GridInterpolations.interpolants(grid::RectangleGrid, x::AbstractVector)
    if any(isnan, x)
        throw(DomainError("Input contains NaN!"))
    end
    cut_counts = grid.cut_counts
    cuts = grid.cuts

    # Reset the values in index and weight:
    fill!(grid.index,0)
    fill!(grid.index2,0)
    fill!(grid.weight,0)
    fill!(grid.weight2,0)
    grid.index[1] = 1
    grid.index2[1] = 1
    grid.weight[1] = 1.
    grid.weight2[1] = 1.

    l = 1
    subblock_size = 1
    cut_i = 1
    n = 1
    for d = 1:length(x)
        coord = x[d]
        lasti = cut_counts[d]+cut_i-1
        ii = cut_i

        if coord <= cuts[ii]
            i_lo, i_hi = ii, ii
        elseif coord >= cuts[lasti]
            i_lo, i_hi = lasti, lasti
        else
            while cuts[ii] < coord
                ii = ii + 1
            end
            if cuts[ii] == coord
                i_lo, i_hi = ii, ii
            else
                i_lo, i_hi = (ii-1), ii
            end
        end

        if i_lo == i_hi
            for i = 1:l
                grid.index[i] += (i_lo - cut_i)*subblock_size
            end
        else
            low = (1 - (coord - cuts[i_lo])/(cuts[i_hi]-cuts[i_lo]))
            for i = 1:l
                grid.index2[i  ] = grid.index[i] + (i_lo-cut_i)*subblock_size
                grid.index2[i+l] = grid.index[i] + (i_hi-cut_i)*subblock_size
            end
            copyto!(grid.index,grid.index2)
            for i = 1:l
                grid.weight2[i  ] = grid.weight[i]*low
                grid.weight2[i+l] = grid.weight[i]*(1-low)
            end
            copyto!(grid.weight,grid.weight2)
            l = l*2
            n = n*2
        end
        cut_i = cut_i + cut_counts[d]
        subblock_size = subblock_size*(cut_counts[d])
    end

    if l<length(grid.index)
        # This is true if we don't need to interpolate all dimensions because we're on a boundary:
        return get_view(grid.index::Vector{Int},l), get_view(grid.weight::Vector{Float64},l)
        # return  grid.index[1:l]
        # return @views grid.index[1:l], grid.weight[1:l]
        # v = 0.0
        # for i in 1:l
        #     data_ind = grid.index[i]
        #     v += data[data_ind]*grid.weight[i]
        # end
        # # println("HG")
        # return v
    end
    get_view(grid.index::Vector{Int}), get_view(grid.weight::Vector{Float64})
    # @views grid.index, grid.weight
end

function get_view(x)
    return @view(x[:])
end
function get_view(x,l)
    return @view(x[1:l])
end
