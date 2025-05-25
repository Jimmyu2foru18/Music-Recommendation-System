using Statistics
using LinearAlgebra

"""Calculate Root Mean Square Error."""
function rmse(predictions::Vector{Float64}, targets::Vector{Float64})
    if length(predictions) != length(targets)
        error("Predictions and targets must have the same length")
    end
    return sqrt(mean((predictions .- targets).^2))
end

"""Calculate Mean Absolute Error."""
function mae(predictions::Vector{Float64}, targets::Vector{Float64})
    if length(predictions) != length(targets)
        error("Predictions and targets must have the same length")
    end
    return mean(abs.(predictions .- targets))
end

"""Calculate Precision at K."""
function precision_at_k(recommended_items::Vector{Int},
                      relevant_items::Vector{Int},
                      k::Int)
    if k <= 0
        error("k must be positive")
    end
    
    k = min(k, length(recommended_items))
    recommended_k = recommended_items[1:k]
    relevant_and_recommended = intersect(Set(recommended_k), Set(relevant_items))
    
    return length(relevant_and_recommended) / k
end

"""Calculate Recall at K."""
function recall_at_k(recommended_items::Vector{Int},
                    relevant_items::Vector{Int},
                    k::Int)
    if k <= 0
        error("k must be positive")
    end
    if isempty(relevant_items)
        return 0.0
    end
    
    k = min(k, length(recommended_items))
    recommended_k = recommended_items[1:k]
    relevant_and_recommended = intersect(Set(recommended_k), Set(relevant_items))
    
    return length(relevant_and_recommended) / length(relevant_items)
end

"""Calculate Mean Average Precision."""
function mean_average_precision(recommended_items::Vector{Int},
                              relevant_items::Vector{Int})
    if isempty(relevant_items)
        return 0.0
    end
    
    precision_sum = 0.0
    num_hits = 0
    
    for (i, item) in enumerate(recommended_items)
        if item in relevant_items
            num_hits += 1
            precision_sum += num_hits / i
        end
    end
    
    return precision_sum / length(relevant_items)
end

"""Calculate Normalized Discounted Cumulative Gain."""
function ndcg_at_k(recommended_items::Vector{Int},
                   relevant_items::Vector{Int},
                   k::Int)
    if k <= 0
        error("k must be positive")
    end
    if isempty(relevant_items)
        return 0.0
    end
    
    k = min(k, length(recommended_items))
    recommended_k = recommended_items[1:k]
    
    # Calculate DCG
    dcg = 0.0
    for (i, item) in enumerate(recommended_k)
        if item in relevant_items
            dcg += 1.0 / log2(i + 1)
        end
    end
    
    # Calculate IDCG
    idcg = sum(1.0 / log2(i + 1) for i in 1:min(k, length(relevant_items)))
    
    return dcg / idcg
end

"""Calculate Coverage of recommendations."""
function catalog_coverage(all_recommendations::Vector{Vector{Int}},
                        total_items::Int)
    unique_recommended = unique(vcat(all_recommendations...))
    return length(unique_recommended) / total_items
end

"""Calculate Diversity of recommendations using average pairwise distance."""
function diversity_score(recommended_items::Vector{Int},
                        item_features::Matrix{Float64})
    if length(recommended_items) < 2
        return 0.0
    end
    
    n_items = length(recommended_items)
    total_distance = 0.0
    n_pairs = 0
    
    for i in 1:n_items
        for j in (i+1):n_items
            # Calculate cosine distance between items
            item1 = item_features[recommended_items[i], :]
            item2 = item_features[recommended_items[j], :]
            similarity = dot(item1, item2) / (norm(item1) * norm(item2))
            distance = 1 - similarity
            
            total_distance += distance
            n_pairs += 1
        end
    end
    
    return total_distance / n_pairs
end