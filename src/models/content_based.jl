using LinearAlgebra
using Statistics
using DataFrames
using SparseArrays

"""Content-based Recommender System implementation."""
mutable struct ContentBasedRecommender
    feature_matrix::Matrix{Float64}
    similarity_matrix::Matrix{Float64}
    song_indices::Dict{Int, Int}
    features::Vector{Symbol}
    
    function ContentBasedRecommender(features::Vector{Symbol})
        new(Matrix{Float64}(undef, 0, 0),
            Matrix{Float64}(undef, 0, 0),
            Dict{Int, Int}(),
            features)
    end
end

"""Compute cosine similarity between feature vectors."""
function cosine_similarity(A::Matrix{Float64})
    # Normalize the matrix
    norms = sqrt.(sum(A.^2, dims=2))
    norms[norms .== 0] .= 1  # Avoid division by zero
    normalized = A ./ norms
    
    # Compute similarity matrix
    return normalized * normalized'
end

"""Fit the content-based recommender with song features."""
function fit!(recommender::ContentBasedRecommender,
             songs::DataFrame,
             song_features::Matrix{Float64})
    # Store feature matrix
    recommender.feature_matrix = song_features
    
    # Create song index mapping
    recommender.song_indices = Dict(song_id => i for (i, song_id) in enumerate(songs.song_id))
    
    # Compute similarity matrix
    recommender.similarity_matrix = cosine_similarity(song_features)
end

"""Get similar songs based on content features."""
function get_similar_songs(recommender::ContentBasedRecommender,
                         song_id::Int,
                         n_recommendations::Int=5)
    if !haskey(recommender.song_indices, song_id)
        error("Song ID $song_id not found in the dataset")
    end
    
    # Get song index
    idx = recommender.song_indices[song_id]
    
    # Get similarities for this song
    similarities = recommender.similarity_matrix[idx, :]
    
    # Sort similarities (excluding the song itself)
    similarities[idx] = -1  # Exclude the input song
    top_n_idx = partialsortperm(similarities, 1:n_recommendations, rev=true)
    
    return top_n_idx, similarities[top_n_idx]
end

"""Get personalized recommendations based on user's listening history."""
function get_recommendations(recommender::ContentBasedRecommender,
                           user_history::DataFrame,
                           n_recommendations::Int=5)
    if isempty(user_history)
        error("User history is empty")
    end
    
    # Calculate weighted average of song features based on play counts
    total_plays = sum(user_history.play_count)
    weighted_features = zeros(size(recommender.feature_matrix, 2))
    
    for row in eachrow(user_history)
        if haskey(recommender.song_indices, row.song_id)
            idx = recommender.song_indices[row.song_id]
            weight = row.play_count / total_plays
            weighted_features .+= recommender.feature_matrix[idx, :] .* weight
        end
    end
    
    # Compute similarities between user profile and all songs
    similarities = cosine_similarity(reshape(weighted_features, 1, :) * recommender.feature_matrix')
    
    # Exclude songs that the user has already listened to
    listened_indices = [recommender.song_indices[id] for id in user_history.song_id if haskey(recommender.song_indices, id)]
    similarities[listened_indices] .= -1
    
    # Get top N recommendations
    top_n_idx = partialsortperm(similarities, 1:n_recommendations, rev=true)
    
    return top_n_idx, similarities[top_n_idx]
end