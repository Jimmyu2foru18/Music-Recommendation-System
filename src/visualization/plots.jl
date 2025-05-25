using Plots
using DataFrames
using Statistics
using StatsBase
using Colors

"""Plot user listening history distribution."""
function plot_listening_history(user_history::DataFrame)
    # Create histogram of play counts
    p1 = histogram(user_history.play_count,
                  title="Play Count Distribution",
                  xlabel="Play Count",
                  ylabel="Frequency",
                  legend=false,
                  color=:blue,
                  alpha=0.6)
    
    # Create time series of listening activity
    if hasproperty(user_history, :timestamp)
        sorted_history = sort(user_history, :timestamp)
        p2 = plot(sorted_history.timestamp,
                  sorted_history.play_count,
                  title="Listening Activity Over Time",
                  xlabel="Time",
                  ylabel="Play Count",
                  legend=false,
                  color=:green)
        
        return plot(p1, p2, layout=(2,1), size=(800,600))
    end
    
    return p1
end

"""Plot song similarity heatmap."""
function plot_similarity_matrix(similarity_matrix::Matrix{Float64},
                              song_names::Vector{String};
                              max_songs::Int=50)
    # If there are too many songs, take a subset
    if length(song_names) > max_songs
        indices = 1:max_songs
        similarity_matrix = similarity_matrix[indices, indices]
        song_names = song_names[indices]
    end
    
    heatmap(similarity_matrix,
            xticks=(1:length(song_names), song_names),
            yticks=(1:length(song_names), song_names),
            xrotation=45,
            title="Song Similarity Matrix",
            color=:viridis,
            aspect_ratio=1,
            size=(800,800))
end

"""Plot recommendation results."""
function plot_recommendations(recommendations::DataFrame;
                            score_col::Symbol=:score,
                            name_col::Symbol=:song_name)
    # Sort recommendations by score
    sorted_recs = sort(recommendations, score_col, rev=true)
    
    # Create bar plot
    bar(sorted_recs[!, name_col],
        sorted_recs[!, score_col],
        title="Top Recommendations",
        xlabel="Songs",
        ylabel="Recommendation Score",
        xrotation=45,
        legend=false,
        color=:orange,
        size=(800,400))
end

"""Plot feature importance analysis."""
function plot_feature_importance(feature_matrix::Matrix{Float64},
                               feature_names::Vector{String},
                               target_variable::Vector{Float64})
    # Calculate correlation between features and target
    importance = [cor(feature_matrix[:,i], target_variable) for i in 1:size(feature_matrix,2)]
    
    # Sort features by absolute importance
    sorted_idx = sortperm(abs.(importance), rev=true)
    
    # Create bar plot
    bar(feature_names[sorted_idx],
        abs.(importance[sorted_idx]),
        title="Feature Importance Analysis",
        xlabel="Features",
        ylabel="Absolute Correlation",
        xrotation=45,
        legend=false,
        color=:purple,
        size=(800,400))
end

"""Plot user preference profile."""
function plot_user_profile(user_history::DataFrame,
                         song_features::DataFrame;
                         feature_cols::Vector{Symbol})
    # Calculate average feature values weighted by play count
    weighted_features = Dict{Symbol,Float64}()
    total_plays = sum(user_history.play_count)
    
    for feature in feature_cols
        weighted_sum = 0.0
        for row in eachrow(user_history)
            song_idx = findfirst(==(row.song_id), song_features.song_id)
            if !isnothing(song_idx)
                weighted_sum += row.play_count * song_features[song_idx, feature]
            end
        end
        weighted_features[feature] = weighted_sum / total_plays
    end
    
    # Create radar plot
    angles = range(0, 2π, length=length(feature_cols)+1)[1:end-1]
    values = [weighted_features[f] for f in feature_cols]
    
    # Normalize values to [0,1]
    values = (values .- minimum(values)) ./ (maximum(values) - minimum(values))
    
    # Create plot
    plot(angles,
         values,
         seriestype=:line,
         proj=:polar,
         title="User Preference Profile",
         label="",
         xticks=(angles, string.(feature_cols)),
         size=(600,600))
end