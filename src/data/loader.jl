using CSV
using DataFrames

"""Load raw song data from CSV file."""
function load_songs(file_path::String)
    try
        return CSV.read(file_path, DataFrame)
    catch e
        @error "Error loading song data: $e"
        return nothing
    end
 end

"""Load user listening history data."""
function load_user_history(file_path::String)
    try
        return CSV.read(file_path, DataFrame)
    catch e
        @error "Error loading user history: $e"
        return nothing
    end
end

"""Load or create user-song interaction matrix."""
function create_interaction_matrix(user_history::DataFrame)
    # Convert user history to sparse matrix format
    users = unique(user_history.user_id)
    songs = unique(user_history.song_id)
    
    # Create mapping dictionaries
    user_to_idx = Dict(user => i for (i, user) in enumerate(users))
    song_to_idx = Dict(song => i for (i, song) in enumerate(songs))
    
    # Initialize interaction matrix
    interactions = zeros(Float32, length(users), length(songs))
    
    # Fill interaction matrix
    for row in eachrow(user_history)
        user_idx = user_to_idx[row.user_id]
        song_idx = song_to_idx[row.song_id]
        interactions[user_idx, song_idx] = row.play_count
    end
    
    return interactions, user_to_idx, song_to_idx
end