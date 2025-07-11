def mood_summary_prompt(track_features):
    return f"""
A track has the following characteristics:
- Valence: {track_features['valence']}
- Energy: {track_features['energy']}
- Danceability: {track_features['danceability']}
- Tempo: {track_features['tempo']} BPM

Describe the emotional mood or atmosphere of this track. Be poetic but clear.
"""

def genre_match_prompt(track_features):
    return f"""
Based on these features:
- Energy: {track_features['energy']}
- Acousticness: {track_features['acousticness']}
- Instrumentalness: {track_features['instrumentalness']}
- Tempo: {track_features['tempo']}

What electronic/EDM sub-genres would this track likely fall under? Justify briefly.
"""

def spiritual_context_prompt(track_features):
    return f"""
Given a track with:
- Energy: {track_features['energy']}
- Valence: {track_features['valence']}
- Tempo: {track_features['tempo']}

What kind of spiritual or meditative context would this track suit?
(E.g., breathwork, chakra meditation, ecstatic dance, etc.)
"""

def listening_context_prompt(track_features):
    return f"""
This song has the following attributes:
- Danceability: {track_features['danceability']}
- Valence: {track_features['valence']}
- Tempo: {track_features['tempo']}

Suggest when and where someone should listen to this song.
(e.g., morning workout, sunset beach, after meditation)
"""
