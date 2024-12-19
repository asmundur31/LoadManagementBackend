from pymediainfo import MediaInfo

# Path to the video file
video_path = 'TestVideo3.mp4'

# Get detailed metadata
media_info = MediaInfo.parse(video_path)

# Extract and print the creation date (if available)
for track in media_info.tracks:
    if track.track_type == "General":  # Look in the 'General' track for file metadata
        if hasattr(track, 'encoded_date'):
            print(f"Encoded Date: {track.encoded_date}")
        if hasattr(track, 'file_created_date'):
            print(f"File Created Date: {track.file_created_date}")
        if hasattr(track, 'file_modified_date'):
            print(f"File Modified Date: {track.file_modified_date}")
