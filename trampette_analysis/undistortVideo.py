import numpy as np
import cv2 as cv
import glob
import os

def get_camera_matrix(images, chessboard_size=(7, 6)):
    '''
    Takes in photos from camera and creates the calibration matrix and distortion coefficients.
    '''
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboard_size[0]*chessboard_size[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboard_size[0],0:chessboard_size[1]].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    shape = (100, 100) # Do not know the shape before hand
    
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)
    
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
    
            corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
    
            # Draw and display the corners
            cv.drawChessboardCorners(img, chessboard_size, corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(500)
            shape = gray.shape[::-1]
    cv.destroyAllWindows()

    print(f'Used {len(imgpoints)} images to calculate the calibration.')

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, shape, None, None)
    
    # Lets calculate the mean error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error

    print( "total error: {}".format(mean_error/len(objpoints)) )

    return mtx, dist

def undistort_image(mtx, dist, img_path):
    '''
    Function that undistorts an image. The image will be saved at the location data/undistorted_images
    '''
    img_name = os.path.basename(img_path).split('.')[0]
    # Create the output folder if it doesn't exist
    output_folder = 'static/undistorted_images'
    os.makedirs(output_folder, exist_ok=True)
    
    # Construct the output video path
    output_image_path = os.path.join(output_folder, f'undistorted_{img_name}.PNG') 
    
    # Now we undisort an image
    img = cv.imread(img_path)
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    # undistort
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)

    cv.imwrite(output_image_path, dst)

    print(f"Undistorted image saved as: {output_image_path}")

    

def undistort_video(mtx, dist, video_path):
    '''
    Function that creates a undistorted video of the video given with video path. The video will be saved at the location data/undistorted_videos
    '''
    # Extract the video name without extension
    video_name = os.path.basename(video_path).split('.')[0]

    # Create the output folder if it doesn't exist
    base, ext = os.path.splitext(video_path)
    output_path = base + "_undistorted" + ext

    # Open the input video
    cap = cv.VideoCapture(video_path)

    # Check if the video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        exit()

    # Get video properties
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv.CAP_PROP_FPS))
    frame_size = (frame_width, frame_height)

    # Prepare the output video writer
    fourcc = cv.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    out = cv.VideoWriter(output_path, fourcc, fps, frame_size)

    # Compute the optimal new camera matrix
    new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(mtx, dist, frame_size, 1, frame_size)

    # Process each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        
        # Undistort the frame
        undistorted_frame = cv.undistort(frame, mtx, dist, None, new_camera_matrix)
        
        # Write the undistorted frame to the output video
        out.write(undistorted_frame)

    # Release resources
    cap.release()
    out.release()
    cv.destroyAllWindows()

    print(f"Undistorted video saved as: {output_path}")


def main():
    load_coeffs = True
    camera_matrix, distortion_coef = None, None
    if load_coeffs:
        # Load calibration data
        data = np.load("static/camera_calibration.npz")
        camera_matrix = data["camera_matrix"]
        distortion_coef = data["dist_coeffs"]
    else:
        images = glob.glob('static/checkerboards/*.png') # Each square in the photos is 1.56 cm
        camera_matrix, distortion_coef = get_camera_matrix(images)
        # Save to a file
        np.savez("camera_calibration.npz", camera_matrix=camera_matrix, dist_coeffs=distortion_coef)
        print("Calibration saved!")


    # We undistort video
    base_folder = '/Users/asmundur/Local/Documents/Master Thesis/Videos/'
    user_ids = range(12, 16)
    excluded_names = {"Comp", "Semi", "Soft"}

    # Find all .mp4 and .mov files
    video_files = []

    for user_id in user_ids:
        folder_path = os.path.join(base_folder, str(user_id))
        mp4_files = glob.glob(os.path.join(folder_path, "**/*.mp4"), recursive=True)
        mov_files = glob.glob(os.path.join(folder_path, "**/*.mov"), recursive=True)
        video_files.extend(mp4_files + mov_files)

    filtered_videos = [
        path for path in video_files
        if os.path.splitext(os.path.basename(path))[0] not in excluded_names
    ]

    if not filtered_videos:
        print("No videos found.")
        return

    for video_path in filtered_videos:
        undistort_video(camera_matrix, distortion_coef, video_path)

if __name__ == "__main__":
    main()