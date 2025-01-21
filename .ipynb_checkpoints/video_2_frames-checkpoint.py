import cv2, os, argparse
from tqdm import tqdm

def main(args):
    for video in os.listdir(args.video_dir):
        video_name = video.split(".")[0]
        video_path = os.path.join(args.video_dir, video)

        cap = cv2.VideoCapture(video_path)
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = round(cap.get(cv2.CAP_PROP_FPS))
        print("video name:", video, "video length:", video_length, "fps:", fps)
        
        save_dir = os.path.join(args.image_save_dir, video_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        pbar = tqdm(total=int(video_length / fps))
        curr_frame_idx = 0
        while curr_frame_idx < video_length:
            cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame_idx)
            ret, frame = cap.read()
            if ret:
                img = cv2.resize(frame, (623, 350)) # We resize the original image to 623 x 350 to save storage space and speed up data loading.
                save_path = os.path.join(save_dir, f"{int(curr_frame_idx / fps)}.jpg")
                cv2.imwrite(save_path, img)
            else:
                print(f"frame {curr_frame_idx} read failed")
            curr_frame_idx += fps # down sample to 1 FPS
            pbar.set_postfix({"process": f"{curr_frame_idx + 1} / {video_length}"})
            pbar.update(1)
    print("all done")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser('experiment configs', add_help=False)
    parser.add_argument('--image_save_dir', type=str)
    parser.add_argument('--video_dir', type=str)
    args = parser.parse_args()

    main(args)






