## How to Run Live Mode on a Remote Server
The motivation is that some laptops don’t have sufficient GPU resources to run the STA frontend, making it difficult to run live mode directly.

To address this, the laptop can instead be used only for video capture and visualization. The following instructions briefly describe how to set this up.

### Streaming the Camera
First, on the laptop, use cam_test.py to check which camera device should be used.
Then, in `live.py`, stream the selected camera to `127.0.0.1:5000/video`. And run
```bash
python live.py
```

To verify the video stream, open `http://127.0.0.1:5000/video` in a browser and check whether the video feed is displayed.

### Port Forwarding
Since in many cases the server cannot directly access the laptop via SSH, reverse port forwarding is required to make the video stream accessible from the server:
```bash
ssh -R 5000:127.0.0.1:5000 -R 9876:127.0.0.1:9876 USER@SERVER_IP
```
Here, `5000` is the port for the video stream, and `9876` is the port used for `rerun` visualization.
To visualize the results locally, open `rerun` on the laptop.

### Running ViSTA-SLAM in Live Mode
Finally, run ViSTA-SLAM in live mode on the remote server:
```bash
python run_live.py --config configs/default.yaml --camera "http://127.0.0.1:5000/video" --output output/live --vis
```