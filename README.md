# motion-tracking-project
Motion detection and tracking system exploiting background subtraction and SORT.

The material contains two videos from the Little version of Stanford University Drone Dataset.

## Dependencies

Install all required libraries with:

```bash
pip install opencv-python numpy matplotlib scikit-learn
```
## How to run:

The order to run the scripts:

1. ```bash
   python detection_bg_sub.py
   ```
   Performs motion detection and tracking, generates a video with bounding boxes around tracked items and several files with trajectories coords.
2. ```bash
   python create_image.py
   ```
   Exploiting matplotlib and numpy creates visival feedback of the results. Generates histogram for MSE and drawing of outliers trajectories.
3. ```bash
   python frequent_path.py
   ```
   Performs the frequent path anaylisis and return a decreasing by value dictionary where the first element is the most frequent path.

Full explanation of how the system works and its results is in the project report and in the demo video. 
