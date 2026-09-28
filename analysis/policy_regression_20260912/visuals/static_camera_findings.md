# Initial scene geometry and depth comparison

Analyzed all **50 demonstrations, 10 original rollouts, and 7 recent rollouts**, using the first RGB frame from each camera. Compared table texture against original rollout 1 using SIFT ratio matching and a RANSAC homography; masks exclude blocks, bin, robot, and curtains. Depth summaries use a temporal median of the first five raw depth frames and table patches mapped by the RGB homography. No hardware was accessed.

## Main environmental change

The block color-to-position assignment is completely reversed: **50/50 demonstrations and 10/10 original rollouts have pink on image-left and blue on image-right in the wrist view; 7/7 recent rollouts have blue on image-left and pink on image-right.** The two occupied spatial slots remain close to the original positions, around x=110 and x=366 at 640×360 resolution. This categorical change is much larger than the measured camera/table alignment changes and is consistent with a learned color-to-location association failing on the second pick. Image position does not establish physical coordinates without calibration.

## Table texture alignment

The table and views are broadly stable, with modest apparent image shifts:

| Comparison to original rollout 1 | Camera 0 | Wrist camera 1 | Camera 2 |
|---|---:|---:|---:|
| Median displacement of three fixed table anchors, original recordings | 0.10 px | 0.10 px | 0.09 px |
| Median displacement, recent recordings | **3.90 px** | **0.77 px** | **1.81 px** |
| Median displacement, demonstrations | 3.50 px | 4.76 px | 1.99 px |

Recent comparisons have median 68, 80, and 60 RANSAC inliers per camera. Median inlier reprojection errors are 0.40, 0.31, and 0.33 pixels. These are matches on visible table paint/grain, not block correspondences. The recent wrist-view change is smaller than ordinary wrist-view variation among the demonstrations. There is no large camera swap, image rotation, zoom, or table displacement visible in this measurement.

These homographies measure **relative table-to-camera image alignment**, not camera motion independently. They cannot distinguish a small camera move from a table move or assign physical millimeters. The chosen three table anchors are listed in `static_camera_analysis.py`; homographies and per-record diagnostics are in `static_camera_metrics.json`. The visualization `table_feature_registration.jpg` shows matched table features for original 1 versus recent 0.

Bin-template matching finds only modest shifts. Compared with the original-group median, recent group median template shifts differ by approximately (-3,-1), (-3,+4), and (-6,+1) pixels for cameras 0, 1, and 2. Template normalized correlation medians are 0.957, 0.876, and 0.941 for recent frames. These measurements also include the small table/camera shifts and bin perspective changes; they do not support a major new bin location. `bin_template_metrics.json` contains raw comparisons against original rollout 1.

## Depth and appearance

Two corresponding, unobstructed wrist-view tabletop patches have median raw depth values:

| Table patch | Demonstrations | Original rollouts | Recent rollouts |
|---|---:|---:|---:|
| Far/upper central table | 317 | 318 | 317 |
| Near/lower central table | 355.5 | 356 | 356 |

The wrist tabletop distance is therefore very similar in the recorded native depth units. **Depth scale, camera serials/intrinsics, and extrinsics are not stored with these recordings**, so these are not calibrated physical-distance measurements. The publisher computes intrinsics at runtime, but the recorder does not persist them or depth scale.

Other-view depth is less reliable. One camera 0 patch has extreme old-recording values around 18,892 despite being a nearby tabletop, and a camera 2 far-table patch differs substantially while its near-table patch changes by only a few units. Grazing angle, missing/incorrect depth, and scene occlusion make these unsuitable for inferring physical table displacement. These raw values remain in the JSON for transparency and are not used to claim altered robot/table geometry.

Median grayscale intensity in the table masks changes only slightly: original→recent 220→221 (camera 0), 143→140 (wrist), 242→244 (camera 2), on a 0–255 scale. This does not rule out local lighting/white-balance changes, but there is no large overall table brightness change. Camera source does not lock RGB exposure or white balance, so those runtime settings remain unrecorded.

## Interpretation limits

Small camera/table/bin differences exist and could affect a visually sensitive policy. However, they are modest relative to the color-slot reversal; the stable wrist table depth and robot state/response analysis do not support a gross environment-height or robot-dynamics change. A controlled repeat restoring pink-left/blue-right is the most direct test of the observed environmental shift. Record exact camera calibration and depth scale if physical geometric measurements are needed.
