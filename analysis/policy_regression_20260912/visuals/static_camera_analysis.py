"""Read-only table-texture registration and raw-depth comparison of initial frames.

Pixels and native uint16 depth units only: recordings do not contain depth scale
or camera extrinsics. Homographies register table texture, not moving blocks/bin.
"""

from pathlib import Path
import json

import cv2
import h5py
import numpy as np

OUT = Path(__file__).resolve().parent
DATA = Path('/data3/extracted_data')
REFERENCE = 'demonstration_base_prompt_no_intervention_1'
frames = np.load(OUT / 'initial_frames.npz')
metadata = json.loads((OUT / 'initial_scene_metrics.json').read_text())
meta = {row['run']: row for row in metadata}
sift = cv2.SIFT_create(nfeatures=2500, contrastThreshold=0.015)
matcher = cv2.BFMatcher(cv2.NORM_L2)


def table_mask(image, cam):
    mask = np.zeros(image.shape[:2], np.uint8)
    if cam == 0:
        # Table top only; exclude bin and blocks, which can move independently.
        mask[186:235, 5:635] = 255
        mask[:211, 170:390] = 0
    elif cam == 1:
        mask[:230, :600] = 255
        mask[43:175, 38:181] = 0
        mask[45:180, 300:431] = 0
    else:
        cv2.fillConvexPoly(mask, np.array([[276, 156], [380, 156], [410, 350], [202, 350]]), 255)
        mask[190:284, 235:354] = 0
    return mask


def features(image, cam):
    return sift.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), table_mask(image, cam))


reference = frames[REFERENCE]
reference_features = [features(im, cam) for cam, im in enumerate(reference)]
anchors = [
    np.array([[90, 215], [300, 220], [525, 215]], np.float32),
    np.array([[230, 35], [235, 130], [465, 205]], np.float32),
    np.array([[330, 180], [370, 290], [270, 320]], np.float32),
]
patches = [
    {'table_left': [60, 202, 140, 228], 'table_right': [450, 200, 550, 228]},
    {'table_center_far': [210, 20, 280, 65], 'table_center_near': [215, 175, 290, 220]},
    {'table_far': [320, 161, 352, 187], 'table_near': [340, 285, 383, 330]},
]


def folder(name):
    group = meta[name]['group']
    return DATA / {'demo': 'both_in_bin_interleaved', 'original': 'base_rollouts_original', 'recent': ''}[group] / name


def load_depth(name, cam):
    with h5py.File(folder(name) / f'cam_{cam}_depth.h5', 'r') as handle:
        # A small temporal median reduces one-frame depth noise; first frames are
        # before significant task motion. Read only five frames, not whole videos.
        raw = handle['depth_images'][:5].astype(float)
        raw[raw == 0] = np.nan
        return np.nanmedian(raw, axis=0)


reference_depths = [load_depth(REFERENCE, c) for c in range(3)]
rows = []
match_visuals = []
for name in frames.files:
    images = frames[name]
    for cam, image in enumerate(images):
        row = {'group': meta[name]['group'], 'run': name, 'camera': cam}
        kp1, des1 = reference_features[cam]
        kp2, des2 = features(image, cam)
        pairs = matcher.knnMatch(des1, des2, k=2) if des1 is not None and des2 is not None else []
        good = [a for a, b in pairs if a.distance < .7 * b.distance]
        row['sift_ratio_matches'] = len(good)
        homography = None
        if len(good) >= 8:
            source = np.float32([kp1[x.queryIdx].pt for x in good])
            target = np.float32([kp2[x.trainIdx].pt for x in good])
            homography, inliers = cv2.findHomography(source, target, cv2.RANSAC, 1.5)
            if homography is not None:
                inliers = inliers.ravel().astype(bool)
                transformed = cv2.perspectiveTransform(source[:, None], homography)[:, 0]
                residual = np.linalg.norm(transformed - target, axis=1)
                shift = cv2.perspectiveTransform(anchors[cam][:, None], homography)[:, 0] - anchors[cam]
                row.update({
                    'homography_reference_to_frame': homography.tolist(),
                    'inliers': int(inliers.sum()),
                    'inlier_error_median_px': float(np.median(residual[inliers])),
                    'inlier_direct_displacement_median_px': np.median(target[inliers] - source[inliers], axis=0).tolist(),
                    'anchor_shift_px': shift.tolist(),
                    'median_anchor_displacement_px': float(np.median(np.linalg.norm(shift, axis=1))),
                })
                if name == 'demonstration_dracorex_base_0':
                    visual = cv2.drawMatches(reference[cam], kp1, image, kp2,
                        [g for g, keep in zip(good, inliers) if keep], None,
                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                    cv2.putText(visual, f'cam {cam}: table inlier matches, original 1 -> recent 0',
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)
                    match_visuals.append(visual)
        depth = load_depth(name, cam)
        row['raw_depth_patches'] = {}
        for patch_name, (x1, y1, x2, y2) in patches[cam].items():
            ref_mask = np.zeros(depth.shape, np.uint8)
            ref_mask[y1:y2, x1:x2] = 1
            mask = ref_mask
            if homography is not None and row.get('inliers', 0) >= 8:
                mask = cv2.warpPerspective(ref_mask, homography, (640, 360), flags=cv2.INTER_NEAREST)
            values = depth[mask.astype(bool)]
            ref_values = reference_depths[cam][ref_mask.astype(bool)]
            row['raw_depth_patches'][patch_name] = {
                'median_native_units': float(np.nanmedian(values)),
                'median_difference_from_reference_native_units': float(np.nanmedian(values) - np.nanmedian(ref_values)),
                'valid_fraction': float(np.isfinite(values).mean()),
            }
        # Fixed gray-table patch brightness; luminance comparison is descriptive
        # and not a calibrated exposure or illumination measurement.
        mask = table_mask(image, cam) > 0
        row['table_gray_median_8bit'] = float(np.median(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[mask]))
        rows.append(row)
    print(name, flush=True)

(OUT / 'static_camera_metrics.json').write_text(json.dumps(rows, indent=2))
if match_visuals:
    cv2.imwrite(str(OUT / 'table_feature_registration.jpg'), np.concatenate(match_visuals, axis=0))

summary = []
for group in ['demo', 'original', 'recent']:
    for cam in range(3):
        rr = [r for r in rows if r['group'] == group and r['camera'] == cam]
        usable = [r for r in rr if r.get('inliers', 0) >= 8]
        result = {'group': group, 'camera': cam, 'n': len(rr), 'registration_n': len(usable)}
        if usable:
            shifts = np.array([r['inlier_direct_displacement_median_px'] for r in usable])
            displacements = np.array([r['median_anchor_displacement_px'] for r in usable])
            result.update({
                'median_direct_xy_shift_px': np.median(shifts, axis=0).tolist(),
                'median_anchor_displacement_px': float(np.median(displacements)),
                'anchor_displacement_p10_p90_px': np.percentile(displacements, [10, 90]).tolist(),
                'median_inlier_count': float(np.median([r['inliers'] for r in usable])),
            })
        result['depth_patch_group_medians_native_units'] = {
            p: float(np.nanmedian([r['raw_depth_patches'][p]['median_native_units'] for r in rr])) for p in patches[cam]
        }
        result['gray_median'] = float(np.median([r['table_gray_median_8bit'] for r in rr]))
        summary.append(result)
(OUT / 'static_camera_summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
