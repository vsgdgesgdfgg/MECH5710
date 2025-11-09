"""
Batch process Xiangqi board images in indoor and outdoor folders
Use SIFT feature matching to align to blank board template and extract grid patches
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Configuration
FEATURE_EXTRACT = Path("./blank_temp.jpg")
DATA_DIR = Path("./data")
OUTPUT_BASE = Path("./processed_boards")

# Board intersection coordinates (9 rows × 10 columns)
BOARD_POINTS = np.array([
    [[114,165],[194,166],[275,163],[350,165],[428,167],[505,165],[586,163],[665,163],[747,163],[830,165]],
    [[115,240],[193,240],[272,240],[350,240],[430,240],[510,240],[585,240],[665,240],[747,240],[832,240]],
    [[110,320],[193,320],[270,320],[350,320],[430,320],[507,320],[588,320],[667,320],[747,320],[835,320]],
    [[110,395],[190,395],[270,395],[350,395],[430,395],[505,395],[585,395],[667,395],[750,395],[835,395]],
    [[110,470],[190,470],[270,470],[350,470],[430,470],[505,470],[585,470],[667,470],[750,470],[835,470]],
    [[110,545],[190,545],[270,545],[350,545],[430,545],[505,545],[585,545],[667,545],[750,545],[835,545]],
    [[110,620],[190,620],[270,620],[350,620],[430,620],[505,620],[585,620],[667,620],[750,620],[835,620]],
    [[110,705],[190,705],[270,705],[350,705],[430,705],[505,705],[585,705],[667,705],[750,705],[835,705]],
    [[110,790],[190,790],[270,790],[350,790],[430,790],[505,790],[585,790],[667,790],[750,790],[835,790]]
])

# Ensure output directory exists
OUTPUT_BASE.mkdir(exist_ok=True)

class BoardAligner:
    """Xiangqi board aligner"""
    
    def __init__(self, template_path):
        """Initialize and load blank board template"""
        self.template = cv2.imread(str(template_path))
        if self.template is None:
            raise FileNotFoundError(f"Template file does not exist: {template_path}")
        
        self.h_t, self.w_t = self.template.shape[:2]
        print(f"Loaded blank board template: {self.w_t}x{self.h_t}")
        
        # Initialize SIFT
        self.sift = cv2.SIFT_create(
            nfeatures=5000,
            contrastThreshold=0.03,
            edgeThreshold=15
        )
        
        # Extract template features
        r = cv2.split(self.template)[2]
        _, mask = cv2.threshold(r, 100, 255, cv2.THRESH_BINARY)
        self.kp_template, self.desc_template = self.sift.detectAndCompute(
            self.template, mask=mask
        )
        print(f"Template keypoints: {len(self.kp_template)}")
        
        # Initialize FLANN matcher
        self.flann = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=8),
            dict(checks=128)
        )
    
    def align_image(self, img_path, visualize=False):
        """
        Align single image to blank board template
        
        Args:
            img_path: Input image path
            visualize: Whether to visualize results
            
        Returns:
            aligned: Aligned image (932x932)
            best_angle: Best rotation angle
            best_inliers: RANSAC inliers count
        """
        img_raw = cv2.imread(str(img_path))
        if img_raw is None:
            raise FileNotFoundError(f"Unable to read image: {img_path}")
        
        print(f"\nProcessing: {img_path.name}")
        print(f"  Original size: {img_raw.shape[1]}x{img_raw.shape[0]}")
        
        # Test 4 rotation directions
        best_angle = 0
        best_inliers = 0
        best_H = None
        
        for angle in [0, 90, 180, 270]:
            # Rotate original image
            if angle == 90:
                rotated = cv2.rotate(img_raw, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                rotated = cv2.rotate(img_raw, cv2.ROTATE_180)
            elif angle == 270:
                rotated = cv2.rotate(img_raw, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                rotated = img_raw
            
            # Extract features
            r_ch = cv2.split(rotated)[2]
            _, m = cv2.threshold(r_ch, 100, 255, cv2.THRESH_BINARY)
            kp, desc = self.sift.detectAndCompute(rotated, mask=m)
            
            if desc is None or len(desc) < 4:
                continue
            
            # Match
            matches = self.flann.knnMatch(self.desc_template, desc, k=2)
            good = [m for m, n in matches if len([m, n]) == 2 and m.distance < 0.75 * n.distance]
            
            if len(good) < 4:
                continue
            
            # Calculate homography
            src_pts = np.float32([self.kp_template[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            H, mask_h = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)
            inliers = int(mask_h.sum()) if mask_h is not None else 0
            
            print(f"  {angle:3d}°: Features={len(kp):4d}, Matches={len(good):3d}, Inliers={inliers:3d}")
            
            if inliers > best_inliers:
                best_inliers = inliers
                best_angle = angle
                best_H = H
        
        # Check matching quality
        if best_inliers < 10:
            print(f"  Matching failed, insufficient inliers: {best_inliers}")
            return None, best_angle, best_inliers
        
        print(f"  Best direction: {best_angle}°, inliers: {best_inliers}")
        
        # Apply best transformation
        if best_angle == 90:
            img_rotated = cv2.rotate(img_raw, cv2.ROTATE_90_CLOCKWISE)
        elif best_angle == 180:
            img_rotated = cv2.rotate(img_raw, cv2.ROTATE_180)
        elif best_angle == 270:
            img_rotated = cv2.rotate(img_raw, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            img_rotated = img_raw
        
        # Align to blank board
        H_inv = np.linalg.inv(best_H)
        aligned = cv2.warpPerspective(img_rotated, H_inv, (self.w_t, self.h_t))
        
        # Visualize
        if visualize:
            self._visualize_alignment(img_raw, img_rotated, aligned, best_angle, best_inliers)
        
        return aligned, best_angle, best_inliers
    
    def _visualize_alignment(self, img_raw, img_rotated, aligned, angle, inliers):
        """Visualize alignment result"""
        plt.figure(figsize=(18, 5))
        
        plt.subplot(141)
        plt.imshow(cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(142)
        plt.imshow(cv2.cvtColor(img_rotated, cv2.COLOR_BGR2RGB))
        plt.title(f'Rotated {angle}°')
        plt.axis('off')
        
        plt.subplot(143)
        plt.imshow(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))
        plt.title(f'Aligned\n{self.w_t}x{self.h_t}')
        plt.axis('off')
        
        plt.subplot(144)
        overlay = cv2.addWeighted(aligned, 0.5, self.template, 0.5, 0)
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.title(f'Overlay\nInliers={inliers}')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()


def extract_grid_patches(aligned_img, board_points, output_dir, img_name, half=62):
    """
    Extract grid patches from aligned board image
    
    Args:
        aligned_img: Aligned image
        board_points: Grid intersection coordinates (9x10x2)
        output_dir: Output directory (crops root directory)
        img_name: Image name (for naming subfolder)
        half: Patch radius
    """
    # Create subfolder for each image
    img_crop_dir = output_dir / img_name
    img_crop_dir.mkdir(parents=True, exist_ok=True)
    
    patch_count = 0
    for r in range(9):  # 9 rows
        for c in range(10):  # 10 columns
            x, y = board_points[r, c]
            crop = aligned_img[y-half:y+half+1, x-half:x+half+1]
            
            # Save patch (without image name prefix)
            filename = f"r{r+1}_c{c+1}.jpg"
            cv2.imwrite(str(img_crop_dir / filename), crop)
            patch_count += 1
    
    return patch_count


def process_folder(aligner, folder_path, folder_name, visualize_first=True):
    """
    Process all images in a folder
    
    Args:
        aligner: BoardAligner instance
        folder_path: Folder path
        folder_name: Folder name (for output)
        visualize_first: Whether to visualize first image
    """
    print("\n" + "="*70)
    print(f"Processing folder: {folder_path}")
    print("="*70)
    
    # Create output directory
    aligned_dir = OUTPUT_BASE / folder_name / "aligned"
    crops_dir = OUTPUT_BASE / folder_name / "crops"
    aligned_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all images
    image_files = sorted(folder_path.glob("*.jpg")) + sorted(folder_path.glob("*.png"))
    print(f"Found {len(image_files)} images")
    
    if len(image_files) == 0:
        print("Warning: No image files found")
        return
    
    # Statistics
    results = []
    total_patches = 0
    
    for idx, img_path in enumerate(image_files, 1):
        try:
            # Align image
            visualize = (idx == 1 and visualize_first)
            aligned, best_angle, best_inliers = aligner.align_image(img_path, visualize)
            
            if aligned is None:
                results.append({
                    'file': img_path.name,
                    'success': False,
                    'angle': best_angle,
                    'inliers': best_inliers
                })
                continue
            
            # Save aligned image
            aligned_path = aligned_dir / img_path.name
            cv2.imwrite(str(aligned_path), aligned)
            
            # Extract grid patches
            img_stem = img_path.stem
            patch_count = extract_grid_patches(
                aligned, BOARD_POINTS, crops_dir, img_stem
            )
            total_patches += patch_count
            
            results.append({
                'file': img_path.name,
                'success': True,
                'angle': best_angle,
                'inliers': best_inliers,
                'patches': patch_count
            })
            
        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                'file': img_path.name,
                'success': False,
                'error': str(e)
            })
    
    # Save statistics
    stats = {
        'folder': folder_name,
        'total_images': len(image_files),
        'success_count': sum(1 for r in results if r['success']),
        'fail_count': sum(1 for r in results if not r['success']),
        'total_patches': total_patches,
        'results': results
    }
    
    stats_path = OUTPUT_BASE / folder_name / "stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "-"*70)
    print(f"【{folder_name} Processing Complete】")
    print(f"  Total images: {stats['total_images']}")
    print(f"  Success: {stats['success_count']}")
    print(f"  Failed: {stats['fail_count']}")
    print(f"  Extracted patches: {stats['total_patches']}")
    print(f"  Aligned images saved to: {aligned_dir}")
    print(f"  Patches saved to: {crops_dir}")
    print(f"  Statistics saved to: {stats_path}")
    print("-"*70)


def main():
    print("="*70)
    print("Xiangqi Board Batch Processing - Indoor & Outdoor")
    print("="*70)
    
    # Check template file
    if not FEATURE_EXTRACT.exists():
        print(f"Error: Blank board template does not exist: {FEATURE_EXTRACT}")
        return
    
    # Initialize aligner
    print("\n【Initialization】")
    aligner = BoardAligner(FEATURE_EXTRACT)
    
    # Process indoor folder
    indoor_base = DATA_DIR / "indoor"
    if indoor_base.exists():
        for scene_dir in sorted(indoor_base.iterdir()):
            if scene_dir.is_dir():
                folder_name = f"indoor_{scene_dir.name}"
                process_folder(aligner, scene_dir, folder_name, visualize_first=False)
    else:
        print(f"Warning: Indoor folder does not exist: {indoor_base}")
    
    # Process outdoor folder
    outdoor_base = DATA_DIR / "outdoor"
    if outdoor_base.exists():
        for scene_dir in sorted(outdoor_base.iterdir()):
            if scene_dir.is_dir():
                folder_name = f"outdoor_{scene_dir.name}"
                process_folder(aligner, scene_dir, folder_name, visualize_first=False)
    else:
        print(f"Warning: Outdoor folder does not exist: {outdoor_base}")
    
    print("\n" + "="*70)
    print("All Processing Complete!")
    print("="*70)


if __name__ == "__main__":
    main()

