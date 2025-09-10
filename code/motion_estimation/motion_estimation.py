#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
import datetime

class TransparentMotionAnalyzer:
    def __init__(self, results_base_dir=None):
        # Camera calibration parameters
        self.K1 = np.array([[723.3451333199889, 0, 239.25581247528572],
                           [0, 718.1038999521772, 242.28373163545928],
                           [0, 0, 1]], dtype=np.float64)
        
        self.K2 = np.array([[393.14017015407893, 0, 239.65905562946952],
                           [0, 390.5183594744296, 183.66732162650945],
                           [0, 0, 1]], dtype=np.float64)
        
        self.T_reference = np.array([-0.01358692, 0.00978159, 0.00966281], dtype=np.float64)
        self.R_reference = np.array([[0.99997791, 0.00637036, 0.0018983],
                                    [-0.00638274, 0.99995793, 0.0065878],
                                    [-0.00185626, -0.00659977, 0.9999765]], dtype=np.float64)
        
        # Projection matrices
        self.P1 = self.K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
        self.P2 = self.K2 @ np.hstack([self.R_reference, self.T_reference.reshape(-1, 1)])
        
        # SIFT detector
        self.sift = cv2.SIFT_create(nfeatures=20000, contrastThreshold=0.02, edgeThreshold=8)
        
        # Storage for results
        self.convergence_history = {}
        self.matches_data = {}
        self.correspondences_data = {}
        self.current_pair_name = ""
        self.current_directory_path = ""
        
        # Results directories
        self.results_base_dir = results_base_dir or r"C:\Users\orkle\Desktop\Results"
        self.current_results_dir = None
        self.analysis_log = []
        
        # Addition: Store separate scale for each camera
        self.scale_wide = 1.0
        self.scale_ultra = 1.0

    def setup_results_directory(self, dataset_path):
        """Create organized directory structure for results"""
        dataset_name = os.path.basename(dataset_path)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_results_dir = os.path.join(self.results_base_dir, f"{dataset_name}_{timestamp}")
        
        os.makedirs(self.current_results_dir, exist_ok=True)
        os.makedirs(os.path.join(self.current_results_dir, "visualizations"), exist_ok=True)
        os.makedirs(os.path.join(self.current_results_dir, "sift_features"), exist_ok=True)
        
        self.analysis_log = []

    def save_sift_visualization(self, img, keypoints, filename_prefix, frame_info=""):
        """Save SIFT feature visualization"""
        if not self.current_results_dir or not keypoints:
            return
        
        img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, 
                                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        img_rgb = cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB)
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(img_rgb)
        ax.set_title(f'SIFT Features - {filename_prefix} {frame_info}\n{len(keypoints)} features detected')
        ax.axis('off')
        
        save_path = os.path.join(self.current_results_dir, "sift_features", f"{filename_prefix}_sift.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def load_image(self, path, camera, max_width=800):
        """Load and optionally resize image"""
        img = cv2.imread(path)
        h, w = img.shape[:2]
        
        scale = 1.0
        if w > max_width:
            scale = max_width / w
            img = cv2.resize(img, (max_width, int(h * scale)))
        
        # Store scale by camera type
        if camera == 'wide':
            self.scale_wide = scale
        else:  # 'ultra'
            self.scale_ultra = scale
        
        return img

    def build_scaled_projections(self):
        """Build projection matrices with separate scaling for each camera"""
        self.K1_scaled = self.K1.copy()
        self.K2_scaled = self.K2.copy()
        self.K1_scaled[:2] *= self.scale_wide
        self.K2_scaled[:2] *= self.scale_ultra
        
        self.P1_scaled = self.K1_scaled @ np.hstack([np.eye(3), np.zeros((3, 1))])
        self.P2_scaled = self.K2_scaled @ np.hstack([self.R_reference, self.T_reference.reshape(-1, 1)])

    def epipolar_distance(self, F, pt1, pt2):
        """Calculate epipolar distance between corresponding points"""
        if F is None:
            return 0.0
        
        x1 = np.array([pt1[0], pt1[1], 1.0], dtype=np.float64)
        x2 = np.array([pt2[0], pt2[1], 1.0], dtype=np.float64)
        
        epiline = F @ x1
        distance = abs(epiline[0]*x2[0] + epiline[1]*x2[1] + epiline[2]) / np.sqrt(epiline[0]**2 + epiline[1]**2 + 1e-12)
        return distance

    def find_matches(self, img1, img2, matches_key=None, stage_description=""):
        """Find SIFT matches between two images"""
        print(f"   Finding SIFT features for {stage_description}...")
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)
        
        # Save SIFT visualizations
        if matches_key and self.current_results_dir:
            self.save_sift_visualization(img1, kp1, f"{matches_key}_img1", "Image 1")
            self.save_sift_visualization(img2, kp2, f"{matches_key}_img2", "Image 2")
        
        if des1 is None or des2 is None:
            return np.array([]), np.array([]), None
        
        print(f"   Found {len(des1)} and {len(des2)} features")
        
        # Brute force matching
        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(des1, des2, k=2)
        
        # Lowe's ratio test
        good = []
        distances = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good.append((kp1[m.queryIdx].pt, kp2[m.trainIdx].pt))
                    distances.append(m.distance)
        
        print(f"   Passed ratio test: {len(good)} matches")
        
        if len(good) < 12:
            return np.array([]), np.array([]), None
        
        pts1 = np.array([pt1 for pt1, pt2 in good])
        pts2 = np.array([pt2 for pt1, pt2 in good])
        
        # RANSAC filtering
        try:
            F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 
                                           ransacReprojThreshold=0.8, confidence=0.999, maxIters=10000)
            if mask is not None:
                inlier_pts1 = pts1[mask.ravel() == 1]
                inlier_pts2 = pts2[mask.ravel() == 1]
                inlier_distances = [distances[i] for i in range(len(mask)) if mask[i] == 1]
                
                print(f"   RANSAC inliers: {len(inlier_pts1)} matches")
                
                # Store for visualization
                storage_key = matches_key if matches_key else self.current_pair_name
                self.matches_data[storage_key] = {
                    'pts1': inlier_pts1,
                    'pts2': inlier_pts2,
                    'distances': inlier_distances,
                    'img1': img1.copy(),
                    'img2': img2.copy()
                }
                
                return inlier_pts1, inlier_pts2, F
        except Exception as e:
            print(f"   RANSAC error: {e}")
        
        return pts1, pts2, None

    def create_correspondences(self, wide1, ultra1, wide2, ultra2):
        """Create 4-point correspondences"""
        print("   Creating multi-camera correspondences...")
        
        # Temporal matching: Wide F1 -> Wide F2
        pts_w1_w2_1, pts_w1_w2_2, F_temporal = self.find_matches(wide1, wide2, f"{self.current_pair_name}_temporal", "Temporal")
        
        # Stereo matching F1: Wide <-> Ultra
        pts_w1_u1_1, pts_w1_u1_2, F_stereo_f1 = self.find_matches(wide1, ultra1, f"{self.current_pair_name}_stereo_f1", "Stereo F1")
        
        # Stereo matching F2: Wide <-> Ultra
        pts_w2_u2_1, pts_w2_u2_2, F_stereo_f2 = self.find_matches(wide2, ultra2, f"{self.current_pair_name}_stereo_f2", "Stereo F2")
        
        print(f"   Matches: Temporal={len(pts_w1_w2_1)}, Stereo F1={len(pts_w1_u1_1)}, Stereo F2={len(pts_w2_u2_1)}")
        
        if len(pts_w1_w2_1) < 10 or len(pts_w1_u1_1) < 10 or len(pts_w2_u2_1) < 10:
            raise ValueError("Insufficient matches")
        
        # Link to full correspondences with uniqueness
        correspondences = []
        threshold_pixels = 3.0
        epipolar_threshold = 2.0
        
        used_u1_indices = set()
        used_u2_indices = set()
        
        for w1, w2 in zip(pts_w1_w2_1, pts_w1_w2_2):
            # Find closest Ultra F1 match
            best_u1, best_dist1, best_idx1 = None, float('inf'), -1
            for idx, (w1_s, u1) in enumerate(zip(pts_w1_u1_1, pts_w1_u1_2)):
                if idx in used_u1_indices:
                    continue
                dist = np.linalg.norm(np.array(w1) - np.array(w1_s))
                if dist < threshold_pixels and dist < best_dist1:
                    # Check epipolar constraint
                    epi_dist = self.epipolar_distance(F_stereo_f1, w1, u1)
                    if epi_dist < epipolar_threshold:
                        best_dist1, best_u1, best_idx1 = dist, u1, idx
            
            if best_u1 is None:
                continue
            
            # Find closest Ultra F2 match
            best_u2, best_dist2, best_idx2 = None, float('inf'), -1
            for idx, (w2_s, u2) in enumerate(zip(pts_w2_u2_1, pts_w2_u2_2)):
                if idx in used_u2_indices:
                    continue
                dist = np.linalg.norm(np.array(w2) - np.array(w2_s))
                if dist < threshold_pixels and dist < best_dist2:
                    # Check epipolar constraint
                    epi_dist = self.epipolar_distance(F_stereo_f2, w2, u2)
                    if epi_dist < epipolar_threshold:
                        best_dist2, best_u2, best_idx2 = dist, u2, idx
            
            if best_u2 is None:
                continue
            
            correspondences.append((w1, best_u1, w2, best_u2))
            used_u1_indices.add(best_idx1)
            used_u2_indices.add(best_idx2)
        
        if len(correspondences) < 8:
            raise ValueError(f"Insufficient correspondences: {len(correspondences)}")
        
        print(f"   Created {len(correspondences)} full correspondences")
        return correspondences

    def triangulate_point_manual(self, pt_wide, pt_ultra):
        """Manual triangulation using DLT"""
        try:
            u1, v1 = pt_wide[0], pt_wide[1]
            u2, v2 = pt_ultra[0], pt_ultra[1]
            
            # Build linear system
            p11, p12, p13, p14 = self.P1_scaled[0, :]
            p21, p22, p23, p24 = self.P1_scaled[1, :]
            p31, p32, p33, p34 = self.P1_scaled[2, :]
            
            q11, q12, q13, q14 = self.P2_scaled[0, :]
            q21, q22, q23, q24 = self.P2_scaled[1, :]
            q31, q32, q33, q34 = self.P2_scaled[2, :]
            
            A = np.array([
                [p11 - u1*p31, p12 - u1*p32, p13 - u1*p33],
                [p21 - v1*p31, p22 - v1*p32, p23 - v1*p33],
                [q11 - u2*q31, q12 - u2*q32, q13 - u2*q33],
                [q21 - v2*q31, q22 - v2*q32, q23 - v2*q33]
            ], dtype=np.float64)
            
            b = np.array([
                u1*p34 - p14, v1*p34 - p24, u2*q34 - q14, v2*q34 - q24
            ], dtype=np.float64)
            
            # Solve least squares
            ATA = A.T @ A
            if abs(np.linalg.det(ATA)) < 1e-10:
                return None
            
            pt_3d = np.linalg.solve(ATA, A.T @ b)
            
            # Check physical plausibility
            if 0.1 < pt_3d[2] < 10 and abs(pt_3d[0]) < 5 and abs(pt_3d[1]) < 5:
                return pt_3d
            
            return None
        except:
            return None

    def project_f1_to_ultra_f2(self, pt_3d_f1_wide, R_motion, t_motion):
        """Project point from Wide F1 to Ultra F2"""
        try:
            # Transform to Wide F2
            pt_3d_f2_wide = R_motion @ pt_3d_f1_wide + t_motion
            
            # Transform to Ultra F2
            pt_3d_f2_ultra = self.R_reference @ pt_3d_f2_wide + self.T_reference
            
            if pt_3d_f2_ultra[2] > 0.01:
                # Project to pixels
                projected = self.K2_scaled @ pt_3d_f2_ultra
                if abs(projected[2]) > 1e-10:
                    return projected[:2] / projected[2]
            
            return None
        except:
            return None

    def euler_to_rotation_matrix(self, alpha, beta, gamma):
        """Convert Euler angles to rotation matrix"""
        ca, sa = np.cos(np.radians(alpha)), np.sin(np.radians(alpha))
        cb, sb = np.cos(np.radians(beta)), np.sin(np.radians(beta))
        cc, sc = np.cos(np.radians(gamma)), np.sin(np.radians(gamma))
        
        Rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]], dtype=np.float64)
        Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]], dtype=np.float64)
        Rz = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]], dtype=np.float64)
        
        return Rz @ Ry @ Rx

    def compute_reprojection_error(self, params, correspondences):
        """Compute reprojection error"""
        alpha, beta, gamma, tx, ty, tz = params
        
        # Physical bounds check
        if (abs(alpha) > 45 or abs(beta) > 45 or abs(gamma) > 45 or
            abs(tx) > 0.3 or abs(ty) > 0.3 or abs(tz) > 0.3):
            return 1e6
        
        R_motion = self.euler_to_rotation_matrix(alpha, beta, gamma)
        t_motion = np.array([tx, ty, tz], dtype=np.float64)
        
        errors = []
        valid_count = 0
        
        for wide_f1, ultra_f1, wide_f2, ultra_f2 in correspondences:
            # Triangulate in F1
            pt_3d_f1_wide = self.triangulate_point_manual(wide_f1, ultra_f1)
            if pt_3d_f1_wide is None:
                continue
            
            valid_count += 1
            
            # Predict in Ultra F2
            ultra_f2_predicted = self.project_f1_to_ultra_f2(pt_3d_f1_wide, R_motion, t_motion)
            if ultra_f2_predicted is None:
                errors.append(50.0)
                continue
            
            # Compare with observed
            observed_ultra_f2 = np.array(ultra_f2)
            error = np.linalg.norm(ultra_f2_predicted - observed_ultra_f2)
            errors.append(error)
        
        if valid_count < 4:
            return 1e6
        
        return np.sqrt(np.mean(np.array(errors)**2))

    def compute_gradient(self, params, correspondences):
        """Numerical gradient computation"""
        gradient = np.zeros(6, dtype=np.float64)
        current_error = self.compute_reprojection_error(params, correspondences)
        epsilons = [0.008, 0.008, 0.008, 0.0003, 0.0003, 0.0003]
        
        for i in range(6):
            params_plus = params.copy()
            params_plus[i] += epsilons[i]
            error_plus = self.compute_reprojection_error(params_plus, correspondences)
            gradient[i] = (error_plus - current_error) / epsilons[i]
        
        return gradient

    def get_comprehensive_starting_points(self):
        """Complete set of starting points for all motion types"""
        guesses = []
        
        # Zero point
        guesses.append([0, 0, 0, 0, 0, 0])
        
        # Basic rotations
        rotation_angles = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 18, 20, 25,
                          -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -15, -18, -20, -25]
        
        for angle in rotation_angles:
            guesses.append([angle, 0, 0, 0, 0, 0])  # X rotation
            guesses.append([0, angle, 0, 0, 0, 0])  # Y rotation
            guesses.append([0, 0, angle, 0, 0, 0])  # Z rotation
        
        # Basic translations
        translation_values = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.12, 0.15,
                             -0.005, -0.01, -0.015, -0.02, -0.025, -0.03, -0.04, -0.05, -0.06, -0.08, -0.1, -0.12, -0.15]
        
        for trans in translation_values:
            guesses.append([0, 0, 0, trans, 0, 0])  # X translation
            guesses.append([0, 0, 0, 0, trans, 0])  # Y translation
            guesses.append([0, 0, 0, 0, 0, trans])  # Z translation
        
        # Extended rotations for challenging cases
        extended_x_rotations = [30, 35, -30, -35]
        for angle in extended_x_rotations:
            guesses.append([angle, 0, 0, 0, 0, 0])
        
        extended_z_rotations = [30, 35, -30, -35]
        for angle in extended_z_rotations:
            guesses.append([0, 0, angle, 0, 0, 0])
        
        # Extended Y translations for vertical motions
        extended_y_translations = [0.18, 0.20, 0.25, 0.30, -0.18, -0.20, -0.25, -0.30]
        for trans in extended_y_translations:
            guesses.append([0, 0, 0, 0, trans, 0])
        
        # Smart combinations for natural motion patterns
        smart_combinations = [
            # Y rotations with corrections
            [0, 5, 0, 0, 0, 0], [0, 10, 0, 0, 0, 0], [0, 15, 0, 0, 0, 0], [0, 20, 0, 0, 0, 0],
            [0, -5, 0, 0, 0, 0], [0, -10, 0, 0, 0, 0], [0, -15, 0, 0, 0, 0], [0, -20, 0, 0, 0, 0],
            
            # Vertical motions with rotation corrections
            [-1, 0, 0, 0, 0.10, 0], [-2, 0, 0, 0, 0.15, 0], [-3, 0, 0, 0, 0.20, 0],
            [1, 0, 0, 0, -0.10, 0], [2, 0, 0, 0, -0.15, 0], [3, 0, 0, 0, -0.20, 0],
            
            # Large rotations with translation corrections
            [15, 0, 0, 0.01, 0, 0], [20, 0, 0, 0.015, 0, 0], [25, 0, 0, 0.02, 0, 0],
            [0, 0, 12, -0.02, 0, 0], [0, 0, 15, -0.03, 0, 0], [0, 0, 18, -0.04, 0, 0]
        ]
        
        guesses.extend(smart_combinations)
        
        print(f"   Created {len(guesses)} starting points")
        return guesses

    def find_global_minimum(self, correspondences):
        """Multi-start global optimization"""
        print("   Searching for global minimum...")
        
        initial_guesses = self.get_comprehensive_starting_points()
        best_params = None
        best_error = float('inf')
        best_attempt = -1
        convergence_data = []
        
        for attempt, initial_params in enumerate(initial_guesses):
            params = np.array(initial_params, dtype=np.float64)
            learning_rate = 0.002
            
            if (attempt + 1) % 30 == 0:
                print(f"      Attempt {attempt+1}/{len(initial_guesses)}")
            
            attempt_convergence = []
            
            # Gradient descent
            for iteration in range(200):
                current_error = self.compute_reprojection_error(params, correspondences)
                attempt_convergence.append(current_error)
                
                if current_error > 1e5:
                    break
                
                gradient = self.compute_gradient(params, correspondences)
                gradient_norm = np.linalg.norm(gradient)
                
                if gradient_norm < 1e-15:
                    break
                
                # Gradient clipping
                max_gradient_norm = 4.0
                if gradient_norm > max_gradient_norm:
                    gradient = gradient * (max_gradient_norm / gradient_norm)
                
                new_params = params - learning_rate * gradient
                new_error = self.compute_reprojection_error(new_params, correspondences)
                
                # Adaptive learning rate
                if new_error < current_error:
                    params = new_params
                    learning_rate = min(learning_rate * 1.003, 0.008)
                else:
                    learning_rate *= 0.92
                    if learning_rate < 1e-12:
                        break
                
                # Convergence check
                if len(attempt_convergence) > 50:
                    recent_errors = attempt_convergence[-25:]
                    if max(recent_errors) - min(recent_errors) < 0.0003:
                        break
            
            final_error = self.compute_reprojection_error(params, correspondences)
            
            if final_error < best_error:
                best_error = final_error
                best_params = params.copy()
                best_attempt = attempt + 1
                convergence_data = attempt_convergence.copy()
        
        if best_params is None or best_error > 100:
            return None
        
        # Store convergence data
        self.convergence_history[self.current_pair_name] = {
            'convergence_data': convergence_data,
            'attempt_number': best_attempt,
            'total_attempts': len(initial_guesses)
        }
        
        alpha, beta, gamma, tx, ty, tz = best_params
        print(f"   Global minimum: {best_error:.3f}px (starting point #{best_attempt})")
        print(f"   Rotation: α={alpha:+6.2f}°, β={beta:+6.2f}°, γ={gamma:+6.2f}°")
        print(f"   Translation: X={tx*100:+6.1f}cm, Y={ty*100:+6.1f}cm, Z={tz*100:+6.1f}cm")
        
        return {
            'rotation_deg': [alpha, beta, gamma],
            'translation_cm': [tx*100, ty*100, tz*100],
            'pixel_error': best_error,
            'best_attempt': best_attempt,
            'success': True
        }

    def analyze_frame_pair(self, wide1, ultra1, wide2, ultra2, pair_name):
        """Analyze single frame pair"""
        print(f"\nAnalyzing: {pair_name}")
        print("=" * 50)
        
        self.current_pair_name = pair_name
        
        try:
            correspondences = self.create_correspondences(wide1, ultra1, wide2, ultra2)
            
            # Store for visualization
            self.correspondences_data[pair_name] = {
                'correspondences': correspondences,
                'wide_f1': wide1.copy(),
                'ultra_f1': ultra1.copy(),
                'wide_f2': wide2.copy(),
                'ultra_f2': ultra2.copy()
            }
            
            result = self.find_global_minimum(correspondences)
            
            if result is None:
                return {'frame_pair': pair_name, 'success': False}
            
            result['frame_pair'] = pair_name
            return result
            
        except Exception as e:
            print(f"   Error: {e}")
            return {'frame_pair': pair_name, 'success': False}

    def visualize_cross_frame_matches(self, pair_name):
        """Show Wide F1 -> Ultra F2 matches"""
        if pair_name not in self.correspondences_data:
            return
        
        data = self.correspondences_data[pair_name]
        correspondences = data['correspondences']
        wide_f1_img = data['wide_f1']
        ultra_f2_img = data['ultra_f2']
        
        # Create combined image
        img1_color = cv2.cvtColor(wide_f1_img, cv2.COLOR_BGR2RGB)
        img2_color = cv2.cvtColor(ultra_f2_img, cv2.COLOR_BGR2RGB)
        
        h1, w1 = img1_color.shape[:2]
        h2, w2 = img2_color.shape[:2]
        
        if h1 != h2:
            if h1 > h2:
                img2_color = cv2.resize(img2_color, (w2, h1))
            else:
                img1_color = cv2.resize(img1_color, (w1, h2))
                h1 = h2
        
        combined_img = np.hstack([img1_color, img2_color])
        
        # Extract points
        pts_wide_f1 = [corr[0] for corr in correspondences]
        pts_ultra_f2 = [corr[3] for corr in correspondences]
        
        # Visualize
        num_matches = min(15, len(pts_wide_f1))
        indices = np.random.choice(len(pts_wide_f1), num_matches, replace=False)
        colors = plt.cm.tab20(np.linspace(0, 1, num_matches))
        
        fig, ax = plt.subplots(1, 1, figsize=(18, 10))
        ax.imshow(combined_img)
        ax.set_title(f'{num_matches} Wide F1 → Ultra F2 Matches - {pair_name}')
        
        w1_updated = img1_color.shape[1]
        
        for i, idx in enumerate(indices):
            pt1 = pts_wide_f1[idx]
            pt2 = pts_ultra_f2[idx]
            color = colors[i]
            
            pt2_adj = (pt2[0] + w1_updated, pt2[1])
            
            # Draw line and points
            ax.plot([pt1[0], pt2_adj[0]], [pt1[1], pt2_adj[1]], '-', color=color, linewidth=2.5)
            
            circle1 = plt.Circle((pt1[0], pt1[1]), radius=8, facecolor=color, alpha=0.9, edgecolor='white', linewidth=2)
            circle2 = plt.Circle(pt2_adj, radius=8, facecolor=color, alpha=0.9, edgecolor='white', linewidth=2)
            ax.add_patch(circle1)
            ax.add_patch(circle2)
            
            # Add numbers
            ax.text(pt1[0]+10, pt1[1]-10, str(i+1), fontsize=11, fontweight='bold', 
                   color='white', bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.9))
            ax.text(pt2_adj[0]+10, pt2_adj[1]-10, str(i+1), fontsize=11, fontweight='bold',
                   color='white', bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.9))
        
        # Separator line
        ax.axvline(x=w1_updated, color='red', linestyle='--', linewidth=3, alpha=0.8)
        
        # Labels
        frame_nums = pair_name.split("→")
        ax.text(w1_updated//2, 30, f'Wide Camera - Frame {frame_nums[0]}', ha='center', fontsize=16, fontweight='bold', 
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        ax.text(w1_updated + img2_color.shape[1]//2, 30, f'Ultra Camera - Frame {frame_nums[1]}', ha='center', fontsize=16, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.8))
        
        ax.set_xlim(0, combined_img.shape[1])
        ax.set_ylim(combined_img.shape[0], 0)
        ax.axis('off')
        
        # Save figure
        if self.current_results_dir:
            save_path = os.path.join(self.current_results_dir, "visualizations", f"{pair_name}_cross_matches.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.tight_layout()
        plt.show()

    def visualize_convergence(self, pair_name):
        """Show convergence plot"""
        if pair_name not in self.convergence_history:
            return
        
        conv_info = self.convergence_history[pair_name]
        conv_data = conv_info['convergence_data']
        attempt_num = conv_info['attempt_number']
        total_attempts = conv_info['total_attempts']
        
        if len(conv_data) < 2:
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 9))
        fig.suptitle(f'Global Search - Best Point #{attempt_num}/{total_attempts} - {pair_name}', fontsize=16, fontweight='bold')
        
        iterations = range(len(conv_data))
        
        # Plot convergence
        ax.plot(iterations, conv_data, 'b-', linewidth=3, label='Reprojection Error', alpha=0.9)
        ax.fill_between(iterations, conv_data, alpha=0.3, color='lightblue')
        
        final_error = conv_data[-1]
        initial_error = conv_data[0]
        
        # Mark important points
        ax.plot(0, initial_error, 'o', markersize=12, color='orange', 
                label=f'Initial: {initial_error:.3f}px', zorder=5)
        ax.plot(len(conv_data)-1, final_error, 'ro', markersize=15, 
                label=f'Final: {final_error:.3f}px', zorder=5)
        
        # Reference lines
        ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Excellent: 1.0px')
        ax.axhline(y=2.0, color='orange', linestyle=':', alpha=0.7, label='Good: 2.0px')
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Reprojection Error (pixels)')
        ax.grid(True, alpha=0.4)
        ax.legend()
        
        # Save figure
        if self.current_results_dir:
            save_path = os.path.join(self.current_results_dir, "visualizations", f"{pair_name}_convergence.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.tight_layout()
        plt.show()

    def save_analysis_report(self, results):
        """Save detailed analysis report"""
        if not self.current_results_dir:
            return
        
        report_path = os.path.join(self.current_results_dir, "analysis_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("MOTION ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Dataset: {os.path.basename(self.current_directory_path)}\n")
            f.write(f"Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Method: Multi-start gradient descent with SIFT features\n\n")
            
            # Camera parameters
            f.write("CAMERA PARAMETERS:\n")
            f.write(f"Wide Camera: fx={self.K1[0,0]:.2f}, fy={self.K1[1,1]:.2f}\n")
            f.write(f"Ultra Camera: fx={self.K2[0,0]:.2f}, fy={self.K2[1,1]:.2f}\n")
            f.write(f"Baseline: {np.linalg.norm(self.T_reference)*100:.2f}cm\n\n")
            
            # SIFT parameters used
            f.write("SIFT CONFIGURATION:\n")
            f.write(f"Max features: 20,000\n")
            f.write(f"Contrast threshold: 0.02\n")
            f.write(f"Edge threshold: 8\n")
            f.write(f"RANSAC threshold: 0.8 pixels\n")
            f.write(f"Epipolar threshold: 2.0 pixels\n\n")
            
            # Results for each frame pair
            f.write("FRAME PAIR RESULTS:\n")
            f.write("-" * 30 + "\n")
            
            successful = [r for r in results if r.get('success', False)]
            
            for result in results:
                frame_pair = result.get('frame_pair', 'Unknown')
                f.write(f"\n{frame_pair}:\n")
                
                # Feature matching statistics
                temporal_key = f"{frame_pair}_temporal"
                stereo_f1_key = f"{frame_pair}_stereo_f1"  
                stereo_f2_key = f"{frame_pair}_stereo_f2"
                
                f.write("  FEATURE MATCHING:\n")
                if temporal_key in self.matches_data:
                    temporal_matches = len(self.matches_data[temporal_key]['pts1'])
                    f.write(f"    Temporal (Wide F1→F2): {temporal_matches} matches\n")
            
                if stereo_f1_key in self.matches_data:
                    stereo_f1_matches = len(self.matches_data[stereo_f1_key]['pts1'])
                    f.write(f"    Stereo F1 (Wide↔Ultra): {stereo_f1_matches} matches\n")
                
                if stereo_f2_key in self.matches_data:
                    stereo_f2_matches = len(self.matches_data[stereo_f2_key]['pts1'])
                    f.write(f"    Stereo F2 (Wide↔Ultra): {stereo_f2_matches} matches\n")
                
                if frame_pair in self.correspondences_data:
                    correspondences_count = len(self.correspondences_data[frame_pair]['correspondences'])
                    f.write(f"    Final correspondences: {correspondences_count}\n")
                
                    # Calculate survival rate for this pair
                    if temporal_key in self.matches_data:
                        initial_matches = temporal_matches
                        survival_rate = (correspondences_count / initial_matches) * 100 if initial_matches > 0 else 0
                        f.write(f"    Survival rate: {survival_rate:.1f}% ({correspondences_count}/{initial_matches})\n")
 
                # Motion results
                f.write("  MOTION ESTIMATION:\n")
                if result.get('success', False):
                    rot = result['rotation_deg']
                    trans = result['translation_cm']
                    error = result['pixel_error']
                    attempt = result.get('best_attempt', 'N/A')
                    
                    f.write(f"  Status: SUCCESS\n")
                    f.write(f"  Reprojection Error: {error:.4f} pixels\n")
                    f.write(f"  Best starting point: #{attempt}/120\n")
                    f.write(f"  Rotation (deg): α={rot[0]:+7.3f}, β={rot[1]:+7.3f}, γ={rot[2]:+7.3f}\n")
                    f.write(f"  Translation (cm): X={trans[0]:+7.2f}, Y={trans[1]:+7.2f}, Z={trans[2]:+7.2f}\n")
                    
                    quality = "EXCELLENT" if error < 1.0 else "GOOD" if error < 2.0 else "FAIR" if error < 3.0 else "POOR"
                    f.write(f"  Quality: {quality}\n")
                    
                    # Convergence info if available
                    if frame_pair in self.convergence_history:
                        conv_info = self.convergence_history[frame_pair]
                        iterations = len(conv_info['convergence_data'])
                        f.write(f"    Convergence: {iterations} iterations\n")    
                else:
                    f.write(f"  Status: FAILED\n")
                    f.write(f"    Reason: Insufficient matches or convergence failure\n")
        
            # Overall statistics
            f.write(f"\nOVERALL STATISTICS:\n")
            f.write(f"Success Rate: {len(successful)}/{len(results)} ({100*len(successful)/len(results):.1f}%)\n")
    
            if successful:
                errors = [r['pixel_error'] for r in successful]
                f.write(f"Average Error: {np.mean(errors):.4f} ± {np.std(errors):.4f} pixels\n")
                
                # Starting point analysis
                attempts = [r.get('best_attempt', 0) for r in successful if r.get('best_attempt')]
                if attempts:
                    f.write(f"Starting points used: {min(attempts)} to {max(attempts)} (avg: {np.mean(attempts):.1f})\n")
      
                
                motion = self.describe_motion_enhanced(results)
                f.write(f"Detected Motion: {motion}\n")
                
                # Correspondence statistics across all pairs
                total_correspondences = []
                for pair_name in [r['frame_pair'] for r in successful]:
                    if pair_name in self.correspondences_data:
                        count = len(self.correspondences_data[pair_name]['correspondences'])
                        total_correspondences.append(count)
            
                if total_correspondences:
                     f.write(f"Correspondences per pair: {min(total_correspondences)} to {max(total_correspondences)} (avg: {np.mean(total_correspondences):.1f})\n")
        
            f.write(f"\nFiles saved in: {self.current_results_dir}\n")
            
            # Add processing summary
            f.write(f"\nPROCESSING PIPELINE SUMMARY:\n")
            f.write(f"1. SIFT feature detection (max 20,000 per image)\n")
            f.write(f"2. Brute force matching with Lowe's ratio test (0.7)\n") 
            f.write(f"3. RANSAC geometric filtering (0.8px threshold)\n")
            f.write(f"4. Multi-camera correspondence linking (3.0px + 2.0px epipolar)\n")
            f.write(f"5. Global optimization (120 starting points)\n")
            f.write(f"6. Gradient descent with adaptive learning rate\n")

    def describe_motion_enhanced(self, results):
        """Identify dominant motion type"""
        successful = [r for r in results if r['success']]
        if not successful:
            return 'failed'
        
        # Calculate statistics
        rotations = {
            'alpha': [r['rotation_deg'][0] for r in successful],
            'beta': [r['rotation_deg'][1] for r in successful], 
            'gamma': [r['rotation_deg'][2] for r in successful]
        }
        
        translations = {
            'x': [r['translation_cm'][0] for r in successful],
            'y': [r['translation_cm'][1] for r in successful],
            'z': [r['translation_cm'][2] for r in successful]
        }
        
        # Find dominant motion
        max_rotation_val = 0
        max_rotation_axis = None
        dominant_rotation = None
        
        for axis, values in rotations.items():
            max_abs = max(abs(v) for v in values)
            if max_abs > max_rotation_val and max_abs > 2.0:
                max_rotation_val = max_abs
                max_rotation_axis = axis
                dominant_rotation = np.median(values)
        
        max_translation_val = 0
        max_translation_axis = None
        dominant_translation = None
        
        for axis, values in translations.items():
            threshold = 1.0 if axis == 'y' else 1.5
            max_abs = max(abs(v) for v in values)
            if max_abs > max_translation_val and max_abs > threshold:
                max_translation_val = max_abs
                max_translation_axis = axis
                dominant_translation = np.median(values)
        
        # Determine motion type
        if max_rotation_val > 2.0 and max_rotation_val > max_translation_val * 0.7:
            if max_rotation_axis == 'beta':
                return f"y_rotation ({dominant_rotation:+.1f}°)"
            elif max_rotation_axis == 'alpha':
                return f"x_rotation ({dominant_rotation:+.1f}°)"
            else:
                return f"z_rotation ({dominant_rotation:+.1f}°)"
        
        elif max_translation_val > 1.0:
            if max_translation_axis == 'z':
                direction = "forward" if dominant_translation < 0 else "backward"
                return f"{direction}_motion ({dominant_translation:+.1f}cm)"
            elif max_translation_axis == 'x':
                direction = "left" if dominant_translation > 0 else "right"
                return f"{direction}_motion ({dominant_translation:+.1f}cm)"
            else:
                direction = "up" if dominant_translation > 0 else "down"
                return f"{direction}_motion ({dominant_translation:+.1f}cm)"
        
        return "minimal_motion"

    def analyze_sequence(self, directory_path):
        """Analyze complete image sequence"""
        print(f"\nAnalyzing sequence: {os.path.basename(directory_path)}")
        print("=" * 60)
        
        # Setup results directory
        self.setup_results_directory(directory_path)
        self.current_directory_path = directory_path
        
        # Find image files
        directory = Path(directory_path)
        image_files = {'wide': {}, 'ultra': {}}
        
        for file_path in directory.glob("*"):
            if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                filename = file_path.name.lower()
                
                camera_type = 'wide' if 'wide' in filename else 'ultra' if 'ultra' in filename else None
                if not camera_type:
                    continue
                
                for i in range(1, 5):
                    if f'f{i}' in filename:
                        image_files[camera_type][f'F{i}'] = str(file_path)
                        break
        
        # Validate files
        required_frames = ['F1', 'F2', 'F3', 'F4']
        for camera_type in ['wide', 'ultra']:
            missing = [f for f in required_frames if f not in image_files[camera_type]]
            if missing:
                raise ValueError(f"Missing {camera_type} files: {missing}")
        
        print("All files found")
        
        # Load images with separate scaling
        wide_frames = []
        ultra_frames = []
        
        for frame in required_frames:
            wide_img = self.load_image(image_files['wide'][frame], camera='wide')
            ultra_img = self.load_image(image_files['ultra'][frame], camera='ultra')
            self.build_scaled_projections()  # Build matrices after loading both images
            wide_frames.append(wide_img)
            ultra_frames.append(ultra_img)
        
        # Analyze frame pairs
        results = []
        frame_pairs = [("F1→F2", 0, 1), ("F2→F3", 1, 2), ("F3→F4", 2, 3)]
        
        for pair_name, idx1, idx2 in frame_pairs:
            result = self.analyze_frame_pair(
                wide_frames[idx1], ultra_frames[idx1],
                wide_frames[idx2], ultra_frames[idx2],
                pair_name
            )
            
            self.last_analysis_result = result
            results.append(result)
            
            # Create visualizations if successful
            if result['success']:
                print(f"\nCreating visualizations for {pair_name}...")
                try:
                    self.visualize_cross_frame_matches(pair_name)
                    self.visualize_convergence(pair_name)
                except Exception as e:
                    print(f"Visualization error: {e}")
        
        # Save analysis report
        self.save_analysis_report(results)
        
        # Print summary
        successful = [r for r in results if r['success']]
        print(f"\nResults: {len(successful)}/{len(results)} successful")
        
        if successful:
            motion = self.describe_motion_enhanced(results)
            print(f"Motion detected: {motion}")
            print(f"Results saved to: {self.current_results_dir}")
            return motion
        
        return 'failed'


def main():
    """Main function to process all datasets"""
    print("Motion Analyzer - Processing All Datasets")
    print("=" * 50)
    
    test_paths = [
        r"C:\Users\orkle\Desktop\Left_0-5-10-15_cm",
        r"C:\Users\orkle\Desktop\Forward_0-5-10-15_cm", 
        r"C:\Users\orkle\Desktop\Counterclockwise_0-10-20-30_degrees",
        r"C:\Users\orkle\Desktop\Right_0-5-10-15_cm",
        r"C:\Users\orkle\Desktop\Back_0-5-10-15_cm",
        r"C:\Users\orkle\Desktop\clockwise_0-10-20-30_degrees",
        r"C:\Users\orkle\Desktop\Pitch_Clockwise_0-10-20-30_degrees",
        r"C:\Users\orkle\Desktop\Upward_0-5-10-15_cm",
        r"C:\Users\orkle\Desktop\Downward_0-5-10-15_cm",
        r"C:\Users\orkle\Desktop\Roll_Clockwise_0-10-20-30_degrees"
    ]
    
    expected_motions = [
        'left_motion', 'forward_motion', 'y_rotation',
        'right_motion', 'backward_motion', 'y_rotation', 
        'x_rotation', 'up_motion', 'down_motion', 'z_rotation'
    ]
    
    detected_motions = []
    master_results_dir = r"C:\Users\orkle\Desktop\Results"
    os.makedirs(master_results_dir, exist_ok=True)
    
    # Process each dataset
    for i, path in enumerate(test_paths, 1):
        print(f"\nProcessing dataset {i}: {os.path.basename(path)}")
        print("=" * 60)
        
        try:
            analyzer = TransparentMotionAnalyzer(master_results_dir)
            detected_motion = analyzer.analyze_sequence(path)
            detected_motions.append(detected_motion)
        except Exception as e:
            print(f"Error: {e}")
            detected_motions.append('error')
    
    # Create master summary
    master_log = os.path.join(master_results_dir, f"master_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    with open(master_log, 'w') as f:
        f.write("MASTER ANALYSIS SUMMARY\n")
        f.write("=" * 30 + "\n\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total datasets: {len(test_paths)}\n\n")
        
        successful = 0
        for i, (detected, expected) in enumerate(zip(detected_motions, expected_motions), 1):
            f.write(f"Dataset {i:02d}: {os.path.basename(test_paths[i-1])}\n")
            f.write(f"  Expected: {expected}\n")
            f.write(f"  Detected: {detected}\n")
            f.write(f"  Match: {'YES' if expected in detected else 'NO'}\n\n")
            
            if detected not in ['failed', 'error'] and expected in detected:
                successful += 1
        
        f.write(f"Success rate: {successful}/{len(test_paths)} ({100*successful/len(test_paths):.1f}%)\n")
    
    # Print final summary
    print(f"\nFinal Summary:")
    print("=" * 30)
    
    successful = 0
    for i, (detected, expected) in enumerate(zip(detected_motions, expected_motions), 1):
        if detected not in ['failed', 'error']:
            if expected in detected:
                print(f"Dataset {i}: SUCCESS - {detected}")
                successful += 1
            else:
                print(f"Dataset {i}: MISMATCH - detected {detected}, expected {expected}")
        else:
            print(f"Dataset {i}: FAILED - {detected}")
    
    success_rate = (successful / len(test_paths)) * 100
    print(f"\nOverall success: {successful}/{len(test_paths)} ({success_rate:.0f}%)")
    print(f"Master summary: {master_log}")
    print(f"All results: {master_results_dir}")
    
    return detected_motions, successful

if __name__ == "__main__":
    results, success_count = main()