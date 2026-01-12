"""
Multi-Cue Similarity Scoring System

Combines appearance (ReID), position, and motion cues for robust player matching.

Scoring Breakdown (Occlusion-Optimized):
- 40% Appearance (ReID cosine similarity) - düşürüldü, gölgelemede güvenilmez
- 35% Position (exponential spatial proximity) - artırıldı, gölgelemede daha güvenilir
- 25% Motion (velocity direction + speed consistency) - artırıldı, hareket ipucu önemli

Time Penalty: -0.3% per frame occluded (düşürüldü)

Thresholds:
- Hard threshold (0.78): Direct ACTIVE profile match (düşürüldü)
- Soft threshold (0.50): FROZEN profile candidate creation (düşürüldü)
- Confirmation (0.65): Candidate → ACTIVE promotion
"""

import math
import numpy as np
from typing import Tuple, Optional, Dict


class SimilarityScorer:
    """
    Multi-cue scoring for player re-identification.

    Weights are tuned for basketball tracking where appearance is dominant
    but position/motion provide important spatial-temporal constraints.
    """

    # Scoring weights (must sum to 1.0) - Occlusion-optimized
    WEIGHT_APPEARANCE = 0.40  # Düşürüldü - gölgelemede appearance güvenilmez
    WEIGHT_POSITION = 0.35    # Artırıldı - gölgelemede en güvenilir ipucu
    WEIGHT_MOTION = 0.25      # Artırıldı - hareket yönü önemli ipucu

    # Matching thresholds - Daha esnek eşleşme için düşürüldü
    APPEARANCE_HARD_THRESHOLD = 0.78   # ACTIVE profile direct match (0.85'ten düşürüldü)
    APPEARANCE_SOFT_THRESHOLD = 0.50   # FROZEN profile candidate creation (0.55'ten düşürüldü)

    # Spatial parameters
    MAX_POSITION_DISTANCE = 600  # pixels - artırıldı, gölgelemede daha geniş arama
    MAX_VELOCITY_DIFF = 200      # px/frame (motion matching)

    # Time penalty - Düşürüldü, gölgeleme sonrası daha uzun süre eşleşme şansı
    OCCLUSION_PENALTY_PER_FRAME = 0.003  # -0.3% per frame occluded (0.5%'ten düşürüldü)

    def __init__(self):
        """Initialize scorer with default parameters"""
        # Validate weights sum to 1.0
        total_weight = self.WEIGHT_APPEARANCE + self.WEIGHT_POSITION + self.WEIGHT_MOTION
        assert abs(total_weight - 1.0) < 1e-6, f"Weights must sum to 1.0 (got {total_weight})"

    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Cosine similarity between two normalized embeddings.

        Args:
            emb1: First embedding (assumed normalized)
            emb2: Second embedding (assumed normalized)

        Returns:
            Similarity in [0, 1] (dot product for normalized vectors)
        """
        sim = float(np.dot(emb1, emb2))
        return max(0.0, min(1.0, sim))  # Clamp to [0, 1]

    def appearance_score(self, embedding: np.ndarray, profile_embedding_mean: np.ndarray) -> float:
        """
        Appearance similarity score using ReID embeddings.

        Args:
            embedding: Current detection embedding (256-dim, normalized)
            profile_embedding_mean: Profile's mean embedding (256-dim, normalized)

        Returns:
            Score in [0.0, 1.0] where 1.0 is perfect match
        """
        return self.cosine_similarity(embedding, profile_embedding_mean)

    def position_score(self, detection_pos: Tuple[float, float],
                      profile_pos: Tuple[float, float]) -> float:
        """
        Position proximity score using exponential decay.

        Args:
            detection_pos: Detection center (x, y) in pixels
            profile_pos: Profile last known position (x, y) in pixels

        Returns:
            Score in [0.0, 1.0] where 1.0 is same position

        Formula: exp(-distance / MAX_POSITION_DISTANCE)
        - At distance=0: score=1.0
        - At distance=MAX_POSITION_DISTANCE (600px): score≈0.37
        - At distance=1200px: score≈0.14
        """
        distance = math.sqrt((detection_pos[0] - profile_pos[0])**2 +
                           (detection_pos[1] - profile_pos[1])**2)

        # Exponential decay
        score = math.exp(-distance / self.MAX_POSITION_DISTANCE)
        return max(0.0, min(1.0, score))

    def motion_score(self, detection_velocity: Tuple[float, float],
                    profile_velocity: Tuple[float, float]) -> float:
        """
        Motion similarity score combining direction and speed.

        Args:
            detection_velocity: Detection velocity (vx, vy) in px/frame
            profile_velocity: Profile velocity (vx, vy) in px/frame

        Returns:
            Score in [0.0, 1.0] where 1.0 is identical motion

        Components:
        - 50% Direction similarity (cosine of velocity vectors)
        - 50% Speed ratio (min/max speed ratio)

        Special cases:
        - Both stationary (<1 px/frame): 1.0 (perfect match)
        - One stationary, one moving: 0.5 (neutral)
        """
        det_vx, det_vy = detection_velocity
        prof_vx, prof_vy = profile_velocity

        det_speed = math.sqrt(det_vx**2 + det_vy**2)
        prof_speed = math.sqrt(prof_vx**2 + prof_vy**2)

        # Handle stationary cases
        if det_speed < 1.0 and prof_speed < 1.0:
            return 1.0  # Both stationary = perfect match

        if det_speed < 1.0 or prof_speed < 1.0:
            return 0.5  # One moving, one stationary = neutral

        # Direction similarity (cosine of velocity vectors)
        dot_product = det_vx * prof_vx + det_vy * prof_vy
        direction_sim = dot_product / (det_speed * prof_speed)
        direction_sim = max(-1.0, min(1.0, direction_sim))  # Clamp to [-1, 1]
        direction_score = (direction_sim + 1.0) / 2.0  # Map to [0, 1]

        # Speed ratio similarity
        speed_ratio = min(det_speed, prof_speed) / max(det_speed, prof_speed)

        # Combine (equal weight to direction and speed)
        motion_score = 0.5 * direction_score + 0.5 * speed_ratio

        return max(0.0, min(1.0, motion_score))

    def combined_score(self,
                      embedding: np.ndarray,
                      detection_pos: Tuple[float, float],
                      detection_velocity: Tuple[float, float],
                      profile_embedding_mean: np.ndarray,
                      profile_pos: Tuple[float, float],
                      profile_velocity: Tuple[float, float],
                      frames_occluded: int = 0) -> Tuple[float, Dict]:
        """
        Calculate combined multi-cue score.

        Args:
            embedding: Detection ReID embedding (256-dim)
            detection_pos: Detection center (x, y) px
            detection_velocity: Detection velocity (vx, vy) px/frame
            profile_embedding_mean: Profile mean embedding (256-dim)
            profile_pos: Profile position (x, y) px
            profile_velocity: Profile velocity (vx, vy) px/frame
            frames_occluded: Frames since profile last seen (for time penalty)

        Returns:
            Tuple of (total_score, score_breakdown_dict)

        Score breakdown dict contains:
        - 'appearance': Appearance score (0-1)
        - 'position': Position score (0-1)
        - 'motion': Motion score (0-1)
        - 'total': Weighted total before time penalty
        - 'time_penalty': Penalty amount applied
        - 'final': Final score after time penalty
        """
        # Calculate individual scores
        app_score = self.appearance_score(embedding, profile_embedding_mean)
        pos_score = self.position_score(detection_pos, profile_pos)
        mot_score = self.motion_score(detection_velocity, profile_velocity)

        # Weighted combination
        total_score = (self.WEIGHT_APPEARANCE * app_score +
                      self.WEIGHT_POSITION * pos_score +
                      self.WEIGHT_MOTION * mot_score)

        # Apply time penalty for occlusion
        time_penalty = 0.0
        if frames_occluded > 0:
            time_penalty = frames_occluded * self.OCCLUSION_PENALTY_PER_FRAME
            total_score = max(0.0, total_score - time_penalty)

        # Score breakdown for debugging
        breakdown = {
            'appearance': app_score,
            'position': pos_score,
            'motion': mot_score,
            'weighted_total': (self.WEIGHT_APPEARANCE * app_score +
                             self.WEIGHT_POSITION * pos_score +
                             self.WEIGHT_MOTION * mot_score),
            'time_penalty': time_penalty,
            'final': total_score
        }

        return total_score, breakdown

    def is_hard_match(self, combined_score: float) -> bool:
        """Check if score meets hard threshold (direct ACTIVE profile match)"""
        return combined_score >= self.APPEARANCE_HARD_THRESHOLD

    def is_soft_match(self, combined_score: float) -> bool:
        """Check if score meets soft threshold (FROZEN profile candidate)"""
        return combined_score >= self.APPEARANCE_SOFT_THRESHOLD

    def format_breakdown(self, breakdown: Dict) -> str:
        """
        Format score breakdown for console output.

        Returns:
            Human-readable string like:
            "App: 0.912 | Pos: 0.856 | Mot: 0.823 | Total: 0.897"
        """
        return (f"App: {breakdown['appearance']:.3f} | "
                f"Pos: {breakdown['position']:.3f} | "
                f"Mot: {breakdown['motion']:.3f} | "
                f"Total: {breakdown['final']:.3f}")

    def tune_weights(self, appearance_weight: float, position_weight: float, motion_weight: float):
        """
        Adjust scoring weights (for parameter tuning).

        Args:
            appearance_weight: Weight for appearance score [0, 1]
            position_weight: Weight for position score [0, 1]
            motion_weight: Weight for motion score [0, 1]

        Raises:
            ValueError: If weights don't sum to 1.0
        """
        total = appearance_weight + position_weight + motion_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total}")

        self.WEIGHT_APPEARANCE = appearance_weight
        self.WEIGHT_POSITION = position_weight
        self.WEIGHT_MOTION = motion_weight

    def get_config(self) -> Dict:
        """Get current scorer configuration for debugging"""
        return {
            'weights': {
                'appearance': self.WEIGHT_APPEARANCE,
                'position': self.WEIGHT_POSITION,
                'motion': self.WEIGHT_MOTION
            },
            'thresholds': {
                'hard': self.APPEARANCE_HARD_THRESHOLD,
                'soft': self.APPEARANCE_SOFT_THRESHOLD
            },
            'parameters': {
                'max_position_distance': self.MAX_POSITION_DISTANCE,
                'max_velocity_diff': self.MAX_VELOCITY_DIFF,
                'occlusion_penalty_per_frame': self.OCCLUSION_PENALTY_PER_FRAME
            }
        }
