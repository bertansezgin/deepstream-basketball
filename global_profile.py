"""
Global Profile + ReID System - Data Structures

This module defines the core data structures for maintaining persistent player
identities across DeepSORT tracker ID changes using appearance features (ReID embeddings).

Key Principle: "Hemen eşleştirme yapma" (Never rush matching)
- Use evidence accumulation over multiple frames
- Require confirmation before binding IDs
- Maintain state transitions: FROZEN → ACTIVE → RETIRED
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List, Tuple
from collections import deque
import numpy as np


class ProfileState(Enum):
    """Player profile lifecycle states"""
    FROZEN = "frozen"    # Uncertain, gathering evidence
    ACTIVE = "active"    # Confirmed, tracking normally
    RETIRED = "retired"  # Player left the scene


@dataclass
class GlobalProfile:
    """
    Persistent player identity across tracker ID changes.

    Maintains appearance history (ReID embeddings), position/velocity history,
    and confidence metrics for robust long-term player tracking.
    """

    # Core identity
    profile_id: int  # Unique global ID (never changes)
    state: ProfileState

    # Appearance features (ReID embeddings)
    embedding_history: deque  # maxlen=30, stores 256-dim ReID embeddings
    embedding_mean: np.ndarray  # Rolling average embedding (256-dim, normalized)

    # Position/motion history (from existing velocity system)
    position_history: deque  # maxlen=20, stores (x, y, frame) tuples
    velocity: Tuple[float, float]  # Current (vx, vy) in pixels per frame

    # Confidence tracking
    confidence_score: float  # 0.0-1.0, increases with successful matches
    frames_since_seen: int  # Increments when not matched
    total_frames_tracked: int  # Total frames this profile has been tracked

    # Current tracker binding
    current_tracker_id: Optional[int]  # Current DeepSORT tracker ID (can change)
    last_seen_frame: int  # Last frame this profile was observed

    # Metadata
    created_frame: int  # Frame when profile was created
    last_match_score: float  # Last multi-cue match score (0.0-1.0)

    def __post_init__(self):
        """Validate profile state after initialization"""
        assert 0.0 <= self.confidence_score <= 1.0, "Confidence must be in [0, 1]"
        assert self.frames_since_seen >= 0, "Frames since seen must be non-negative"
        assert self.total_frames_tracked >= 0, "Total frames tracked must be non-negative"
        assert self.embedding_mean.shape == (256,), "Embedding must be 256-dim"

        # Ensure embedding is normalized
        norm = np.linalg.norm(self.embedding_mean)
        if norm > 0:
            self.embedding_mean = self.embedding_mean / norm

    def update_frames_unseen(self, current_frame: int):
        """Update frames_since_seen counter"""
        self.frames_since_seen = current_frame - self.last_seen_frame

    def should_retire(self, current_frame: int, max_frames_unseen: int = 150) -> bool:
        """Check if profile should be retired (5 seconds @ 30fps)"""
        return (current_frame - self.last_seen_frame) > max_frames_unseen


@dataclass
class CandidateID:
    """
    Uncertain match requiring evidence accumulation before confirmation.

    Implements the "never rush matching" principle by tracking scores across
    multiple frames and requiring consistent evidence before binding to a profile.
    """

    # Identification
    tracker_id: int  # Candidate tracker ID (from DeepSORT)

    # Evidence accumulation
    profile_candidates: Dict[int, List[Tuple[float, int]]] = field(default_factory=dict)
    # Maps profile_id → [(score, frame), ...] for all potential matches

    # Tracking state
    total_evidence_frames: int = 0  # Total frames since candidate created
    best_profile_id: Optional[int] = None  # Best matching profile so far
    best_profile_score: float = 0.0  # Best average score so far

    # Confirmation requirements - Gölgeleme için hızlandırıldı
    min_evidence_frames: int = 3  # Minimum frames before confirmation (5'ten düşürüldü)
    confirmation_threshold: float = 0.60  # Average score needed for confirmation (0.65'ten düşürüldü)

    def add_evidence(self, profile_id: int, score: float, frame: int):
        """
        Accumulate evidence for a profile match.

        Args:
            profile_id: Global profile ID being matched
            score: Multi-cue match score (0.0-1.0)
            frame: Current frame number
        """
        if profile_id not in self.profile_candidates:
            self.profile_candidates[profile_id] = []

        self.profile_candidates[profile_id].append((score, frame))
        self.total_evidence_frames += 1

        # Update best profile tracking
        avg_score = sum(s for s, _ in self.profile_candidates[profile_id]) / len(self.profile_candidates[profile_id])
        if avg_score > self.best_profile_score:
            self.best_profile_id = profile_id
            self.best_profile_score = avg_score

    def should_confirm(self) -> Optional[int]:
        """
        Check if enough evidence accumulated to confirm a profile.

        Returns:
            profile_id if confirmation criteria met, None otherwise

        Confirmation requires:
        - At least min_evidence_frames frames of evidence
        - Average score >= confirmation_threshold for best profile
        """
        if self.total_evidence_frames < self.min_evidence_frames:
            return None

        # Calculate average score per profile
        profile_avg_scores = {}
        for profile_id, evidence in self.profile_candidates.items():
            if len(evidence) > 0:
                avg_score = sum(score for score, _ in evidence) / len(evidence)
                profile_avg_scores[profile_id] = avg_score

        # Find best profile
        if profile_avg_scores:
            best_profile = max(profile_avg_scores, key=profile_avg_scores.get)
            best_score = profile_avg_scores[best_profile]

            if best_score >= self.confirmation_threshold:
                return best_profile

        return None

    def get_evidence_summary(self) -> Dict:
        """Get summary of accumulated evidence for debugging"""
        summary = {
            'total_frames': self.total_evidence_frames,
            'num_profiles': len(self.profile_candidates),
            'best_profile_id': self.best_profile_id,
            'best_profile_score': self.best_profile_score,
            'profile_scores': {}
        }

        for profile_id, evidence in self.profile_candidates.items():
            if len(evidence) > 0:
                avg_score = sum(score for score, _ in evidence) / len(evidence)
                summary['profile_scores'][profile_id] = {
                    'avg_score': avg_score,
                    'num_frames': len(evidence),
                    'min_score': min(score for score, _ in evidence),
                    'max_score': max(score for score, _ in evidence)
                }

        return summary
