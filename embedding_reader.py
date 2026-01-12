"""
ReID Embedding Reader - File-Based Extraction

Reads 256-dimensional ReID embeddings from DeepSORT tracker output files.
This is a stepping stone to custom Python bindings for direct metadata extraction.

File Format (expected from DeepSORT outputReidTensor):
- int32: num_tracks
- For each track:
  - int32: tracker_id
  - float32[256]: embedding

Note: If DeepSORT's outputReidTensor doesn't produce this exact format,
this reader will need to be adjusted based on actual output format.
"""

import struct
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import warnings


class EmbeddingReader:
    """
    Read ReID embeddings from DeepSORT file output.

    Maintains a rolling cache of recent frames to reduce disk I/O.
    Automatically normalizes embeddings for cosine similarity matching.
    """

    def __init__(self, output_path: str = "/tmp/deepstream_reid/", feature_dim: int = 256):
        """
        Initialize embedding reader.

        Args:
            output_path: Directory where DeepSORT writes embedding files
            feature_dim: Embedding dimension (default: 256 for ResNet50)
        """
        self.output_path = Path(output_path)
        self.feature_dim = feature_dim
        self.embedding_cache = {}  # frame → {tracker_id → embedding}
        self.cache_size = 100  # Keep last 100 frames in memory

        # Create output directory if it doesn't exist
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Track warnings to avoid spam
        self._warned_missing = False
        self._warned_format = False

    def read_frame_embeddings(self, frame_num: int) -> Dict[int, np.ndarray]:
        """
        Read all embeddings for a frame from binary file.

        Args:
            frame_num: Frame number to read

        Returns:
            Dictionary mapping tracker_id → normalized embedding (256-dim numpy array)
            Returns empty dict if file doesn't exist or cannot be read.
        """
        # Check cache first
        if frame_num in self.embedding_cache:
            return self.embedding_cache[frame_num]

        # Construct file path (format: frame_000001.bin)
        file_path = self.output_path / f"frame_{frame_num:06d}.bin"

        if not file_path.exists():
            # Only warn once about missing files
            if not self._warned_missing:
                warnings.warn(
                    f"ReID embedding file not found: {file_path}\n"
                    f"Ensure DeepSORT config has:\n"
                    f"  outputReidTensor: 1\n"
                    f"  outputReidPath: \"{self.output_path}\""
                )
                self._warned_missing = True
            return {}

        embeddings = {}
        try:
            with open(file_path, 'rb') as f:
                # Read number of tracks
                num_tracks_bytes = f.read(4)
                if len(num_tracks_bytes) < 4:
                    raise ValueError("File too short to contain track count")

                num_tracks = struct.unpack('i', num_tracks_bytes)[0]

                if num_tracks < 0 or num_tracks > 1000:  # Sanity check
                    raise ValueError(f"Invalid track count: {num_tracks}")

                # Read each track's embedding
                for track_idx in range(num_tracks):
                    # Read tracker ID
                    tracker_id_bytes = f.read(4)
                    if len(tracker_id_bytes) < 4:
                        warnings.warn(f"Incomplete tracker ID at track {track_idx}")
                        break

                    tracker_id = struct.unpack('i', tracker_id_bytes)[0]

                    # Read embedding
                    embedding_bytes = f.read(self.feature_dim * 4)  # 256 float32s
                    if len(embedding_bytes) < self.feature_dim * 4:
                        warnings.warn(f"Incomplete embedding for tracker {tracker_id}")
                        break

                    embedding = np.frombuffer(embedding_bytes, dtype=np.float32)

                    if embedding.shape[0] != self.feature_dim:
                        warnings.warn(f"Wrong embedding dimension: {embedding.shape[0]} (expected {self.feature_dim})")
                        continue

                    # Normalize embedding for cosine similarity
                    norm = np.linalg.norm(embedding)
                    if norm > 1e-8:  # Avoid division by zero
                        embedding = embedding / norm
                    else:
                        warnings.warn(f"Zero-norm embedding for tracker {tracker_id}")
                        continue

                    embeddings[tracker_id] = embedding

        except struct.error as e:
            if not self._warned_format:
                warnings.warn(f"Binary format error reading {file_path}: {e}")
                self._warned_format = True
            return {}
        except Exception as e:
            warnings.warn(f"Error reading embeddings for frame {frame_num}: {e}")
            return {}

        # Cache the result
        self.embedding_cache[frame_num] = embeddings

        # Cleanup old frames (keep last cache_size frames)
        if len(self.embedding_cache) > self.cache_size:
            oldest_frame = min(self.embedding_cache.keys())
            del self.embedding_cache[oldest_frame]

        return embeddings

    def get_embedding(self, frame_num: int, tracker_id: int) -> Optional[np.ndarray]:
        """
        Get specific embedding for a tracker ID at a frame.

        Args:
            frame_num: Frame number
            tracker_id: DeepSORT tracker ID

        Returns:
            Normalized 256-dim embedding or None if not available
        """
        if frame_num not in self.embedding_cache:
            self.read_frame_embeddings(frame_num)

        return self.embedding_cache.get(frame_num, {}).get(tracker_id)

    def clear_cache(self):
        """Clear embedding cache (useful for memory management)"""
        self.embedding_cache.clear()
        self._warned_missing = False
        self._warned_format = False

    def get_cache_info(self) -> Dict:
        """Get cache statistics for debugging"""
        total_embeddings = sum(len(embs) for embs in self.embedding_cache.values())
        return {
            'cached_frames': len(self.embedding_cache),
            'total_embeddings': total_embeddings,
            'cache_size_limit': self.cache_size,
            'output_path': str(self.output_path),
            'feature_dim': self.feature_dim
        }


# Alternative: Mock reader for testing without DeepSORT output
class MockEmbeddingReader(EmbeddingReader):
    """
    Mock embedding reader for testing without actual DeepSORT output.
    Generates random embeddings for testing the pipeline.
    """

    def __init__(self, output_path: str = "/tmp/deepstream_reid/", feature_dim: int = 256):
        super().__init__(output_path, feature_dim)
        self.persistent_embeddings = {}  # tracker_id → embedding

    def read_frame_embeddings(self, frame_num: int) -> Dict[int, np.ndarray]:
        """Generate mock embeddings (for testing only)"""
        # For testing, generate random but consistent embeddings per tracker ID
        # In real usage, this class should NOT be used

        # Return empty dict to simulate "embeddings not available yet"
        # This allows testing the position-only fallback
        return {}

    def create_mock_embedding(self, tracker_id: int) -> np.ndarray:
        """Create a consistent mock embedding for a tracker ID"""
        if tracker_id not in self.persistent_embeddings:
            np.random.seed(tracker_id)  # Deterministic per ID
            embedding = np.random.randn(self.feature_dim).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            self.persistent_embeddings[tracker_id] = embedding

        return self.persistent_embeddings[tracker_id].copy()
