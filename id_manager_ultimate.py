"""
Ultimate ID Manager
Voting (noise filtering) + Optimal Matching (baseline normalization) + Global Profile ReID
"""

import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import Counter, deque, defaultdict
from global_profile import GlobalProfile, CandidateID, ProfileState
from embedding_reader import EmbeddingReader
from similarity_scorer import SimilarityScorer

class UltimateIDManager:
    def __init__(self,
                 learning_frames=90,
                 voting_window=7,
                 position_threshold=250,
                 use_reid=True):
        """
        Args:
            learning_frames: Baseline öğrenme süresi (frame)
            voting_window: Voting pencere boyutu
            position_threshold: Max pozisyon farkı (piksel)
        """
        self.learning_frames = learning_frames
        self.voting_window = voting_window
        self.position_threshold = position_threshold

        # VELOCITY PARAMETERS - Gölgeleme için optimize edildi
        self.velocity_alpha = 0.7  # Exponential smoothing factor
        self.base_threshold = 300  # Base matching threshold (px) - artırıldı, ID kaybını önle
        self.max_threshold = 700  # Max adaptive threshold (px) - occlusion bonus için artırıldı
        self.velocity_scale_factor = 0.6  # Threshold increase per pixel/frame velocity - artırıldı
        self.max_occlusion_frames = 75  # Max frames to extrapolate - artırıldı (2.5 saniye @30fps)
        self.min_velocity_history = 2  # Frames needed before using velocity
        self.greedy_threshold = 150  # Greedy nearest neighbor matching threshold (px)
        
        # VOTING STATE
        self.voting_history = {}  # {position_key: deque([id1, id2, ...])}
        self.voting_positions = {}

        # VELOCITY TRACKING STATE
        self.velocities = {}  # {id: (vx, vy)} - pixels per frame
        self.position_history = {}  # {id: deque([(x,y,frame), ...], maxlen=3)}
        self.last_seen_frame = {}  # {id: frame_number} - for occlusion detection
        self.occluded_ids = {}  # {id: occlusion_start_frame}

        # NORMALIZATION STATE
        self.baseline_ids = set()
        self.baseline_positions = {}  # Current/predicted positions
        self.last_real_positions = {}  # ← YENİ: Son gerçek görülen pozisyonlar (predict için)
        self.virtual_to_real = {}

        # ID REMAPPING (to start from 1)
        self.tracker_to_sequential = {}  # Maps tracker IDs to 1, 2, 3...
        self.next_sequential_id = 1
        
        # CONTROL
        self.frame_count = 0
        self.learning_complete = False
        self.baseline_locked = False  # ← YENİ: Baseline kilidi (learning sonrası True)
        
        # STATS
        self.stats = {
            'voting_corrections': 0,
            'optimal_mappings': 0,
            'greedy_matches': 0,  # ← YENİ: Greedy matching sayısı
            'multi_mapping_events': 0,
            'baseline_players': 0,
            'velocity_tracked_ids': 0,
            'occlusion_events': 0,
            'avg_player_speed': 0.0
        }

        # GLOBAL PROFILE STATE (ReID System)
        self.use_reid = use_reid
        if self.use_reid:
            self.profiles = {}  # profile_id → GlobalProfile
            self.candidates = {}  # tracker_id → CandidateID
            self.next_profile_id = 1
            self.embedding_reader = EmbeddingReader()
            self.scorer = SimilarityScorer()

            # ReID stats
            self.stats['reid_matches'] = 0
            self.stats['candidate_confirmations'] = 0
            self.stats['active_profiles'] = 0
            self.stats['frozen_profiles'] = 0
    
    def update(self, detections):
        """
        3-Stage Pipeline:
        1. Voting (ID stability)
        2. Learning/Tracking phase
        3. Normalization
        """
        self.frame_count += 1
        
        # STAGE 1: VOTING - Kısa süreli noise temizle
        voted_detections = self._apply_voting(detections)
        
        # STAGE 2 & 3: Learning veya Tracking
        if self.frame_count <= self.learning_frames:
            return self._learning_phase(voted_detections)
        
        if not self.learning_complete:
            self._finalize_learning()
        
        return self._tracking_phase(voted_detections)
    
    def _apply_voting(self, detections):
        """
        Stage 1: Voting ile ID stabilize et
        """
        voted = []
        current_keys = set()
        
        for det in detections:
            center = (det['x'] + det['w']/2, det['y'] + det['h']/2)
            key = self._find_voting_key(center)
            
            if key is None:
                key = center
                self.voting_history[key] = deque(maxlen=self.voting_window)
                self.voting_positions[key] = center
            else:
                self.voting_positions[key] = center
            
            current_keys.add(key)
            
            # Voting
            raw_id = det['id']
            self.voting_history[key].append(raw_id)
            
            if len(self.voting_history[key]) >= 3:
                vote_counts = Counter(self.voting_history[key])
                voted_id = vote_counts.most_common(1)[0][0]
                
                if voted_id != raw_id:
                    self.stats['voting_corrections'] += 1
            else:
                voted_id = raw_id
            
            voted.append({
                **det,
                'id': voted_id,  # ID'yi voted ile değiştir
                'raw_id': raw_id
            })
        
        # Cleanup
        self._cleanup_voting_keys(current_keys)
        
        return voted
    
    def _find_voting_key(self, pos):
        """Voting için en yakın key bul"""
        min_dist = float('inf')
        nearest = None
        
        for key in self.voting_positions:
            dist = math.sqrt((pos[0]-key[0])**2 + (pos[1]-key[1])**2)
            if dist < min_dist and dist < 150:  # Voting threshold
                min_dist = dist
                nearest = key
        
        return nearest
    
    def _cleanup_voting_keys(self, current_keys):
        """Eski voting key'leri temizle"""
        to_remove = [k for k in self.voting_history if k not in current_keys]
        for k in to_remove:
            del self.voting_history[k]
            if k in self.voting_positions:
                del self.voting_positions[k]
    
    def _learning_phase(self, detections):
        """Learning: Baseline oluştur + Profile initialization"""
        normalized = []

        # Extract embeddings for this frame (if ReID enabled)
        embeddings = {}
        if self.use_reid:
            embeddings = self.embedding_reader.read_frame_embeddings(self.frame_count)

        for det in detections:
            tracker_id = det['id']

            # Map tracker ID to sequential ID (1, 2, 3...)
            if tracker_id not in self.tracker_to_sequential:
                self.tracker_to_sequential[tracker_id] = self.next_sequential_id
                print(f"🔢 ID MAPPING: Tracker {tracker_id} → Sequential {self.next_sequential_id}")
                self.next_sequential_id += 1

            det_id = self.tracker_to_sequential[tracker_id]
            pos = (det['x'] + det['w']/2, det['y'] + det['h']/2)

            if det_id not in self.baseline_ids:
                self.baseline_ids.add(det_id)
                self.stats['baseline_players'] = len(self.baseline_ids)

            self.baseline_positions[det_id] = pos
            self.last_real_positions[det_id] = pos  # ← YENİ: Gerçek pozisyonu kaydet

            # Add velocity tracking during learning
            self._update_velocity(det_id, pos, self.frame_count)

            # NEW: Create GlobalProfile during learning (if ReID enabled and embedding available)
            if self.use_reid and det_id not in [p.current_tracker_id for p in self.profiles.values()]:
                embedding = embeddings.get(tracker_id)  # Use original tracker_id for embedding lookup
                if embedding is not None:
                    # Create ACTIVE profile during learning phase
                    profile = GlobalProfile(
                        profile_id=det_id,  # Use sequential ID as profile ID
                        state=ProfileState.ACTIVE,
                        embedding_history=deque(maxlen=30),
                        embedding_mean=embedding.copy(),
                        position_history=deque(maxlen=20),
                        velocity=self.velocities.get(det_id, (0.0, 0.0)),
                        confidence_score=1.0,
                        frames_since_seen=0,
                        total_frames_tracked=1,
                        current_tracker_id=det_id,  # Sequential ID
                        last_seen_frame=self.frame_count,
                        created_frame=self.frame_count,
                        last_match_score=1.0
                    )
                    profile.embedding_history.append(embedding)
                    profile.position_history.append((pos[0], pos[1], self.frame_count))

                    self.profiles[det_id] = profile  # Use sequential ID as key

                    print(f"🆔 PROFILE CREATED: ID {det_id} (Tracker {tracker_id} → Sequential {det_id})")

            normalized.append({
                **det,
                'normalized_id': det_id,
                'voted_id': det_id,
                'profile_id': self._get_profile_id_for_tracker(det_id) if self.use_reid else None,
                'phase': 'learning'
            })

        return normalized
    
    def _finalize_learning(self):
        """Learning tamamlandı - Baseline artık KİLİTLİ"""
        self.learning_complete = True
        self.baseline_locked = True  # ← YENİ: Baseline kilitlendi
        print("\n" + "="*70)
        print("🎓 ÖĞRENME TAMAMLANDI!")
        print("="*70)
        print(f"📊 Baseline Oyuncular: {len(self.baseline_ids)}")
        print(f"🆔 Baseline ID'ler: {sorted(self.baseline_ids)}")
        print("🔒 Baseline KİLİTLENDİ - Yeni ID baseline'a eklenemez!")
        print("="*70)
        print("🔍 Voting + Optimal Matching aktif...")
        print("="*70 + "\n")

    def _tracking_phase(self, detections):
        """Tracking: ReID-aware matching with 5-stage pipeline"""
        normalized = []

        # ← DEĞIŞIKLIK: Yeni tracker ID'lere sequential ID verme!
        # Sadece baseline'daki ID'ler kullanılacak
        # Yeni ID'ler optimal matching ile baseline'a map edilecek

        # Sadece zaten bilinen (baseline) tracker ID'leri için sequential ID kullan
        current_ids = set()
        for det in detections:
            original_tracker_id = det['id']
            if original_tracker_id in self.tracker_to_sequential:
                current_ids.add(self.tracker_to_sequential[original_tracker_id])

        # Extract embeddings (if ReID enabled)
        embeddings = {}
        if self.use_reid:
            frame_embeddings = self.embedding_reader.read_frame_embeddings(self.frame_count)
            # Map embeddings from tracker IDs to sequential IDs
            for original_tracker_id, embedding in frame_embeddings.items():
                if original_tracker_id in self.tracker_to_sequential:
                    sequential_id = self.tracker_to_sequential[original_tracker_id]
                    embeddings[sequential_id] = embedding

        # Handle occlusions (existing velocity system)
        occluded_ids = self._handle_occlusions(detections)

        # Track matched IDs to avoid double-matching
        matched_tracker_ids = set()
        matched_profile_ids = set()

        # STAGE 1: Update existing profile bindings (sadece baseline ID'ler için)
        if self.use_reid:
            for det in detections:
                original_tracker_id = det['id']
                # Sadece baseline'a zaten map edilmiş ID'ler için
                if original_tracker_id not in self.tracker_to_sequential:
                    continue  # Yeni ID - ReID stage atla
                tracker_id = self.tracker_to_sequential[original_tracker_id]
                embedding = embeddings.get(tracker_id)

                if embedding is None:
                    continue  # No embedding available

                pos = (det['x'] + det['w']/2, det['y'] + det['h']/2)
                velocity = self.velocities.get(tracker_id, (0.0, 0.0))

                # Check if tracker already bound to an active profile
                existing_profile = self._get_profile_for_tracker(tracker_id)
                if existing_profile and existing_profile.state == ProfileState.ACTIVE:
                    # Update existing profile
                    self._update_profile(existing_profile, embedding, pos, velocity, tracker_id)
                    matched_tracker_ids.add(tracker_id)
                    matched_profile_ids.add(existing_profile.profile_id)
                    continue

        # STAGE 2: Match unmatched detections to ACTIVE profiles (hard threshold)
        if self.use_reid:
            # Sadece baseline'da olan ID'ler için ReID matching
            unmatched_detections = [det for det in detections
                                   if det['id'] in self.tracker_to_sequential and
                                   self.tracker_to_sequential[det['id']] not in matched_tracker_ids]
            active_profiles = [p for p in self.profiles.values()
                              if p.state == ProfileState.ACTIVE and p.profile_id not in matched_profile_ids]

            for det in unmatched_detections:
                original_tracker_id = det['id']
                tracker_id = self.tracker_to_sequential[original_tracker_id]
                embedding = embeddings.get(tracker_id)
                if embedding is None:
                    continue

                pos = (det['x'] + det['w']/2, det['y'] + det['h']/2)
                velocity = self.velocities.get(tracker_id, (0.0, 0.0))

                best_profile = None
                best_score = 0.0
                best_breakdown = None

                for profile in active_profiles:
                    if profile.profile_id in matched_profile_ids:
                        continue  # Already matched

                    score, breakdown = self.scorer.combined_score(
                        embedding, pos, velocity,
                        profile.embedding_mean,
                        profile.position_history[-1][:2] if profile.position_history else pos,
                        profile.velocity,
                        frames_occluded=self.frame_count - profile.last_seen_frame
                    )

                    if score >= self.scorer.APPEARANCE_HARD_THRESHOLD and score > best_score:
                        best_score = score
                        best_profile = profile
                        best_breakdown = breakdown

                if best_profile:
                    # HARD MATCH - Update profile immediately
                    self._update_profile(best_profile, embedding, pos, velocity, tracker_id)
                    matched_tracker_ids.add(tracker_id)
                    matched_profile_ids.add(best_profile.profile_id)
                    self.stats['reid_matches'] += 1
                    print(f"🎯 REID MATCH: ID {tracker_id} → Profile {best_profile.profile_id} (score: {best_score:.3f})")
                    print(f"   ├─ {self.scorer.format_breakdown(best_breakdown)}")

        # STAGE 3: Create/update candidates for remaining unmatched (soft threshold)
        if self.use_reid:
            # Sadece baseline'da olan ID'ler için candidate matching
            unmatched_detections = [det for det in detections
                                   if det['id'] in self.tracker_to_sequential and
                                   self.tracker_to_sequential[det['id']] not in matched_tracker_ids]
            frozen_profiles = [p for p in self.profiles.values()
                              if p.state == ProfileState.FROZEN and p.profile_id not in matched_profile_ids]

            for det in unmatched_detections:
                original_tracker_id = det['id']
                tracker_id = self.tracker_to_sequential[original_tracker_id]
                embedding = embeddings.get(tracker_id)
                if embedding is None:
                    continue

                pos = (det['x'] + det['w']/2, det['y'] + det['h']/2)
                velocity = self.velocities.get(tracker_id, (0.0, 0.0))

                # Check if candidate already exists
                if tracker_id not in self.candidates:
                    self.candidates[tracker_id] = CandidateID(
                        tracker_id=tracker_id,
                        profile_candidates={},
                        total_evidence_frames=0,
                        best_profile_id=None,
                        best_profile_score=0.0
                    )

                candidate = self.candidates[tracker_id]

                # Accumulate evidence for all frozen profiles above soft threshold
                for profile in frozen_profiles:
                    score, breakdown = self.scorer.combined_score(
                        embedding, pos, velocity,
                        profile.embedding_mean,
                        profile.position_history[-1][:2] if profile.position_history else pos,
                        profile.velocity,
                        frames_occluded=self.frame_count - profile.last_seen_frame
                    )

                    if score >= self.scorer.APPEARANCE_SOFT_THRESHOLD:
                        candidate.add_evidence(profile.profile_id, score, self.frame_count)

                # Check if candidate ready to confirm
                confirmed_profile_id = candidate.should_confirm()
                if confirmed_profile_id:
                    confirmed_profile = self.profiles[confirmed_profile_id]
                    self._update_profile(confirmed_profile, embedding, pos, velocity, tracker_id)
                    confirmed_profile.state = ProfileState.ACTIVE
                    matched_tracker_ids.add(tracker_id)
                    matched_profile_ids.add(confirmed_profile_id)
                    del self.candidates[tracker_id]
                    self.stats['candidate_confirmations'] += 1
                    print(f"✅ CANDIDATE CONFIRMED: ID {tracker_id} → Profile {confirmed_profile_id} (evidence: {candidate.total_evidence_frames} frames)")

        # STAGE 4: REMOVED - Yeni profile oluşturma kaldırıldı
        # ← DEĞIŞIKLIK: Baseline kilitliyken yeni profile oluşturulmaz
        # Baseline dışındaki ID'ler sadece geçici ID olarak gösterilir
        # Bu sayede sadece ilk 90 frame'deki oyuncular/hakemler/seyirciler tracked olur

        # STAGE 5: Position-based matching for ALL unmatched detections
        missing_baseline_ids = self.baseline_ids - current_ids - occluded_ids
        if self.use_reid:
            missing_baseline_ids = missing_baseline_ids - matched_profile_ids

        # ← DEĞIŞIKLIK: Tüm yeni/bilinmeyen tracker ID'leri matching'e dahil et
        unmatched_detections = []
        for det in detections:
            original_tracker_id = det['id']

            # Eğer bu tracker ID zaten baseline'da ise (bilinen ID)
            if original_tracker_id in self.tracker_to_sequential:
                sequential_id = self.tracker_to_sequential[original_tracker_id]
                if (sequential_id not in matched_tracker_ids and
                    sequential_id not in self.baseline_ids and
                    sequential_id not in self.virtual_to_real):
                    det_copy = det.copy()
                    det_copy['sequential_id'] = sequential_id
                    det_copy['original_tracker_id'] = original_tracker_id
                    unmatched_detections.append(det_copy)
            else:
                # Yeni tracker ID - henüz sequential ID'si yok
                det_copy = det.copy()
                det_copy['sequential_id'] = None  # Henüz yok
                det_copy['original_tracker_id'] = original_tracker_id
                unmatched_detections.append(det_copy)

        # ═══════════════════════════════════════════════════════════════════
        # STAGE 5a: GREEDY NEAREST NEIGHBOR PRE-MATCHING (YENİ!)
        # Her detection için yakın çevredeki baseline ID'yi bul
        # Bu, obvious matches'ı hızlıca çözer
        # ═══════════════════════════════════════════════════════════════════
        greedy_mappings = {}
        greedy_matched_baselines = set()

        for det in unmatched_detections[:]:  # Copy to allow modification
            det_pos = (det['x'] + det['w']/2, det['y'] + det['h']/2)
            original_tracker_id = det['original_tracker_id']

            best_baseline = None
            best_distance = float('inf')

            for baseline_id in missing_baseline_ids:
                if baseline_id in greedy_matched_baselines:
                    continue  # Zaten eşleşti

                # Son gerçek pozisyonu kullan
                if baseline_id in self.last_real_positions:
                    baseline_pos = self.last_real_positions[baseline_id]
                elif baseline_id in self.baseline_positions:
                    baseline_pos = self.baseline_positions[baseline_id]
                else:
                    continue

                distance = math.sqrt((det_pos[0] - baseline_pos[0])**2 +
                                    (det_pos[1] - baseline_pos[1])**2)

                if distance < self.greedy_threshold and distance < best_distance:
                    best_distance = distance
                    best_baseline = baseline_id

            if best_baseline is not None:
                greedy_mappings[original_tracker_id] = best_baseline
                greedy_matched_baselines.add(best_baseline)
                print(f"⚡ GREEDY MATCH: Tracker {original_tracker_id} → Baseline {best_baseline} ({best_distance:.1f}px)")

        # Apply greedy mappings
        for original_tracker_id, baseline_id in greedy_mappings.items():
            self.tracker_to_sequential[original_tracker_id] = baseline_id
            matched_tracker_ids.add(baseline_id)
            self.stats['greedy_matches'] += 1  # ← Greedy stats

        # Remove greedy-matched detections and baselines from further processing
        unmatched_detections = [d for d in unmatched_detections
                                if d['original_tracker_id'] not in greedy_mappings]
        missing_baseline_ids = missing_baseline_ids - greedy_matched_baselines

        # ═══════════════════════════════════════════════════════════════════
        # STAGE 5b: OPTIMAL MATCHING (Hungarian Algorithm)
        # Kalan zor vakalar için
        # ═══════════════════════════════════════════════════════════════════
        if len(unmatched_detections) > 0 and len(missing_baseline_ids) > 0:
            mappings = self._optimal_matching_v2(unmatched_detections, missing_baseline_ids)
            for original_tracker_id, baseline_id in mappings.items():
                # Bu tracker ID'yi baseline ID ile ilişkilendir
                self.tracker_to_sequential[original_tracker_id] = baseline_id
                matched_tracker_ids.add(baseline_id)
                self.stats['optimal_mappings'] += 1
        elif len(unmatched_detections) > 0:
            # DEBUG: Neden map edilemedi?
            print(f"⚠️  MAP EDİLEMEDİ: {len(unmatched_detections)} detection, missing_baseline: {len(missing_baseline_ids)}")
            print(f"   ├─ current_ids: {len(current_ids)}/{len(self.baseline_ids)} → {sorted(current_ids)[:5]}...")
            print(f"   ├─ occluded: {len(occluded_ids)} → {sorted(occluded_ids)[:5]}...")
            print(f"   ├─ matched_tracker_ids: {len(matched_tracker_ids)}")
            # Tüm baseline görünür mü?
            visible_baseline = current_ids | occluded_ids | matched_tracker_ids
            print(f"   └─ visible+occluded: {len(visible_baseline)}/{len(self.baseline_ids)}")

        # Build normalized output
        for det in detections:
            original_tracker_id = det['id']
            pos = (det['x'] + det['w']/2, det['y'] + det['h']/2)

            # Determine normalized ID
            profile_id = None

            # Eğer bu tracker ID baseline'a map edilmişse
            if original_tracker_id in self.tracker_to_sequential:
                baseline_id = self.tracker_to_sequential[original_tracker_id]

                if baseline_id in self.baseline_ids:
                    normalized_id = baseline_id
                    mapping_type = 'baseline'

                    # Position ve velocity güncelle
                    self.baseline_positions[normalized_id] = pos
                    self.last_real_positions[normalized_id] = pos  # ← YENİ: Gerçek pozisyonu kaydet
                    self._update_velocity(normalized_id, pos, self.frame_count)
                else:
                    # Bu olmamalı ama fallback
                    normalized_id = -1
                    mapping_type = 'unknown'
            else:
                # ← YENİ: Bu tracker ID baseline'a map edilemedi → -1 ver
                normalized_id = -1
                mapping_type = 'unmatched'

            normalized.append({
                **det,
                'normalized_id': normalized_id,
                'voted_id': normalized_id,  # baseline'a map edilen ID
                'profile_id': profile_id,
                'mapping_type': mapping_type,
                'phase': 'tracking'
            })

        # Cleanup: Retire old profiles
        if self.use_reid:
            self._retire_old_profiles()

        return normalized
    
    def _optimal_matching(self, virtual_detections, missing_baseline_ids):
        """Hungarian Algorithm ile optimal matching"""
        if not virtual_detections or not missing_baseline_ids:
            return {}
        
        n_virtual = len(virtual_detections)
        n_missing = len(missing_baseline_ids)
        missing_list = list(missing_baseline_ids)
        
        cost_matrix = np.full((n_virtual, n_missing), 999999.0)
        
        for i, virt_det in enumerate(virtual_detections):
            virt_pos = (virt_det['x'] + virt_det['w']/2, 
                       virt_det['y'] + virt_det['h']/2)
            
            for j, baseline_id in enumerate(missing_list):
                if baseline_id in self.baseline_positions:
                    # Use predicted position instead of last position
                    predicted_pos = self._predict_position(baseline_id, frames_ahead=1)
                    if predicted_pos is None:
                        predicted_pos = self.baseline_positions[baseline_id]

                    # Calculate distance to predicted position
                    distance = math.sqrt((virt_pos[0] - predicted_pos[0])**2 +
                                       (virt_pos[1] - predicted_pos[1])**2)

                    # Use adaptive threshold based on velocity
                    adaptive_threshold = self._calculate_adaptive_threshold(baseline_id)

                    if distance < adaptive_threshold:
                        cost_matrix[i, j] = distance
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        mappings = {}
        for i, j in zip(row_ind, col_ind):
            baseline_id = missing_list[j]
            adaptive_threshold = self._calculate_adaptive_threshold(baseline_id)

            if cost_matrix[i, j] < adaptive_threshold:
                virtual_id = virtual_detections[i]['id']  # Sequential ID
                real_id = missing_list[j]
                mappings[virtual_id] = real_id

                # Get original tracker ID for display
                original_id = virtual_detections[i].get('original_id', virtual_id)

                print(f"🔄 MAPPING: ID {virtual_id} (Tracker {original_id}) → Baseline {real_id}")

                # Add velocity debugging info
                if real_id in self.velocities:
                    vx, vy = self.velocities[real_id]
                    speed = math.sqrt(vx**2 + vy**2)
                    print(f"   ├─ Distance: {cost_matrix[i, j]:.1f}px | Speed: {speed:.1f}px/frame | Threshold: {adaptive_threshold:.1f}px")
                else:
                    print(f"   ├─ Distance: {cost_matrix[i, j]:.1f}px (no velocity)")
        
        if len(mappings) > 1:
            self.stats['multi_mapping_events'] += 1
            print(f"⚡ MULTI-MAPPING: {len(mappings)} ID!")

        return mappings

    def _optimal_matching_v2(self, unmatched_detections, missing_baseline_ids):
        """
        Optimal matching V2 - original_tracker_id → baseline_id döndürür
        Hungarian Algorithm ile yeni tracker ID'leri baseline'a map eder
        """
        if not unmatched_detections or not missing_baseline_ids:
            return {}

        n_detections = len(unmatched_detections)
        n_missing = len(missing_baseline_ids)
        missing_list = list(missing_baseline_ids)

        cost_matrix = np.full((n_detections, n_missing), 999999.0)

        # DEBUG: Tüm mesafeleri logla
        debug_distances = []

        for i, det in enumerate(unmatched_detections):
            det_pos = (det['x'] + det['w']/2, det['y'] + det['h']/2)
            original_tracker_id = det['original_tracker_id']

            for j, baseline_id in enumerate(missing_list):
                if baseline_id in self.baseline_positions:
                    # Calculate frames unseen for this baseline ID
                    frames_unseen = 0
                    if baseline_id in self.last_seen_frame:
                        frames_unseen = self.frame_count - self.last_seen_frame[baseline_id]

                    # Use predicted position (now limited)
                    predicted_pos = self._predict_position(baseline_id, frames_ahead=frames_unseen)
                    if predicted_pos is None:
                        predicted_pos = self.baseline_positions[baseline_id]

                    distance = math.sqrt((det_pos[0] - predicted_pos[0])**2 +
                                       (det_pos[1] - predicted_pos[1])**2)

                    # Adaptive threshold includes occlusion bonus now
                    adaptive_threshold = self._calculate_adaptive_threshold(baseline_id, frames_unseen)

                    # DEBUG: Her çift için mesafe kaydet
                    debug_distances.append({
                        'tracker': original_tracker_id,
                        'baseline': baseline_id,
                        'distance': distance,
                        'threshold': adaptive_threshold,
                        'det_pos': det_pos,
                        'pred_pos': predicted_pos,
                        'accepted': distance < adaptive_threshold
                    })

                    if distance < adaptive_threshold:
                        cost_matrix[i, j] = distance

        # DEBUG: Eşleşme durumunu logla
        if len(debug_distances) > 0:
            accepted = [d for d in debug_distances if d['accepted']]
            rejected = [d for d in debug_distances if not d['accepted']]

            if len(rejected) > 0 and len(accepted) == 0:
                print(f"\n🔍 DEBUG MATCHING (Frame {self.frame_count}):")
                print(f"   ├─ {n_detections} detection vs {n_missing} missing baseline")
                # En yakın 3 eşleşmeyi göster
                sorted_by_dist = sorted(debug_distances, key=lambda x: x['distance'])[:3]
                for d in sorted_by_dist:
                    status = "✅" if d['accepted'] else "❌"
                    print(f"   ├─ {status} Tracker {d['tracker']} ↔ Baseline {d['baseline']}: {d['distance']:.1f}px (thr: {d['threshold']:.1f}px)")
                    print(f"   │     det_pos: ({d['det_pos'][0]:.0f}, {d['det_pos'][1]:.0f}) | pred_pos: ({d['pred_pos'][0]:.0f}, {d['pred_pos'][1]:.0f})")

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # original_tracker_id → baseline_id
        mappings = {}
        for i, j in zip(row_ind, col_ind):
            baseline_id = missing_list[j]
            # Calculate frames unseen for threshold
            frames_unseen = 0
            if baseline_id in self.last_seen_frame:
                frames_unseen = self.frame_count - self.last_seen_frame[baseline_id]
            adaptive_threshold = self._calculate_adaptive_threshold(baseline_id, frames_unseen)

            if cost_matrix[i, j] < adaptive_threshold:
                original_tracker_id = unmatched_detections[i]['original_tracker_id']
                mappings[original_tracker_id] = baseline_id

                print(f"🔄 MAPPING: Tracker {original_tracker_id} → Baseline {baseline_id}")

                if baseline_id in self.velocities:
                    vx, vy = self.velocities[baseline_id]
                    speed = math.sqrt(vx**2 + vy**2)
                    print(f"   ├─ Distance: {cost_matrix[i, j]:.1f}px | Speed: {speed:.1f}px/frame | Threshold: {adaptive_threshold:.1f}px")
                else:
                    print(f"   ├─ Distance: {cost_matrix[i, j]:.1f}px (no velocity)")

        if len(mappings) > 1:
            self.stats['multi_mapping_events'] += 1
            print(f"⚡ MULTI-MAPPING: {len(mappings)} ID!")

        # DEBUG: Hiç mapping olmazsa uyar
        if len(mappings) == 0 and n_detections > 0 and n_missing > 0:
            print(f"⚠️  MAPPING BAŞARISIZ: {n_detections} detection, {n_missing} missing baseline - threshold aşıldı!")

        return mappings

    def _update_velocity(self, player_id, position, current_frame):
        """
        Update velocity for a player using exponential moving average.

        Args:
            player_id: Tracker ID
            position: (x, y) tuple - center position
            current_frame: Current frame number

        Returns:
            (vx, vy) tuple or None if insufficient history
        """
        # Initialize position history if new ID
        if player_id not in self.position_history:
            self.position_history[player_id] = deque(maxlen=3)
            self.velocities[player_id] = (0.0, 0.0)

        # Add current position
        self.position_history[player_id].append((position[0], position[1], current_frame))
        self.last_seen_frame[player_id] = current_frame

        # Need at least 2 positions to calculate velocity
        if len(self.position_history[player_id]) < 2:
            return None

        # Calculate instantaneous velocity from last 2 positions
        history = list(self.position_history[player_id])
        pos_prev = history[-2]
        pos_curr = history[-1]

        dt = pos_curr[2] - pos_prev[2]  # Frame delta
        if dt == 0:
            return self.velocities[player_id]

        vx_instant = (pos_curr[0] - pos_prev[0]) / dt
        vy_instant = (pos_curr[1] - pos_prev[1]) / dt

        # Apply exponential moving average for smoothing
        old_vx, old_vy = self.velocities[player_id]
        vx_smooth = self.velocity_alpha * vx_instant + (1 - self.velocity_alpha) * old_vx
        vy_smooth = self.velocity_alpha * vy_instant + (1 - self.velocity_alpha) * old_vy

        self.velocities[player_id] = (vx_smooth, vy_smooth)

        return (vx_smooth, vy_smooth)

    def _predict_position(self, player_id, frames_ahead=1):
        """
        Predict future position using constant velocity model.
        LIMITED extrapolation to avoid unrealistic predictions.

        Args:
            player_id: Tracker ID
            frames_ahead: Number of frames to predict ahead

        Returns:
            (x, y) predicted position or None
        """
        # ← DÜZELTME: last_real_positions kullan (kümülatif hata önlenir)
        if player_id not in self.last_real_positions:
            if player_id in self.baseline_positions:
                return self.baseline_positions[player_id]
            return None

        if player_id not in self.velocities:
            return self.last_real_positions[player_id]

        # Get REAL last known position (not predicted!) and velocity
        last_pos = self.last_real_positions[player_id]
        vx, vy = self.velocities[player_id]

        # Check if we have sufficient velocity history
        if player_id in self.position_history and len(self.position_history[player_id]) < self.min_velocity_history:
            return last_pos  # Use position-only matching

        # ← YENİ: Extrapolation'ı sınırla (max 10 frame ileri tahmin)
        # Daha uzun occlusion'larda velocity güvenilmez
        max_extrapolation_frames = 10
        effective_frames = min(frames_ahead, max_extrapolation_frames)

        # Predict: pos_predicted = pos_last + velocity * dt
        predicted_x = last_pos[0] + vx * effective_frames
        predicted_y = last_pos[1] + vy * effective_frames

        # ← YENİ: Max extrapolation mesafesini sınırla (300px)
        # Çok uzak tahminler güvenilmez
        max_extrapolation_distance = 300
        dx = predicted_x - last_pos[0]
        dy = predicted_y - last_pos[1]
        distance = math.sqrt(dx**2 + dy**2)

        if distance > max_extrapolation_distance:
            # Mesafeyi sınırla, yönü koru
            scale = max_extrapolation_distance / distance
            predicted_x = last_pos[0] + dx * scale
            predicted_y = last_pos[1] + dy * scale

        return (predicted_x, predicted_y)

    def _calculate_adaptive_threshold(self, player_id, frames_unseen=0):
        """
        Calculate adaptive matching threshold based on player velocity AND occlusion duration.
        Fast-moving players AND long occlusions get larger search radius.

        Args:
            player_id: Tracker ID
            frames_unseen: How many frames since player was last seen

        Returns:
            Threshold in pixels
        """
        base = self.base_threshold

        # Factor 1: Velocity-based expansion
        velocity_bonus = 0
        if player_id in self.velocities:
            vx, vy = self.velocities[player_id]
            speed = math.sqrt(vx**2 + vy**2)  # Pixels per frame
            velocity_bonus = speed * self.velocity_scale_factor

        # Factor 2: Occlusion-based expansion (YENİ!)
        # Uzun occlusion = daha geniş arama alanı
        occlusion_bonus = 0
        if frames_unseen > 0:
            # Her 10 frame için +50px threshold
            occlusion_bonus = (frames_unseen / 10) * 50
            occlusion_bonus = min(occlusion_bonus, 300)  # Max +300px occlusion bonus

        adaptive_threshold = base + velocity_bonus + occlusion_bonus

        # Clamp to max threshold
        return min(adaptive_threshold, self.max_threshold)

    def _handle_occlusions(self, current_detections):
        """
        Handle occluded baseline IDs by extrapolating their trajectory.
        Updates baseline_positions with predicted positions for missing IDs.

        Args:
            current_detections: List of current detection dicts

        Returns:
            Set of IDs that are currently occluded
        """
        # ← DÜZELTME: Baseline ID'lerini kullan (original tracker ID değil)
        current_ids = set()
        for det in current_detections:
            if det['id'] in self.tracker_to_sequential:
                current_ids.add(self.tracker_to_sequential[det['id']])
        occluded_baseline_ids = set()

        for baseline_id in self.baseline_ids:
            if baseline_id not in current_ids:
                # ID is missing - check if occluded
                if baseline_id in self.last_seen_frame:
                    occlusion_duration = self.frame_count - self.last_seen_frame[baseline_id]

                    if occlusion_duration <= self.max_occlusion_frames:
                        # Still within tracking window - extrapolate position
                        predicted_pos = self._predict_position(baseline_id, frames_ahead=occlusion_duration)

                        if predicted_pos is not None:
                            # Update baseline position to predicted position
                            self.baseline_positions[baseline_id] = predicted_pos
                            occluded_baseline_ids.add(baseline_id)

                            # Track occlusion state
                            if baseline_id not in self.occluded_ids:
                                self.occluded_ids[baseline_id] = self.frame_count
                    else:
                        # Occlusion too long - stop tracking
                        if baseline_id in self.occluded_ids:
                            del self.occluded_ids[baseline_id]
                else:
                    # No last seen frame - can't extrapolate
                    pass
            else:
                # ID is detected - clear occlusion state
                if baseline_id in self.occluded_ids:
                    occlusion_duration = self.frame_count - self.occluded_ids[baseline_id]
                    print(f"✅ ID {baseline_id} reappeared after {occlusion_duration} frames")
                    del self.occluded_ids[baseline_id]

        return occluded_baseline_ids

    def _get_profile_for_tracker(self, tracker_id: int):
        """
        Find profile currently bound to a tracker ID.

        Args:
            tracker_id: DeepSORT tracker ID

        Returns:
            GlobalProfile if found, None otherwise
        """
        for profile in self.profiles.values():
            if profile.current_tracker_id == tracker_id:
                return profile
        return None

    def _get_profile_id_for_tracker(self, tracker_id: int):
        """
        Get profile ID for a tracker ID.

        Args:
            tracker_id: DeepSORT tracker ID

        Returns:
            Profile ID (int) if found, None otherwise
        """
        profile = self._get_profile_for_tracker(tracker_id)
        return profile.profile_id if profile else None

    def _update_profile(self, profile, embedding: np.ndarray,
                       position, velocity,
                       tracker_id: int):
        """
        Update profile with new observation.

        Args:
            profile: GlobalProfile to update
            embedding: New ReID embedding (256-dim)
            position: (x, y) tuple
            velocity: (vx, vy) tuple
            tracker_id: Current tracker ID
        """
        # Update embedding history and mean
        profile.embedding_history.append(embedding)
        profile.embedding_mean = np.mean(list(profile.embedding_history), axis=0)

        # Normalize the mean
        norm = np.linalg.norm(profile.embedding_mean)
        if norm > 1e-8:
            profile.embedding_mean = profile.embedding_mean / norm

        # Update position/velocity
        profile.position_history.append((position[0], position[1], self.frame_count))
        profile.velocity = velocity

        # Update tracking metadata
        profile.current_tracker_id = tracker_id
        profile.last_seen_frame = self.frame_count
        profile.frames_since_seen = 0
        profile.total_frames_tracked += 1
        profile.confidence_score = min(1.0, profile.confidence_score + 0.01)

    def _retire_old_profiles(self):
        """
        Retire profiles that haven't been seen for too long.

        Profiles in RETIRED state are kept in memory for potential reappearance
        but no longer actively matched.
        """
        MAX_FRAMES_UNSEEN = 150  # 5 seconds @ 30fps

        for profile in self.profiles.values():
            if profile.state != ProfileState.RETIRED:
                frames_unseen = self.frame_count - profile.last_seen_frame

                if frames_unseen > MAX_FRAMES_UNSEEN:
                    profile.state = ProfileState.RETIRED
                    profile.current_tracker_id = None
                    print(f"👻 PROFILE RETIRED: Profile {profile.profile_id} (unseen for {frames_unseen} frames)")

    def get_stats(self):
        """İstatistikler (including ReID system)"""
        # Calculate average speed
        if self.velocities:
            speeds = [math.sqrt(vx**2 + vy**2) for vx, vy in self.velocities.values()]
            avg_speed = sum(speeds) / len(speeds) if speeds else 0.0
        else:
            avg_speed = 0.0

        self.stats['velocity_tracked_ids'] = len(self.velocities)
        self.stats['occlusion_events'] = len(self.occluded_ids)
        self.stats['avg_player_speed'] = avg_speed

        # ReID stats
        if self.use_reid:
            active_profiles = sum(1 for p in self.profiles.values() if p.state == ProfileState.ACTIVE)
            frozen_profiles = sum(1 for p in self.profiles.values() if p.state == ProfileState.FROZEN)
            self.stats['active_profiles'] = active_profiles
            self.stats['frozen_profiles'] = frozen_profiles

        return {
            'frame_count': self.frame_count,
            'learning_complete': self.learning_complete,
            **self.stats
        }

