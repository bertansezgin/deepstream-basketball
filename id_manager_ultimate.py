"""
Ultimate ID Manager
Voting (noise filtering) + Optimal Matching (baseline normalization)
"""

import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import Counter, deque, defaultdict

class UltimateIDManager:
    def __init__(self, 
                 learning_frames=90,
                 voting_window=7,
                 position_threshold=250):
        """
        Args:
            learning_frames: Baseline öğrenme süresi (frame)
            voting_window: Voting pencere boyutu
            position_threshold: Max pozisyon farkı (piksel)
        """
        self.learning_frames = learning_frames
        self.voting_window = voting_window
        self.position_threshold = position_threshold
        
        # VOTING STATE
        self.voting_history = {}  # {position_key: deque([id1, id2, ...])}
        self.voting_positions = {}
        
        # NORMALIZATION STATE
        self.baseline_ids = set()
        self.baseline_positions = {}
        self.virtual_to_real = {}
        
        # CONTROL
        self.frame_count = 0
        self.learning_complete = False
        
        # STATS
        self.stats = {
            'voting_corrections': 0,
            'optimal_mappings': 0,
            'multi_mapping_events': 0,
            'baseline_players': 0
        }
    
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
        """Learning: Baseline oluştur"""
        normalized = []
        
        for det in detections:
            det_id = det['id']
            pos = (det['x'] + det['w']/2, det['y'] + det['h']/2)
            
            if det_id not in self.baseline_ids:
                self.baseline_ids.add(det_id)
                self.stats['baseline_players'] = len(self.baseline_ids)
            
            self.baseline_positions[det_id] = pos
            
            normalized.append({
                **det,
                'normalized_id': det_id,
                'voted_id': det_id,
                'phase': 'learning'
            })
        
        return normalized
    
    def _finalize_learning(self):
        """Learning tamamlandı"""
        self.learning_complete = True
        print("\n" + "="*70)
        print("🎓 ÖĞRENME TAMAMLANDI!")
        print("="*70)
        print(f"📊 Baseline Oyuncular: {len(self.baseline_ids)}")
        print(f"🆔 Baseline ID'ler: {sorted(self.baseline_ids)}")
        print("="*70)
        print("🔍 Voting + Optimal Matching aktif...")
        print("="*70 + "\n")
    
    def _tracking_phase(self, detections):
        """Tracking: Optimal matching ile normalize et"""
        normalized = []
        current_ids = {det['id'] for det in detections}
        
        # Kayıp baseline ID'ler
        missing_baseline_ids = self.baseline_ids - current_ids
        
        # Yeni virtual ID'ler
        new_virtual_ids = []
        for det in detections:
            if det['id'] not in self.baseline_ids and det['id'] not in self.virtual_to_real:
                new_virtual_ids.append(det)
        
        # OPTIMAL MATCHING
        if len(new_virtual_ids) > 0 and len(missing_baseline_ids) > 0:
            mappings = self._optimal_matching(new_virtual_ids, missing_baseline_ids)
            
            for virtual_id, real_id in mappings.items():
                self.virtual_to_real[virtual_id] = real_id
                self.stats['optimal_mappings'] += 1
                print(f"�� MAPPING: Virtual {virtual_id} → Real {real_id}")
        
        # Normalize all detections
        for det in detections:
            voted_id = det['id']
            pos = (det['x'] + det['w']/2, det['y'] + det['h']/2)
            
            if voted_id in self.baseline_ids:
                normalized_id = voted_id
                self.baseline_positions[voted_id] = pos
                mapping_type = 'baseline'
            elif voted_id in self.virtual_to_real:
                normalized_id = self.virtual_to_real[voted_id]
                self.baseline_positions[normalized_id] = pos
                mapping_type = 'mapped'
            else:
                normalized_id = voted_id
                self.baseline_ids.add(voted_id)
                mapping_type = 'new_player'
                print(f"🆕 YENİ OYUNCU: ID {voted_id}")
            
            normalized.append({
                **det,
                'normalized_id': normalized_id,
                'voted_id': voted_id,
                'mapping_type': mapping_type,
                'phase': 'tracking'
            })
        
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
                    base_pos = self.baseline_positions[baseline_id]
                    distance = math.sqrt((virt_pos[0] - base_pos[0])**2 + 
                                       (virt_pos[1] - base_pos[1])**2)
                    
                    if distance < self.position_threshold:
                        cost_matrix[i, j] = distance
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        mappings = {}
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < self.position_threshold:
                virtual_id = virtual_detections[i]['id']
                real_id = missing_list[j]
                mappings[virtual_id] = real_id
                print(f"   ├─ Distance: {cost_matrix[i, j]:.1f}px")
        
        if len(mappings) > 1:
            self.stats['multi_mapping_events'] += 1
            print(f"⚡ MULTI-MAPPING: {len(mappings)} ID!")
        
        return mappings
    
    def get_stats(self):
        """İstatistikler"""
        return {
            'frame_count': self.frame_count,
            'learning_complete': self.learning_complete,
            **self.stats
        }

