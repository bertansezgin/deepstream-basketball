"""
Hybrid ID Manager
Arkadaşınızın Slot System + Bizim Voting + Optimal Matching
"""

import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import Counter, deque

MAX_SLOTS = 10
MAX_MEMORY_DIST = 200.0
MAX_LOST_TIME = 150
VOTING_WINDOW = 7

class PlayerStats:
    """İstatistik tracking (arkadaşınızın sistemi)"""
    def __init__(self, slot_id):
        self.id = slot_id
        self.total_distance = 0.0
        self.frame_count = 0
        self.last_pos = None
        self.active_now = False

player_data = {}

class HybridIDManager:
    def __init__(self):
        # SLOT SYSTEM (Arkadaşınızdan)
        self.slots = {i: None for i in range(MAX_SLOTS)}
        
        # VOTING SYSTEM (Bizden)
        self.voting_history = {}
        self.voting_positions = {}
        
        # STATS
        self.frame_count = 0
        self.stats = {
            'voting_corrections': 0,
            'greedy_matches': 0,
            'optimal_matches': 0
        }
    
    def update(self, current_detections, frame_num):
        """Ana update fonksiyonu"""
        self.frame_count = frame_num
        
        # STEP 1: VOTING - ID stabilize et
        voted_detections = self._apply_voting(current_detections)
        
        # STEP 2: SLOT MATCHING
        return self._slot_matching(voted_detections, frame_num)
    
    def _apply_voting(self, detections):
        """Voting ile kısa vadeli hataları düzelt"""
        voted = []
        
        for det in detections:
            center = (det['x'], det['y'])
            key = self._find_voting_key(center)
            
            if key is None:
                key = center
                self.voting_history[key] = deque(maxlen=VOTING_WINDOW)
                self.voting_positions[key] = center
            
            raw_id = det['ds_id']
            self.voting_history[key].append(raw_id)
            
            # Voting
            if len(self.voting_history[key]) >= 3:
                vote_counts = Counter(self.voting_history[key])
                voted_id = vote_counts.most_common(1)[0][0]
                
                if voted_id != raw_id:
                    self.stats['voting_corrections'] += 1
            else:
                voted_id = raw_id
            
            voted.append({
                **det,
                'ds_id': voted_id
            })
        
        return voted
    
    def _find_voting_key(self, pos):
        """Voting key bul"""
        min_dist = float('inf')
        nearest = None
        
        for key in self.voting_positions:
            dist = math.sqrt((pos[0]-key[0])**2 + (pos[1]-key[1])**2)
            if dist < min_dist and dist < 150:
                min_dist = dist
                nearest = key
        
        return nearest
    
    def _slot_matching(self, voted_detections, frame_num):
        """
        Slot-based matching (arkadaşınızın mantığı)
        ARTIK OPTIMAL MATCHING ile!
        """
        unassigned = []
        assigned_slots = set()
        detection_map = {d['ds_id']: d for d in voted_detections}
        
        # 1. DIREKT MATCH
        for s_id in range(MAX_SLOTS):
            slot = self.slots[s_id]
            if slot is not None:
                ds_id = slot['ds_id']
                if ds_id in detection_map:
                    det = detection_map[ds_id]
                    self._update_slot(s_id, det, frame_num)
                    assigned_slots.add(s_id)
        
        # Unassigned detections
        for det in voted_detections:
            is_assigned = any(
                self.slots[s]['ds_id'] == det['ds_id'] 
                for s in range(MAX_SLOTS) 
                if self.slots[s] is not None
            )
            if not is_assigned:
                unassigned.append(det)
        
        # 2. POSITION-BASED MATCHING
        # YENİ: Eğer 2+ unassigned varsa → OPTIMAL MATCHING
        available_slots = [
            s for s in range(MAX_SLOTS)
            if s not in assigned_slots and self.slots[s] is not None
            and (frame_num - self.slots[s]['last_seen']) <= MAX_LOST_TIME
        ]
        
        if len(unassigned) > 1 and len(available_slots) > 1:
            # OPTIMAL MATCHING (Hungarian)
            mappings = self._optimal_slot_matching(unassigned, available_slots, frame_num)
            
            for det_idx, slot_id in mappings.items():
                det = unassigned[det_idx]
                self._update_slot(slot_id, det, frame_num)
                assigned_slots.add(slot_id)
                self.stats['optimal_matches'] += 1
            
            # Matched olanları çıkar
            unassigned = [d for i, d in enumerate(unassigned) if i not in mappings]
        
        # Kalan unassigned'lar için GREEDY
        possible_matches = []
        for i, det in enumerate(unassigned):
            for s_id in available_slots:
                if s_id in assigned_slots:
                    continue
                
                slot = self.slots[s_id]
                dist = math.sqrt((det['x'] - slot['x'])**2 + 
                               (det['y'] - slot['y'])**2)
                
                if dist < MAX_MEMORY_DIST:
                    possible_matches.append((dist, s_id, i))
        
        possible_matches.sort()
        used_indices = set()
        
        for dist, s_id, det_idx in possible_matches:
            if s_id in assigned_slots or det_idx in used_indices:
                continue
            
            det = unassigned[det_idx]
            self._update_slot(s_id, det, frame_num)
            assigned_slots.add(s_id)
            used_indices.add(det_idx)
            self.stats['greedy_matches'] += 1
        
        # 3. YENİ OYUNCU - BOŞ SLOT
        for i, det in enumerate(unassigned):
            if i in used_indices:
                continue
            
            for s_id in range(MAX_SLOTS):
                slot = self.slots[s_id]
                is_expired = (slot is not None and 
                            (frame_num - slot['last_seen']) > MAX_LOST_TIME)
                
                if s_id not in assigned_slots and (slot is None or is_expired):
                    self.slots[s_id] = {
                        'ds_id': det['ds_id'],
                        'last_seen': frame_num,
                        'x': det['x'],
                        'y': det['y']
                    }
                    assigned_slots.add(s_id)
                    break
        
        # RESULT MAP
        result_map = {}
        for s_id in range(MAX_SLOTS):
            slot = self.slots[s_id]
            if slot and (frame_num - slot['last_seen']) < 10:
                result_map[slot['ds_id']] = s_id
        
        return result_map
    
    def _optimal_slot_matching(self, detections, slot_ids, frame_num):
        """Hungarian algorithm ile optimal slot matching"""
        n_det = len(detections)
        n_slots = len(slot_ids)
        
        cost_matrix = np.full((n_det, n_slots), 999999.0)
        
        for i, det in enumerate(detections):
            for j, s_id in enumerate(slot_ids):
                slot = self.slots[s_id]
                dist = math.sqrt((det['x'] - slot['x'])**2 + 
                               (det['y'] - slot['y'])**2)
                
                if dist < MAX_MEMORY_DIST:
                    cost_matrix[i, j] = dist
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        mappings = {}
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < MAX_MEMORY_DIST:
                mappings[i] = slot_ids[j]
        
        return mappings
    
    def _update_slot(self, slot_id, detection, frame_num):
        """Slot güncelle + stats tracking"""
        if self.slots[slot_id] is None:
            self.slots[slot_id] = {}
        
        old_pos = (self.slots[slot_id].get('x', 0), 
                   self.slots[slot_id].get('y', 0))
        new_pos = (detection['x'], detection['y'])
        
        # Stats update
        if slot_id not in player_data:
            player_data[slot_id] = PlayerStats(slot_id)
        
        stats = player_data[slot_id]
        
        if stats.last_pos is not None:
            distance = math.sqrt((new_pos[0] - stats.last_pos[0])**2 + 
                               (new_pos[1] - stats.last_pos[1])**2)
            stats.total_distance += distance
        
        stats.last_pos = new_pos
        stats.frame_count += 1
        stats.active_now = True
        
        # Slot update
        self.slots[slot_id].update({
            'ds_id': detection['ds_id'],
            'last_seen': frame_num,
            'x': detection['x'],
            'y': detection['y']
        })

id_manager = HybridIDManager()
