"""
Advanced ID Normalizer - Multiple ID Loss Handling
Aynı anda birden fazla ID kaybolduğunda optimal matching
"""

import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

class AdvancedIDNormalizer:
    def __init__(self, learning_frames=90, position_threshold=250):
        """
        Args:
            learning_frames: İlk kaç frame baseline oluşturacak
            position_threshold: Max pozisyon farkı (piksel)
        """
        self.learning_frames = learning_frames
        self.position_threshold = position_threshold
        
        # Baseline ID'ler (gerçek oyuncular)
        self.baseline_ids = set()
        self.baseline_positions = {}  # {real_id: (x, y)}
        
        # Aktif mapping
        self.virtual_to_real = {}  # {virtual_id: real_id}
        
        # İstatistikler
        self.frame_count = 0
        self.learning_complete = False
        self.total_mappings = 0
        self.stats = defaultdict(int)
        
    def update(self, detections):
        """Ana fonksiyon: Her frame'de ID normalizasyonu yap"""
        self.frame_count += 1
        
        # PHASE 1: LEARNING
        if self.frame_count <= self.learning_frames:
            return self._learning_phase(detections)
        
        # İlk kez tracking'e geçiş
        if not self.learning_complete:
            self._finalize_learning()
        
        # PHASE 2: TRACKING
        return self._tracking_phase(detections)
    
    def _learning_phase(self, detections):
        """Öğrenme fazı: Tüm görülen ID'leri baseline'a ekle"""
        normalized = []
        
        for det in detections:
            det_id = det['id']
            pos = (det['x'] + det['w']/2, det['y'] + det['h']/2)
            
            if det_id not in self.baseline_ids:
                self.baseline_ids.add(det_id)
                self.stats['baseline_players'] += 1
            
            self.baseline_positions[det_id] = pos
            
            normalized.append({
                **det,
                'normalized_id': det_id,
                'virtual_id': det_id,
                'is_baseline': True,
                'mapping_type': 'baseline'
            })
        
        return normalized
    
    def _finalize_learning(self):
        """Öğrenme fazı bitti"""
        self.learning_complete = True
        print("\n" + "="*70)
        print("🎓 ÖĞRENME TAMAMLANDI!")
        print("="*70)
        print(f"📊 Baseline Oyuncular: {len(self.baseline_ids)}")
        print(f"🆔 Baseline ID'ler: {sorted(self.baseline_ids)}")
        print("="*70)
        print("🔍 Tracking fazına geçildi...")
        print("="*70 + "\n")
    
    def _tracking_phase(self, detections):
        """Tracking fazı: Yeni ID'leri baseline'a map'le"""
        normalized = []
        current_ids = {det['id'] for det in detections}
        
        # Hangi baseline ID'ler kayıp?
        missing_baseline_ids = self.baseline_ids - current_ids
        
        # Hangi ID'ler yeni (virtual)?
        new_virtual_ids = []
        for det in detections:
            if det['id'] not in self.baseline_ids and det['id'] not in self.virtual_to_real:
                new_virtual_ids.append(det)
        
        # ÇOK ÖNEMLİ: Birden fazla yeni ID varsa OPTIMAL MATCHING
        if len(new_virtual_ids) > 0 and len(missing_baseline_ids) > 0:
            mappings = self._optimal_matching(new_virtual_ids, missing_baseline_ids)
            
            # Yeni mappings'i kaydet
            for virtual_id, real_id in mappings.items():
                self.virtual_to_real[virtual_id] = real_id
                self.total_mappings += 1
                self.stats['successful_mappings'] += 1
                print(f"🔄 OPTIMAL MAPPING: Virtual ID {virtual_id} → Real ID {real_id}")
        
        # Şimdi tüm detections'ları normalize et
        for det in detections:
            virtual_id = det['id']
            pos = (det['x'] + det['w']/2, det['y'] + det['h']/2)
            
            # CASE 1: Baseline ID
            if virtual_id in self.baseline_ids:
                normalized_id = virtual_id
                self.baseline_positions[virtual_id] = pos
                mapping_type = 'baseline'
                is_virtual = False
            
            # CASE 2: Mapped virtual ID
            elif virtual_id in self.virtual_to_real:
                normalized_id = self.virtual_to_real[virtual_id]
                self.baseline_positions[normalized_id] = pos
                mapping_type = 'mapped'
                is_virtual = True
            
            # CASE 3: Gerçekten yeni oyuncu (mapping bulunamadı)
            else:
                normalized_id = virtual_id
                self.baseline_ids.add(virtual_id)
                self.stats['new_baseline_players'] += 1
                mapping_type = 'new_player'
                is_virtual = False
                print(f"🆕 YENİ OYUNCU: ID {virtual_id}")
            
            normalized.append({
                **det,
                'normalized_id': normalized_id,
                'virtual_id': virtual_id,
                'is_virtual': is_virtual,
                'mapping_type': mapping_type
            })
        
        return normalized
    
    def _optimal_matching(self, virtual_detections, missing_baseline_ids):
        """
        Hungarian Algorithm ile optimal matching
        
        Args:
            virtual_detections: Yeni gelen virtual ID'lerin detection'ları
            missing_baseline_ids: Kayıp baseline ID'ler
        
        Returns:
            {virtual_id: real_id, ...}
        """
        if not virtual_detections or not missing_baseline_ids:
            return {}
        
        # Cost matrix oluştur (mesafe bazlı)
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
                    
                    # Eşik dahilindeyse cost matrix'e ekle
                    if distance < self.position_threshold:
                        cost_matrix[i, j] = distance
        
        # Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Mapping oluştur (sadece valid olanlar)
        mappings = {}
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < self.position_threshold:
                virtual_id = virtual_detections[i]['id']
                real_id = missing_list[j]
                mappings[virtual_id] = real_id
                
                # Debug log
                dist = cost_matrix[i, j]
                self.stats[f'mapping_distance_{virtual_id}'] = dist
                print(f"   ├─ Virtual {virtual_id} ↔ Real {real_id} (distance: {dist:.1f}px)")
        
        if len(mappings) > 1:
            self.stats['multi_mapping_events'] += 1
            print(f"⚡ MULTI-MAPPING: {len(mappings)} ID aynı anda eşleştirildi!")
        
        return mappings
    
    def get_stats(self):
        """İstatistikleri döndür"""
        return {
            'frame_count': self.frame_count,
            'learning_complete': self.learning_complete,
            'baseline_players': len(self.baseline_ids),
            'total_mappings': self.total_mappings,
            'virtual_ids_count': len(self.virtual_to_real),
            **self.stats
        }
    
    def get_baseline_ids(self):
        """Baseline ID'leri döndür"""
        return sorted(self.baseline_ids)
    
    def get_mapping_table(self):
        """Virtual → Real mapping tablosu"""
        return self.virtual_to_real.copy()

