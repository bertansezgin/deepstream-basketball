"""
ID Stabilizer - Sliding Window Voting
DeepStream'den gelen ID'lerdeki kısa süreli hataları düzeltir
"""

from collections import Counter, deque
import math

class IDStabilizer:
    def __init__(self, window_size=7, position_threshold=150):
        """
        Args:
            window_size: Kaç frame geriye bakacak (7 = ~0.23 saniye @ 30fps)
            position_threshold: Max hareket mesafesi (piksel)
        """
        self.window_size = window_size
        self.position_threshold = position_threshold
        self.history = {}  # {key: deque([id1, id2, ...])}
        self.positions = {}  # {key: (x, y)}
        self.stats = {
            'total_corrections': 0,
            'active_players': 0
        }
        
    def stabilize(self, detections):
        """
        Ana fonksiyon: Her detection için stabil ID döndür
        
        Args:
            detections: [{'id': int, 'x': float, 'y': float, 'w': float, 'h': float}, ...]
        
        Returns:
            [{'id': int, 'stable_id': int, 'raw_id': int, ...}, ...]
        """
        stable_detections = []
        current_keys = set()
        
        for det in detections:
            # Merkez pozisyon hesapla
            center_x = det['x'] + det['w'] / 2
            center_y = det['y'] + det['h'] / 2
            center = (center_x, center_y)
            
            # En yakın mevcut key'i bul
            key = self._find_nearest_key(center)
            
            if key is None:
                # Yeni oyuncu - yeni key oluştur
                key = center
                self.history[key] = deque(maxlen=self.window_size)
                self.positions[key] = center
            else:
                # Pozisyonu güncelle
                self.positions[key] = center
            
            current_keys.add(key)
            
            # ID'yi history'e ekle
            raw_id = det['id']
            self.history[key].append(raw_id)
            
            # Voting ile stabil ID hesapla
            if len(self.history[key]) >= 3:  # En az 3 frame gerekli
                vote_counts = Counter(self.history[key])
                stable_id = vote_counts.most_common(1)[0][0]
                
                # ID değişti mi?
                if stable_id != raw_id:
                    self.stats['total_corrections'] += 1
            else:
                # Yeterli veri yok, raw ID kullan
                stable_id = raw_id
            
            # Sonuç detection'ı oluştur
            stable_det = {
                **det,
                'stable_id': stable_id,
                'raw_id': raw_id,
                'corrected': stable_id != raw_id
            }
            stable_detections.append(stable_det)
        
        # Eski key'leri temizle (görünmeyen oyuncular)
        self._cleanup_old_keys(current_keys)
        
        # İstatistikleri güncelle
        self.stats['active_players'] = len(self.history)
        
        return stable_detections
    
    def _find_nearest_key(self, pos, threshold=None):
        """En yakın pozisyondaki key'i bul"""
        if threshold is None:
            threshold = self.position_threshold
            
        min_dist = float('inf')
        nearest = None
        
        for key in self.positions:
            dist = math.sqrt((pos[0] - key[0])**2 + (pos[1] - key[1])**2)
            if dist < min_dist and dist < threshold:
                min_dist = dist
                nearest = key
        
        return nearest
    
    def _cleanup_old_keys(self, current_keys):
        """Kullanılmayan key'leri temizle"""
        keys_to_remove = []
        for key in self.history:
            if key not in current_keys:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.history[key]
            if key in self.positions:
                del self.positions[key]
    
    def get_stats(self):
        """İstatistikleri döndür"""
        return self.stats.copy()
    
    def reset(self):
        """Tüm history'i temizle"""
        self.history.clear()
        self.positions.clear()
        self.stats = {
            'total_corrections': 0,
            'active_players': 0
        }

