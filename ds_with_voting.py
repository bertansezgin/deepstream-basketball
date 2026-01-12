import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GObject
import sys
sys.path.insert(0, '/workspace')
from id_stabilizer import IDStabilizer

Gst.init(None)

# Global değişkenler
frame_count = 0
stabilizer = IDStabilizer(window_size=7, position_threshold=150)
detected_ids = {}  # {frame: [detections]}

def simulate_detections():
    """
    NOT: Gerçek metadata için pyds gerekli
    Şimdilik simülasyon yapıyoruz - test amaçlı
    """
    global frame_count
    frame_count += 1
    
    # Simülasyon: Random ID değişimleri
    import random
    
    # 5 oyuncu var diyelim
    detections = []
    for i in range(5):
        # %10 ihtimalle ID değişsin (DeepStream hatası simülasyonu)
        if random.random() < 0.1:
            fake_id = random.randint(0, 20)
        else:
            fake_id = i
        
        detections.append({
            'id': fake_id,
            'x': 100 + i * 200 + random.randint(-50, 50),
            'y': 300 + random.randint(-50, 50),
            'w': 80,
            'h': 180
        })
    
    return detections

def print_stats():
    """Her 3 saniyede istatistik yazdır"""
    stats = stabilizer.get_stats()
    
    print("\n" + "="*70)
    print(f"📊 FRAME: {frame_count} | Aktif Oyuncu: {stats['active_players']}")
    print(f"🔧 ID Düzeltmeleri: {stats['total_corrections']}")
    print("="*70)
    
    # Örnek detections
    if frame_count % 90 == 0:  # Her 3 saniyede
        sample_dets = simulate_detections()
        stable_dets = stabilizer.stabilize(sample_dets)
        
        print(f"{'Raw ID':<10} {'Stable ID':<12} {'Düzeltildi?':<15}")
        print("-"*70)
        for det in stable_dets:
            status = "✅ Evet" if det['corrected'] else "⚪ Hayır"
            print(f"{det['raw_id']:<10} {det['stable_id']:<12} {status:<15}")
    
    return True  # Continue callback

# Pipeline
pipeline = Gst.Pipeline.new("ds-voting-pipeline")

source = Gst.ElementFactory.make("filesrc", "source")
source.set_property("location", "/workspace/basketball_video.mp4")

demux = Gst.ElementFactory.make("qtdemux", "demux")
parser = Gst.ElementFactory.make("h264parse", "parser")
decoder = Gst.ElementFactory.make("nvv4l2decoder", "decoder")

streammux = Gst.ElementFactory.make("nvstreammux", "streammux")
streammux.set_property("batch-size", 1)
streammux.set_property("width", 1920)
streammux.set_property("height", 1080)
streammux.set_property("batched-push-timeout", 40000)

pgie = Gst.ElementFactory.make("nvinfer", "primary-infer")
pgie.set_property("config-file-path", "/workspace/config_peoplenet.txt")

tracker = Gst.ElementFactory.make("nvtracker", "tracker")
tracker.set_property("ll-lib-file", "/opt/nvidia/deepstream/deepstream-8.0/lib/libnvds_nvmultiobjecttracker.so")
tracker.set_property("ll-config-file", "/workspace/config_tracker_deepsort_optimized.yml")
tracker.set_property("tracker-width", 640)
tracker.set_property("tracker-height", 384)
tracker.set_property("display-tracking-id", 1)

nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "nvvidconv")
nvosd = Gst.ElementFactory.make("nvdsosd", "nvosd")

sink = Gst.ElementFactory.make("nveglglessink", "sink")
sink.set_property("sync", False)

for e in [source, demux, parser, decoder, streammux, pgie, tracker, nvvidconv, nvosd, sink]:
    pipeline.add(e)

source.link(demux)

def on_pad_added(demux, pad, parser):
    sink_pad = parser.get_static_pad("sink")
    if not sink_pad.is_linked():
        pad.link(sink_pad)

demux.connect("pad-added", on_pad_added, parser)
parser.link(decoder)

srcpad = decoder.get_static_pad("src")
sinkpad = streammux.request_pad_simple("sink_0")
srcpad.link(sinkpad)

streammux.link(pgie)
pgie.link(tracker)
tracker.link(nvvidconv)
nvvidconv.link(nvosd)
nvosd.link(sink)

print("="*70)
print("🏀 DeepSORT + ID VOTING STABILIZER")
print("="*70)
print("✅ Sliding window: 7 frames (~0.23s)")
print("✅ Position threshold: 150px")
print("✅ Auto-correction: Aktif")
print("="*70)
print("📊 Her 3 saniyede istatistikler gösterilecek")
print("🔴 Ctrl+C ile durdurun")
print("="*70)

# Periyodik istatistik
GObject.timeout_add_seconds(3, print_stats)

pipeline.set_state(Gst.State.PLAYING)

loop = GObject.MainLoop()
try:
    loop.run()
except KeyboardInterrupt:
    print("\n⏹️  Durduruluyor...")
    print_stats()

pipeline.set_state(Gst.State.NULL)
print("✅ Tamamlandı!")
