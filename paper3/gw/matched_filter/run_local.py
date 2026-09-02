import sys
import numpy as np
import h5py
from gwpy.timeseries import TimeSeries

H1_PATH = "H-H1_GWOSC_4KHZ_R1-1126257415-4096.hdf5"
L1_PATH = "L-L1_GWOSC_4KHZ_R1-1126257415-4096.hdf5"
LOCAL_GPS_START = 1126257415.0
LOCAL_DURATION = 4096.0

_cache = {}


def _load_full(path):
    if path not in _cache:
        with h5py.File(path, 'r') as f:
            strain = np.asarray(f['strain']['Strain'][:], dtype=np.float64)
            gps_start = float(f['meta']['GPSstart'][()])
            duration = float(f['meta']['Duration'][()])
        sample_rate = len(strain) / duration
        _cache[path] = (strain, gps_start, sample_rate)
        print(f"[local_fetch] заредено {path}: {len(strain)} samples @ {sample_rate:.1f}Hz, "
              f"GPS start {gps_start:.1f}")
    return _cache[path]


def local_fetch_open_data(detector, start, end, sample_rate=4096, cache=False, **kwargs):
    path = H1_PATH if detector == "H1" else L1_PATH
    strain, gps_start, sr = _load_full(path)
    i0 = int(round((start - gps_start) * sr))
    i1 = int(round((end - gps_start) * sr))
    if i0 < 0 or i1 > len(strain):
        raise ValueError(f"Сегмент [{start},{end}] извън локалния файл (GPS {gps_start} + {len(strain)/sr:.0f}s)")
    seg = strain[i0:i1]
    return TimeSeries(seg, sample_rate=sr, t0=start)


TimeSeries.fetch_open_data = staticmethod(local_fetch_open_data)
print("[local_fetch] gwpy.TimeSeries.fetch_open_data пренасочен към локални файлове (без мрежа)")

# --- зареди целия injection_recovery_lambda.py БЕЗ да го оставяш да
# автоматично извика main() -- за да можем да патчнем sample_epoch()
# ПРЕДИ реалния run, ограничавайки избора на epoch само до реално
# наличния локален диапазон (segments_cache.json съдържа epochs от
# целия O1+O2, много по-широко от тази 4096s локална слика) ---
_src = open("injection_recovery_lambda.py").read()
_src = _src.replace(
    'if __name__ == "__main__":\n    main()',
    '# main() автоматичното извикване премахнато от run_local.py'
)
exec(compile(_src, "injection_recovery_lambda.py", "exec"))

_seg_half = SEGMENT_HALF  # дефинирано вътре в exec'натия скрипт
_margin = 5.0


def patched_sample_epoch(usable, rng):
    lo = LOCAL_GPS_START + _seg_half + _margin
    hi = LOCAL_GPS_START + LOCAL_DURATION - _seg_half - _margin
    return float(rng.uniform(lo, hi))


sample_epoch = patched_sample_epoch
print(f"[local_fetch] epoch sampling ограничен до [{LOCAL_GPS_START + _seg_half + _margin:.1f}, "
      f"{LOCAL_GPS_START + LOCAL_DURATION - _seg_half - _margin:.1f}]")

sys.argv = ["injection_recovery_lambda.py"] + sys.argv[1:]
main()
