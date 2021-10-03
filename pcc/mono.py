# Shortcut for mono ui, moving imports into main invocation to prevent 
# unnecessarily loading UI modules unless they're needed
if __name__ == '__main__':
    from .ui import mono
    mono.run_mono_calibration()
