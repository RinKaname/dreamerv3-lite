import sys
import imageio
import crafter.run_gui

# Store original mimsave
original_mimsave = imageio.mimsave

def safe_mimsave(uri, ims, **kwargs):
    # If the Crafter recorder tries to save an mp4, swap it to a GIF!
    if isinstance(uri, str) and uri.endswith('.mp4'):
        uri = uri.replace('.mp4', '.gif')
        # Force a safe frame rate for gifs
        if 'fps' in kwargs:
            kwargs['fps'] = 20
        print(f"Intercepted video save! Redirecting to {uri} to bypass PyAV codec bug...")
    return original_mimsave(uri, ims, **kwargs)

# Monkey-patch imageio globally to fix the Windows PyAV bug
imageio.mimsave = safe_mimsave

if __name__ == "__main__":
    # Launch the official Crafter GUI with our hotfix applied
    crafter.run_gui.main()
