
import PyInstaller.__main__

PyInstaller.__main__.run([
    'main.py',
    '--onefile',
    '--name', 'Aegis',
    '--windowed' #Use '--console' instead of '--windowed' for console application
])
