import sys
import os
import site

print("--- Python Debug Info ---")
print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")
print(f"Current Working Directory: {os.getcwd()}")

print("\nsys.path:")
for path in sys.path:
    print(f"  - {path}")

print("\nsite-packages:")
for path in site.getsitepackages():
    print(f"  - {path}")

print("--- End Debug Info ---\n")

try:
    print("Attempting import: from tunix.rl.agentic.agents import model_agent")
    from tunix.rl.agentic.agents import model_agent
    print("SUCCESS: Imported model_agent")
except ModuleNotFoundError as e:
    print(f"FAILED to import: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    import traceback
    traceback.print_exc()